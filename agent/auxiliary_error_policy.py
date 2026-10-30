"""What the auxiliary client does with a failed provider call, read from the shared classifier.

``classify_api_error`` (agent/error_classifier.py) is the one reader of an API error's status, code and
wording, and plugin ``transform_api_error_classification`` hooks and ``ProviderProfile.classify_api_error``
steer it. The auxiliary recovery ladder asks this module what to do: retry the same provider, refresh or
rotate the credential, bench the provider, or fall back. Every answer comes from one ``ClassifiedError``
per exception and route, classified against the physical attempt that raised it: the backend, model,
endpoint and credential of the client actually sent (``attempt`` at the transport seam, else the
enclosing ``error_route``). The verdict is cached on the exception, keyed by that route identity (a
credential fingerprint, never the secret), so the ladder's rungs and the telemetry scope agree.

Named reasons keep the fallback labels the logs, the unhealthy marks and the candidate quarantine TTLs
(agent/auxiliary_health.py) have always used; any other fallback reason is labelled by its enum value.
"""

from __future__ import annotations

import contextlib
import hashlib
from contextvars import ContextVar
from typing import Any, Iterator, NamedTuple, Optional

import httpx

from agent.error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_api_error,
    is_max_tokens_rejection,
    is_structured_output_rejection,
    is_unsupported_parameter_error,
    rejects_reasoning_disable,
    rejects_reasoning_field,
)

_R = FailoverReason

AUTH_ERROR = "auth error"
PAYMENT_ERROR = "payment error"
RATE_LIMIT = "rate limit"
MODEL_NOT_FOUND = "model not found"
MODEL_INCOMPATIBLE = "model incompatible with route"
INVALID_RESPONSE = "invalid provider response"
STRUCTURED_ERROR = "structured provider error"
TIMED_OUT = "request timed out"
CONNECTION_ERROR = "connection error"
PROVIDER_OVERLOADED = "provider overloaded"
SERVER_ERROR = "server error"

# Same provider, same request, a moment later: a blip the provider recovers from by itself.
_TRANSIENT = frozenset({_R.timeout, _R.overloaded, _R.server_error})

# Deterministic for THIS request on any route: the caller's request shape, not the provider, failed.
# The ladder falls back on one only after a parameter rung changed the request and it still failed
# (main-loop parity: the shrink/reasoning "spent" fallbacks in agent/turn_api_error.py).
_REQUEST_SHAPE = frozenset({
    _R.format_error, _R.role_alternation, _R.multimodal_tool_content_unsupported, _R.reasoning_mandatory,
    _R.llama_cpp_grammar_pattern, _R.image_too_large, _R.image_corrupt, _R.invalid_encrypted_content,
    _R.thinking_signature, _R.data_policy_denied, _R.context_overflow, _R.payload_too_large,
    _R.oauth_long_context_beta_forbidden,
})

_LABELS = {
    _R.auth: AUTH_ERROR, _R.auth_permanent: AUTH_ERROR, _R.billing: PAYMENT_ERROR,
    _R.rate_limit: RATE_LIMIT, _R.upstream_rate_limit: RATE_LIMIT,
    _R.model_not_found: MODEL_NOT_FOUND, _R.model_entitlement: MODEL_INCOMPATIBLE,
    _R.incomplete_response: INVALID_RESPONSE, _R.overloaded: PROVIDER_OVERLOADED, _R.server_error: SERVER_ERROR,
    # The main loop compresses into the standard tier; an auxiliary request cannot, so it moves on.
    _R.long_context_tier: "long context tier",
}

# Raises from a stream negotiation that are the provider's failure, not streaming's: re-raised as-is
# rather than retried non-streaming.
_PROVIDER_REFUSALS = frozenset({_R.auth, _R.auth_permanent, _R.billing, _R.rate_limit, _R.upstream_rate_limit})

# Tasks on a user-visible critical path (compression blocks resuming an oversized session; vision
# stalls the serialised turn queue). A same-provider retry after a full-budget timeout costs another
# whole ``timeout`` window, so they skip straight to fallback; fast blips (a dropped stream, a 5xx)
# still retry (#54465). Title generation joins them because its retries multiplied the user's
# ``auxiliary.title_generation.timeout`` (~4x) on a slow local model (#89445, #66251).
TIMEOUT_NO_RETRY_TASKS = frozenset({"compression", "vision", "title_generation"})

_VERDICTS_ATTR = "_hermes_aux_verdicts"
_ATTEMPT_ATTR = "_hermes_aux_attempt_route"
_RUNG_SPENT_ATTR = "_hermes_aux_rung_spent"


class _Route(NamedTuple):
    provider: str
    model: str
    base_url: str
    api_key: Optional[str]

    def __repr__(self) -> str:  # never render the credential
        return f"_Route(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"

    def identity(self) -> tuple[str, str, str, str]:
        """The verdict cache key: backend, model, normalized endpoint and a credential fingerprint."""
        key = hashlib.sha256(self.api_key.encode()).hexdigest()[:16] if self.api_key else ""
        return self.provider, self.model, self.base_url.strip().rstrip("/").lower(), key


def _route(provider: Optional[str], model: Optional[str], base_url: Any, api_key: Any) -> _Route:
    # Only a literal key: a callable credential provider is never invoked just to classify an error.
    key = api_key if isinstance(api_key, str) and api_key else None
    return _Route(_concrete(provider), str(model or ""), str(base_url or ""), key)


_UNBOUND = _Route("", "", "", None)
_ROUTE: ContextVar[Optional[_Route]] = ContextVar("auxiliary_error_route", default=None)


def _concrete(provider: Optional[str]) -> str:
    """The provider slug the classifier's profile hook and provider-native code maps key on; the
    routing label ``auto`` names no backend."""
    slug = str(provider or "").strip().lower()
    return "" if slug == "auto" else slug


@contextlib.contextmanager
def error_route(
    provider: Optional[str], model: Optional[str] = None, base_url: Optional[str] = None, api_key: Any = None,
) -> Iterator[None]:
    """Classify failures raised inside this block against this backend."""
    token = _ROUTE.set(_route(provider, model, base_url, api_key))
    try:
        yield
    finally:
        _ROUTE.reset(token)


def _setattr(exc: BaseException, name: str, value: Any) -> None:
    with contextlib.suppress(AttributeError, TypeError):
        setattr(exc, name, value)


def _client_base_url(client: Any) -> str:
    url = getattr(client, "base_url", None)  # str or an SDK ``httpx.URL``; never a mock's auto-attribute
    return str(url) if isinstance(url, (str, httpx.URL)) else ""


@contextlib.contextmanager
def attempt(client: Any, provider: Optional[str], model: Any) -> Iterator[None]:
    """One physical provider request: everything classified inside (the stream-negotiation check, the
    affordable-token retry) and a failure raised out of it is bound to the backend, endpoint and
    credential ``client`` actually used (a refreshed or rotated client, a fallback candidate), not the
    enclosing route's defaults. The first binding wins, so the innermost attempt names the exception."""
    ambient = _ROUTE.get() or _UNBOUND
    # ``_route`` keeps only a literal string key (a callable credential provider or a mock's attribute
    # falls back to the route's key).
    bound = _route(_concrete(provider) or ambient.provider, model if isinstance(model, str) and model else ambient.model,
                   _client_base_url(client) or ambient.base_url, getattr(client, "api_key", None))
    token = _ROUTE.set(bound if bound.api_key else bound._replace(api_key=ambient.api_key))
    try:
        yield
    except Exception as exc:
        if getattr(exc, _ATTEMPT_ATTR, None) is None:
            _setattr(exc, _ATTEMPT_ATTR, _ROUTE.get())
        raise
    finally:
        _ROUTE.reset(token)


def verdict(exc: Exception, *, provider: Optional[str] = None) -> ClassifiedError:
    """The classifier's verdict for ``exc`` on the route of the attempt that raised it (else the enclosing
    route; ``provider`` overrides the backend when the caller knows better, e.g. the credential pool).
    Classified once per exception and route identity; the plugin and provider-profile hooks run inside
    ``classify_api_error``."""
    route = getattr(exc, _ATTEMPT_ATTR, None)
    if route is None:
        route = _ROUTE.get()
        if route is not None:  # the first read inside a route binds the exception to it
            _setattr(exc, _ATTEMPT_ATTR, route)
    route = route or _UNBOUND
    if _concrete(provider):
        route = route._replace(provider=_concrete(provider))
    cache = getattr(exc, _VERDICTS_ATTR, None)
    key = route.identity()
    if isinstance(cache, dict) and key in cache:
        return cache[key]
    classified = classify_api_error(
        exc, provider=route.provider, model=route.model, base_url=route.base_url, api_key=route.api_key)
    if not isinstance(cache, dict):
        cache = {}
        _setattr(exc, _VERDICTS_ATTR, cache)
    cache[key] = classified
    return classified


def same_attempt(copy: Exception, original: Exception) -> Exception:
    """``copy`` (a reshaped ``original``) classifies on the attempt route ``original`` was raised on."""
    _setattr(copy, _ATTEMPT_ATTR, getattr(original, _ATTEMPT_ATTR, None))
    return copy


def reported_verdict(exc: Exception, *, provider: str = "", model: str = "") -> ClassifiedError:
    """The verdict telemetry reports for one attempt error: the ladder's own for that attempt, so the
    hooks run once per exception and route. An error no route saw classifies against the reporting
    scope's provider and model."""
    if getattr(exc, _ATTEMPT_ATTR, None) is None and _ROUTE.get() is None:
        with error_route(provider, model):
            return verdict(exc)
    return verdict(exc)


def note_rung_spent(exc: Exception) -> None:
    """A parameter rung already changed the request ``exc`` failed on (a fallback candidate's own repair)."""
    _setattr(exc, _RUNG_SPENT_ATTR, True)


def noted_rung_spent(exc: Exception) -> bool:
    return getattr(exc, _RUNG_SPENT_ATTR, False) is True


def transport(exc: Exception) -> Optional[str]:
    """``timeout`` | ``connection`` for a failure below the HTTP layer (or a stream that ran out of
    time), None for a provider answer."""
    return verdict(exc).error_context.get("transport")


def terminal_network(exc: Exception) -> bool:
    """No provider answer can come back on this network path: a transport failure, or a TLS certificate
    the client rejected (which skips the same-provider retry but is no provider's verdict either)."""
    return transport(exc) is not None or verdict(exc).reason is _R.ssl_cert_verification


def same_provider_retry(exc: Exception) -> bool:
    """A transient blip worth re-sending to the same provider: timeout, connection, overload, 5xx, unless
    the verdict (e.g. a plugin or provider-profile hook) marks it not retryable, as the main loop honours."""
    classified = verdict(exc)
    return classified.reason in _TRANSIENT and classified.retryable


def skip_same_provider_retry(task: Optional[str], exc: Exception) -> bool:
    """A critical-path task that spent a whole time budget skips the same-provider retry. A zero-output
    first-token failure (``AuxNoProgressTimeout``) is cheap and keeps it."""
    from agent.auxiliary_stream_watchdog import AuxNoProgressTimeout

    return (task in TIMEOUT_NO_RETRY_TASKS and not isinstance(exc, AuxNoProgressTimeout)
            and transport(exc) == "timeout")


def auth_refresh(exc: Exception, *, provider: Optional[str] = None) -> bool:
    """A credential the provider rejected as such (401-class): refresh it, then rotate the pool. A
    generic 403 is not one; it falls back without touching the credential."""
    classified = verdict(exc, provider=provider)
    return classified.reason is _R.auth and classified.should_rotate_credential


def rotates_credential(exc: Exception, *, provider: Optional[str] = None) -> bool:
    """Another key in the pool may serve this request (dead key, spent credit, key-level throttle)."""
    return verdict(exc, provider=provider).should_rotate_credential


def same_key_retry(exc: Exception) -> bool:
    """A per-key throttle that clears within seconds: wait and re-send on the same key first. Not a
    spent quota window and not an upstream model throttle (the key is healthy, so fall back)."""
    classified = verdict(exc)
    return classified.reason is _R.rate_limit and not classified.error_context.get("quota_exhausted")


def benches_provider(exc: Exception, *, provider: Optional[str] = None) -> bool:
    """Credit or a long quota window is spent: mark the provider unhealthy so later aux calls skip it."""
    classified = verdict(exc, provider=provider)
    return classified.reason is _R.billing or bool(classified.error_context.get("quota_exhausted"))


def model_not_found(exc: Exception) -> bool:
    return verdict(exc).reason is _R.model_not_found


def fallback_label(exc: Exception, *, rung_spent: bool = False) -> Optional[str]:
    """Why another provider should take this request (the label logs and quarantine TTLs key on), or None
    when the failure is the request's own and would follow it to any provider.

    Transient reasons are included: the ladder only runs once the same-provider retries are spent.
    ``rung_spent``: a parameter rung already changed the request and it still failed."""
    classified = verdict(exc)
    reason, context = classified.reason, classified.error_context
    if benches_provider(exc):
        return PAYMENT_ERROR
    if reason is _R.timeout:
        return CONNECTION_ERROR if context.get("transport") == "connection" else TIMED_OUT
    if reason in _LABELS:
        return _LABELS[reason]
    if context.get("route_rejects_model"):
        return MODEL_INCOMPATIBLE
    if context.get("rejects_parameter"):  # status-less too: the parameter follows the request anywhere
        return MODEL_INCOMPATIBLE if rung_spent else None
    if reason in _REQUEST_SHAPE:
        return MODEL_INCOMPATIBLE if rung_spent else (STRUCTURED_ERROR if context.get("statusless_structured") else None)
    if classified.should_fallback:
        return reason.value.replace("_", " ")
    return STRUCTURED_ERROR if context.get("statusless_structured") else None


def ladder_accepts(exc: Exception, *, rung_spent: bool = False) -> bool:
    """A later recovery rung (credential rotation, provider fallback) can act on ``exc``; anything else
    re-raises to the caller."""
    return rotates_credential(exc) or fallback_label(exc, rung_spent=rung_spent) is not None


def param_rung_accepts(exc: Exception) -> bool:
    """After a parameter-strip retry: fall through to the next rung for another rejected field (rungs
    chain in any order: a reasoning-strip retry can 400 on temperature, a temperature-strip retry on
    max_tokens) or for anything the credential/fallback rungs handle, including a request-shape
    failure now that the request was changed once."""
    return (ladder_accepts(exc, rung_spent=True) or is_max_tokens_rejection(exc)
            or is_unsupported_parameter_error(exc, "temperature") or rejects_reasoning_field(exc)
            or rejects_reasoning_disable(exc) or is_structured_output_rejection(exc))


def stream_negotiation_reraises(exc: Exception) -> bool:
    """A failed stream negotiation that is the provider's answer, not a streaming-specific rejection:
    re-raise it unchanged instead of re-sending the request non-streaming (a 429 or a welcome-tier
    refusal included, whatever its reason). A timeout (the stream watchdog's own included) is transient
    by construction and skips the classification."""
    if isinstance(exc, TimeoutError):
        return True
    classified = verdict(exc)
    return (classified.reason in _TRANSIENT or classified.reason in _PROVIDER_REFUSALS
            or classified.status_code == 429 or "welcome_refusal" in classified.error_context
            or bool(classified.error_context.get("quota_exhausted")))
