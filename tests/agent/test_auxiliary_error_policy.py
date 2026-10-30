"""Auxiliary recovery decisions come from the shared classifier (the Code Health plan).

The aux ladder used to decide retry/fallback/refresh with private substring predicates and consulted
``classify_api_error`` only for telemetry, so a plugin or provider-profile classifier never steered an
aux decision and the two drifted (an OpenRouter budget-limit 403 was billing for the main loop but not
for the aux ladder). These tests drive ``call_llm`` end to end with a scripted client; each
``behaviour change`` case fails on the predicate implementation it replaced.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from agent import auxiliary_error_policy as aux_errors
from agent.auxiliary_client import async_call_llm, call_llm
from agent.error_classifier import FailoverReason
from agent.errors import InvalidProviderResponseError

_REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
_MESSAGES = [{"role": "user", "content": "hi"}]
_SDK_BY_STATUS = {400: openai.BadRequestError, 401: openai.AuthenticationError, 403: openai.PermissionDeniedError,
                  404: openai.NotFoundError, 429: openai.RateLimitError}


def sdk(status, message, *, wrapped=False, **error_fields):
    """An error shaped like openai-python's ``_make_status_error`` output (``wrapped``: a relay that
    keeps the ``{"error": ...}`` envelope on the exception body)."""
    body = {"message": message, **error_fields}
    if wrapped:
        body = {"error": body}
    cls = _SDK_BY_STATUS.get(status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return cls(f"Error code: {status} - {{'error': {body!r}}}", response=httpx.Response(status, request=_REQ), body=body)


class _Plain(Exception):
    pass


def _ok(text="fallback"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def run_ladder(error, *, provider="openrouter", task=None, retries=1, client=None, resolved_key=None,
               is_async=False, **call_kw):
    """``call_llm`` (``async_call_llm`` when ``is_async``) on a client whose every request raises ``error``
    (a list: one outcome per request); returns what the ladder did. ``client`` overrides the primary
    client's attributes; ``call_kw`` (temperature, max_tokens) goes to the call."""
    create = AsyncMock if is_async else MagicMock
    primary = MagicMock()
    primary.base_url = "https://openrouter.ai/api/v1"
    primary.api_key = "sk-primary"
    for name, value in (client or {}).items():
        setattr(primary, name, value)
    primary.chat.completions.create = create(side_effect=error)
    fallback = MagicMock()
    fallback.base_url = "https://api.example.com/v1"
    fallback.chat.completions.create = create(return_value=_ok())
    marks, rotations = [], []

    def _recover(pool_provider, exc, **_kw):
        rotations.append(exc)
        return False

    resolved = (provider, "some-model", None, resolved_key, None)
    refreshes = []
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=resolved),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "some-model")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._transient_retry_count", return_value=retries),
        patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value="openrouter"),
        patch("agent.auxiliary_client._recover_provider_pool", side_effect=_recover),
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **kw: marks.append(a)),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=(fallback, "fb-model", "fallback")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb-model", "fallback")),
        patch("agent.auxiliary_rate_limit_backoff.rate_limit_wait", return_value=0.0),
        patch("agent.auxiliary_client._refresh_provider_credentials",
              side_effect=lambda slug, **_kw: refreshes.append(slug) or True),
    ):
        try:
            if is_async:
                result = asyncio.run(async_call_llm(task=task, messages=_MESSAGES, **call_kw))
            else:
                result = call_llm(task=task, messages=_MESSAGES, **call_kw)
        except Exception as exc:  # the outcome under test
            result = exc
    return SimpleNamespace(
        attempts=primary.chat.completions.create.call_count,
        fell_back=fallback.chat.completions.create.call_count > 0,
        unhealthy=bool(marks), rotated=bool(rotations), refreshed=refreshes, result=result)


# ── accepted behaviour changes (each fails on the replaced predicates) ──────
# (case id, error, route provider, expected outcome)
BEHAVIOUR_CHANGES = [
    # DNS / refused connections are transient: same-provider retry, then fallback.
    ("dns-refused", _Plain("[Errno 8] nodename nor servname provided, or not known"), "openrouter",
     dict(attempts=2, fell_back=True)),
    # 5xx, 503/529 overload and 408 fall back once the retries are spent.
    ("http-500", sdk(500, "Internal Server Error"), "openrouter", dict(attempts=2, fell_back=True)),
    ("http-503", sdk(503, "Service Unavailable"), "openrouter", dict(attempts=2, fell_back=True)),
    ("http-408", sdk(408, "Request Timeout"), "openrouter", dict(attempts=2, fell_back=True)),
    # a 400-coded billing body benches the provider and falls back.
    ("400-billing-body", sdk(400, "Insufficient credits. Add more using https://openrouter.ai/credits"), "openrouter",
     dict(attempts=1, fell_back=True, unhealthy=True, rotated=True)),
    # a missing model falls back on every route, not only Nous.
    ("404-model-not-found", sdk(404, "model_not_found: The model 'x' does not exist"), "openrouter", dict(attempts=1, fell_back=True)),
    ("404-no-tool-endpoints", sdk(404, "No endpoints found that support tool use."), "openrouter", dict(attempts=1, fell_back=True)),
    # a generic 403 on an auto route falls back (no credential refresh).
    ("403-generic-auto", sdk(403, "Forbidden"), "auto", dict(attempts=1, fell_back=True, rotated=False)),
    # a 403 upstream_unavailable is an overload: retried, then fallback.
    ("403-upstream-unavailable", sdk(403, "upstream down", code="upstream_unavailable"), "openrouter", dict(attempts=2, fell_back=True)),
    # a structured usage-limit code on a 429 is billing: the provider is benched.
    ("429-usage-limit-code", sdk(429, "usage limit reached", code="usage_limit_reached"), "openrouter",
     dict(fell_back=True, unhealthy=True)),
    # a 400-coded rate limit rotates the pool and falls back.
    ("400-rate-limit", sdk(400, "Rate limit reached for requests"), "openrouter", dict(fell_back=True, rotated=True)),
    # a 429 'temporarily overloaded' backs off on the same provider; the key is never rotated.
    ("429-overloaded", sdk(429, "The service may be temporarily overloaded, please try again later"), "openrouter",
     dict(attempts=2, fell_back=True, rotated=False)),
    # safety, WAF, data-policy and certificate failures fall back without retrying.
    ("400-data-inspection", sdk(400, "Input data may contain inappropriate content.", type="data_inspection_failed"), "openrouter",
     dict(attempts=1, fell_back=True)),
    ("403-waf", sdk(403, "Attention Required! | Cloudflare — Sorry, you have been blocked"), "openrouter",
     dict(attempts=1, fell_back=True)),
    ("ssl-cert", _Plain("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"), "openrouter",
     dict(attempts=1, fell_back=True)),
    # the Codex account entitlement 400 rotates the pool before falling back.
    ("codex-entitlement", sdk(400, "The 'claude-sonnet' model is not supported when using Codex with a ChatGPT account."),
     "openrouter", dict(attempts=1, fell_back=True, rotated=True)),
    # a server-injected prompt_cache_retention 400 is a server flake: retried.
    ("400-server-injected-param", sdk(400, "prompt_cache_retention is not supported on this model"), "openrouter",
     dict(attempts=2, fell_back=True)),
    # a status-less in-stream rate limit is a rate limit: the pool rotates.
    ("statusless-rate-limit", openai.APIError("Rate limit exceeded", _REQ, body={"code": "rate_limit_exceeded", "message": "slow"}),
     "openrouter", dict(fell_back=True, rotated=True)),
    # a request-validation 5xx is the request's fault: no pointless same-provider retry.
    ("5xx-request-validation", sdk(502, "Unknown parameter: 'frequency_penalty'"), "openrouter", dict(attempts=1, fell_back=False)),
    # an OpenRouter-wrapped upstream 429 leaves a healthy key alone and falls back.
    ("openrouter-upstream-429", sdk(429, "Provider returned error", wrapped=True, code=429,
                metadata={"raw": "x is rate-limited upstream", "provider_name": "DeepInfra"}),
     "openrouter", dict(attempts=1, fell_back=True, rotated=False)),
    # Codex "Failed to extract accountId from token" is a credential failure: the pool acts on it.
    ("codex-account-id", _Plain("Failed to extract accountId from token"), "auto", dict(fell_back=True, rotated=True)),
]


@pytest.mark.parametrize("case,error,provider,expected", BEHAVIOUR_CHANGES,
                         ids=[c[0] for c in BEHAVIOUR_CHANGES])
def test_behaviour_change(case, error, provider, expected):
    outcome = run_ladder(error, provider=provider)
    observed = {key: getattr(outcome, key) for key in expected}
    assert observed == expected, (case, outcome.result)


# ── hooks now steer aux decisions ───────────────────────────────────────


def test_plugin_classification_hook_steers_the_aux_ladder(monkeypatch):
    """A ``transform_api_error_classification`` plugin calling a 403 billing makes aux bench the
    provider and fall back; on main the hook only reached the telemetry label."""
    from hermes_cli import plugins

    def _hook(**kw):
        if kw["status_code"] == 403 and "org wallet empty" in kw["error_message"]:
            return {"reason": FailoverReason.billing, "retryable": False, "should_fallback": True,
                    "should_rotate_credential": True}
        return None

    monkeypatch.setattr(plugins, "get_plugin_error_classification", _hook)
    outcome = run_ladder(sdk(403, "org wallet empty"), provider="openrouter")
    assert (outcome.fell_back, outcome.unhealthy, outcome.rotated) == (True, True, True)


def test_provider_profile_classifier_steers_the_aux_ladder(monkeypatch):
    """``ProviderProfile.classify_api_error`` on the concrete route provider decides the aux outcome."""
    import providers

    class _Profile:
        @staticmethod
        def classify_api_error(error, **_kw):
            if "custom gateway hiccup" in str(error):
                return {"reason": "overloaded"}
            return None

    real = providers.get_provider_profile
    monkeypatch.setattr(providers, "get_provider_profile",
                        lambda slug: _Profile() if slug == "openrouter" else real(slug))
    outcome = run_ladder(sdk(400, "custom gateway hiccup"), provider="openrouter")
    # A plain 400 aborts on the built-in pipeline; the profile's overload verdict retries, then falls back.
    assert (outcome.attempts, outcome.fell_back) == (2, True)


# ── decisions moved from the private predicates (unchanged behaviour) ───────


@pytest.mark.parametrize("error", [
    sdk(402, "Payment Required"),
    _Plain("Budget limit exceeded (monthly limit). Contact your org admin."),
    sdk(403, "this model requires a subscription, upgrade for access: https://ollama.com/upgrade"),
    sdk(404, "Model 'gpt-5' is not available on the free tier. Upgrade."),
    sdk(429, "Too many tokens per day, please wait"),
])
def test_billing_and_spent_quota_bench_the_provider(error):
    assert aux_errors.benches_provider(error)
    assert aux_errors.fallback_label(error) == aux_errors.PAYMENT_ERROR


@pytest.mark.parametrize("spelling", ["RESOURCE_EXHAUSTED", "ResourceExhausted", "resource-exhausted"])
@pytest.mark.parametrize("status", [None, 429])
def test_resource_exhausted_without_a_quota_window_is_a_rate_limit(spelling, status):
    """NIM's concurrency cap names no window, so it rotates and falls back but does not
    bench the provider for the long unhealthy TTL (main pinned this as payment)."""
    error = _Plain(f"{spelling}: Worker local total request limit reached (32/32)")
    if status is not None:
        error.status_code = status
    assert aux_errors.verdict(error).reason is FailoverReason.rate_limit
    assert not aux_errors.benches_provider(error)
    assert aux_errors.fallback_label(error) == aux_errors.RATE_LIMIT


def test_generic_404_is_not_a_provider_failure():
    error = sdk(404, "Not Found")
    assert not aux_errors.benches_provider(error) and aux_errors.fallback_label(error) is None


@pytest.mark.parametrize("error", [
    sdk(404, "Model 'gpt-5.4-mini' not found. The requested model does not exist in our configuration or "
             "OpenRouter catalog."),
    sdk(400, "openrouter/foo/bar is not a valid model ID"),
])
def test_missing_model_heals_and_falls_back(error):
    assert aux_errors.model_not_found(error)
    assert aux_errors.fallback_label(error) == aux_errors.MODEL_NOT_FOUND


@pytest.mark.parametrize("error", [
    sdk(404, "Your API key is blocked or out of funds. model_not_found"),
    sdk(400, "insufficient credits: model is not supported on free tier"),
])
def test_billing_wording_wins_over_model_wording(error):
    assert not aux_errors.model_not_found(error)
    assert aux_errors.verdict(error).reason is FailoverReason.billing


def test_rate_limit_error_class_without_status_is_a_rate_limit():
    class RateLimitError(Exception):
        pass

    error = RateLimitError("rate limit exceeded")
    assert aux_errors.same_key_retry(error) and aux_errors.rotates_credential(error)


def test_statusless_structured_body_falls_back_but_a_plain_message_does_not():
    structured = openai.APIError("stream error", _REQ, body={"type": "server_error", "message": "upstream failed"})
    assert aux_errors.fallback_label(structured) is not None
    assert aux_errors.fallback_label(_Plain("stream_error: mid_stream_failure")) is None


def test_transport_kinds_drive_the_critical_path_skip_and_client_eviction():
    from agent.auxiliary_stream_watchdog import AuxNoProgressTimeout

    timed_out = _Plain("Request timed out.")
    refused = _Plain("Connection refused")
    assert aux_errors.transport(timed_out) == "timeout" and aux_errors.skip_same_provider_retry("compression", timed_out)
    assert aux_errors.transport(refused) == "connection"
    assert not aux_errors.skip_same_provider_retry("compression", refused)
    assert not aux_errors.skip_same_provider_retry("title_generation", AuxNoProgressTimeout("no output in 30s"))
    assert not aux_errors.skip_same_provider_retry("session_search", timed_out)
    assert aux_errors.transport(sdk(400, "Bad Request: invalid model")) is None


def test_verdict_is_classified_once_per_exception_and_backend(monkeypatch):
    calls = []
    import agent.auxiliary_error_policy as policy

    real = policy.classify_api_error
    monkeypatch.setattr(policy, "classify_api_error", lambda exc, **kw: calls.append(kw["provider"]) or real(exc, **kw))
    error = sdk(500, "boom")
    with aux_errors.error_route("auto"):
        assert aux_errors.same_provider_retry(error) and aux_errors.fallback_label(error)
    assert aux_errors.verdict(error, provider="openrouter").reason is FailoverReason.server_error
    assert calls == ["", "openrouter"]  # the routing label "auto" never reaches the classifier


# ── recovery attempts classify as themselves ───────────────────────────


def _invalid_response(*_args, **_kwargs):
    try:
        raise AttributeError("missing choices[0].message")
    except AttributeError as exc:
        raise InvalidProviderResponseError("LLM returned invalid response") from exc


@pytest.mark.parametrize("retry_outcome,label", [
    (ConnectionError("Connection refused"), aux_errors.CONNECTION_ERROR),
    (TimeoutError("timed out"), aux_errors.TIMED_OUT),
    (RuntimeError("Connection reset by peer"), aux_errors.CONNECTION_ERROR),
    (_invalid_response, aux_errors.INVALID_RESPONSE),
])
def test_transport_failure_after_a_401_refresh_falls_back_without_another_rotation(retry_outcome, label):
    """401 -> credential refresh -> the retry cannot connect or answers garbage: the retry's own failure
    decides (no second credential rotation, provider fallback), not the 401 it was raised while handling."""
    outcomes = iter([sdk(401, "invalid api key"), retry_outcome])
    raised = []

    def create(*args, **kwargs):
        outcome = next(outcomes, retry_outcome)
        try:
            if callable(outcome) and not isinstance(outcome, BaseException):
                outcome()
            raise outcome
        except Exception as exc:
            raised.append(exc)
            raise

    outcome = run_ladder(create)
    assert outcome.refreshed == ["openrouter"]
    assert (outcome.rotated, outcome.fell_back) == (False, True)
    assert aux_errors.fallback_label(raised[-1]) == label


def test_route_level_model_rejection_falls_back_without_touching_the_credential():
    outcome = run_ladder(sdk(400, "This model is not supported for image input"))
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (1, True, False)
    assert aux_errors.fallback_label(sdk(400, "model_not_supported: tools")) == aux_errors.MODEL_INCOMPATIBLE


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("message", [
    "Streaming is not supported for this account", "Image input is not supported for this account",
    "Tool use is not supported with this model",
    # parameter advice after the restriction does not make a parameter its subject
    "Streaming is not supported for this account. Please check your request parameters.",
    "Tool use is not supported with this model. Remove the tools parameter.",
    "Streaming is not supported for this account. Set the stream parameter to false.",
])
def test_account_or_model_feature_restriction_falls_back_as_route_incompatible(message, is_async):
    """A feature of the model or account: another provider takes the request, the credential stays."""
    error = sdk(400, message)
    outcome = run_ladder(error, is_async=is_async)
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (1, True, False)
    assert aux_errors.fallback_label(error) == aux_errors.MODEL_INCOMPATIBLE
    assert aux_errors.verdict(error).reason is FailoverReason.format_error  # the main loop's verdict


_PARAMETER_NAMED = [
    "Unsupported parameter: 'temperature' is not supported with this model.",
    "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
    "top_p is not supported with this model", "The 'tools' parameter is not supported with this model",
    "tool_choice is not supported with this model", "Parameter 'response_format' is not supported with this model",
    "temperature is not supported for this account",
    "The model parameter temperature is not supported for this account",
]


def _statusless(message):
    return openai.APIError(message, _REQ, body={"message": message, "type": "invalid_request_error"})


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("error", [sdk(400, m) for m in _PARAMETER_NAMED] + [_statusless(m) for m in _PARAMETER_NAMED[:4]]
                         + [_Plain(_PARAMETER_NAMED[2])],
                         ids=[f"400-{i}" for i in range(len(_PARAMETER_NAMED))] + [f"statusless-{i}" for i in range(4)]
                         + ["plain-2"])
def test_parameter_named_rejection_with_no_rung_reraises(error, is_async):
    """The request's own parameter, not the route: with no parameter rung to apply (the field is not on
    the wire) it would follow the request to any provider, so it re-raises without a fallback."""
    outcome = run_ladder(error, is_async=is_async)
    assert outcome.result is error
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (1, False, False)
    assert aux_errors.fallback_label(error) is None
    assert aux_errors.fallback_label(error, rung_spent=True) == aux_errors.MODEL_INCOMPATIBLE


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("final", [sdk(400, "Unsupported parameter: 'top_p' is not supported with this model."),
                                   _statusless("Unsupported parameter: 'top_p' is not supported with this model.")],
                         ids=["400", "statusless"])
def test_parameter_named_rejection_after_a_spent_rung_falls_back(final, is_async):
    """The temperature rung ran and the repaired request failed on another parameter: fall back."""
    first = sdk(400, "Unsupported parameter: 'temperature' is not supported with this model.")
    outcome = run_ladder([first, final], is_async=is_async, temperature=0.3)
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (2, True, False)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("message,overrides", [
    (_PARAMETER_NAMED[0], {"temperature": None}),  # temperature not on the wire: its rung cannot apply
    ("top_p is not supported with this model", {}),
], ids=["temperature", "top_p"])
def test_candidate_parameter_named_rejection_with_no_rung_reraises(is_async, message, overrides):
    """A candidate's parameter-named 400 with no rung to apply re-raises, like the primary's."""
    client = _candidate([sdk(400, message)], is_async=is_async)
    with pytest.raises(openai.BadRequestError):
        _run_candidate(client, is_async=is_async, **overrides)
    assert client.chat.completions.create.call_count == 1


def test_account_gated_model_with_a_tagged_id_falls_back():
    outcome = run_ladder(sdk(400, "The model qwen3:8b is not supported when using this account"))
    assert (outcome.attempts, outcome.fell_back) == (1, True)


class _Pool:
    """A credential pool that records rotations (the only path that persists a key or key+model cooldown)."""

    def __init__(self):
        self.rotations = []

    def has_credentials(self):
        return True

    def try_refresh_current(self):
        return None

    def mark_exhausted_and_rotate(self, **kwargs):
        self.rotations.append(kwargs)
        return object()


def _pool(monkeypatch):
    from agent import auxiliary_client as ac

    pool = _Pool()
    monkeypatch.setattr(ac, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda _provider: None)
    return ac, pool


@pytest.mark.parametrize("error", [
    sdk(400, "This model is not supported for image input"), sdk(400, "model_not_supported: tools"),
    sdk(400, "Streaming is not supported for this account"), sdk(400, "Tool use is not supported with this model"),
    # a feature or parameter restriction that mentions the model first is still not model gating
    sdk(400, "Streaming with this model is not supported for this account"),
    _Plain("Streaming with this model is not supported for this account"),
    sdk(400, "The model parameter temperature is not supported for this account"),
    _Plain("Parameter reasoning_effort for model o3 is not supported for this account"),
    # a parameter right after "model" is not a model id
    sdk(400, "The model parameter is not supported for this account"),
    _Plain("The model parameter is not supported for this account"),
    sdk(400, "The model tool_choice is not supported for this account"),
    _statusless("The model tool_choice is not supported for this account"),
], ids=lambda e: f"{getattr(e, 'status_code', None)}-{(getattr(e, 'body', None) or {}).get('message', str(e))[:56]}")
def test_route_level_model_rejection_records_no_pool_cooldown(monkeypatch, error):
    """A model_entitlement rotation would bench key+model for a year, even a sole key."""
    ac, pool = _pool(monkeypatch)
    assert ac._recover_provider_pool("openrouter", error, failed_api_key="sk-only") is False
    assert pool.rotations == []


def test_pool_recovery_classifies_once_for_the_pools_provider(monkeypatch):
    """Inside another route (an alias, ``main``, ``custom:x``) every pool decision reads one verdict for the
    pool's own provider: the hooks run once and the decisions cannot split."""
    import agent.auxiliary_error_policy as policy

    calls = []
    real = policy.classify_api_error
    monkeypatch.setattr(policy, "classify_api_error", lambda exc, **kw: calls.append(kw["provider"]) or real(exc, **kw))
    ac, pool = _pool(monkeypatch)
    with aux_errors.error_route("custom:my-proxy"):
        assert ac._recover_provider_pool("openrouter", sdk(429, "Rate limit exceeded for this API key"),
                                         failed_api_key="sk-a") is True
    assert calls == ["openrouter"]
    assert [r["status_code"] for r in pool.rotations] == [429]


def test_auto_routed_nous_classifies_with_the_credential_the_client_sent(monkeypatch):
    """The anonymous Nous JWT on the auto-routed client (not the resolved-key override) marks the 429 a
    welcome-tier gate: fall back, never rotate the tier's only credential."""
    from tests.hermes_cli.anon_portal import make_jwt

    monkeypatch.setattr("agent.auxiliary_client._refresh_nous_recommended_model", lambda **_kw: None)
    gate = sdk(429, "model not available on the free tier", reason="model_not_free")
    outcome = run_ladder(gate, provider="auto", resolved_key="sk-configured-override", client={
        "api_key": make_jwt(), "base_url": "https://inference-api.nousresearch.com/v1",
        "_hermes_aux_effective_provider": "nous"})
    assert (outcome.rotated, outcome.fell_back) == (False, True)
    assert "welcome_refusal" in aux_errors.verdict(gate).error_context


# ── one verdict per exception and attempt route ─────────────────────────


def test_one_failed_attempt_runs_the_classification_hook_once(monkeypatch):
    """Telemetry and the recovery ladder read the same verdict for the attempt."""
    from hermes_cli import plugins

    seen = []
    monkeypatch.setattr(plugins, "get_plugin_error_classification", lambda **kw: seen.append(kw["error"]))
    outcome = run_ladder(sdk(403, "Forbidden"), provider="auto")
    assert outcome.fell_back and len(seen) == 1


def test_route_identity_keys_model_endpoint_and_credential_without_the_secret():
    def identity(**route):
        base = dict(provider="nous", model="m", base_url="https://inference-api.nousresearch.com/v1", api_key="sk-a")
        return aux_errors._route(**{**base, **route}).identity()

    assert identity() == identity(base_url="https://INFERENCE-api.nousresearch.com/v1/")
    assert len({identity(), identity(model="n"), identity(base_url="https://welcome-api.nousresearch.com/v1"),
                identity(api_key="sk-b"), identity(provider="openrouter")}) == 5
    assert "sk-a" not in repr(identity()) + repr(aux_errors._route("nous", "m", "", "sk-a"))
    assert aux_errors._route("nous", "m", "", lambda: "sk-called").api_key is None  # never invoked


def test_telemetry_reports_the_attempt_verdict_not_the_first_cached_one(monkeypatch):
    """The credential pool asks about another provider first; telemetry still reports the verdict for the
    route the attempt ran on."""
    from agent.auxiliary_call_outcome import error_class
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "get_plugin_error_classification",
                        lambda **kw: {"reason": FailoverReason.overloaded, "retryable": True}
                        if kw["provider"] == "gemini" else None)
    error = _Plain("upstream said no")
    with aux_errors.error_route("custom"):
        assert aux_errors.verdict(error, provider="gemini").reason is FailoverReason.overloaded
        assert aux_errors.verdict(error).reason is FailoverReason.unknown
    assert error_class(error, provider="custom") == "unknown"


# ── fallback candidates: spent repairs and refreshed credentials ────────


def _candidate(outcomes, *, is_async):
    client = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-fallback")
    client.chat.completions.create = (AsyncMock if is_async else MagicMock)(side_effect=outcomes)
    return client


def _run_candidate(client, *, is_async, retry_client=None, label="openrouter", **overrides):
    from agent.auxiliary_client import _call_fallback_candidate_async, _call_fallback_candidate_sync

    marks = []
    kw = dict(task="title_generation", messages=_MESSAGES, temperature=0.3, max_tokens=16, tools=None,
              effective_timeout=30.0, effective_extra_body={}, reasoning_config=None)
    kw.update(overrides)
    with (
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **k: marks.append((a, k))),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=retry_client is not None),
        patch("agent.auxiliary_client._get_cached_client", return_value=(retry_client, "fb-model")),
    ):
        if is_async:
            result = asyncio.run(_call_fallback_candidate_async(client, "fb-model", label, **kw))
        else:
            result = _call_fallback_candidate_sync(client, "fb-model", label, **kw)
    return result, marks


_TEMPERATURE_400 = "Error code: 400 - Unsupported value: 'temperature' does not support 0.3 with this model."


@pytest.mark.parametrize("is_async", [False, True])
def test_candidate_whose_parameter_repair_still_fails_is_skipped(is_async):
    client = _candidate([sdk(400, _TEMPERATURE_400), sdk(400, "Bad Request")], is_async=is_async)
    result, marks = _run_candidate(client, is_async=is_async)
    assert result is None and client.chat.completions.create.call_count == 2
    assert [k["reason"] for _a, k in marks] == [aux_errors.MODEL_INCOMPATIBLE]


@pytest.mark.parametrize("is_async", [False, True])
def test_candidate_unmodified_request_shape_failure_still_raises(is_async):
    client = _candidate([sdk(400, "Bad Request")], is_async=is_async)
    with pytest.raises(openai.BadRequestError):
        _run_candidate(client, is_async=is_async)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("final,reason,ttl", [
    (sdk(503, "Service Unavailable"), aux_errors.PROVIDER_OVERLOADED, 60.0),
    (sdk(401, "invalid api key"), "stale fallback credential", None),
    (_invalid_response, aux_errors.INVALID_RESPONSE, 60.0),
    (RuntimeError("Connection reset by peer"), aux_errors.CONNECTION_ERROR, 60.0),
])
def test_refreshed_candidate_is_quarantined_for_its_final_failure(is_async, final, reason, ttl):
    client = _candidate([sdk(401, "invalid api key")], is_async=is_async)
    retry_client = _candidate(final if callable(final) else [final], is_async=is_async)
    result, marks = _run_candidate(client, is_async=is_async, retry_client=retry_client)
    assert result is None and retry_client.chat.completions.create.call_count == 1
    assert [(k["reason"], k["ttl"]) for _a, k in marks] == [(reason, ttl)]


@pytest.mark.parametrize("is_async", [False, True])
def test_fallback_candidate_classifies_with_its_own_credential(monkeypatch, is_async):
    """A candidate's failure is classified with the key the candidate sent, not the primary's."""
    import agent.auxiliary_error_policy as policy

    keys = []
    real = policy.classify_api_error
    monkeypatch.setattr(policy, "classify_api_error", lambda exc, **kw: keys.append(kw["api_key"]) or real(exc, **kw))
    client = _candidate([sdk(503, "Service Unavailable")], is_async=is_async)
    with aux_errors.error_route("nous", "m", "https://inference-api.nousresearch.com/v1", "sk-primary"):
        result, _marks = _run_candidate(client, is_async=is_async)
    assert result is None and keys and set(keys) == {"sk-fallback"}


def _free_tier_gate():
    return sdk(429, "model not available on the free tier", reason="model_not_free")


def _affordable_402():
    return sdk(402, "This request requires more credits: requested 8000 tokens, but can only afford 4000")


@pytest.mark.parametrize("is_async,streamed,make_error", [
    (False, True, _free_tier_gate), (True, True, _free_tier_gate),
    (False, False, _affordable_402),  # the affordable-token retry is sync-only
], ids=["stream-negotiation-sync", "stream-negotiation-async", "affordable-token-retry"])
def test_checks_inside_an_attempt_classify_with_the_clients_own_credential(monkeypatch, is_async, streamed, make_error):
    """A refreshed client sends its own key to its own endpoint: the stream-negotiation check and the
    affordable-token retry classify against that attempt (an anonymous Nous JWT makes the free-tier 429
    a welcome-tier gate), never the enclosing route's stale key or URL."""
    import agent.auxiliary_client as ac
    import agent.auxiliary_error_policy as policy
    from tests.hermes_cli.anon_portal import make_jwt

    seen = []
    real = policy.classify_api_error
    monkeypatch.setattr(policy, "classify_api_error",
                        lambda exc, **kw: seen.append((kw["api_key"], kw["base_url"])) or real(exc, **kw))
    monkeypatch.setattr(ac, "_aux_progress_active", lambda: streamed)
    error = make_error()
    jwt, url = make_jwt(), "https://inference-api.nousresearch.com/v1"
    client = MagicMock(base_url=url, api_key=jwt)
    client.chat.completions.create = (AsyncMock if is_async else MagicMock)(side_effect=[error, error])
    kwargs = {"model": "m", "messages": _MESSAGES, "max_tokens": 8000}
    with aux_errors.error_route("nous", "m", "https://stale.example/v1", "sk-stale"), \
            pytest.raises(type(error)) as raised:
        if is_async:
            asyncio.run(ac._relay_async_completion(client, kwargs, provider="nous"))
        else:
            ac._relay_sync_completion(client, kwargs, provider="nous")
    assert seen and set(seen) == {(jwt, url)}
    if streamed:  # the welcome-tier refusal is the provider's answer: not re-sent non-streaming
        assert "welcome_refusal" in aux_errors.verdict(raised.value).error_context
        assert client.chat.completions.create.call_count == 1


def test_error_classification_hook_runs_under_the_bounded_hook_timeout():
    from hermes_cli.plugins_dispatch import _hook_uses_callback_timeout

    assert _hook_uses_callback_timeout("transform_api_error_classification", 5.0)


# ── an invalid answer after a refresh is the refreshed attempt's failure ──


def _wire_client(status_then, *, key, is_async):
    """A real SDK client over ``httpx.MockTransport``: one canned response per request."""
    responses = iter(status_then)

    def handle(request):
        status, body = next(responses)
        return httpx.Response(status, json=body, request=request)

    url, transport = "https://openrouter.ai/api/v1", httpx.MockTransport(handle)
    if is_async:
        return openai.AsyncOpenAI(api_key=key, base_url=url, max_retries=0,
                                  http_client=httpx.AsyncClient(transport=transport))
    return openai.OpenAI(api_key=key, base_url=url, max_retries=0, http_client=httpx.Client(transport=transport))


_401 = (401, {"error": {"message": "invalid api key"}})
_NO_CHOICES = (200, {"id": "x", "object": "chat.completion", "created": 0, "model": "fb-model-new", "choices": []})


def _hook_for_refreshed_model(monkeypatch):
    """A model-scoped classification hook (overloaded, for the refreshed model only) and a record of
    the route every invalid answer classified on."""
    import agent.auxiliary_error_policy as policy
    from hermes_cli import plugins

    routes = []
    real = policy.classify_api_error
    monkeypatch.setattr(policy, "classify_api_error", lambda exc, **kw: (
        routes.append((kw["model"], kw["api_key"])) if isinstance(exc, InvalidProviderResponseError) else None)
        or real(exc, **kw))
    monkeypatch.setattr(plugins, "get_plugin_error_classification", lambda **kw: {
        "reason": FailoverReason.overloaded, "retryable": True} if kw["model"] == "fb-model-new" else None)
    return routes


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_invalid_answer_from_a_refreshed_candidate_classifies_on_the_refreshed_route(monkeypatch, is_async):
    from agent.auxiliary_client import _call_fallback_candidate_async, _call_fallback_candidate_sync

    routes = _hook_for_refreshed_model(monkeypatch)
    stale = _wire_client([_401], key="sk-stale", is_async=is_async)
    fresh = _wire_client([_NO_CHOICES], key="sk-fresh", is_async=is_async)
    marks = []
    kw = dict(task="title_generation", messages=_MESSAGES, temperature=None, max_tokens=16, tools=None,
              effective_timeout=30.0, effective_extra_body={}, reasoning_config=None)
    with (
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **k: marks.append(k)),
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True),
        patch("agent.auxiliary_client._get_cached_client", return_value=(fresh, "fb-model-new")),
    ):
        if is_async:
            result = asyncio.run(_call_fallback_candidate_async(stale, "fb-model", "openrouter", **kw))
        else:
            result = _call_fallback_candidate_sync(stale, "fb-model", "openrouter", **kw)
    assert result is None
    assert routes and set(routes) == {("fb-model-new", "sk-fresh")}
    assert [k["reason"] for k in marks] == [aux_errors.PROVIDER_OVERLOADED]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_invalid_answer_from_a_refreshed_primary_retry_classifies_on_the_refreshed_route(monkeypatch, is_async):
    from agent.auxiliary_client import _retry_same_provider_async, _retry_same_provider_sync

    routes = _hook_for_refreshed_model(monkeypatch)
    fresh = _wire_client([_NO_CHOICES], key="sk-fresh", is_async=is_async)
    request = {"model": "fb-model-new", "messages": _MESSAGES}
    with patch("agent.auxiliary_client._prepare_same_provider_retry", return_value=(fresh, request)), \
            aux_errors.error_route("openrouter", "fb-model", "https://openrouter.ai/api/v1", "sk-stale"), \
            pytest.raises(InvalidProviderResponseError) as raised:
        retry = _retry_same_provider_async if is_async else _retry_same_provider_sync
        outcome = retry(resolved_provider="openrouter", resolved_api_mode=None, task="title_generation")
        if is_async:
            asyncio.run(outcome)
    assert aux_errors.verdict(raised.value).reason is FailoverReason.overloaded
    assert routes and set(routes) == {("fb-model-new", "sk-fresh")}


# ── compression keeps the session on a TLS certificate failure ──────────


def _botocore_ssl_error():
    import ssl

    exceptions = pytest.importorskip("botocore.exceptions")
    return exceptions.SSLError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
                               error=ssl.SSLCertVerificationError(1, "certificate verify failed"))


def _cert_verification_error():
    import ssl

    return ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")


class EndpointConnectionError(Exception):
    """botocore's class name, outside the builtin ConnectionError/OSError hierarchy like botocore's own."""


_BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude/converse"


def _endpoint_connection_error():
    return EndpointConnectionError(f'Could not connect to the endpoint URL: "{_BEDROCK_URL}"')


def _botocore_endpoint_connection_error():
    exceptions = pytest.importorskip("botocore.exceptions")
    return exceptions.EndpointConnectionError(endpoint_url=_BEDROCK_URL)


@pytest.mark.parametrize("make_error", [
    _cert_verification_error, _botocore_ssl_error, _endpoint_connection_error, _botocore_endpoint_connection_error,
], ids=["ssl", "botocore-ssl", "endpoint-connection", "botocore-endpoint-connection"])
def test_compression_preserves_the_messages_after_exhausted_network_failures(make_error):
    """A TLS failure or an unreachable endpoint is a dead network for the compressor (as before): it
    aborts and keeps every message instead of committing a placeholder summary over the dropped middle."""
    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                       abort_on_summary_failure=False)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
    error = make_error()
    with patch("agent.context_compressor.call_llm", side_effect=error):
        result = compressor.compress(msgs, current_tokens=999999, force=True)
    assert result == msgs
    assert compressor._last_summary_network_failure is True
    assert compressor._last_compress_aborted is True
    assert compressor._last_summary_dropped_count == 0


# ── an unreachable Bedrock endpoint is a connection failure ─────────────


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("make_error", [_endpoint_connection_error, _botocore_endpoint_connection_error],
                         ids=["named", "botocore"])
def test_unreachable_endpoint_retries_then_falls_back(make_error, is_async):
    outcome = run_ladder([make_error(), make_error()], provider="bedrock", retries=1, is_async=is_async)
    assert outcome.attempts == 2 and outcome.fell_back
    assert not outcome.rotated and not outcome.unhealthy


# ── a backoff retry's own failure is the one the ladder acts on ─────────


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_generic_403_after_a_rate_limit_backoff_reraises_on_an_explicit_provider(is_async):
    """429 then a generic 403 on an explicit provider with no task fallback chain is a 403: re-raised, no
    provider hop (a 429 would have fallen back)."""
    outcome = run_ladder([sdk(429, "Rate limit exceeded"), sdk(403, "Forbidden")], is_async=is_async)
    assert isinstance(outcome.result, openai.PermissionDeniedError)
    assert outcome.attempts == 2 and not outcome.fell_back


# ── every attempt classifies against its concrete backend ───────────────


def _record_classified_providers(monkeypatch):
    import agent.auxiliary_error_policy as policy

    seen = []
    real = policy.classify_api_error
    monkeypatch.setattr(policy, "classify_api_error", lambda exc, **kw: seen.append(kw["provider"]) or real(exc, **kw))
    return seen


def _go_upstream_429():
    import model_tools  # registers the bundled provider profiles

    body = {"error": {"type": "server_error", "message": "Upstream request failed: Rate limit exceeded."}}
    response = httpx.Response(429, json=body, request=httpx.Request("POST", "https://opencode.ai/zen/go/v1/x"))
    return openai.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_unavailable_primary_classifies_its_chain_fallback_on_the_concrete_backend(monkeypatch, is_async):
    """An explicit provider with no credentials runs on its task fallback_chain entry; that entry's
    failure classifies on its provider (the Go profile reads an upstream throttle), not on the
    ``fallback_chain[0](opencode-go)`` display label."""
    seen = _record_classified_providers(monkeypatch)
    error = _go_upstream_429()
    go = MagicMock(base_url="https://opencode.ai/zen/go/v1", api_key="sk-go")
    go.chat.completions.create = (AsyncMock if is_async else MagicMock)(side_effect=error)
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("anthropic", "m", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(None, None)),
        patch("agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
              return_value=(go, "mock-model", "fallback_chain[0](opencode-go)")),
        patch("agent.auxiliary_client._to_async_client", return_value=(go, "mock-model")),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        pytest.raises(openai.RateLimitError),
    ):
        if is_async:
            asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        else:
            call_llm(task="title_generation", messages=_MESSAGES)
    assert seen and set(seen) == {"opencode-go"}
    assert aux_errors.verdict(error).reason is FailoverReason.upstream_rate_limit


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_auto_candidate_classifies_on_the_provider_it_routed_to(monkeypatch, is_async):
    seen = _record_classified_providers(monkeypatch)
    client = _candidate([_go_upstream_429()], is_async=is_async)
    client._hermes_aux_effective_provider = "opencode-go"
    result, marks = _run_candidate(client, is_async=is_async, label="auto")
    assert result is None and marks
    assert seen and set(seen) == {"opencode-go"}


def _nous_ladder(primary, *, is_async, refreshed=None, healed_model=None):
    """``call_llm`` / ``async_call_llm`` on a Nous route whose primary is a real SDK client; the 401
    refresh hands back ``refreshed`` and the catalog heal names ``healed_model``. Returns the result."""
    fallback = MagicMock(base_url="https://api.example.com/v1")
    fallback.chat.completions.create = (AsyncMock if is_async else MagicMock)(return_value=_ok())
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "old-model", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "old-model")),
        patch("agent.auxiliary_client._refresh_nous_auxiliary_client", return_value=(refreshed, "fb-model-new")),
        patch("agent.auxiliary_client._refresh_nous_recommended_model", return_value=healed_model),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb", "fallback")),
    ):
        if is_async:
            return asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        return call_llm(task="title_generation", messages=_MESSAGES)


_404_MODEL = (404, {"error": {"message": "The model `old-model` does not exist", "code": "model_not_found"}})


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("step", ["credential-refresh", "model-heal"])
def test_invalid_answer_from_a_nous_ladder_step_classifies_on_that_steps_request(monkeypatch, is_async, step):
    """The Nous 401 refresh sends the refreshed key and model, the catalog heal the healed model: an
    HTTP-200 answer with no choices is that request's failure, classified on its own model and key."""
    routes = _hook_for_refreshed_model(monkeypatch)
    if step == "credential-refresh":
        primary = _wire_client([_401], key="sk-stale", is_async=is_async)
        refreshed, healed, expected = _wire_client([_NO_CHOICES], key="sk-fresh", is_async=is_async), None, "sk-fresh"
    else:
        primary = _wire_client([_404_MODEL, _NO_CHOICES], key="sk-primary", is_async=is_async)
        refreshed, healed, expected = None, "fb-model-new", "sk-primary"
    result = _nous_ladder(primary, is_async=is_async, refreshed=refreshed, healed_model=healed)
    assert result.choices[0].message.content == "fallback"
    assert routes and set(routes) == {("fb-model-new", expected)}


# ── a stream error after a credential refresh is the refreshed attempt's own ──


def _sse_wire_client(error_json, *, key, is_async):
    """A real SDK client whose every request answers HTTP 200 with one SSE ``error`` event."""
    def handle(request):
        return httpx.Response(200, content=f'data: {{"error": {error_json}}}\n\n'.encode(),
                              headers={"content-type": "text/event-stream"}, request=request)

    url, transport = "https://inference-api.nousresearch.com/v1", httpx.MockTransport(handle)
    if is_async:
        return openai.AsyncOpenAI(api_key=key, base_url=url, max_retries=0,
                                  http_client=httpx.AsyncClient(transport=transport))
    return openai.OpenAI(api_key=key, base_url=url, max_retries=0, http_client=httpx.Client(transport=transport))


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("error_json", ['"backend worker failed"', '{"message": "backend worker failed"}'],
                         ids=["string-body", "object-body"])
def test_stream_error_after_a_nous_credential_refresh_falls_back(is_async, error_json):
    """401, Nous refresh, then the refreshed key's stream carries an SSE ``error`` event: the SDK raises a
    status-less APIError while the 401 is handled. That is the refreshed attempt's own failure, so a string
    or object body falls back as a status-less structured error instead of re-raising an inherited 401."""
    primary = _wire_client([_401], key="sk-stale", is_async=is_async)
    refreshed = _sse_wire_client(error_json, key="sk-fresh", is_async=is_async)
    with patch("agent.auxiliary_client._aux_progress_active", return_value=True):  # every attempt streams
        result = _nous_ladder(primary, is_async=is_async, refreshed=refreshed)
    assert result.choices[0].message.content == "fallback"


# ── a refreshed candidate's own bad answer is not a stale credential ────

_SHIM_COMPLETION = {"id": "x", "object": "chat.completion", "created": 0, "model": "fb-model-new",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "Connect timeout, please try again later."}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}}
_BAD_ANSWERS = {  # (body, content type, streamed)
    "malformed-sse": (b"data: {bad json}\n\n", "text/event-stream", True),
    "json-null": (b"null", "application/json", False),
    "error-envelope": (b'{"success": false, "data": {"error": {"message": "upstream worker crashed"}}}',
                       "application/json", False),
    "timeout-shim": (json.dumps(_SHIM_COMPLETION).encode(), "application/json", False),
}


def _bad_answer_client(answer, *, key, is_async):
    """A real SDK client whose every request answers HTTP 200 with one of ``_BAD_ANSWERS``."""
    body, content_type, _streamed = _BAD_ANSWERS[answer]
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, content=body, headers={"content-type": content_type}, request=request))
    if is_async:
        return openai.AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=transport))
    return openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0,
                         http_client=httpx.Client(transport=transport))


def _candidate_outcome(first, refreshed, *, is_async, streamed):
    """``(raised type or None, [(quarantine reason, ttl)])`` of one fallback candidate call."""
    from agent.auxiliary_client import _call_fallback_candidate_async, _call_fallback_candidate_sync

    marks, raised = [], None
    kw = dict(task="title_generation", messages=_MESSAGES, temperature=None, max_tokens=16, tools=None,
              effective_timeout=30.0, effective_extra_body={}, reasoning_config=None)
    with (
        patch("agent.auxiliary_client._aux_progress_active", return_value=streamed),
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **k: marks.append(k)),
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True),
        patch("agent.auxiliary_client._get_cached_client", return_value=(refreshed, "fb-model-new")),
    ):
        try:
            if is_async:
                asyncio.run(_call_fallback_candidate_async(first, "fb-model", "openrouter", **kw))
            else:
                _call_fallback_candidate_sync(first, "fb-model", "openrouter", **kw)
        except Exception as exc:
            raised = type(exc)
    return raised, [(k["reason"], k["ttl"]) for k in marks]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("answer", list(_BAD_ANSWERS))
def test_refreshed_candidates_own_bad_answer_is_not_a_stale_credential(is_async, answer):
    """401, credential refresh, then the refreshed key's answer is malformed, null, an error envelope or a
    timeout shim. The answer check raises while the 401 is handled, but the failure is the retry's own: it
    gets the outcome the same answer gets with no 401 before it, never the stale-credential hold."""
    streamed = _BAD_ANSWERS[answer][2]
    stale = _wire_client([_401], key="sk-stale", is_async=is_async)
    after_refresh = _candidate_outcome(stale, _bad_answer_client(answer, key="sk-fresh", is_async=is_async),
                                       is_async=is_async, streamed=streamed)
    alone = _candidate_outcome(_bad_answer_client(answer, key="sk-fresh", is_async=is_async), None,
                               is_async=is_async, streamed=streamed)
    assert ("stale fallback credential", None) not in after_refresh[1]
    assert after_refresh == alone


# ── a sync→async client conversion keeps the client's backend identity ──


def _sync_client(name):
    """One sync client per ``_to_async_client`` conversion branch, built when the test runs."""
    from agent import auxiliary_client as aux
    from agent.gemini_native_adapter import GeminiNativeClient

    plain = openai.OpenAI(api_key="sk", base_url="https://opencode.ai/zen/go/v1")
    return {
        "openai": lambda: plain,
        "codex": lambda: aux.CodexAuxiliaryClient(plain, "m"),
        "anthropic": lambda: aux.AnthropicAuxiliaryClient(MagicMock(), "m", "sk", "https://api.anthropic.com"),
        "bedrock": lambda: aux.BedrockAuxiliaryClient("us-east-1", "m"),
        "gemini": lambda: GeminiNativeClient(api_key="sk"),
    }[name]()


@pytest.mark.parametrize("name", ["openai", "codex", "anthropic", "bedrock", "gemini"])
def test_async_conversion_keeps_the_client_identity(name):
    from agent.auxiliary_client import _FallbackDestination, _classification_provider, _to_async_client

    sync_client = _sync_client(name)
    destination = _FallbackDestination("opencode-go", "https://opencode.ai/zen/go/v1", None, "m")
    sync_client._hermes_aux_effective_provider = "opencode-go"
    sync_client._hermes_fallback_destination = destination
    converted, _model = _to_async_client(sync_client, "m")
    assert converted is not sync_client
    assert converted._hermes_aux_effective_provider == "opencode-go"
    assert converted._hermes_fallback_destination is destination
    assert _classification_provider(converted, "auto") == "opencode-go"


def _go_429_transport():
    """An OpenCode Go endpoint whose every request is an upstream 429."""
    body = {"error": {"type": "server_error", "message": "Upstream request failed: Rate limit exceeded."}}
    return httpx.MockTransport(lambda request: httpx.Response(429, json=body, request=request))


@pytest.mark.parametrize("tag", ["effective-provider", "fallback-destination"])
def test_async_auto_candidate_classifies_on_its_backend_through_the_real_conversion(monkeypatch, tag):
    """``async_call_llm`` converts a sync fallback candidate to async before sending: the converted client
    keeps the backend its sync original was tagged with, so the Go profile reads the upstream throttle."""
    import model_tools  # registers the bundled provider profiles
    import agent.auxiliary_error_policy as policy
    from agent.auxiliary_client import _FallbackDestination

    seen, real = [], policy.classify_api_error

    def record(exc, **kw):
        classified = real(exc, **kw)
        if "Upstream request failed" in str(exc):
            seen.append((kw["provider"], classified.reason))
        return classified

    monkeypatch.setattr(policy, "classify_api_error", record)
    transport = _go_429_transport()
    go = openai.OpenAI(api_key="sk-go", base_url="https://opencode.ai/zen/go/v1", max_retries=0,
                       http_client=httpx.Client(transport=transport))
    if tag == "effective-provider":
        go._hermes_aux_effective_provider = "opencode-go"
    else:
        go._hermes_fallback_destination = _FallbackDestination("opencode-go", str(go.base_url), None, "go-model")
    primary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-primary")
    primary.chat.completions.create = AsyncMock(side_effect=sdk(503, "Service Unavailable"))
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openrouter", "m", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("agent.auxiliary_client._openai_http_client_kwargs",
              return_value={"http_client": httpx.AsyncClient(transport=transport)}),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(go, "go-model", "auto")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        pytest.raises(openai.APIError),
    ):
        asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
    assert seen and set(seen) == {("opencode-go", FailoverReason.upstream_rate_limit)}


# ── an embedded HTTP-200 error classifies on its attempt's route ─────────


def test_embedded_200_error_telemetry_classifies_on_the_attempt_route(monkeypatch):
    from agent.auxiliary_call_outcome import PROVIDER_ERROR_ATTR, error_class

    seen = _record_classified_providers(monkeypatch)
    error = InvalidProviderResponseError("LLM returned invalid response")
    setattr(error, PROVIDER_ERROR_ATTR, {"message": "Upstream request failed: Rate limit exceeded.", "code": 429})
    client = SimpleNamespace(base_url="https://opencode.ai/zen/go/v1", api_key="sk-go")
    with pytest.raises(InvalidProviderResponseError), aux_errors.attempt(client, "opencode-go", "go-model"):
        raise error
    error_class(error, provider="nous", model="scope-model")
    assert seen == ["opencode-go"]


def test_embedded_200_error_runs_the_hooks_once_per_error_read(monkeypatch):
    """The per-attempt note, the ladder and the terminal report all read one failed attempt: the hook runs
    once for the answer the ladder routes and once for the error object telemetry reports, not per read."""
    import agent.auxiliary_error_policy as policy
    from agent.auxiliary_call_outcome import PROVIDER_ERROR_ATTR, error_class
    from hermes_cli import plugins

    hooked, real = [], policy.classify_api_error
    monkeypatch.setattr(plugins, "get_plugin_error_classification", lambda **kw: hooked.append(kw["provider"]))
    monkeypatch.setattr(policy, "classify_api_error", lambda exc, **kw: hooked.append(type(exc).__name__) or real(exc, **kw))
    error = InvalidProviderResponseError("LLM returned invalid response")
    setattr(error, PROVIDER_ERROR_ATTR, {"message": "Rate limit exceeded", "code": 429})
    client = SimpleNamespace(base_url="https://opencode.ai/zen/go/v1", api_key="sk-go")
    with aux_errors.error_route("nous", "scope-model", "https://inference-api.nousresearch.com/v1", "sk-nous"):
        with pytest.raises(InvalidProviderResponseError), aux_errors.attempt(client, "opencode-go", "go-model"):
            inside = error_class(error, provider="nous", model="scope-model")
            raise error
        ladder = aux_errors.verdict(error).reason
    terminal = error_class(error, provider="fallback_chain[0](x)", model="scope-model")
    assert hooked == ["_EmbeddedProviderError", "opencode-go", "InvalidProviderResponseError", "opencode-go"]
    assert inside == terminal == "rate_limit" and ladder is FailoverReason.incomplete_response


def test_same_provider_retry_honours_a_not_retryable_verdict():
    from agent.error_classifier import ClassifiedError

    error = RuntimeError("upstream 500")
    for retryable, expected in ((True, True), (False, False)):
        verdict = ClassifiedError(reason=FailoverReason.server_error, retryable=retryable)
        with patch.object(aux_errors, "verdict", return_value=verdict):
            assert aux_errors.same_provider_retry(error) is expected


def test_stream_deadline_raised_from_a_401_never_refreshes_the_credential():
    from agent.auxiliary_stream_watchdog import AuxChatStreamTimeout

    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    try:
        try:
            raise openai.AuthenticationError(
                "Error code: 401", response=httpx.Response(401, request=req), body={"message": "invalid api key"})
        except openai.AuthenticationError as exc:
            raise AuxChatStreamTimeout("Auxiliary chat stream stalled: no new output for 60.0s (timed out)") from exc
    except AuxChatStreamTimeout as deadline:
        assert not aux_errors.auth_refresh(deadline) and aux_errors.transport(deadline) == "timeout"
