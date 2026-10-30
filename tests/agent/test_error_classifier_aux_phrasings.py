"""Phrasings and shape facts the auxiliary client used to match privately, now in the shared classifier.

Most cases here classified differently (or not at all) on the main loop before the auxiliary ladder's
private predicates were folded into ``classify_api_error``; the rest (free-tier code, the parameter-rung
detectors, the negative cases) guard what must not change. The main loop and the auxiliary client now
read the same verdict.
"""
import json

import httpx
import openai
import pytest

from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
    is_max_tokens_rejection,
    is_structured_output_rejection,
    is_unsupported_parameter_error,
    rejects_reasoning_disable,
    rejects_reasoning_field,
)
from agent.errors import EmptyStreamError, InvalidProviderResponseError

_R = FailoverReason
_REQ = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


class _Err(Exception):
    def __init__(self, msg, status=None, body=None):
        super().__init__(msg)
        if status is not None:
            self.status_code = status
        if body is not None:
            self.body = body


# ── (a)+(d): phrasings the main loop missed too ─────────────────────────


def test_404_free_tier_wording_is_billing():
    assert classify_api_error(_Err("This model isn't available on the free tier", 404)).reason is _R.billing


@pytest.mark.parametrize("message", [
    "Quota exceeded for this key",
    "this model requires a subscription, upgrade for access: https://ollama.com/upgrade",
])
def test_403_quota_and_subscription_walls_are_billing(message):
    verdict = classify_api_error(_Err(message, 403))
    assert verdict.reason is _R.billing
    assert verdict.should_fallback is True


@pytest.mark.parametrize("err", [
    _Err("unauthenticated:bad-credentials", 403),
    _Err("Error code: 403 - unauthenticated:bad-credentials"),
])
def test_xai_bad_credentials_is_a_rotating_auth_failure(err):
    verdict = classify_api_error(err, provider="xai-oauth")
    assert verdict.reason is _R.auth
    assert verdict.should_rotate_credential is True


def test_statusless_account_model_gating_is_model_entitlement():
    err = RuntimeError("model kimi-k2-thinking is not supported when using this account")
    verdict = classify_api_error(err)
    assert verdict.reason is _R.model_entitlement
    assert verdict.should_fallback is True


def test_400_account_model_gating_is_model_entitlement_but_a_param_rejection_is_not():
    assert classify_api_error(_Err("This model is not supported for this account.", 400)).reason is _R.model_entitlement
    rejected = classify_api_error(_Err("Unsupported parameter: 'temperature' is not supported with this model.", 400))
    assert rejected.reason is _R.format_error


@pytest.mark.parametrize("status", [400, None])
@pytest.mark.parametrize("message", [
    "The model qwen3:8b is not supported when using this account",
    "model qwen3:8b is not supported for this account",
    "Error code: 400 - {'error': {'message': \"The 'kimi-k2' model is not supported for this account.\"}}",
])
def test_the_model_itself_as_the_rejected_subject_is_model_entitlement(message, status):
    """An id token (an Ollama tag carries a colon) between the subject and the verb is still the model."""
    verdict = classify_api_error(_Err(message, status))
    assert (verdict.reason, verdict.should_rotate_credential) == (_R.model_entitlement, True)


@pytest.mark.parametrize("err", [
    _Err("This model is not supported for image input", 400),
    _Err("model_not_supported: tools", 400),
    _Err("The endpoint does not support this model"),
    # Account or model restrictions of a feature: the route cannot serve this request, the key is fine.
    _Err("Streaming is not supported for this account", 400),
    _Err("Image input is not supported for this account"),
    _Err("Tool use is not supported with this model", 400),
    _Err("Streaming with this model is not supported for this account", 400),
])
def test_route_level_model_rejection_never_rotates_or_benches_the_key(err):
    """The route cannot serve the model for this request; the credential is fine. A model_entitlement
    verdict would rotate the key and bench key+model in the pool for a year."""
    verdict = classify_api_error(err, provider="openrouter")
    assert verdict.reason is not _R.model_entitlement
    assert verdict.should_rotate_credential is False
    assert verdict.error_context["route_rejects_model"] is True


@pytest.mark.parametrize("status", [400, None])
@pytest.mark.parametrize("message", [
    "Unsupported parameter: 'temperature' is not supported with this model.",
    "top_p is not supported with this model", "The 'tools' parameter is not supported with this model",
    "temperature is not supported for this account", "reasoning_effort is not supported with this model",
])
def test_route_wording_about_a_request_parameter_is_the_parameter_not_the_route(message, status):
    """A parameter as the subject is the request's own failure: no route_rejects_model fact, no rotation."""
    verdict = classify_api_error(_Err(message, status), provider="openrouter")
    assert "route_rejects_model" not in verdict.error_context
    assert verdict.should_rotate_credential is False
    assert verdict.error_context["rejects_parameter"] is True


@pytest.mark.parametrize("status", [400, None])
@pytest.mark.parametrize("message", [
    "Streaming is not supported for this account. Please check your request parameters.",
    "Tool use is not supported with this model. Remove the tools parameter.",
    "Streaming is not supported for this account. Set the stream parameter to false.",
    "Image input is not supported for this account; see the parameters docs",
])
def test_parameter_advice_after_a_feature_restriction_keeps_the_feature_subject(message, status):
    """Only the subject of the unsupported clause decides: a parameter named in later advice text does
    not turn a feature restriction into the request's own parameter failure."""
    verdict = classify_api_error(_Err(message, status), provider="openrouter")
    assert verdict.error_context.get("route_rejects_model") is True
    assert "rejects_parameter" not in verdict.error_context
    assert verdict.should_rotate_credential is False


def _repeat(unit, size=200_000):
    return (unit * (size // len(unit) + 1))[:size]


# Every scan this classifier runs over a whole body (route and account-gating wording, clause stops,
# parameter tokens, model-id tokens, quota and wall tables), fed the input that would make it rescan.
_ADVERSARIAL_BODIES = {
    "separator-runs": _repeat("." * 158 + " is not supported for this account " + "model " + "-/" * 60
                              + " is not supported when using this account. The model " + "_." * 40 + " model "),
    "route-wording-no-stops": _repeat("tool use is not supported with this model "),
    "parameter-then-route-wording": _repeat("the temperature parameter model is not supported "),
    "gating-tail-runs": _repeat("model is not supported for this account "),
    "model-tokens": _repeat("model " * 20 + "the model gpt-4.1 model "),
    "parameter-tokens": _repeat("parameter top_p max_tokens tool_choice "),
    "stop-runs": _repeat(". ; ! ? \n" * 10 + "does not support this model "),
    "id-separators": _repeat("model " + "a-" * 64 + ": " + "/" * 129 + " "),
    "quota-and-walls": _repeat("quota daily limit credits budget limit requires a subscription "),
}


@pytest.mark.parametrize("status", [400, None], ids=["400", "statusless"])
@pytest.mark.parametrize("body", list(_ADVERSARIAL_BODIES.values()), ids=list(_ADVERSARIAL_BODIES))
def test_adversarial_200kb_bodies_classify_in_linear_time(body, status):
    """A hostile or odd 200 KB body must not backtrack or rescan prefixes: async aux classifies on the
    event loop."""
    import time

    classify_api_error(_Err("warm up", 400))
    started = time.perf_counter()
    classify_api_error(_Err(body, status), provider="openrouter", model="gpt-4.1")
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize("message,expected", [
    ("Unsupported parameter: 'temperature' is not supported with this model.", True),
    ("Streaming is not supported for this account. Set the stream parameter to false.", False),
    ("Tool use is not supported with this model\nRemove the tools parameter.", False),
    ("The model parameter max_tokens; is not supported with this model", False),
    ("first clause. top_p is not supported with this model. second clause model is not supported", True),
    ("model is not supported. Then reasoning_effort is not supported with this model", True),
])
def test_only_the_unsupported_clauses_own_subject_names_a_parameter(message, expected):
    from agent.error_classifier import _rejected_subject_is_parameter

    assert _rejected_subject_is_parameter(message.lower()) is expected


def test_free_tier_model_code_stays_billing_not_entitlement():
    assert classify_api_error(_Err("model_not_supported_on_free_tier", 400)).reason is _R.billing


def test_response_ended_prematurely_is_a_transport_failure():
    verdict = classify_api_error(_Err("RemoteProtocolError: response ended prematurely"))
    assert verdict.reason is _R.timeout
    assert verdict.error_context["transport"] == "connection"


# ── RESOURCE_EXHAUSTED is a rate limit unless it names a long quota window ──


@pytest.mark.parametrize("status", [None, 429])
def test_bare_resource_exhausted_is_a_rate_limit(status):
    verdict = classify_api_error(_Err("RESOURCE_EXHAUSTED: Worker local total request limit reached (32/32)", status))
    assert verdict.reason is _R.rate_limit
    assert "quota_exhausted" not in verdict.error_context


@pytest.mark.parametrize("status", [None, 429])
@pytest.mark.parametrize("message", [
    "RESOURCE_EXHAUSTED: You have reached your daily limit.",
    "You've hit your weekly usage limit.",
    "Too many tokens per day",
])
def test_named_daily_or_weekly_window_without_reset_is_billing_and_quota_exhausted(status, message):
    verdict = classify_api_error(_Err(message, status))
    assert verdict.reason is _R.billing
    assert verdict.error_context["quota_exhausted"] is True


@pytest.mark.parametrize("message", [
    "Quota exceeded for quota metric 'Generate Content API requests per day'",
    # DashScope's per-minute allocation throttle: still retried by the main loop.
    'event:error :HTTP_STATUS/429 data:{"code":"Throttling.AllocationQuota","message":"Allocated quota exceeded."}',
])
def test_exceeded_quota_without_a_named_window_stays_a_rate_limit_but_is_flagged(message):
    verdict = classify_api_error(_Err(message, 429))
    assert verdict.reason is _R.rate_limit
    assert verdict.error_context["quota_exhausted"] is True


def test_named_quota_window_with_reset_stays_rate_limit_but_quota_exhausted():
    verdict = classify_api_error(_Err("Too many tokens per day, please wait before retrying", 429))
    assert verdict.reason is _R.rate_limit
    assert verdict.error_context["quota_exhausted"] is True


def test_plugin_verdict_is_not_reinterpreted_as_quota_exhausted(monkeypatch):
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "get_plugin_error_classification",
                        lambda **_kw: {"reason": _R.rate_limit, "should_fallback": True})
    verdict = classify_api_error(_Err("You have reached your daily limit.", 429))
    assert verdict.reason is _R.rate_limit
    assert "quota_exhausted" not in verdict.error_context


# ── shape facts the auxiliary decisions branch on ───────────────────────


def test_transport_sub_kind_separates_timeouts_from_connection_failures():
    timed_out = classify_api_error(openai.APITimeoutError(request=_REQ))
    refused = classify_api_error(httpx.ConnectError("[Errno 61] Connection refused"))
    dns = classify_api_error(_Err("[Errno 8] nodename nor servname provided, or not known"))
    assert (timed_out.reason, timed_out.error_context["transport"]) == (_R.timeout, "timeout")
    assert (refused.reason, refused.error_context["transport"]) == (_R.timeout, "connection")
    assert (dns.reason, dns.error_context["transport"]) == (_R.timeout, "connection")
    assert "transport" not in classify_api_error(_Err("Bad request", 400)).error_context


def test_statusless_structured_body_is_flagged():
    err = openai.APIError("stream error", _REQ, body={"type": "server_error", "message": "upstream failed"})
    assert classify_api_error(err).error_context["statusless_structured"] is True
    assert "statusless_structured" not in classify_api_error(_Err("plain failure")).error_context


def test_invalid_provider_response_is_incomplete_response():
    verdict = classify_api_error(InvalidProviderResponseError("Auxiliary compression: LLM returned invalid response"))
    assert verdict.reason is _R.incomplete_response
    assert verdict.should_fallback is True


@pytest.mark.parametrize("status,code,reason,rotate", [
    (503, "ServiceUnavailableException", _R.overloaded, False),
    (403, "AccessDeniedException", _R.auth, False),
    (429, "ThrottlingException", _R.rate_limit, True),
])
def test_botocore_status_is_read_from_the_response_metadata(status, code, reason, rotate):
    """Bedrock's ClientError keeps its status in the parsed response: the main loop reads it too."""
    class ClientError(Exception):
        def __init__(self):
            super().__init__(f"An error occurred ({code}) when calling the Converse operation")
            self.response = {"ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": code}}

    verdict = classify_api_error(ClientError())
    assert (verdict.status_code, verdict.reason, verdict.should_rotate_credential) == (status, reason, rotate)


def test_status_carried_only_on_the_response_is_read():
    response = httpx.Response(429, request=_REQ)
    verdict = classify_api_error(httpx.HTTPStatusError("Too Many Requests", request=_REQ, response=response))
    assert (verdict.status_code, verdict.reason) == (429, _R.rate_limit)


# ── parameter-rung detectors (moved from the auxiliary client, behaviour unchanged) ──


def test_parameter_rung_detectors_are_status_gated():
    assert is_unsupported_parameter_error(_Err("Unsupported parameter: 'temperature'", 400), "temperature")
    assert not is_unsupported_parameter_error(_Err("Unsupported parameter: 'temperature'", 400), "")
    assert is_structured_output_rejection(_Err("response_format json_schema is not supported", 422))
    assert not is_structured_output_rejection(_Err("response_format json_schema is not supported", 500))
    assert rejects_reasoning_field(_Err("Unrecognized request argument supplied: reasoning_effort", 400))
    assert not rejects_reasoning_field(_Err("Unrecognized request argument supplied: reasoning_effort", 503))
    assert rejects_reasoning_disable(_Err("Reasoning is mandatory for this endpoint and cannot be disabled.", 400))
    assert is_max_tokens_rejection(_Err("code 1210 bad request"), "https://open.bigmodel.cn/api/paas/v4")
    assert not is_max_tokens_rejection(_Err("code 1210 bad request"), "https://api.example.com/v1")


# ── an account restriction on a parameter or feature is not model gating ──


@pytest.mark.parametrize("err", [
    _Err("Streaming is not supported for this account", 400),
    _Err("Unsupported parameter: temperature is not supported for this account"),
    _Err("Streaming is not supported for this account"),
    # "model" before the restriction does not make the model its subject.
    _Err("Streaming with this model is not supported for this account", 400),
    _Err("Streaming with this model is not supported for this account"),
    _Err("For this model, streaming is not supported for this account", 400),
    _Err("The model parameter temperature is not supported for this account", 400),
    _Err("Parameter reasoning_effort for model o3 is not supported for this account", 400),
    _Err("Parameter reasoning_effort for model o3 is not supported for this account"),
    _Err("Error: Streaming model is not supported for this account"),
    _Err("Vision: model is not supported for this account"),
    # a parameter right after "model" is the subject, not a model id
    _Err("The model parameter is not supported for this account", 400),
    _Err("The model parameter is not supported for this account"),
    _Err("The model tool_choice is not supported for this account", 400),
    _Err("The model tool_choice is not supported for this account"),
    _Err("The model temperature is not supported when using this account", 400),
    _Err("The model temperature is not supported when using this account"),
])
def test_account_scoped_parameter_or_feature_restriction_never_rotates(err):
    verdict = classify_api_error(err)
    assert verdict.reason is not _R.model_entitlement
    assert verdict.should_rotate_credential is False


# ── an exception raised while handling another is its own failure ───────


def _raised_while_handling_a_401(outer):
    try:
        try:
            raise openai.AuthenticationError(
                "Error code: 401", response=httpx.Response(401, request=_REQ), body={"message": "invalid api key"})
        except openai.AuthenticationError:
            raise outer
    except Exception as exc:
        return exc


@pytest.mark.parametrize("outer", [
    ConnectionError("Connection refused"), TimeoutError("timed out"), openai.APIConnectionError(request=_REQ),
])
def test_transport_failure_raised_during_a_401_retry_does_not_inherit_the_401(outer):
    """A credential refresh retried inside the 401's handler: the new attempt's transport failure carries
    the 401 only as implicit ``__context__`` and is not an auth failure."""
    verdict = classify_api_error(_raised_while_handling_a_401(outer), provider="openrouter")
    assert (verdict.reason, verdict.status_code, verdict.should_rotate_credential) == (_R.timeout, None, False)


def _invalid_response_during_a_401_retry():
    def validate():
        try:
            raise AttributeError("missing choices[0].message")
        except AttributeError as exc:
            raise InvalidProviderResponseError("LLM returned invalid response") from exc
    try:
        validate()
    except InvalidProviderResponseError as exc:
        return exc


@pytest.mark.parametrize("outer", [
    RuntimeError("Connection reset by peer"),
    RuntimeError("peer closed connection without sending complete message body (incomplete chunked read)"),
    RuntimeError("[Errno 61] Connection refused"),
])
def test_message_only_transport_failure_during_a_401_retry_does_not_inherit_the_401(outer):
    verdict = classify_api_error(_raised_while_handling_a_401(outer), provider="openrouter")
    assert (verdict.reason, verdict.status_code, verdict.should_rotate_credential) == (_R.timeout, None, False)


def test_explicit_cause_raised_during_a_401_retry_does_not_inherit_the_401():
    """A wrapper raised ``from`` an AttributeError that itself carries the 401 as implicit context: the
    explicit cause is followed, the cause's implicit context is not (it belongs to the earlier attempt)."""
    def retry():
        try:
            raise AttributeError("missing choices[0].message")
        except AttributeError as exc:
            raise RuntimeError("provider call failed") from exc

    try:
        try:
            raise openai.AuthenticationError(
                "Error code: 401", response=httpx.Response(401, request=_REQ), body={"message": "invalid api key"})
        except openai.AuthenticationError:
            retry()
    except RuntimeError as exc:
        err = exc
    assert err.__cause__.__context__.status_code == 401
    verdict = classify_api_error(err, provider="openrouter")
    assert (verdict.status_code, verdict.should_rotate_credential) == (None, False)


def test_invalid_response_during_a_401_retry_does_not_inherit_the_401():
    """An HTTP-200 answer with no usable message is the retry's own failure, whatever its cause chain."""
    try:
        try:
            raise openai.AuthenticationError(
                "Error code: 401", response=httpx.Response(401, request=_REQ), body={"message": "invalid api key"})
        except openai.AuthenticationError:
            raise _invalid_response_during_a_401_retry()
    except InvalidProviderResponseError as exc:
        err = exc
    assert err.__cause__.__context__.status_code == 401  # the shape that used to inherit the 401
    verdict = classify_api_error(err, provider="openrouter")
    assert (verdict.reason, verdict.status_code, verdict.should_rotate_credential) == (
        _R.incomplete_response, None, False)


def test_explicit_cause_and_a_bare_rewrap_still_read_the_wrapped_status():
    explicit = RuntimeError("provider call failed")
    explicit.__cause__ = _raised_while_handling_a_401(ValueError("unused")).__context__
    bare = _raised_while_handling_a_401(RuntimeError("provider call failed: invalid api key"))
    assert classify_api_error(explicit).status_code == 401
    assert classify_api_error(bare).status_code == 401


# ── the auxiliary client's base predicates, by class name and wording ───


def _named(class_name, message="request failed", **attrs):
    """An exception whose class NAME is an SDK's (botocore, aiohttp, websocket) but not its hierarchy."""
    exc = type(class_name, (Exception,), {})(message)
    for name, value in attrs.items():
        setattr(exc, name, value)
    return exc


_BEDROCK = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude/converse"


@pytest.mark.parametrize("exc,transport", [
    (_named("EndpointConnectionError", f'Could not connect to the endpoint URL: "{_BEDROCK}"'), "connection"),
    (_named("ProxyConnectionError", 'Failed to connect to proxy URL: "http://proxy:8080"'), "connection"),
    (_named("ConnectionClosedError", "Connection was closed before we received a valid response"), "connection"),
    (_named("ClientConnectorDNSError", "Cannot connect to host api.example.com:443"), "connection"),
    (_named("WebSocketConnectionClosedException", "socket is already closed."), "connection"),
    (_named("DNSLookupError"), "connection"),
    (_named("ConnectTimeoutError", f'Connect timeout on endpoint URL: "{_BEDROCK}"'), "timeout"),
    (_named("ReadTimeoutError", f'Read timeout on endpoint URL: "{_BEDROCK}"'), "timeout"),
    (_named("GatewayTimeout", "the upstream gateway did not answer"), "timeout"),
], ids=lambda v: getattr(type(v), "__name__", v))
def test_sdk_transport_class_names_are_transport_failures(exc, transport):
    verdict = classify_api_error(exc, provider="bedrock")
    assert verdict.reason is _R.timeout and verdict.retryable
    assert verdict.error_context["transport"] == transport


def test_real_botocore_endpoint_connection_error_is_a_connection_failure():
    exceptions = pytest.importorskip("botocore.exceptions")
    verdict = classify_api_error(exceptions.EndpointConnectionError(endpoint_url=_BEDROCK), provider="bedrock")
    assert verdict.reason is _R.timeout
    assert verdict.error_context["transport"] == "connection"


def test_a_status_settles_a_transport_named_class():
    assert classify_api_error(_named("ConnectionAbortedByProxy", "Forbidden", status_code=403)).reason is _R.auth


@pytest.mark.parametrize("class_name", ["AuthenticationError", "AnthropicAuthenticationError"])
def test_statusless_authentication_error_class_is_a_401(class_name):
    verdict = classify_api_error(_named(class_name, "invalid x-api-key"))
    assert verdict.reason is _R.auth and verdict.should_rotate_credential
    assert verdict.status_code == 401


@pytest.mark.parametrize("status", [400, 403, None])
def test_xai_bad_credentials_is_auth_whatever_the_status(status):
    verdict = classify_api_error(_Err("Error code: unauthenticated:bad-credentials", status))
    assert verdict.reason is _R.auth and verdict.should_rotate_credential


@pytest.mark.parametrize("status", [403, 404, 429, None])
@pytest.mark.parametrize("wording", [
    "You have run out of credits.", "Budget limit reached for this organization.", "Key limit exceeded.",
    "This model requires a subscription, upgrade for access.", "Upgrade for higher limits.",
])
def test_account_walls_are_billing_on_the_statuses_that_carry_them(wording, status):
    verdict = classify_api_error(_Err(wording, status), provider="openrouter")
    assert verdict.reason is _R.billing and verdict.should_rotate_credential


@pytest.mark.parametrize("status", [403, 404])
def test_resource_exhausted_on_a_403_or_404_is_an_account_wall(status):
    assert classify_api_error(_Err("RESOURCE_EXHAUSTED: project quota", status)).reason is _R.billing


@pytest.mark.parametrize("status", [400, 500])
def test_bare_credits_wording_outside_the_wall_statuses_stays_unbilled(status):
    assert classify_api_error(_Err("Request used 12 credits.", status)).reason is not _R.billing


@pytest.mark.parametrize("status", [400, 404, None])
def test_openrouter_catalog_wording_is_a_missing_model(status):
    verdict = classify_api_error(_Err("Model 'x/y' is missing from the OpenRouter catalog.", status))
    assert verdict.reason is _R.model_not_found


@pytest.mark.parametrize("word", ["credits", "billing", "free tier", "free-tier"])
def test_a_wall_word_keeps_model_wording_off_the_route_rejection(word):
    verdict = classify_api_error(_Err(f"model is not supported ({word})", 400))
    assert "route_rejects_model" not in verdict.error_context


# ── every SDK error body shape raised during a 401 retry is its own failure ──


class _TextOnlyResponse:
    """An HTTP client error's response with no status and a non-JSON body."""
    text = "upstream worker crashed"

    def json(self):
        raise ValueError("not json")


def _with_response(exc, response):
    exc.response = response
    return exc


def _stream_error(body, message="An error occurred during streaming"):
    return openai.APIError(message, request=_REQ, body=body)


_OWN_FAILURE_SHAPES = {
    "string-body": lambda: _stream_error("backend worker failed"),
    "error-string-envelope": lambda: _stream_error({"error": "backend worker failed"}),
    "error-object-envelope": lambda: _stream_error({"error": {"message": "backend worker failed", "type": "server"}}),
    "flat-object": lambda: _stream_error({"message": "backend worker failed"}),
    "list-body": lambda: _stream_error([{"message": "backend worker failed"}]),
    "bytes-body": lambda: _stream_error(b"backend worker failed"),
    "no-body-own-message": lambda: _stream_error(None, "backend worker failed"),
    "status-error-no-body": lambda: openai.InternalServerError(
        "Error code: 500", response=httpx.Response(500, request=_REQ), body=None),
    "httpx-decoding-error": lambda: httpx.DecodingError("Error -3 while decompressing data", request=_REQ),
    "httpx-error-no-request": lambda: httpx.LocalProtocolError("Illegal header value"),
    "httpx-stream-error": lambda: httpx.StreamClosed(),
    "response-text-only": lambda: _with_response(Exception("request failed"), _TextOnlyResponse()),
}


@pytest.mark.parametrize("shape", list(_OWN_FAILURE_SHAPES))
def test_sdk_error_of_any_body_shape_during_a_401_retry_is_its_own_failure(shape):
    """A status-less (or other-status) SDK / HTTP-client error raised while a 401 is handled carries its
    own body or message: it is the retry's failure, never the inherited 401."""
    verdict = classify_api_error(_raised_while_handling_a_401(_OWN_FAILURE_SHAPES[shape]()), provider="openrouter")
    assert verdict.status_code != 401
    assert verdict.reason not in (_R.auth, _R.auth_permanent) and verdict.should_rotate_credential is False


@pytest.mark.parametrize("outer", [
    _stream_error("", ""), _stream_error("   ", ""), RuntimeError("provider call failed"),
], ids=["empty-body-and-message", "blank-body", "bare-rewrap"])
def test_a_rewrap_with_nothing_of_its_own_still_reads_the_401(outer):
    assert classify_api_error(_raised_while_handling_a_401(outer), provider="openrouter").status_code == 401


def _raise(exc):
    raise exc


def _called_while_handling_a_401(exc):
    """``exc`` raised by a call made inside the 401's handler (a retry after a refresh, its answer check)."""
    try:
        try:
            raise openai.AuthenticationError(
                "Error code: 401", response=httpx.Response(401, request=_REQ), body={"message": "invalid api key"})
        except openai.AuthenticationError:
            _raise(exc)
    except Exception as raised:
        return raised


_RETRY_OWN_FAILURES = {
    "answer-check": lambda: RuntimeError("Auxiliary title_generation: LLM returned None response"),
    "codex-no-final-response": lambda: RuntimeError("Codex auxiliary Responses stream did not return a final response"),
    "empty-stream": lambda: EmptyStreamError("Anthropic stream ended before message_stop"),
    "request-contract": lambda: ValueError("Codex Responses input item 0 has no role"),
    "bare-runtime-error": lambda: RuntimeError("provider call failed"),
}


@pytest.mark.parametrize("shape", list(_RETRY_OWN_FAILURES))
def test_failure_raised_by_a_call_inside_a_401_handler_is_its_own(shape):
    """Only a re-wrap raised in the frame that caught the 401 reads it: anything a call made inside the
    handler raises is that call's failure, classified as if no 401 came before it."""
    verdict = classify_api_error(_called_while_handling_a_401(_RETRY_OWN_FAILURES[shape]()), provider="openrouter")
    alone = classify_api_error(_RETRY_OWN_FAILURES[shape](), provider="openrouter")
    assert verdict.status_code is None and verdict.should_rotate_credential is False
    assert verdict.reason is alone.reason and verdict.reason not in (_R.auth, _R.auth_permanent)


@pytest.mark.parametrize("outer", [
    json.JSONDecodeError("Expecting property name enclosed in double quotes", "{bad json}", 1),
    ValueError("expected ident at line 1 column 2"),
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
], ids=["json-decode-error", "stream-parse-error", "unicode-decode-error"])
def test_malformed_payload_during_a_401_retry_is_its_own_failure(outer):
    verdict = classify_api_error(_raised_while_handling_a_401(outer), provider="openrouter")
    assert verdict.status_code is None and verdict.reason not in (_R.auth, _R.auth_permanent)


def test_a_rewrap_with_a_hand_set_context_still_reads_the_401():
    """No frames to compare (never raised): the re-wrap reading holds."""
    rewrap = RuntimeError("provider call failed")
    rewrap.__context__ = openai.AuthenticationError(
        "Error code: 401", response=httpx.Response(401, request=_REQ), body={"message": "invalid api key"})
    assert classify_api_error(rewrap, provider="openrouter").status_code == 401


# ── chained-exception edges: deadlines own their verdict; helper re-wraps are their own failure ──


def _auth_401():
    return openai.AuthenticationError(
        "Error code: 401", response=httpx.Response(401, request=_REQ), body={"message": "invalid api key"})


def test_deadline_timeout_raised_from_a_401_is_a_timeout():
    class _StreamDeadline(TimeoutError):
        pass

    for timeout_cls in (TimeoutError, _StreamDeadline):
        try:
            try:
                raise _auth_401()
            except openai.AuthenticationError as exc:
                raise timeout_cls("stream stalled: no new output for 60.0s (timed out)") from exc
        except TimeoutError as deadline:
            verdict = classify_api_error(deadline, provider="openrouter")
        assert verdict.reason is _R.timeout and not verdict.should_rotate_credential, timeout_cls


def test_explicit_cause_still_carries_the_status_of_an_ordinary_wrapper():
    try:
        try:
            raise _auth_401()
        except openai.AuthenticationError as exc:
            raise RuntimeError("provider call failed") from exc
    except RuntimeError as wrapped:
        assert classify_api_error(wrapped, provider="openrouter").reason is _R.auth


def _raise_friendly(exc):
    del exc  # a helper handed the caught error, re-wrapping it without ``from``
    raise RuntimeError("provider call failed")


class _Rewrap:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if exc_info[1] is not None:
            raise RuntimeError("provider call failed")


@pytest.mark.parametrize("shape", ["helper", "exit"])
def test_rewrap_outside_the_handler_frame_is_its_own_failure(shape):
    """Deliberate: only a re-wrap in the frame that caught the error (or ``from``) inherits its status."""
    try:
        if shape == "helper":
            try:
                raise _auth_401()
            except openai.AuthenticationError as exc:
                _raise_friendly(exc)
        else:
            with _Rewrap():
                raise _auth_401()
    except RuntimeError as wrapped:
        assert classify_api_error(wrapped, provider="openrouter").reason is not _R.auth
