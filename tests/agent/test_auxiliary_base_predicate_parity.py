"""Every class-name and wording rule of the auxiliary client's removed error predicates, on the policy.

Each fixture row is a synthetic exception built for one rule of the predicates the auxiliary ladder used
before it read ``classify_api_error`` (``_is_connection_error``, ``_is_timeout_error``, ``_is_auth_error``,
``_is_payment_error``, ``_is_rate_limit_error``, the model and request-shape predicates): an SDK class
NAME such as botocore's ``EndpointConnectionError`` outside the builtin hierarchy, or a needle in a
realistic body, on each status the rule was gated on. ``expected`` is what the policy decides now;
``why`` says whether that is base's decision (``base``) or a named, accepted change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import auxiliary_error_policy as aux_errors

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "aux_base_predicate_parity.jsonl"
_ROWS = [json.loads(line) for line in _FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
_PARENTS = {"Exception": Exception, "RuntimeError": RuntimeError, "ValueError": ValueError}
_INPUT_COLUMNS = ("class", "parent", "status", "status_attr", "message")  # what ``_build`` reads


def _build(row):
    exc = type(row["class"], (_PARENTS[row["parent"]],), {})(row["message"])
    if row["status"] is not None:
        if row["status_attr"] == "response":
            exc.response = type("Response", (), {"status_code": row["status"], "headers": {}})()
        else:
            setattr(exc, row["status_attr"], row["status"])
    return exc


def _decisions(exc):
    return [aux_errors.same_provider_retry(exc), aux_errors.transport(exc), aux_errors.fallback_label(exc),
            aux_errors.auth_refresh(exc), aux_errors.rotates_credential(exc) or aux_errors.auth_refresh(exc),
            aux_errors.benches_provider(exc), aux_errors.model_not_found(exc)]


def test_every_row_is_base_or_a_named_change():
    assert len(_ROWS) > 300
    assert all(row["why"] == "base" or row["why"].split(":")[0] in {
        "approved", "ruling", "status decides", "by construction"} for row in _ROWS)
    shapes = [json.dumps([row[k] for k in _INPUT_COLUMNS]) for row in _ROWS]
    assert len(set(shapes)) == len(shapes), "duplicate fixture rows (same exception twice)"


@pytest.mark.parametrize("row", _ROWS, ids=[f"{r['id']}-{r['class']}-{r['status']}" for r in _ROWS])
def test_removed_predicate_rule_maps_to_the_expected_decision(row, monkeypatch):
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "get_plugin_error_classification", lambda **_kw: None)
    assert _decisions(_build(row)) == row["expected"], row["why"]
