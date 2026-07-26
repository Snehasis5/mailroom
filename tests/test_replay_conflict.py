"""
Run with: pytest -q
Uses a temp sqlite file per test session and monkeypatches app.llm.decide so
NOTHING here ever calls a real model or spends a token -- exactly the "test
exact replay, changed-content conflicts, and malformed input without calling
the model" build-order step from the spec.
"""
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

os.environ["MAILROOM_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

from app import main as main_module  # noqa: E402
from app import llm as llm_module  # noqa: E402
from app.models import LLMDecision  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    async def fake_decide(dossier_id, content, attempts=2):
        return LLMDecision(
            action="no_action",
            target={},
            payload={"reason": "stub"},
            evidence=[{"quote": dossier_id, "field": "dossierId"}],
            reasoning="stub decision, no model called",
        )

    monkeypatch.setattr(llm_module, "decide", fake_decide)
    monkeypatch.setattr(main_module, "decide", fake_decide)
    yield


client = TestClient(main_module.app)


def _propose_body(eval_id="eval-1", dossier_text="hello world"):
    return {
        "operation": "propose",
        "evaluationId": eval_id,
        "dossiers": [{"dossierId": "d1", "content": {"body": dossier_text}}],
    }


def test_malformed_json_returns_400():
    resp = client.post("/v1/mailroom/actions", content=b"{not json", headers={"content-type": "application/json"})
    assert resp.status_code == 400


def test_invalid_operation_returns_400():
    resp = client.post("/v1/mailroom/actions", json={"operation": "bogus"})
    assert resp.status_code == 400


def test_duplicate_dossier_id_returns_422():
    body = {
        "operation": "propose",
        "evaluationId": "eval-dup",
        "dossiers": [
            {"dossierId": "d1", "content": {"body": "a"}},
            {"dossierId": "d1", "content": {"body": "b"}},
        ],
    }
    resp = client.post("/v1/mailroom/actions", json=body)
    assert resp.status_code == 422


def test_propose_then_exact_replay_is_byte_equivalent():
    body = _propose_body(eval_id="eval-replay")
    r1 = client.post("/v1/mailroom/actions", json=body)
    r2 = client.post("/v1/mailroom/actions", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()


def test_same_evaluation_id_changed_content_is_409():
    body1 = _propose_body(eval_id="eval-conflict", dossier_text="version A")
    body2 = _propose_body(eval_id="eval-conflict", dossier_text="version B")
    r1 = client.post("/v1/mailroom/actions", json=body1)
    r2 = client.post("/v1/mailroom/actions", json=body2)
    assert r1.status_code == 200
    assert r2.status_code == 409


def test_commit_unknown_evaluation_returns_400():
    resp = client.post("/v1/mailroom/actions", json={
        "operation": "commit",
        "evaluationId": "does-not-exist",
        "receipts": [{"dossierId": "d1", "callId": "x", "proposalDigest": "y", "approved": True}],
    })
    assert resp.status_code == 400


def test_commit_digest_mismatch_is_rejected_not_executed():
    propose_resp = client.post("/v1/mailroom/actions", json=_propose_body(eval_id="eval-mismatch"))
    assert propose_resp.status_code == 200
    commit_resp = client.post("/v1/mailroom/actions", json={
        "operation": "commit",
        "evaluationId": "eval-mismatch",
        "receipts": [{
            "dossierId": "d1", "callId": "wrong-call-id",
            "proposalDigest": "wrong-digest", "approved": True,
        }],
    })
    assert commit_resp.status_code == 200
    outcome = commit_resp.json()["outcomes"][0]
    assert outcome["result"] == "rejected"


def test_commit_replay_is_idempotent():
    propose_resp = client.post("/v1/mailroom/actions", json=_propose_body(eval_id="eval-commit-replay"))
    proposal = propose_resp.json()["proposals"][0]
    commit_body = {
        "operation": "commit",
        "evaluationId": "eval-commit-replay",
        "receipts": [{
            "dossierId": "d1", "callId": proposal["callId"],
            "proposalDigest": proposal["proposalDigest"], "approved": True,
        }],
    }
    r1 = client.post("/v1/mailroom/actions", json=commit_body)
    r2 = client.post("/v1/mailroom/actions", json=commit_body)
    assert r1.json() == r2.json()


def test_unapproved_receipt_without_signal_is_rejected():
    propose_resp = client.post("/v1/mailroom/actions", json=_propose_body(eval_id="eval-noapproval"))
    proposal = propose_resp.json()["proposals"][0]
    commit_resp = client.post("/v1/mailroom/actions", json={
        "operation": "commit",
        "evaluationId": "eval-noapproval",
        "receipts": [{
            "dossierId": "d1", "callId": proposal["callId"],
            "proposalDigest": proposal["proposalDigest"],
        }],
    })
    outcome = commit_resp.json()["outcomes"][0]
    assert outcome["result"] == "rejected"
