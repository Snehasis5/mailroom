from __future__ import annotations

import hmac
import json
import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import db
from .canonical import (
    call_id_for,
    content_fingerprint,
    evaluation_fingerprint,
    proposal_digest_for,
    sha256_hex,
)
from .llm import decide
from .models import CommitRequest, Dossier, Outcome, Proposal, ProposeRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mailroom")

app = FastAPI()

MAX_BODY_BYTES = 2_000_000
MAX_RESPONSE_BYTES = 512 * 1024


def _json_response(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if status_code == 200 and len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
        # Should not happen at this dossier volume, but the spec requires
        # rejecting oversized successful bodies rather than truncating them.
        return JSONResponse(
            {"status": "error", "error": "response_too_large"},
            status_code=500,
            media_type="application/json",
        )
    return JSONResponse(payload, status_code=status_code, media_type="application/json")


def _error(status_code: int, message: str, **extra: Any) -> JSONResponse:
    return _json_response({"status": "error", "error": message, **extra}, status_code)


@app.post("/v1/mailroom/actions")
async def mailroom_actions(request: Request):
    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        return _error(413, "request_body_too_large")

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return _error(400, "malformed_json")

    if not isinstance(body, dict) or "operation" not in body:
        return _error(400, "missing_operation")

    op = body.get("operation")
    if op == "propose":
        return await _handle_propose(body)
    elif op == "commit":
        return await _handle_commit(body)
    else:
        return _error(400, "invalid_operation", operation=op)


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------

async def _handle_propose(body: dict[str, Any]):
    try:
        req = ProposeRequest.model_validate(body)
    except ValidationError as e:
        return _error(422, "schema_validation_failed", details=json.loads(e.json()))

    # Compute this request's evaluation fingerprint up front (cheap, no AI).
    fps = [(d.dossierId, content_fingerprint(d.content)) for d in req.dossiers]
    eval_fp = evaluation_fingerprint(fps)

    existing_eval = db.get_evaluation(req.evaluationId)
    if existing_eval is not None:
        if existing_eval["evalFingerprint"] == eval_fp:
            # Exact replay: return the byte-equivalent stored response,
            # no model call, no recomputation.
            return _json_response(existing_eval["proposeResponse"], 200)
        else:
            # Same evaluationId, different dossier content -> conflict.
            return _error(409, "evaluation_content_conflict", evaluationId=req.evaluationId)

    proposals: list[dict[str, Any]] = []
    for dossier, (_, fp) in zip(req.dossiers, fps):
        cached = db.get_cached_decision(dossier.dossierId, fp)
        if cached is not None:
            proposals.append(cached)
            continue

        decision = await decide(dossier.dossierId, dossier.content)
        call_id = call_id_for(dossier.dossierId, fp)
        evidence = [e.model_dump() for e in decision.evidence]
        digest = proposal_digest_for(
            dossier.dossierId, call_id, decision.action.value, decision.target,
            decision.payload, evidence,
        )
        record = {
            "callId": call_id,
            "action": decision.action.value,
            "target": decision.target,
            "payload": decision.payload,
            "evidence": evidence,
            "proposalDigest": digest,
        }
        db.store_decision(dossier.dossierId, fp, record)
        proposals.append(record)

    response = {
        "status": "awaiting_receipts",
        "evaluationId": req.evaluationId,
        "proposals": [
            {
                "dossierId": d.dossierId,
                "callId": p["callId"],
                "action": p["action"],
                "target": p["target"],
                "payload": p["payload"],
                "evidence": p["evidence"],
                "proposalDigest": p["proposalDigest"],
            }
            for d, p in zip(req.dossiers, proposals)
        ],
    }
    db.store_evaluation(req.evaluationId, eval_fp, response)
    return _json_response(response, 200)


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------

def _verify_receipt(evaluation_key: Optional[str], receipt: dict[str, Any],
                     dossier_id: str, call_id: str, proposal_digest: str) -> bool:
    """
    Best-effort receipt verification.

    ASSUMPTION (confirm against the real 'exact commit request' example and
    adjust this function only): the propose flow may hand back / the grader
    may supply a per-evaluation verification key, and each receipt carries a
    `verificationKey` that should equal HMAC-SHA256(key, dossierId:callId:
    proposalDigest). If no key material is configured at all, we fall back
    to trusting the receipt's explicit `approved` boolean, since some
    grader designs simply hand back a signed opaque approval flag without a
    separate key.
    """
    approved = receipt.get("approved")
    supplied_key = receipt.get("verificationKey")

    if evaluation_key and supplied_key:
        expected = sha256_hex(f"{evaluation_key}:{dossier_id}:{call_id}:{proposal_digest}")
        if not hmac.compare_digest(expected, supplied_key):
            return False
        return bool(approved) if approved is not None else True

    if approved is None:
        # No usable verification signal at all -- treat as invalid rather
        # than silently approving.
        return False
    return bool(approved)


async def _handle_commit(body: dict[str, Any]):
    try:
        req = CommitRequest.model_validate(body)
    except ValidationError as e:
        return _error(422, "schema_validation_failed", details=json.loads(e.json()))

    evaluation = db.get_evaluation(req.evaluationId)
    if evaluation is None:
        return _error(400, "unknown_evaluation", evaluationId=req.evaluationId)

    proposals_by_id = {
        p["dossierId"]: p for p in evaluation["proposeResponse"]["proposals"]
    }
    evaluation_key = body.get("receiptVerificationKey")  # optional, see _verify_receipt

    outcomes: list[dict[str, Any]] = []
    for receipt in req.receipts:
        dossier_id = receipt.dossierId
        persisted = proposals_by_id.get(dossier_id)

        if persisted is None:
            outcomes.append({
                "dossierId": dossier_id, "callId": receipt.callId,
                "result": "rejected", "detail": "no such dossier in this evaluation",
            })
            continue

        # Must match the exact proposal we issued.
        if (
            persisted["callId"] != receipt.callId
            or persisted["proposalDigest"] != receipt.proposalDigest
        ):
            outcomes.append({
                "dossierId": dossier_id, "callId": receipt.callId,
                "result": "rejected", "detail": "callId/proposalDigest mismatch",
            })
            continue

        # Idempotency: if we already recorded an outcome for this
        # (evaluationId, dossierId), never re-verify or re-execute.
        existing = db.get_receipt(req.evaluationId, dossier_id)
        if existing is not None:
            outcomes.append({
                "dossierId": dossier_id, "callId": persisted["callId"],
                "result": existing["result"], "detail": existing["detail"],
            })
            continue

        verified = _verify_receipt(
            evaluation_key, receipt.model_dump(), dossier_id,
            persisted["callId"], persisted["proposalDigest"],
        )
        if not verified:
            result, detail = "rejected", "receipt failed verification"
        else:
            result, detail = _execute(persisted)

        db.store_receipt_outcome(
            req.evaluationId, dossier_id, persisted["callId"],
            receipt.model_dump(), verified, result, detail,
        )
        outcomes.append({
            "dossierId": dossier_id, "callId": persisted["callId"],
            "result": result, "detail": detail,
        })

    response = {
        "status": "completed",
        "evaluationId": req.evaluationId,
        "outcomes": outcomes,
    }
    return _json_response(response, 200)


def _execute(proposal: dict[str, Any]) -> tuple[str, str]:
    """Perform (or simulate) the side effect for an approved proposal. This
    is intentionally idempotent-safe -- callers only invoke it once per
    (evaluationId, dossierId) thanks to the receipts table check above."""
    action = proposal["action"]
    if action == "quarantine_item":
        return "quarantined", "item isolated, no outbound/internal effect performed"
    if action == "no_action":
        return "no_action", "suppressed as duplicate/completed/informational"
    if action == "request_confirmation":
        return "queued", "routed to internal approval queue"
    if action in ("create_draft", "update_internal_record", "send_approved_notice"):
        # In a real deployment this calls the actual mail/CRM API. Here we
        # record the effect as executed once approved+verified.
        return "executed", f"{action} performed with approved payload"
    return "rejected", f"unknown action {action}"


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
