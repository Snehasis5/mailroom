"""
Request/response/domain models for the mailroom agent.

IMPORTANT: The exam page references "Exact propose request and response" /
"Exact commit request and terminal response" examples that were NOT present
in the pasted spec text (they were likely collapsible sections that didn't
copy over). The field names below are a best-effort reconstruction from the
prose description ("dossiers", "evaluationId", "proposals", "receipts",
"outcomes", "callId"). If the real page shows different field names,
this is the ONLY file you should need to edit -- everything else
(canonical.py, db.py, llm.py, main.py) is written against these classes,
not against raw dict keys.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


ACTIONS = (
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
)


class ActionType(str, Enum):
    create_draft = "create_draft"
    update_internal_record = "update_internal_record"
    send_approved_notice = "send_approved_notice"
    request_confirmation = "request_confirmation"
    quarantine_item = "quarantine_item"
    no_action = "no_action"


# ---------------------------------------------------------------------------
# Inbound: propose
# ---------------------------------------------------------------------------

class Dossier(BaseModel):
    dossierId: str
    content: dict[str, Any]

    @field_validator("dossierId")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("dossierId must be non-empty")
        return v


class ProposeRequest(BaseModel):
    operation: Literal["propose"]
    evaluationId: str
    dossiers: list[Dossier] = Field(min_length=1)

    @field_validator("dossiers")
    @classmethod
    def _unique_ids(cls, v: list[Dossier]) -> list[Dossier]:
        ids = [d.dossierId for d in v]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate dossierId in request")
        return v


# ---------------------------------------------------------------------------
# Inbound: commit
# ---------------------------------------------------------------------------

class Receipt(BaseModel):
    dossierId: str
    callId: str
    proposalDigest: str
    # The grader's "unpredictable receipt" -- opaque approval/rejection token.
    # We do not assume its internal shape beyond an "approved" flag and a
    # verification key/signature, both optional so unknown extra fields
    # from the real schema still validate (see model_config below).
    approved: Optional[bool] = None
    verificationKey: Optional[str] = None

    model_config = {"extra": "allow"}


class CommitRequest(BaseModel):
    operation: Literal["commit"]
    evaluationId: str
    receipts: list[Receipt] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    quote: str
    field: str


class Proposal(BaseModel):
    dossierId: str
    callId: str
    action: ActionType
    target: dict[str, Any]
    payload: dict[str, Any]
    evidence: list[Evidence]
    proposalDigest: str


class ProposeResponse(BaseModel):
    status: Literal["awaiting_receipts"]
    evaluationId: str
    proposals: list[Proposal]


class Outcome(BaseModel):
    dossierId: str
    callId: str
    result: Literal["executed", "rejected", "quarantined", "no_action", "queued"]
    detail: str


class CommitResponse(BaseModel):
    status: Literal["completed"]
    evaluationId: str
    outcomes: list[Outcome]


# ---------------------------------------------------------------------------
# What the LLM must return for a single dossier (validated before it becomes
# a Proposal). Kept intentionally narrow/strict so a hallucinated field or
# an attempt to smuggle raw dossier text into an outbound argument fails
# validation instead of silently proceeding.
# ---------------------------------------------------------------------------

class LLMDecision(BaseModel):
    action: ActionType
    target: dict[str, Any]
    payload: dict[str, Any]
    evidence: list[Evidence] = Field(min_length=1, max_length=5)
    reasoning: str = Field(max_length=600)
