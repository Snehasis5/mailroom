"""
Canonical serialization + hashing.

Every fingerprint used for caching, replay-detection, and conflict-detection
in this service is derived from these two functions ONLY. Never hash a
Python repr(), never hash with default dict ordering -- always go through
canonical_json first.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, stable float/str repr."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_fingerprint(content: dict[str, Any]) -> str:
    """Fingerprint of a single dossier's content (identity of *what it says*,
    independent of which evaluationId it arrived under)."""
    return sha256_hex(canonical_json(content))


def evaluation_fingerprint(dossier_ids_and_fingerprints: list[tuple[str, str]]) -> str:
    """Fingerprint of the full set of (dossierId, contentFingerprint) pairs
    submitted under one evaluationId, used to detect 'same evaluationId,
    changed content' -> HTTP 409."""
    ordered = sorted(dossier_ids_and_fingerprints, key=lambda p: p[0])
    return sha256_hex(canonical_json(ordered))


def call_id_for(dossier_id: str, content_fp: str) -> str:
    """Stable callId: depends only on dossierId + content, so the SAME
    dossier produces the SAME callId across different evaluations and later
    Checks, per the spec's 'stable dossiers must produce the same complete
    proposal and callId across evaluations' requirement."""
    return sha256_hex(f"callId:{dossier_id}:{content_fp}")[:32]


def proposal_digest_for(
    dossier_id: str,
    call_id: str,
    action: str,
    target: dict[str, Any],
    payload: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    """Digest of the *decision itself*. Used at commit time to verify the
    receipt refers to exactly the proposal we issued (not a stale or
    tampered one)."""
    body = {
        "dossierId": dossier_id,
        "callId": call_id,
        "action": action,
        "target": target,
        "payload": payload,
        "evidence": evidence,
    }
    return sha256_hex(canonical_json(body))
