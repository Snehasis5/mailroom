"""
Durable persistence. SQLite (WAL mode) behind a process-wide lock -- simple,
survives restarts, and is fine at this request volume (a single Check run is
~64-70 dossiers). Nothing here relies on in-process memory for state that
must survive a request boundary or a redeploy.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

DB_PATH = os.environ.get("MAILROOM_DB_PATH", "/data/mailroom.db")

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.execute("PRAGMA synchronous=NORMAL;")
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS dossier_decisions (
            dossier_id TEXT NOT NULL,
            content_fp TEXT NOT NULL,
            call_id TEXT NOT NULL,
            action TEXT NOT NULL,
            target_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            proposal_digest TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (dossier_id, content_fp)
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            evaluation_id TEXT PRIMARY KEY,
            eval_fingerprint TEXT NOT NULL,
            propose_response_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS receipts (
            evaluation_id TEXT NOT NULL,
            dossier_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            verified INTEGER NOT NULL,
            result TEXT NOT NULL,
            detail TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (evaluation_id, dossier_id)
        );

        CREATE TABLE IF NOT EXISTS commits (
            evaluation_id TEXT PRIMARY KEY,
            commit_response_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """
    )
    conn.commit()


@contextmanager
def tx():
    with _lock:
        conn = _get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ---------------------------------------------------------------------------
# Dossier decision cache (keyed by dossierId + content fingerprint, NOT by
# evaluationId -- this is what makes later Checks/Save free of model calls).
# ---------------------------------------------------------------------------

def get_cached_decision(dossier_id: str, content_fp: str) -> Optional[dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT call_id, action, target_json, payload_json, evidence_json, proposal_digest "
        "FROM dossier_decisions WHERE dossier_id=? AND content_fp=?",
        (dossier_id, content_fp),
    ).fetchone()
    if row is None:
        return None
    return {
        "callId": row[0],
        "action": row[1],
        "target": json.loads(row[2]),
        "payload": json.loads(row[3]),
        "evidence": json.loads(row[4]),
        "proposalDigest": row[5],
    }


def store_decision(dossier_id: str, content_fp: str, decision: dict[str, Any]) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO dossier_decisions "
            "(dossier_id, content_fp, call_id, action, target_json, payload_json, "
            "evidence_json, proposal_digest, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                dossier_id,
                content_fp,
                decision["callId"],
                decision["action"],
                json.dumps(decision["target"]),
                json.dumps(decision["payload"]),
                json.dumps(decision["evidence"]),
                decision["proposalDigest"],
                time.time(),
            ),
        )


# ---------------------------------------------------------------------------
# Evaluations (propose-side replay / conflict detection)
# ---------------------------------------------------------------------------

def get_evaluation(evaluation_id: str) -> Optional[dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT eval_fingerprint, propose_response_json FROM evaluations WHERE evaluation_id=?",
        (evaluation_id,),
    ).fetchone()
    if row is None:
        return None
    return {"evalFingerprint": row[0], "proposeResponse": json.loads(row[1])}


def store_evaluation(evaluation_id: str, eval_fp: str, propose_response: dict[str, Any]) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO evaluations (evaluation_id, eval_fingerprint, "
            "propose_response_json, created_at) VALUES (?,?,?,?)",
            (evaluation_id, eval_fp, json.dumps(propose_response), time.time()),
        )


# ---------------------------------------------------------------------------
# Receipts / commit (idempotent execution + full-commit replay)
# ---------------------------------------------------------------------------

def get_receipt(evaluation_id: str, dossier_id: str) -> Optional[dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT call_id, verified, result, detail FROM receipts "
        "WHERE evaluation_id=? AND dossier_id=?",
        (evaluation_id, dossier_id),
    ).fetchone()
    if row is None:
        return None
    return {"callId": row[0], "verified": bool(row[1]), "result": row[2], "detail": row[3]}


def store_receipt_outcome(
    evaluation_id: str,
    dossier_id: str,
    call_id: str,
    receipt: dict[str, Any],
    verified: bool,
    result: str,
    detail: str,
) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO receipts (evaluation_id, dossier_id, call_id, "
            "receipt_json, verified, result, detail, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                evaluation_id,
                dossier_id,
                call_id,
                json.dumps(receipt),
                1 if verified else 0,
                result,
                detail,
                time.time(),
            ),
        )


def get_commit(evaluation_id: str) -> Optional[dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT commit_response_json FROM commits WHERE evaluation_id=?",
        (evaluation_id,),
    ).fetchone()
    return json.loads(row[0]) if row else None


def store_commit(evaluation_id: str, commit_response: dict[str, Any]) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO commits (evaluation_id, commit_response_json, created_at) "
            "VALUES (?,?,?)",
            (evaluation_id, json.dumps(commit_response), time.time()),
        )
