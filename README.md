# Safe AI Mailroom Agent

FastAPI service implementing the propose/commit contract described in the
assignment. One endpoint: `POST /v1/mailroom/actions`.

## ⚠️ Read this before you submit

The pasted assignment text referenced two collapsible examples —
**"Exact propose request and response"** and **"Exact commit request and
terminal response"** — that did not come through in the copy you gave me
(the surrounding prose survived, the expanded JSON examples didn't). I built
this against a best-effort reconstruction of the field names
(`evaluationId`, `dossiers[].dossierId/content`, `proposals[].callId/action/
target/payload/evidence/proposalDigest`, `receipts[].dossierId/callId/
proposalDigest/approved/verificationKey`, `outcomes[].dossierId/callId/
result/detail`).

**Before you deploy for real, open the actual assignment page and diff the
real JSON examples against `app/models.py`.** That file is the single place
field names live — `canonical.py`, `db.py`, `llm.py`, and `main.py` are all
written against the Pydantic classes, not raw dict keys, so renaming a field
in `models.py` (and the couple of matching dict keys in `main.py`) is
normally enough to match the real schema exactly.

The **highest-risk guess** is the receipt-verification mechanism
(`_verify_receipt` in `app/main.py`): the spec says to "store each
evaluation's supplied receipt-verification key with that evaluation" but
never shows the key or the receipt's signature field. I implemented:

- if the propose body included a `receiptVerificationKey` and a receipt
  supplies a matching `verificationKey`, verify
  `sha256(key:dossierId:callId:proposalDigest) == verificationKey`;
- otherwise fall back to trusting an explicit `approved: true/false` on the
  receipt, and **reject** any receipt with no verifiable signal at all.

Adjust `_verify_receipt` to match the real receipt shape once you see it —
this is the one function most likely to need a rewrite, and a wrong
implementation here is exactly what caps the score at 2/4 per the "failure
to reject an invalid receipt" rule.

## What's implemented

- **Envelopes & validation** (`models.py`): strict Pydantic schemas for
  propose/commit; malformed bodies or duplicate dossier IDs fail with
  400/422 *before* any AI or persistence work (`main.py`).
- **Canonical hashing** (`canonical.py`): all fingerprints (content,
  evaluation-set, callId, proposalDigest) go through one canonical-JSON +
  SHA-256 path, so a stable dossier always gets the same `callId` across
  evaluations/Checks, and a changed dossier under the same `evaluationId`
  is detected as a conflict.
- **Decision cache** (`db.py`): keyed by `(dossierId, content fingerprint)`,
  *not* `evaluationId` — later Checks/Save reuse cached decisions and make
  zero model calls for unchanged dossiers.
- **LLM decision layer** (`llm.py`): DeepSeek call with a system prompt that
  explicitly marks dossier content as untrusted data (not instructions),
  asks for one of the six fixed actions plus 1–5 verbatim evidence quotes,
  strict Pydantic validation of the output, and a `request_confirmation`
  fallback (never a silent send/mutate) if the model output fails
  validation twice.
- **Evidence grounding** (`safety.py`): every cited quote and every payload
  string over 40 chars must appear verbatim in the dossier's own content —
  this is enforced in code, independent of what the model claims, and stops
  injected instructions or secrets from being copied into an action
  argument.
- **Replay / conflict**: exact propose replay returns the stored response
  byte-for-byte with no model call; same `evaluationId` + changed content →
  HTTP 409; commit is idempotent per `(evaluationId, dossierId)` so
  re-committing never re-executes an effect.
- **Digest checking at commit**: a receipt whose `callId` or
  `proposalDigest` doesn't match the persisted proposal is rejected, not
  executed.
- **Bounds**: request body capped, response capped at 512 KiB, explicit
  `Content-Type: application/json`, no redirects.

## Run locally

```bash
cd mailroom-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...
uvicorn app.main:app --reload --port 8080
```

## Test (no model calls, no API key needed)

```bash
pip install pytest
PYTHONPATH=. pytest -q
```

Covers: malformed JSON → 400, invalid operation → 400, duplicate dossier ID
→ 422, exact propose replay is byte-identical, same evaluationId + changed
content → 409, unknown evaluation on commit → 400, digest mismatch on
commit → rejected (not executed), commit replay is idempotent, and a
receipt with no approval/verification signal is rejected.

## Deploy (Fly.io)

```bash
fly launch --no-deploy      # use the included fly.toml, pick a unique app name
fly volumes create mailroom_data --region iad --size 1
fly secrets set DEEPSEEK_API_KEY=sk-...
fly deploy
```

Your submitted URL will be `https://<your-app>.fly.dev/v1/mailroom/actions`
— no credentials, query, or fragment in it, as required.

(Render/Railway: same Dockerfile; add a persistent disk mounted at `/data`
and set `DEEPSEEK_API_KEY` as a secret/env var in their dashboard instead.)

## Known simplifications worth knowing about

- SQLite behind a single process lock is durable and simple but assumes a
  single running instance (`min_machines_running = 1`, no autoscaling) —
  fine for this workload (~70 dossiers/Check), not a general concurrency
  solution.
- `_execute()` in `main.py` records the effect deterministically rather
  than calling a real mail/CRM API, since none exists in this exam context.
- The injection-phrase list in `safety.py` is only ever passed to the model
  as a *hint*, never used to decide the action itself, per the spec's
  warning against keyword-only filtering.
