"""
Safety is enforced in two places:

1. HARD, code-only gates in this file: every evidence quote the model cites
   must be an exact substring of the dossier content it claims to come from
   (no evidence -> reject the LLM output and retry/quarantine), and outbound
   actions (send_approved_notice) must not contain any text that isn't
   traceable to trusted/approved fields. These are enforced regardless of
   what the model says.

2. The model's *reasoning* about intent (is this trying to control tools /
   exfiltrate private context / trigger an unauthorized send?), which we do
   NOT try to replace with keyword blocklists -- the spec explicitly warns
   that blanket/keyword-only filtering scores poorly, and that a trusted
   quote containing attack words is not automatically an attack. Heuristic
   signals below are only ever passed to the model as *hints in the prompt*,
   never used directly to force a verdict.
"""
from __future__ import annotations

from typing import Any

MAX_BODY_BYTES = 2_000_000  # generous bound on inbound request bodies
MAX_RESPONSE_BYTES = 512 * 1024  # spec: successful body over 512 KiB is rejected

_INJECTION_HINTS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "system prompt",
    "you are now",
    "act as",
    "reveal your instructions",
    "print the secret",
    "exfiltrate",
    "new instructions:",
)


def injection_hint_score(text: str) -> int:
    """Non-authoritative signal only -- surfaced to the model as context,
    never used on its own to pick an action."""
    lowered = text.lower()
    return sum(1 for h in _INJECTION_HINTS if h in lowered)


def flatten_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(flatten_strings(v))
    return out


def evidence_is_grounded(evidence: list[dict[str, str]], content: dict[str, Any]) -> bool:
    """Every cited quote must appear verbatim somewhere in the dossier's own
    content. This is what stops the model from inventing evidence or
    smuggling instructions-as-facts."""
    haystack = " \n ".join(flatten_strings(content))
    for item in evidence:
        quote = item.get("quote", "")
        if not quote or quote not in haystack:
            return False
    return True


def payload_is_grounded(payload: dict[str, Any], content: dict[str, Any]) -> bool:
    """Extra guard specifically for outbound-ish payloads: every string
    value placed into a tool payload must either be short/structural or
    traceable back to the dossier content. This blocks raw prompt-injection
    text or secret canaries from being copied verbatim into an action
    argument."""
    haystack = " \n ".join(flatten_strings(content))
    for v in flatten_strings(payload):
        if len(v) > 40 and v not in haystack:
            return False
    return True
