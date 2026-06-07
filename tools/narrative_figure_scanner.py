"""
tools/narrative_figure_scanner.py — Item 7: Narrative figure integrity check.

Extracts numeric figures from agent_findings text and agent_context fields in
risk_store.json, then compares them against the KRI ground truth values computed
deterministically from source CSVs (kri_data_layer output in pipeline state).

A mismatch > TOLERANCE flags the specific claim for the HITL reviewer.
This catches the EM-01 class of error: agent narrative citing a self-computed
figure that differs from the authoritative CSV-derived value.

Deterministic — no LLM calls.
"""

from __future__ import annotations
import json
import re
from pathlib import Path

STORE_PATH  = Path(__file__).parent.parent / "api" / "risk_store.json"

# Relative tolerance — flag when |narrative_fig - kri_val| / |kri_val| > TOLERANCE
TOLERANCE = 0.10   # 10%

# Map KRI names to the canonical value expected in narratives
# (allows aliases — e.g. "6 hours" for an RTO KRI stored as 6.0)
_UNIT_SCALE: dict[str, float] = {
    "%": 1.0,
    "weeks": 1.0,
    "days": 1.0,
    "hours": 1.0,
    "count": 1.0,
    "ratio": 1.0,
    "USD_M": 1.0,
    "USD_B": 1000.0,   # convert billions → millions for comparison
}


def _extract_numbers(text: str) -> list[float]:
    """Extract all stand-alone numeric values from a text string."""
    # Match: optional $ or USD prefix, digits, optional decimal, optional M/B/% suffix
    # Examples: 44.2%, USD 3,400M, 5.2 weeks, 0.048%, 2.8x
    pattern = r"(?:USD\s*|[$])?(\d[\d,]*(?:\.\d+)?)\s*(?:M|B|%|x|pp)?"
    found = []
    for m in re.finditer(pattern, text, re.IGNORECASE):
        raw = m.group(1).replace(",", "")
        try:
            found.append(float(raw))
        except ValueError:
            pass
    return found


def _kri_values_from_state(kri_ground_truth: dict) -> dict[str, float]:
    """
    Flatten kri_ground_truth (state["kri_ground_truth"]) into {kri_name: value}.
    Structure: {risk_id: {kri_name: {value: ..., status: ...}}}
    """
    flat: dict[str, float] = {}
    for risk_id, kris in kri_ground_truth.items():
        if not isinstance(kris, dict):
            continue
        for kri_name, kri_data in kris.items():
            if isinstance(kri_data, dict) and "value" in kri_data:
                try:
                    flat[kri_name] = float(kri_data["value"])
                except (TypeError, ValueError):
                    pass
    return flat


def _kri_values_from_store() -> dict[str, float]:
    """Fallback: read KRI values directly from risk_store.json."""
    flat: dict[str, float] = {}
    try:
        store = json.loads(STORE_PATH.read_text())
        buckets = [
            store.get("operational_risks",  {}),
            store.get("strategic_risks",    {}),
            store.get("financial_risks",    {}),
            store.get("compliance_risks",   {}),
        ]
        for bucket in buckets:
            for risk_id, risk_obj in bucket.items():
                for kri_name, kri_data in risk_obj.get("kris", {}).items():
                    if isinstance(kri_data, dict) and "value" in kri_data:
                        try:
                            flat[kri_name] = float(kri_data["value"])
                        except (TypeError, ValueError):
                            pass
    except Exception:
        pass
    return flat


def scan_findings(
    agent_findings_text: str,
    risk_id: str,
    kri_values: dict[str, float],
    kri_name_hint: str | None = None,
) -> list[dict]:
    """
    Scan a single agent_findings text block for numbers that mismatch KRI values.
    Returns list of mismatch dicts.
    """
    mismatches: list[dict] = []
    if not agent_findings_text or not kri_values:
        return mismatches

    narrative_nums = set(_extract_numbers(agent_findings_text))

    for kri_name, kri_val in kri_values.items():
        if kri_val == 0:
            continue
        # Check if this KRI name (or a fragment) appears in the text
        kri_fragment = kri_name.replace("_", " ").replace("pct", "%")
        risk_mentioned = (
            risk_id.lower() in agent_findings_text.lower() or
            kri_fragment.lower() in agent_findings_text.lower()
        )
        if not risk_mentioned:
            continue

        # Look for a number that should match this KRI value but doesn't
        for num in narrative_nums:
            # Only flag numbers in a plausible range (within 2 orders of magnitude)
            if kri_val > 0 and (num / kri_val) < 0.01:
                continue
            if kri_val > 0 and (num / kri_val) > 100:
                continue
            diff = abs(num - kri_val) / abs(kri_val)
            if diff > TOLERANCE and diff < 10.0:   # >10% off but not obviously a different metric
                mismatches.append({
                    "risk_id":         risk_id,
                    "kri_name":        kri_name,
                    "kri_value":       kri_val,
                    "narrative_value": num,
                    "diff_pct":        round(diff * 100, 1),
                    "detail": (
                        f"{risk_id}/{kri_name}: narrative mentions {num} but "
                        f"KRI ground truth is {kri_val} ({round(diff*100,1)}% discrepancy)"
                    ),
                })
                break  # one flag per KRI per text block is enough

    return mismatches


def run(state: dict) -> dict:
    """
    Main entry point. Called from content_validation_node or as a standalone check.

    state must contain:
      - kri_ground_truth: dict (from kri_data_layer_node)
      - operational_findings, financial_findings, etc. (from domain agents)

    Returns dict with {passed, mismatch_count, mismatches, warnings}.
    """
    kri_values = _kri_values_from_state(state.get("kri_ground_truth", {}))
    if not kri_values:
        kri_values = _kri_values_from_store()

    all_mismatches: list[dict] = []
    scan_targets: list[tuple[str, str]] = []

    # Collect all agent_findings text blocks from domain agents
    domain_map = {
        "operational_findings":  ["O-01", "O-02", "O-03", "O-04"],
        "financial_findings":    ["F-01", "F-02", "F-03", "F-04"],
        "strategic_findings":    ["S-01", "S-02", "S-03"],
        "compliance_findings":   ["C-01", "C-02", "C-03"],
    }

    for findings_key, risk_ids in domain_map.items():
        findings = state.get(findings_key) or {}
        if not findings:
            continue
        # agent_findings is typically a dict keyed by risk_id
        af = findings.get("agent_findings", {})
        if isinstance(af, dict):
            for risk_id, text in af.items():
                if isinstance(text, str) and text.strip():
                    scan_targets.append((risk_id, text))
        # Also scan the narrative / summary field if present
        for field in ["narrative", "summary", "board_text", "findings_text"]:
            text = findings.get(field, "")
            if isinstance(text, str) and text.strip():
                scan_targets.append((findings_key, text))

    for risk_id, text in scan_targets:
        mismatches = scan_findings(text, risk_id, kri_values)
        all_mismatches.extend(mismatches)

    # Deduplicate by (risk_id, kri_name)
    seen: set[tuple] = set()
    deduped = []
    for m in all_mismatches:
        key = (m["risk_id"], m["kri_name"])
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    return {
        "passed":         len(deduped) == 0,
        "mismatch_count": len(deduped),
        "mismatches":     deduped,
        "kri_count_checked": len(kri_values),
        "scan_targets_checked": len(scan_targets),
    }


if __name__ == "__main__":
    import sys
    # Standalone test against risk_store.json
    result = run({})
    print(f"Narrative figure scan: {result['mismatch_count']} mismatch(es) "
          f"across {result['kri_count_checked']} KRIs, "
          f"{result['scan_targets_checked']} text blocks")
    for m in result["mismatches"]:
        print(f"  ⚠ {m['detail']}")
    sys.exit(0 if result["passed"] else 1)
