"""
Board pack completeness gate.

Layer 1 — deterministic pre-flight (always runs, zero LLM cost):
  - KRI registration: every KRI key in store has a row in kri_thresholds.csv
  - exec_rec completeness: owner, deadline, status all non-empty on every entry
  - Override governance: every override_justification has text + ISO date
  - Breach domain coverage: every risk ID with ≥1 breach KRI has ≥1 exec_rec
  - Board summary coverage: every breach-level risk ID named in board_summary

Layer 2 — LLM semantic check (stub; activate via risk_panel_config.json):
  - completeness_checks.layer2_enabled: true
  - completeness_checks.layer2_model: <model id>

Entry point: run_completeness_checks(store, thresholds_path, panel_config) -> dict
"""

from __future__ import annotations

import csv
import pathlib
import re
from typing import Any


# ─── Shared helpers ───────────────────────────────────────────────────────────

_DOMAIN_KEYS = [
    "operational_risks",
    "financial_risks",
    "compliance_risks",
    "strategic_risks",
]


def _finding(severity: str, code: str, message: str, detail: str = "") -> dict:
    return {"severity": severity, "code": code, "message": message, "detail": detail}


def _all_risks(store: dict) -> list[tuple[str, str, dict]]:
    """Yield (domain_key, risk_id, risk_obj) for every risk in the store."""
    for dk in _DOMAIN_KEYS:
        for rid, obj in store.get(dk, {}).items():
            yield dk, rid, obj


# ─── Layer 1 checks ──────────────────────────────────────────────────────────

def _check_kri_registration(store: dict, registered: set[str]) -> list[dict]:
    """Every KRI key in the store must have a threshold row in kri_thresholds.csv."""
    findings = []
    for _, risk_id, risk_obj in _all_risks(store):
        for kri_name in risk_obj.get("kris", {}).keys():
            if kri_name not in registered:
                findings.append(_finding(
                    "HIGH", "KRI_NO_THRESHOLD",
                    f"{risk_id}.{kri_name} has no row in kri_thresholds.csv",
                    "KRI is computed and stored but carries no amber/breach thresholds. "
                    "Add a calibrated row to kri_thresholds.csv before the next board run.",
                ))
    return findings


def _check_exec_rec_completeness(store: dict) -> list[dict]:
    """Every exec_rec entry must have non-empty owner, deadline, and status."""
    findings = []
    for key, rec in store.get("exec_rec_drafts", {}).items():
        if not isinstance(rec, dict):
            continue
        for field in ("owner", "deadline", "status"):
            if not str(rec.get(field, "")).strip():
                findings.append(_finding(
                    "CRITICAL", "EXEC_REC_INCOMPLETE",
                    f"exec_rec '{key}' is missing required field: {field}",
                    f"The board cannot formally approve or assign accountability without a named "
                    f"{field}. Populate this field before submission.",
                ))
    return findings


def _check_override_governance(store: dict) -> list[dict]:
    """Every override_justification must contain substantive text and an ISO date."""
    _DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
    findings = []
    for _, risk_id, risk_obj in _all_risks(store):
        justification = risk_obj.get("override_justification", "")
        if not justification:
            continue
        if len(justification.strip()) < 30:
            findings.append(_finding(
                "HIGH", "OVERRIDE_NO_JUSTIFICATION",
                f"{risk_id} override_justification is too short to be meaningful",
                "Provide a full written rationale including forward-looking evidence "
                "and the specific indicators that justify the override.",
            ))
        elif not _DATE.search(justification):
            findings.append(_finding(
                "HIGH", "OVERRIDE_NO_DATE",
                f"{risk_id} override_justification has no CRO sign-off date (YYYY-MM-DD)",
                "A dated CRO sign-off is mandatory for the audit trail. "
                "Add the date the override was approved.",
            ))
    return findings


def _check_breach_domain_coverage(store: dict) -> list[dict]:
    """Every risk ID with ≥1 breach KRI must be covered by at least one exec_rec."""
    # Collect risk IDs in breach
    breach_ids: set[str] = set()
    for _, risk_id, risk_obj in _all_risks(store):
        for kri_data in risk_obj.get("kris", {}).values():
            if isinstance(kri_data, dict) and kri_data.get("status") == "breach":
                breach_ids.add(risk_id)
                break

    # Collect risk IDs covered by exec_recs (explicit risk_ids list + scan action text)
    covered: set[str] = set()
    for rec in store.get("exec_rec_drafts", {}).values():
        if not isinstance(rec, dict):
            continue
        for rid in rec.get("risk_ids", []):
            covered.add(rid)
        action = rec.get("action", "")
        for _, risk_id, _ in _all_risks(store):
            if risk_id in action:
                covered.add(risk_id)

    findings = []
    for risk_id in sorted(breach_ids - covered):
        findings.append(_finding(
            "HIGH", "BREACH_NO_EXEC_REC",
            f"{risk_id} is in breach but has no exec_rec entry",
            "The board pack has no management action item for this breach. "
            "Add an exec_rec entry with owner, deadline, and status.",
        ))
    return findings


def _check_board_summary_breach_coverage(store: dict) -> list[dict]:
    """Every breach-level risk should be substantively covered in the board_summary.

    Checks for the risk ID code (e.g. 'O-01') OR any meaningful keyword (≥5 chars)
    from the risk name — board summaries typically use descriptive language rather
    than bare IDs.
    """
    _STOPWORDS = {"risk", "other", "their", "with", "from", "this", "that", "and",
                  "the", "for", "key", "person", "attack"}
    board_lower = store.get("board_summary", "").lower()
    findings = []
    for _, risk_id, risk_obj in _all_risks(store):
        is_breach = any(
            isinstance(kri_data, dict) and kri_data.get("status") == "breach"
            for kri_data in risk_obj.get("kris", {}).values()
        )
        if not is_breach:
            continue
        # Pass if risk ID code appears
        if risk_id.lower() in board_lower:
            continue
        # Pass if any meaningful keyword from the risk name appears
        name = risk_obj.get("name", "")
        keywords = [
            w.lower().strip("&-,.")
            for w in name.split()
            if len(w) >= 5 and w.lower().strip("&-,.") not in _STOPWORDS
        ]
        if any(kw in board_lower for kw in keywords):
            continue
        findings.append(_finding(
            "MEDIUM", "BREACH_NOT_IN_NARRATIVE",
            f"{risk_id} ({name}) in breach — no mention in board_summary",
            "The board summary should cover every breach-level risk either by ID or by "
            "descriptive reference. Add a paragraph or sentence addressing this risk.",
        ))
    return findings


def run_layer1(store: dict, thresholds_path: pathlib.Path) -> dict:
    """Run all Layer 1 deterministic checks. Returns structured result."""
    registered: set[str] = set()
    if thresholds_path.exists():
        with thresholds_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = row.get("kri_name", "").strip()
                if name:
                    registered.add(name)

    all_findings = (
        _check_kri_registration(store, registered)
        + _check_exec_rec_completeness(store)
        + _check_override_governance(store)
        + _check_breach_domain_coverage(store)
        + _check_board_summary_breach_coverage(store)
    )

    critical = [f for f in all_findings if f["severity"] == "CRITICAL"]
    high     = [f for f in all_findings if f["severity"] == "HIGH"]
    medium   = [f for f in all_findings if f["severity"] == "MEDIUM"]

    return {
        "layer": 1,
        "total_findings": len(all_findings),
        "critical": critical,
        "high":     high,
        "medium":   medium,
        "findings": all_findings,
        "passed":   len(critical) == 0,
    }


# ─── Layer 2: LLM semantic check (stub) ──────────────────────────────────────

def run_layer2(store: dict, cc_config: dict) -> dict:
    """
    LLM-assisted semantic completeness check — STUB.

    Enable via risk_panel_config.json:
        "completeness_checks": { "layer2_enabled": true, "layer2_model": "claude-haiku-4-5-20251001" }

    When implemented this specialist will verify:
    - Every breach-level KRI is substantively explained in board_summary (not just named)
    - Every exec_rec action maps unambiguously to a KRI in breach
    - All CRIT findings from the prior panel run appear as board resolution items
    - Language in board materials is consistent with KRI status fields
    - Escalation record entries exist for every CRIT-rated finding

    Estimated cost per run: ~$0.02–$0.05 depending on store size.
    Replace the return statement below with the actual LLM call when implementing.
    """
    model = cc_config.get("layer2_model", "claude-haiku-4-5-20251001")
    return {
        "layer":           2,
        "status":          "STUB_NOT_IMPLEMENTED",
        "findings":        [],
        "total_findings":  0,
        "passed":          True,
        "model":           model,
        "note": (
            f"Layer 2 semantic check is configured but not yet implemented "
            f"(model: {model}). Implement run_layer2() in "
            f"tools/board_pack_completeness.py and set layer2_enabled=true "
            f"in risk_panel_config.json to activate."
        ),
    }


# ─── Public entry point ───────────────────────────────────────────────────────

def run_completeness_checks(
    store: dict,
    thresholds_path: pathlib.Path,
    panel_config: dict,
) -> dict:
    """
    Run board pack completeness checks per configuration.

    Layer 1 always runs (deterministic, zero cost).
    Layer 2 runs only when completeness_checks.layer2_enabled = true.

    Returns:
        {
          "layer1":          {...},
          "layer2":          {...} | None,
          "overall_passed":  bool,   # True only if no CRITICAL findings
          "total_critical":  int,
          "total_findings":  int,
          "summary":         str,    # one-line status for pipeline log
        }
    """
    cc_config = panel_config.get("completeness_checks", {})

    layer1 = run_layer1(store, thresholds_path)
    layer2 = run_layer2(store, cc_config) if cc_config.get("layer2_enabled", False) else None

    crit_l2    = len([f for f in (layer2 or {}).get("findings", []) if f.get("severity") == "CRITICAL"])
    total_crit = len(layer1["critical"]) + crit_l2
    total_all  = layer1["total_findings"] + ((layer2 or {}).get("total_findings", 0))

    overall_passed = layer1["passed"] and (layer2 is None or layer2.get("passed", True))

    parts = [
        f"L1: {layer1['total_findings']} findings "
        f"({len(layer1['critical'])} CRIT / {len(layer1['high'])} HIGH / {len(layer1['medium'])} MED)"
    ]
    parts.append(f"L2: {'disabled' if layer2 is None else layer2.get('status', '?')}")

    return {
        "layer1":         layer1,
        "layer2":         layer2,
        "overall_passed": overall_passed,
        "total_critical": total_crit,
        "total_findings": total_all,
        "summary":        " | ".join(parts),
    }
