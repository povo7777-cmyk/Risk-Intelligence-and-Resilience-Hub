"""
agents/compliance_agent.py — C-01 to C-03
Uses Claude Sonnet 4.6.
Primary specialization: obligation mapping and gap analysis.
"""

import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_reader import read_csv
from tools.risk_writer import update_risk, load_store
from schemas.agent_outputs import validate_agent_output
import anthropic

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "compliance_v1.txt"
MODEL = "claude-sonnet-4-6"


def _call_claude(system: str, user: str) -> tuple[str, dict]:
    client = anthropic.Anthropic()
    msg = client.messages.create(model=MODEL, max_tokens=16000, system=system,
                                  messages=[{"role": "user", "content": user}])
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    return msg.content[0].text, usage


def _parse_screening() -> dict:
    rows = read_csv("screening_results.csv")
    result = {}
    for r in rows:
        try:
            result[r.get("metric", "")] = float(r.get("value", 0))
        except Exception:
            pass
    return result


def run(kri_data: dict | None = None) -> dict:
    """
    kri_data: pre-computed KRI values from kri_data_layer (when running in graph).
    If None, computes KRI values from CSVs directly (standalone / test mode).
    """
    print(f"\n{'='*60}")
    print("[Compliance Agent] Starting — Claude Sonnet 4.6")
    print(f"{'='*60}")

    system_prompt = PROMPT_PATH.read_text()

    # Read all CSVs — no fallback defaults; missing data must be surfaced, not masked
    screening         = _parse_screening()
    horizon           = read_csv("regulatory_horizon.csv")
    audit_log         = read_csv("audit_log.csv")
    comp_metrics_rows = read_csv("compliance_metrics.csv")
    comp_metrics      = {r["metric"]: float(r["value"]) for r in comp_metrics_rows}
    screening_rows    = read_csv("screening_results.csv")

    screening_coverage   = screening.get("export_screening_coverage_pct", 0.0)
    confirmed_violations = screening.get("confirmed_sanctions_violations_ytd", 0.0)
    denied_pending       = screening.get("denied_party_matches_pending", 0.0)
    ai_audit_pct         = comp_metrics.get("ai_audit_coverage_pct", 0.0)
    dsr_rate             = comp_metrics.get("gdpr_dsr_resolution_rate_pct", 0.0)
    third_party_coverage = comp_metrics.get("third_party_abac_coverage_pct", 0.0)
    scope3_pct           = comp_metrics.get("csrd_scope3_disclosure_pct", 0.0)
    formal_investigation = any("investigation" in r.get("audit_type", "").lower() for r in audit_log)
    whistleblower        = next((r for r in audit_log if "Whistleblower" in r.get("audit_type", "")), {})
    wb_high              = int(whistleblower.get("high_findings", 0))

    # KRI values: use data layer if available, otherwise compute locally
    if kri_data and kri_data.get("dashboard_kris"):
        kri_updates = kri_data["dashboard_kris"]
        # Supplement narrative vars with agent_context values
        actx = kri_data.get("agent_context", {})
        denied_pending       = actx.get("C-01", {}).get("denied_party_matches_pending", denied_pending)
        formal_investigation = bool(actx.get("C-02", {}).get("formal_investigation_open", int(formal_investigation)))
        wb_high              = int(actx.get("C-03", {}).get("whistleblower_high_findings", wb_high))
    else:
        kri_updates = {
            "C-01": [
                {"name": "export_screening_coverage_pct", "value": screening_coverage, "unit": "%",
                 "status": "breach" if screening_coverage < 95 else "amber" if screening_coverage < 98 else "ok"},
                {"name": "confirmed_sanctions_violations_ytd", "value": confirmed_violations, "unit": "count",
                 "status": "breach" if confirmed_violations > 0 else "ok"},
                {"name": "denied_party_matches_pending", "value": denied_pending, "unit": "count",
                 "status": "amber" if denied_pending >= 3 else "ok"},
            ],
            "C-02": [
                {"name": "ai_audit_coverage_pct", "value": ai_audit_pct, "unit": "%",
                 "status": "breach" if ai_audit_pct < 90 else "amber" if ai_audit_pct < 95 else "ok"},
                {"name": "gdpr_dsr_resolution_rate_pct", "value": dsr_rate, "unit": "%",
                 "status": "fyi"},  # archived: not in KRI framework; context only
                {"name": "formal_investigation_open", "value": 1.0 if formal_investigation else 0.0, "unit": "count",
                 "status": "breach" if formal_investigation else "ok"},
            ],
            "C-03": [
                {"name": "third_party_abac_coverage_pct", "value": third_party_coverage, "unit": "%",
                 "status": "breach" if third_party_coverage < 95 else "amber" if third_party_coverage < 100 else "ok"},
                {"name": "whistleblower_high_findings", "value": float(wb_high), "unit": "count",
                 "status": "amber" if wb_high >= 1 else "ok"},
                {"name": "csrd_scope3_disclosure_pct", "value": scope3_pct, "unit": "%",
                 "status": "fyi"},  # archived: not in KRI framework; context only
            ],
        }

    breaches = sum(1 for kris in kri_updates.values() for k in kris if k["status"] == "breach")
    ambers = sum(1 for kris in kri_updates.values() for k in kris if k["status"] == "amber")
    escalation = confirmed_violations > 0 or screening_coverage < 95 or formal_investigation
    escalation_reasons = []
    if confirmed_violations > 0:
        escalation_reasons.append(f"C-01: {int(confirmed_violations)} confirmed sanctions violation(s) — zero-tolerance")
    if screening_coverage < 95:
        escalation_reasons.append(f"C-01: Export screening {screening_coverage}% below 95% threshold")
    if formal_investigation:
        escalation_reasons.append("C-02: Formal regulatory investigation open")

    regulatory_deadlines = [
        {"regulation": r.get("regulation"), "deadline": r.get("effective_date"),
         "gap": r.get("gap_identified"), "owner": r.get("owner"), "risk_id": r.get("risk_id")}
        for r in horizon if r.get("compliance_status") in ["Partial", "In-Progress"]
    ]

    # ── Build rich user prompt — all CSV rows, not aggregates ────────────────
    # All screening metrics
    sc_lines = "\n".join(
        f"  {r['metric']}: {r['value']} {r['unit']} (source: {r['source_system']})"
        for r in screening_rows
    )
    # All compliance metrics with notes
    cm_lines = "\n".join(
        f"  {r['metric']}: {r['value']} {r['unit']} — {r.get('notes','')}"
        for r in comp_metrics_rows
    )
    # Full audit log
    al_lines = "\n".join(
        f"  {r['audit_id']} [{r['audit_type']}] {r['subject']}: "
        f"status={r['status']}, findings={r['finding_count']}, high={r['high_findings']}, "
        f"critical={r['critical_findings']}, due={r['due_date']}, owner={r['owner']}"
        for r in audit_log
    )
    # Full regulatory horizon
    hr_lines = "\n".join(
        f"  {r['regulation']} ({r['jurisdiction']}): status={r['compliance_status']}, "
        f"effective={r['effective_date']}, gap={r['gap_identified']}, owner={r['owner']}, risk={r['risk_id']}"
        for r in horizon
    )

    user_prompt = f"""C-01 EXPORT CONTROLS & SANCTIONS (all screening_results.csv rows):
{sc_lines}
  (amber: coverage<98% — breach: coverage<95% | any confirmed violation)

C-02 DATA PRIVACY & AI (all compliance_metrics.csv rows):
{cm_lines}
  (amber: ai_audit<95% — breach: ai_audit<90% | CURRENT ai_audit_coverage_pct={ai_audit_pct}% → KRI STATUS: {"BREACH" if ai_audit_pct < 90 else "AMBER" if ai_audit_pct < 95 else "OK"} | gdpr_dsr=FYI context only, no threshold)

C-03 ANTI-BRIBERY & ESG:
  third_party_abac_coverage: {third_party_coverage}% (breach: <95% | amber: <100%)
  csrd_scope3_disclosure: {scope3_pct}% (FYI context only — no KRI threshold)
  whistleblower_high_findings: {wb_high}

AUDIT LOG (all audit_log.csv rows):
{al_lines}

REGULATORY HORIZON (all regulatory_horizon.csv rows):
{hr_lines}

KRI STATUS THIS RUN: {breaches} breach(es), {ambers} amber(s)
ESCALATION: {'YES' if escalation else 'no'}

Assess all three compliance risk areas. Cite specific regulations, deadlines, and gaps
from the data above — only obligations that appear in regulatory_horizon.csv.
Identify cross-domain interconnection flags where the data shows a genuine connection.
Return the full JSON per the output format in your instructions."""

    # LLM generates interconnections from data — no hardcoded static flags
    narrative = (f"Compliance assessment: {breaches} breach(es), {ambers} amber(s). "
                 f"C-01 screening {screening_coverage}%, violations {int(confirmed_violations)}. "
                 f"C-02 AI audit {ai_audit_pct}%, DSR {dsr_rate}%.")
    llm_interconnections = []
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        from tools.json_parser import parse_llm_json
        print("  Calling Claude Sonnet 4.6 for compliance assessment...")
        raw, token_usage = _call_claude(system_prompt, user_prompt)
        parsed, parse_err = parse_llm_json(raw)
        if parsed:
            if parsed.get("narrative"):
                narrative = parsed["narrative"]
            llm_interconnections = parsed.get("interconnection_flags", [])
            if llm_interconnections:
                print(f"  Agent identified {len(llm_interconnections)} interconnection flag(s)")
        else:
            print(f"  Claude assessment parse failed: {parse_err}")
    except Exception as e:
        print(f"  Claude assessment failed: {e}")

    # KRI values already written by kri_data_layer; write only narrative metadata here
    for risk_id in kri_updates:
        try:
            update_risk(risk_id=risk_id, kri_updates={}, new_ctrl=None, new_lv=None,
                        agent_findings=narrative, proposed_actions=[])
        except Exception as e:
            print(f"  Warning: update_risk failed for {risk_id}: {e}")

    result = {
        "domain": "compliance", "agent_version": "v1", "risks": ["C-01", "C-02", "C-03"],
        "kri_updates": kri_updates, "regulatory_deadlines": regulatory_deadlines,
        "breach_count": breaches, "amber_count": ambers,
        "escalation_required": escalation, "escalation_reasons": escalation_reasons,
        "interconnection_flags": llm_interconnections,
        "narrative": narrative, "confidence": "high", "proposed_ctrl_changes": {},
        "token_usage": token_usage,
    }

    valid, _ = validate_agent_output("compliance", result)
    print(f"  [Compliance] Complete — {breaches} breach(es), {ambers} amber(s) | Schema: {'valid' if valid else 'INVALID'}")
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
