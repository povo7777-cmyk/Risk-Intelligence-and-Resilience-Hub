"""
agents/operational_agent.py — O-01 to O-04
Uses Claude Sonnet 4.5.
Primary specialization: structured data evaluation.
"""

import json, os, sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_reader import get_supply_chain_data, get_cyber_data, get_quality_data, get_talent_data
from tools.kri_evaluator import evaluate_kri, compute_residual_score
from tools.risk_writer import update_risk, load_store
from schemas.agent_outputs import validate_agent_output
import anthropic

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "operational_v1.txt"
MODEL = "claude-sonnet-4-5"


def _call_claude(system: str, user: str) -> tuple[str, dict]:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    return msg.content[0].text, usage


def run(kri_data: dict | None = None) -> dict:
    """
    kri_data: pre-computed KRI values from kri_data_layer (when running in graph).
    If None, computes KRI values from CSVs directly (standalone / test mode).
    """
    print(f"\n{'='*60}")
    print("[Operational Agent] Starting — Claude Sonnet 4.5")
    print(f"{'='*60}")

    system_prompt = PROMPT_PATH.read_text()

    # Always read CSVs for narrative context (richer than just KRI values)
    sc = get_supply_chain_data()
    for key in list(sc.keys()):
        if sc[key] is None: sc[key] = 0
    cy = get_cyber_data()
    for key in list(cy.keys()):
        if cy[key] is None: cy[key] = 0
    qu = get_quality_data()
    for key in list(qu.keys()):
        if qu[key] is None: qu[key] = 0
    ta = get_talent_data()
    for key in list(ta.keys()):
        if ta[key] is None: ta[key] = 0

    # KRI values: use data layer if available, otherwise compute locally
    if kri_data and kri_data.get("dashboard_kris"):
        kri_updates = kri_data["dashboard_kris"]
    else:
        kri_updates = {
            "O-01": [
                {"name": "single_source_concentration", "value": sc["overall_single_source_concentration_pct"], "unit": "%",
                 "status": "breach" if sc["overall_single_source_concentration_pct"] >= 50 else "amber" if sc["overall_single_source_concentration_pct"] >= 40 else "ok"},
                {"name": "inventory_cover_weeks", "value": sc["min_inventory_weeks"], "unit": "weeks",
                 "status": "breach" if sc["min_inventory_weeks"] <= 3 else "amber" if sc["min_inventory_weeks"] <= 4 else "ok"},
                {"name": "supplier_distress_flags", "value": float(sc["supplier_distress_flags"]), "unit": "count",
                 "status": "breach" if sc["supplier_distress_flags"] >= 4 else "amber" if sc["supplier_distress_flags"] >= 2 else "ok"},
            ],
            "O-02": [
                {"name": "mttd_days", "value": cy["mttd_days"], "unit": "days",
                 "status": "breach" if cy["mttd_days"] >= 10 else "amber" if cy["mttd_days"] >= 7 else "ok"},
                {"name": "patch_compliance_pct", "value": cy["patch_compliance_pct"], "unit": "%",
                 "status": "breach" if cy["patch_compliance_pct"] < 85 else "amber" if cy["patch_compliance_pct"] < 95 else "ok"},
                {"name": "critical_vulns_open_gt30d", "value": cy["critical_vulns_open_gt30d"], "unit": "count",
                 "status": "breach" if cy["critical_vulns_open_gt30d"] >= 10 else "amber" if cy["critical_vulns_open_gt30d"] >= 5 else "ok"},
                {"name": "it_rto_hours", "value": cy["it_rto_hours"], "unit": "hours",
                 "status": "amber" if cy["it_rto_hours"] > 4 else "ok"},
            ],
            "O-03": [
                {"name": "field_failure_rate_pct", "value": qu["max_field_failure_rate_pct"], "unit": "%",
                 "status": "breach" if qu["max_field_failure_rate_pct"] >= 0.05 else "amber" if qu["max_field_failure_rate_pct"] >= 0.04 else "ok"},
                {"name": "recall_readiness_score_pct", "value": qu["recall_readiness_score_pct"], "unit": "%",
                 "status": "breach" if qu["recall_readiness_score_pct"] <= 80 else "amber" if qu["recall_readiness_score_pct"] <= 85 else "ok"},
                {"name": "safety_incidents_ytd", "value": qu["safety_incidents_ytd"], "unit": "count",
                 "status": "breach" if qu["safety_incidents_ytd"] > 0 else "ok"},
                {"name": "supplier_quality_rejection_rate_pct", "value": qu["supplier_quality_rejection_rate_pct"], "unit": "%",
                 "status": "breach" if qu["supplier_quality_rejection_rate_pct"] >= 1.0 else "amber" if qu["supplier_quality_rejection_rate_pct"] >= 0.5 else "ok"},
            ],
            "O-04": [
                {"name": "tech_attrition_rate_pct", "value": ta["tech_attrition_engineering_pct"], "unit": "%",
                 "status": "breach" if ta["tech_attrition_engineering_pct"] >= 15 else "amber" if ta["tech_attrition_engineering_pct"] >= 13 else "ok"},
                {"name": "critical_open_roles_gt60d", "value": ta["critical_open_roles_gt60d"], "unit": "count",
                 "status": "breach" if ta["critical_open_roles_gt60d"] >= 12 else "amber" if ta["critical_open_roles_gt60d"] >= 8 else "ok"},
                {"name": "svp_succession_coverage_pct", "value": ta["svp_succession_coverage_pct"], "unit": "%",
                 "status": "breach" if ta["svp_succession_coverage_pct"] <= 80 else "amber" if ta["svp_succession_coverage_pct"] <= 90 else "ok"},
            ],
        }

    breaches = sum(1 for risk_kris in kri_updates.values() for k in risk_kris if k["status"] == "breach")
    ambers = sum(1 for risk_kris in kri_updates.values() for k in risk_kris if k["status"] == "amber")
    escalation = (sc["overall_single_source_concentration_pct"] > 75 or
                  cy["mttd_days"] > 14 or qu["safety_incidents_ytd"] > 0 or
                  ta.get("flight_risk_flagged", 0) >= 3)

    # ── Build rich user prompt — pass full CSV context, not aggregates ──────
    # Single-source suppliers with full detail
    single_source_rows = [r for r in sc.get("raw_rows", []) if r.get("single_source") == "true"]
    multi_source_rows  = [r for r in sc.get("raw_rows", []) if r.get("single_source") != "true"]

    ss_lines = "\n".join(
        f"  - {r['supplier_name']} [{r['component_category']}]: "
        f"spend ${r['our_spend_usd_m']}M, inventory {r['inventory_weeks']}wk, "
        f"lead_time {r['lead_time_weeks']}wk, financial_health {r['financial_health_score']}, "
        f"country {r['country']}, OTD {r['on_time_delivery_pct']}%, rejection {r['quality_rejection_pct']}%"
        for r in single_source_rows
    )
    ms_names = ", ".join(r['supplier_name'] for r in multi_source_rows)
    geo = sc.get("geographic_concentration_pct", {})
    geo_str = ", ".join(f"{c} {p}%" for c, p in sorted(geo.items(), key=lambda x: -x[1]))

    # All cyber metrics
    cy_raw = cy.get("raw_metrics", {})
    cy_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(cy_raw.items()))

    # Quality by SKU
    qu_raw = qu.get("raw_metrics", {})
    qu_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(qu_raw.items()))

    # Talent by department
    ta_raw = ta.get("raw_rows", [])
    ta_lines = "\n".join(
        f"  {r['metric']} [{r.get('department','All')}]: {r['value']} {r.get('unit','')}"
        for r in ta_raw
    ) if ta_raw else "\n".join(f"  {k}: {v}" for k, v in ta.items() if k != "raw_rows")

    user_prompt = f"""O-01 SUPPLY CHAIN DATA:
Overall single-source concentration: {sc.get('overall_single_source_concentration_pct',0)}% of ${sc.get('total_spend_usd_m',0)}M spend
  (amber threshold: 40% | breach threshold: 50%)
Minimum inventory cover: {sc.get('min_inventory_weeks',0)} weeks
  (amber threshold: 6wk | breach threshold: 4wk — lower is worse)
Supplier distress flags (financial_health < 65): {sc.get('supplier_distress_flags',0)}
  (amber threshold: 1 | breach threshold: 3)
Geographic concentration: {geo_str}

Single-source suppliers ({len(single_source_rows)} of {sc.get('suppliers_assessed',0)}):
{ss_lines}

Multi-source suppliers: {ms_names}

O-02 CYBER AND IT DATA (all metrics from SIEM/ITSM):
{cy_lines}
  (amber thresholds: MTTD 7d | patch 90% | critical_vulns 5 | RTO 4h)
  (breach thresholds: MTTD 10d | patch 80% | critical_vulns 10 | RTO 8h)

O-03 PRODUCT QUALITY DATA (all metrics from QMS):
{qu_lines}
  (amber thresholds: field_failure 0.05% | recall_readiness 80% | supplier_quality_rejection 0.5%)
  (breach thresholds: field_failure 0.10% | recall_readiness 70% | safety_incidents >0 | supplier_quality_rejection 1.0%)

O-04 TALENT DATA (all metrics from HRIS by department):
{ta_lines}
  (amber thresholds: attrition 12% | open_roles 3 | succession 80%)
  (breach thresholds: attrition 18% | open_roles 6 | succession 70%)

KRI STATUS THIS RUN: {breaches} breach(es), {ambers} amber(s)
ESCALATION: {'YES' if escalation else 'no'}

Assess all four risk areas. Name specific suppliers, systems, and functions.
Surface BCM, revenue, and EBITDA implications where the data supports them.
Identify cross-domain interconnections from the data.
Return the full JSON per the output format in your instructions."""

    # Hardcoded Python interconnections removed — LLM generates them from data above
    print("  Calling Claude Sonnet 4.5 for operational assessment...")
    narrative = "Operational assessment completed. KRI evaluation based on structured data analysis."
    llm_interconnections = []
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        from tools.json_parser import parse_llm_json
        raw, token_usage = _call_claude(system_prompt, user_prompt)
        parsed, parse_err = parse_llm_json(raw)
        if parsed:
            narrative = parsed.get("narrative", narrative)
            llm_interconnections = parsed.get("interconnection_flags", [])
            if llm_interconnections:
                print(f"  Agent identified {len(llm_interconnections)} interconnection flag(s)")
        else:
            print(f"  Claude assessment parse failed: {parse_err} — using computed KRI data")
    except Exception as e:
        print(f"  Claude assessment failed: {e} — using computed KRI data")

    # KRI values already written by kri_data_layer; write only narrative metadata here
    for risk_id in kri_updates:
        update_risk(risk_id=risk_id, kri_updates={}, new_ctrl=None, new_lv=None,
                    agent_findings=narrative, proposed_actions=[])

    result = {
        "domain": "operational",
        "agent_version": "v1",
        "risks": ["O-01", "O-02", "O-03", "O-04"],
        "kri_updates": kri_updates,
        "bcm_updates": {},
        "breach_count": breaches,
        "amber_count": ambers,
        "escalation_required": escalation,
        "escalation_reasons": (["O-03 safety incident — zero tolerance breach"] if qu["safety_incidents_ytd"] > 0 else []) +
                              (["O-01 concentration >75%"] if sc["overall_single_source_concentration_pct"] > 75 else []),
        "interconnection_flags": llm_interconnections,
        "narrative": narrative,
        "confidence": "high",
        "proposed_ctrl_changes": {},
        "token_usage": token_usage,
    }

    valid, _ = validate_agent_output("operational", result)
    print(f"  [Operational] Complete — {breaches} breach(es), {ambers} amber(s) | Schema: {'valid' if valid else 'INVALID'}")
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
