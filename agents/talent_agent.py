"""
talent_agent.py — O-04 Talent Retention & Key-Person Risk
Domain agent reading HRIS data, evaluating KRIs, returning findings.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_reader import get_talent_data
from tools.kri_evaluator import evaluate_kri, compute_residual_score, residual_to_label
from tools.risk_writer import update_risk, load_store


RISK_ID = "O-04"

APPETITE = {
    "domain": "Low appetite for key-person dependency and talent concentration risk.",
    "tolerance": "Tech attrition <15%; succession plans for all SVP+; open critical roles <12.",
    "owner": "CHRO",
}

KRI_SPECS = [
    {
        "store_name": "tech_attrition_rate_pct",
        "label": "Tech role attrition rate",
        "unit": "%",
        "amber": 13.0,
        "red": 15.0,
        "direction": "higher_worse",
    },
    {
        "store_name": "critical_open_roles_gt60d",
        "label": "Critical open roles >60 days",
        "unit": " roles",
        "amber": 8.0,
        "red": 12.0,
        "direction": "higher_worse",
    },
    {
        "store_name": "svp_succession_coverage_pct",
        "label": "SVP+ succession plan coverage",
        "unit": "%",
        "amber": 90.0,
        "red": 80.0,
        "direction": "lower_worse",
    },
]


def run() -> dict:
    print(f"\n{'='*60}")
    print(f"[{RISK_ID}] Talent Agent starting...")
    print(f"{'='*60}")

    data = get_talent_data()
    print(f"  Tech attrition (Engineering): {data['tech_attrition_engineering_pct']}%")
    print(f"  Critical open roles >60d: {data['critical_open_roles_gt60d']}")
    print(f"  SVP+ succession coverage: {data['svp_succession_coverage_pct']}%")
    print(f"  Flight risk employees flagged: {data['flight_risk_flagged']}")
    print(f"  Compensation gap vs market: {data['compensation_gap_pct']}%")

    kri_values = {
        "tech_attrition_rate_pct": data["tech_attrition_engineering_pct"],
        "critical_open_roles_gt60d": data["critical_open_roles_gt60d"],
        "svp_succession_coverage_pct": data["svp_succession_coverage_pct"],
    }

    kri_results = []
    kri_updates = {}
    for spec in KRI_SPECS:
        val = kri_values[spec["store_name"]]
        result = evaluate_kri(
            name=spec["store_name"],
            value=val,
            unit=spec["unit"],
            amber=spec["amber"],
            red=spec["red"],
            direction=spec["direction"],
            label=spec["label"],
        )
        kri_results.append(result)
        kri_updates[spec["store_name"]] = {"value": val, "status": result.status}
        print(f"  KRI [{result.status.upper()}] {result.finding}")

    store = load_store()
    risk = store["operational_risks"][RISK_ID]
    residual = compute_residual_score(risk["l"], risk["i"], risk["ctrl"])
    arith_label = residual_to_label(residual)

    kri_summary = "\n".join(f"  - {r.finding}" for r in kri_results)

    prompt = f"""You are the Talent Risk Agent for the Risk Intelligence and Resilience Hub.
Your role: analyse O-04 (Talent retention & key-person risk) based on live HRIS data.

RISK APPETITE:
{APPETITE['domain']}
Tolerance: {APPETITE['tolerance']}
Owner: {APPETITE['owner']}

OVERRIDE NOTE: This risk carries an expert override — rated High despite arithmetic Low.
Justification: 3 senior engineers flagged as flight risk; no SVP successors documented
for 2 critical roles; compensation gap widening vs hyperscalers.

LIVE HRIS DATA:
- Tech attrition rate (Engineering, annualised): {data['tech_attrition_engineering_pct']}% (red: 15%)
- Critical open roles >60 days: {data['critical_open_roles_gt60d']} (red: 12)
- SVP+ succession plan coverage: {data['svp_succession_coverage_pct']}% (red: <80%)
- Flight risk employees flagged: {data['flight_risk_flagged']} (Trading Desk)
- Compensation gap vs market: +{data['compensation_gap_pct']}% above market paid, or below?
- Avg time to fill critical role: {data['avg_time_to_fill_days']} days
- Employee engagement score: {data['engagement_score_pct']}%
- Counter-offer acceptance rate: {data['counter_offer_acceptance_pct']}%

KRI EVALUATION:
{kri_summary}

CURRENT RISK PARAMETERS:
- Expert label: {risk['lv']} (arithmetic: {arith_label}) — override in place
- Control effectiveness: {risk['ctrl']}%

Produce a structured assessment:
1. OVERALL ASSESSMENT (2 sentences — acknowledge the override and explain why it is justified by the data)
2. KEY FINDINGS (3-4 bullets — focus on the leading indicators: flight risk, comp gap, succession gap)
3. CONTROL EFFECTIVENESS ASSESSMENT (is {risk['ctrl']}% accurate given a 42% counter-offer acceptance rate?)
4. PROPOSED ACTIONS (3 numbered, what/who/when — be specific about retention package timing)
5. ESCALATION REQUIRED (yes/no — focus on the Trading Desk flight risk concentration)

 Direct language."""

    print(f"  Calling Claude for risk narrative...")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    narrative = message.content[0].text

    timestamp = datetime.now(timezone.utc).isoformat()
    update_risk(
        risk_id=RISK_ID,
        kri_updates=kri_updates,
        new_ctrl=None,
        new_lv=None,
        agent_findings=narrative,
        proposed_actions=[],
    )

    breaches = [r for r in kri_results if r.status == "breach"]
    ambers = [r for r in kri_results if r.status == "amber"]

    result = {
        "risk_id": RISK_ID,
        "risk_name": risk["name"],
        "timestamp": timestamp,
        "kri_results": [
            {"name": r.name, "value": r.value, "status": r.status, "finding": r.finding}
            for r in kri_results
        ],
        "breach_count": len(breaches),
        "amber_count": len(ambers),
        "residual_score": residual,
        "arithmetic_label": arith_label,
        "expert_label": risk["lv"],
        "narrative": narrative,
        "escalation_required": data["flight_risk_flagged"] >= 2 or len(breaches) > 0,
        "data_snapshot": {
            "attrition_pct": data["tech_attrition_engineering_pct"],
            "open_roles": data["critical_open_roles_gt60d"],
            "succession_pct": data["svp_succession_coverage_pct"],
            "flight_risk": data["flight_risk_flagged"],
        },
    }

    print(f"  [{RISK_ID}] Complete — {len(breaches)} breach(es), {len(ambers)} amber(s)")
    return result


if __name__ == "__main__":
    result = run()
    print("\n--- AGENT OUTPUT ---")
    print(json.dumps(result, indent=2))
