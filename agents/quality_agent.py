"""
quality_agent.py — O-03 Product Quality & Safety Failure
Domain agent reading QMS data, evaluating KRIs, returning findings.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_reader import get_quality_data
from tools.kri_evaluator import evaluate_kri, compute_residual_score, residual_to_label
from tools.risk_writer import update_risk, load_store


RISK_ID = "O-03"

APPETITE = {
    "domain": "Zero tolerance for major safety incidents. Low appetite for product quality failures.",
    "tolerance": "Field failure rate <0.05%; recall readiness ≥85%; zero confirmed safety incidents.",
    "owner": "Chief Quality Officer / COO",
}

KRI_SPECS = [
    {
        "store_name": "field_failure_rate_pct",
        "label": "Field failure rate",
        "unit": "%",
        "amber": 0.04,
        "red": 0.05,
        "direction": "higher_worse",
    },
    {
        "store_name": "recall_readiness_score_pct",
        "label": "Recall readiness score",
        "unit": "%",
        "amber": 80.0,
        "red": 85.0,
        "direction": "lower_worse",
    },
    {
        "store_name": "safety_incidents_ytd",
        "label": "Confirmed safety incidents YTD",
        "unit": " incidents",
        "amber": 1.0,
        "red": 1.0,
        "direction": "higher_worse",
    },
]


def run() -> dict:
    print(f"\n{'='*60}")
    print(f"[{RISK_ID}] Quality Agent starting...")
    print(f"{'='*60}")

    data = get_quality_data()
    print(f"  Max field failure rate: {data['max_field_failure_rate_pct']}%")
    print(f"  Recall readiness: {data['recall_readiness_score_pct']}%")
    print(f"  Safety incidents YTD: {data['safety_incidents_ytd']}")
    print(f"  Near-miss events YTD: {data['near_miss_events_ytd']}")

    kri_values = {
        "field_failure_rate_pct": data["max_field_failure_rate_pct"],
        "recall_readiness_score_pct": data["recall_readiness_score_pct"],
        "safety_incidents_ytd": data["safety_incidents_ytd"],
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
    recall_months = data.get("recall_simulation_last_run_months_ago", "N/A")

    prompt = f"""You are the Product Quality Risk Agent for the Risk Intelligence and Resilience Hub.
Your role: analyse O-03 (Product quality & safety failure) based on live QMS data.

RISK APPETITE:
{APPETITE['domain']}
Tolerance: {APPETITE['tolerance']}
Owner: {APPETITE['owner']}

LIVE QMS DATA:
- Field failure rate (KRI aggregate — worst SKU across all product lines): {data['max_field_failure_rate_pct']}% (tolerance: <0.05%)
- Warranty claims MTD: {data.get('warranty_claims_mtd', 'N/A')}
- Recall readiness score: {data['recall_readiness_score_pct']}% (tolerance: ≥85%)
- Recall simulation last run: {recall_months} months ago
- Confirmed safety incidents YTD: {data['safety_incidents_ytd']} (zero tolerance)
- Near-miss events YTD: {data.get('near_miss_events_ytd', 'N/A')}

CITATION RULE: In your narrative, cite only the aggregate field failure rate above ({data['max_field_failure_rate_pct']}%).
Do NOT break down by product line or cite individual SKU rates — the KRI is the aggregate.

KRI EVALUATION:
{kri_summary}

CURRENT RISK PARAMETERS:
- Likelihood: {risk['l']}/5, Impact: {risk['i']}/5
- Control effectiveness: {risk['ctrl']}%
- Arithmetic residual: {residual} → {arith_label}
- Expert label: {risk['lv']}

Produce a structured assessment:
1. OVERALL ASSESSMENT (2 sentences)
2. KEY FINDINGS (3-4 bullets with exact numbers)
3. CONTROL EFFECTIVENESS ASSESSMENT (is {risk['ctrl']}% accurate? recall readiness at {data['recall_readiness_score_pct']}% suggests what?)
4. PROPOSED ACTIONS (3 numbered, what/who/when — note recall simulation gap of {recall_months} months)
5. ESCALATION REQUIRED (zero-tolerance domain — even near-miss events matter)

 Be direct."""

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
        "escalation_required": data["safety_incidents_ytd"] > 0 or len(breaches) > 0,
        "data_snapshot": {
            "field_failure_rate_pct": data["max_field_failure_rate_pct"],
            "recall_readiness_pct": data["recall_readiness_score_pct"],
            "safety_incidents": data["safety_incidents_ytd"],
        },
    }

    print(f"  [{RISK_ID}] Complete — {len(breaches)} breach(es), {len(ambers)} amber(s)")
    return result


if __name__ == "__main__":
    result = run()
    print("\n--- AGENT OUTPUT ---")
    print(json.dumps(result, indent=2))
