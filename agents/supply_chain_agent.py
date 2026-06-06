"""
supply_chain_agent.py — O-01 Supply Chain Concentration & Disruption
Domain agent that reads ERP data, evaluates KRIs, calls Claude for
risk narrative, and returns structured findings to the Chief Risk Agent.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_reader import get_supply_chain_data
from tools.kri_evaluator import evaluate_kri, compute_residual_score, residual_to_label
from tools.risk_writer import update_risk, load_store


RISK_ID = "O-01"

APPETITE = {
    "domain": "Low appetite for supply chain disruption.",
    "tolerance": "No single supplier >40% of any component category; dual-source for all Tier-1.",
    "owner": "Chief Procurement Officer",
}

KRI_SPECS = [
    {
        "store_name": "single_source_concentration",
        "label": "Single-source concentration",
        "unit": "%",
        "amber": 40.0,
        "red": 50.0,
        "direction": "higher_worse",
    },
    {
        "store_name": "inventory_cover_weeks",
        "label": "Inventory cover",
        "unit": " weeks",
        "amber": 4.0,
        "red": 3.0,
        "direction": "lower_worse",
    },
    {
        "store_name": "supplier_distress_flags",
        "label": "Supplier financial distress flags",
        "unit": " suppliers",
        "amber": 2.0,
        "red": 4.0,
        "direction": "higher_worse",
    },
]


def run() -> dict:
    print(f"\n{'='*60}")
    print(f"[{RISK_ID}] Supply Chain Agent starting...")
    print(f"{'='*60}")

    # ── 1. Read data ──
    data = get_supply_chain_data()
    print(f"  Suppliers assessed: {data['suppliers_assessed']}")
    print(f"  Single-source concentration: {data['overall_single_source_concentration_pct']}%")
    print(f"  Min inventory cover: {data['min_inventory_weeks']} weeks")
    print(f"  Supplier distress flags: {data['supplier_distress_flags']}")

    # ── 2. Evaluate KRIs ──
    kri_values = {
        "single_source_concentration": data["overall_single_source_concentration_pct"],
        "inventory_cover_weeks": data["min_inventory_weeks"],
        "supplier_distress_flags": float(data["supplier_distress_flags"]),
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

    # ── 3. Compute residual ──
    store = load_store()
    risk = store["operational_risks"][RISK_ID]
    residual = compute_residual_score(risk["l"], risk["i"], risk["ctrl"])
    arith_label = residual_to_label(residual)
    print(f"  Residual score: {residual} → {arith_label}")

    # ── 4. Build context for Claude ──
    kri_summary = "\n".join(f"  - {r.finding}" for r in kri_results)
    geo_str = ", ".join(
        f"{k}: {v}%" for k, v in data["geographic_concentration_pct"].items()
    )
    cat_issues = [
        f"{cat}: {vals['concentration_pct']}% {'(single-source)' if vals['single_source'] else ''}"
        for cat, vals in data["category_concentration"].items()
        if vals["concentration_pct"] > 40 or vals["single_source"]
    ]
    cat_str = "; ".join(cat_issues) if cat_issues else "No category concentration breaches"

    prompt = f"""You are the Supply Chain Risk Agent for the Risk Intelligence and Resilience Hub.
Your role: analyse O-01 (Supply chain concentration & disruption) based on live ERP data and produce a concise, board-quality risk assessment.

RISK APPETITE:
{APPETITE['domain']}
Tolerance: {APPETITE['tolerance']}
Owner: {APPETITE['owner']}

LIVE ERP DATA:
- Overall single-source concentration: {data['overall_single_source_concentration_pct']}% (tolerance: ≤40%)
- Single-source spend: USD {data['single_source_spend_usd_m']}M of USD {data['total_spend_usd_m']}M total
- Minimum inventory cover: {data['min_inventory_weeks']} weeks
- Average inventory cover: {data['avg_inventory_weeks']} weeks
- Supplier distress flags (score <65): {data['supplier_distress_flags']} suppliers
- Geographic concentration: {geo_str}
- Category concentration issues: {cat_str}

KRI EVALUATION:
{kri_summary}

CURRENT RISK PARAMETERS:
- Likelihood: {risk['l']}/5, Impact: {risk['i']}/5
- Control effectiveness: {risk['ctrl']}%
- Arithmetic residual: {residual} → {arith_label}
- Expert label: {risk['lv']}

Produce a structured assessment with exactly these sections:
1. OVERALL ASSESSMENT (2 sentences: current status and primary concern)
2. KEY FINDINGS (3-4 bullet points, specific numbers, no generalities)
3. CONTROL EFFECTIVENESS ASSESSMENT (is {risk['ctrl']}% still accurate given the data? recommend up/down/same with rationale)
4. PROPOSED ACTIONS (3 numbered actions, each with: what, who, by when)
5. ESCALATION REQUIRED (yes/no and why — based on appetite breach)

Be direct. No hedging. Use exact numbers from the data. """

    # ── 5. Call Claude ──
    print(f"  Calling Claude for risk narrative...")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    narrative = message.content[0].text
    print(f"  Claude response received ({len(narrative)} chars)")

    # ── 6. Persist findings ──
    timestamp = datetime.now(timezone.utc).isoformat()
    updated_risk = update_risk(
        risk_id=RISK_ID,
        kri_updates=kri_updates,
        new_ctrl=None,   # agent observes but does not auto-update ctrl without CRO approval
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
        "escalation_required": len(breaches) > 0,
        "data_snapshot": {
            "single_source_pct": data["overall_single_source_concentration_pct"],
            "min_inventory_weeks": data["min_inventory_weeks"],
            "distress_flags": data["supplier_distress_flags"],
        },
    }

    print(f"  [{RISK_ID}] Complete — {len(breaches)} breach(es), {len(ambers)} amber(s)")
    return result


if __name__ == "__main__":
    result = run()
    print("\n--- AGENT OUTPUT ---")
    print(json.dumps(result, indent=2))
