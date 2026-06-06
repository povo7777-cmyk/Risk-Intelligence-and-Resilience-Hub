"""
cyber_agent.py — O-02 Cyber Attack & IT Resilience
Domain agent that reads SIEM/ITSM data, evaluates KRIs,
calls Claude for risk narrative, returns findings to Chief Risk Agent.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_reader import get_cyber_data
from tools.kri_evaluator import evaluate_kri, compute_residual_score, residual_to_label
from tools.risk_writer import update_risk, load_store


RISK_ID = "O-02"

APPETITE = {
    "domain": "Low tolerance for cyber incidents. MTTD must stay below 5 days; patch compliance must exceed 95%.",
    "tolerance": "MTTD <5 days; patch compliance ≥95%; IT system RTO ≤4h; zero firmware compromises.",
    "owner": "CISO / CTO",
}

KRI_SPECS = [
    {
        "store_name": "mttd_days",
        "label": "Mean Time to Detect (MTTD)",
        "unit": " days",
        "amber": 5.0,
        "red": 7.0,
        "direction": "higher_worse",
    },
    {
        "store_name": "patch_compliance_pct",
        "label": "Patch compliance rate",
        "unit": "%",
        "amber": 95.0,
        "red": 85.0,
        "direction": "lower_worse",
    },
    {
        "store_name": "critical_vulns_open_gt30d",
        "label": "Critical vulnerabilities open >30 days",
        "unit": " vulns",
        "amber": 5.0,
        "red": 8.0,
        "direction": "higher_worse",
    },
    {
        "store_name": "it_rto_hours",
        "label": "IT system RTO – trading platform",
        "unit": "h",
        "amber": 4.0,
        "red": 8.0,
        "direction": "higher_worse",
    },
]


def run() -> dict:
    print(f"\n{'='*60}")
    print(f"[{RISK_ID}] Cyber Agent starting...")
    print(f"{'='*60}")

    # ── 1. Read data ──
    data = get_cyber_data()
    print(f"  MTTD: {data['mttd_days']} days")
    print(f"  Patch compliance: {data['patch_compliance_pct']}%")
    print(f"  Critical vulns >30d: {data['critical_vulns_open_gt30d']}")
    print(f"  IT RTO: {data['it_rto_hours']}h")
    print(f"  Privileged access unreviewed: {data.get('raw_metrics', {}).get('privileged_access_unreviewed_days', 'N/A')} days")

    # ── 2. Evaluate KRIs ──
    kri_values = {
        "mttd_days": data["mttd_days"],
        "patch_compliance_pct": data["patch_compliance_pct"],
        "critical_vulns_open_gt30d": data["critical_vulns_open_gt30d"],
        "it_rto_hours": data["it_rto_hours"],
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

    # ── 4. Build context for Claude ──
    kri_summary = "\n".join(f"  - {r.finding}" for r in kri_results)
    breach_kris = [r for r in kri_results if r.status == "breach"]
    mfa = data.get("mfa_coverage_pct", "N/A")
    vendor_assessed = data.get("third_party_vendor_assessed_pct", "N/A")
    priv_days = data.get("raw_metrics", {}).get("privileged_access_unreviewed_days", "N/A")

    prompt = f"""You are the Cyber Risk Agent for the Risk Intelligence and Resilience Hub.
Your role: analyse O-02 (Cyber attack & IT resilience) based on live SIEM/ITSM data.

RISK APPETITE:
{APPETITE['domain']}
Tolerance: {APPETITE['tolerance']}
Owner: {APPETITE['owner']}

LIVE SIEM/ITSM DATA:
- Mean Time to Detect (MTTD): {data['mttd_days']} days (appetite: <5 days)
- Mean Time to Respond (MTTR): {data.get('mttr_days', 'N/A')} days
- Patch compliance: {data['patch_compliance_pct']}% (tolerance: ≥95%)
- Critical vulnerabilities open >30 days: {data['critical_vulns_open_gt30d']}
- High vulnerabilities open >30 days: {data.get('high_vulns_open_gt30d', 'N/A')}
- IT RTO (order management system): {data['it_rto_hours']}h (tolerance: ≤4h)
- Security incidents MTD: {data.get('security_incidents_mtd', 'N/A')}
- MFA coverage: {mfa}%
- Third-party vendor assessed: {vendor_assessed}%
- Privileged access unreviewed: {priv_days} days (tolerance: ≤30 days)

NOTE: Supplier cyber resilience assessment (0% coverage) is tracked as an O-01 supply chain KRI
owned by the COO/Procurement — do not include it in O-02 findings. Cross-reference it only if
discussing compound geopolitical + cyber scenarios where supplier network is the attack vector.

KRI EVALUATION:
{kri_summary}

CURRENT RISK PARAMETERS:
- Likelihood: {risk['l']}/5, Impact: {risk['i']}/5
- Control effectiveness: {risk['ctrl']}%
- Arithmetic residual: {residual} → {arith_label}
- Expert label: {risk['lv']}
- Active KRI breaches: {len(breach_kris)}

Produce a structured assessment with exactly these sections:
1. OVERALL ASSESSMENT (2 sentences: severity and primary concern)
2. KEY FINDINGS (3-4 bullet points, exact numbers, MITRE ATT&CK relevance if applicable)
3. CONTROL EFFECTIVENESS ASSESSMENT (is {risk['ctrl']}% accurate? Factor in patch compliance at {data['patch_compliance_pct']}%, privileged access unreviewed {priv_days} days, and MTTD at {data['mttd_days']} days)
4. PROPOSED ACTIONS (3 numbered actions: what, who, by when — reference NIS2 Article 21 where relevant)
5. ESCALATION REQUIRED (yes/no — with {len(breach_kris)} active breaches, be explicit about Board notification)

Direct language. Exact numbers. """

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

    # ── 6. Persist ──
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
        "escalation_required": len(breaches) > 0,
        "data_snapshot": {
            "mttd_days": data["mttd_days"],
            "patch_compliance_pct": data["patch_compliance_pct"],
            "critical_vulns": data["critical_vulns_open_gt30d"],
            "it_rto_hours": data["it_rto_hours"],
        },
    }

    print(f"  [{RISK_ID}] Complete — {len(breaches)} breach(es), {len(ambers)} amber(s)")
    return result


if __name__ == "__main__":
    result = run()
    print("\n--- AGENT OUTPUT ---")
    print(json.dumps(result, indent=2))
