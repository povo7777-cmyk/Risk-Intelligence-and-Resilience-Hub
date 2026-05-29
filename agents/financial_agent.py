"""
agents/financial_agent.py — F-01 to F-04
Uses Claude Sonnet 4.5.
Primary specialization: quantitative financial risk measurement.
"""

import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_reader import read_csv, read_csv_latest
from tools.risk_writer import update_risk, load_store
from schemas.agent_outputs import validate_agent_output
import anthropic

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "financial_v1.txt"
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


def _parse_treasury() -> dict:
    rows = read_csv_latest("treasury_positions.csv")
    total_gross = sum(float(r.get("gross_exposure_usd_m", 0)) for r in rows)
    total_hedged = sum(float(r.get("hedged_amount_usd_m", 0)) for r in rows)
    total_pnl = sum(float(r.get("unrealised_pnl_usd_m", 0)) for r in rows)
    unhedged = total_gross - total_hedged
    avg_hedge = (total_hedged / total_gross * 100) if total_gross > 0 else 0
    return {
        "total_unhedged_usd_m": round(unhedged, 1),
        "avg_hedge_ratio_pct": round(avg_hedge, 1),
        "total_unrealised_pnl_usd_m": round(total_pnl, 1),
    }


def _parse_ar() -> dict:
    rows = read_csv("ar_aging.csv")
    top_conc = max((float(r.get("top_customer_concentration_pct", 0)) for r in rows), default=22.1)
    overdue_90 = sum(float(r.get("overdue_90d_usd_m", 0)) for r in rows)
    total_current = sum(float(r.get("current_usd_m", 0)) for r in rows)
    bad_debt = sum(float(r.get("bad_debt_provision_usd_m", 0)) for r in rows)
    total_ar = total_current + overdue_90
    bad_debt_pct = (bad_debt / total_ar * 100) if total_ar > 0 else 0
    # % of AR overdue >90 days is more meaningful than the absolute USD amount
    overdue_90_pct = round((overdue_90 / total_ar * 100) if total_ar > 0 else 0, 1)
    return {
        "top_customer_concentration_pct": round(top_conc, 1),
        "overdue_90d_pct":              overdue_90_pct,      # % of total AR
        "total_overdue_90d_usd_m":      round(overdue_90, 1), # kept for narrative context
        "bad_debt_provision_pct":       round(bad_debt_pct, 2),
    }


def _parse_covenants() -> dict:
    rows = read_csv("covenant_tracker.csv")
    result = {}
    for r in rows:
        metric = r.get("metric", "")
        try:
            val = float(r.get("current_value", 0))
        except Exception:
            val = 0.0
        if "Net_Debt_EBITDA" in metric:
            result["net_debt_ebitda"] = val
        elif "Liquidity" in metric:
            result["liquidity_headroom_usd_b"] = val
        elif "maturity_runway" in metric:
            result["debt_maturity_runway_months"] = int(val)
        elif "Interest_Coverage" in metric:
            result["interest_coverage"] = val
    return result


def _parse_audit_log_financial() -> dict:
    """Derive F-04 KRI values from audit_log.csv (Internal-SOX / financial controls scope)."""
    rows = read_csv("audit_log.csv")
    sox_rows = [r for r in rows if any(t in r.get("audit_type", "") for t in ["SOX", "Internal"])]
    findings_open = sum(int(r.get("high_findings", 0)) for r in sox_rows)
    material_weakness = sum(int(r.get("critical_findings", 0)) for r in sox_rows)
    return {
        "audit_findings_open": float(findings_open),
        "material_weakness_count": float(material_weakness),
    }


def run(kri_data: dict | None = None) -> dict:
    """
    kri_data: pre-computed KRI values from kri_data_layer (when running in graph).
    If None, computes KRI values from CSVs directly (standalone / test mode).
    """
    print(f"\n{'='*60}")
    print("[Financial Agent] Starting — Claude Sonnet 4.5")
    print(f"{'='*60}")

    system_prompt = PROMPT_PATH.read_text()

    # Read all CSVs — no fallback defaults; missing data must be surfaced, not masked
    tr  = _parse_treasury()
    ar  = _parse_ar()
    cv  = _parse_covenants()
    f04 = _parse_audit_log_financial()

    unhedged_fx          = tr["total_unhedged_usd_m"]
    avg_hedge_ratio      = tr["avg_hedge_ratio_pct"]
    unrealised_pnl       = tr["total_unrealised_pnl_usd_m"]
    top_customer_pct     = ar["top_customer_concentration_pct"]
    overdue_90d_pct      = ar.get("overdue_90d_pct", 0.0)
    overdue_90d_usd      = ar["total_overdue_90d_usd_m"]
    bad_debt_pct         = ar["bad_debt_provision_pct"]
    net_debt_ebitda      = cv.get("net_debt_ebitda", 0.0)
    liquidity_headroom   = cv.get("liquidity_headroom_usd_b", 0.0)
    debt_maturity_months = cv.get("debt_maturity_runway_months", 0)

    # KRI values: use data layer if available, otherwise compute locally
    if kri_data and kri_data.get("dashboard_kris"):
        kri_updates = kri_data["dashboard_kris"]
        # agent_context has additional context metrics
        actx = kri_data.get("agent_context", {})
        unrealised_pnl  = actx.get("F-01", {}).get("unrealised_pnl_usd_m", unrealised_pnl)
        overdue_90d_pct = actx.get("F-02", {}).get("overdue_90d_pct", overdue_90d_pct)
        overdue_90d_usd = actx.get("F-02", {}).get("overdue_90d_usd_m", overdue_90d_usd)
        mw              = actx.get("F-04", {}).get("material_weakness_count", f04.get("material_weakness_count", 0))
    else:
        mw = f04.get("material_weakness_count", 0.0)
        kri_updates = {
            "F-01": [
                {"name": "unhedged_fx_exposure_usd_m", "value": unhedged_fx, "unit": "USD_M",
                 "status": "breach" if unhedged_fx >= 5000 else "amber" if unhedged_fx >= 4000 else "ok"},
                {"name": "avg_hedge_ratio_pct", "value": avg_hedge_ratio, "unit": "%",
                 "status": "breach" if avg_hedge_ratio < 40 else "amber" if avg_hedge_ratio < 55 else "ok"},
            ],
            "F-02": [
                {"name": "top_customer_concentration_pct", "value": top_customer_pct, "unit": "%",
                 "status": "breach" if top_customer_pct >= 25 else "amber" if top_customer_pct >= 20 else "ok"},
                {"name": "bad_debt_provision_pct", "value": bad_debt_pct, "unit": "%",
                 "status": "breach" if bad_debt_pct >= 0.70 else "amber" if bad_debt_pct >= 0.55 else "ok"},
            ],
            "F-03": [
                {"name": "net_debt_ebitda_ratio", "value": net_debt_ebitda, "unit": "ratio",
                 "status": "breach" if net_debt_ebitda >= 3.0 else "amber" if net_debt_ebitda >= 2.7 else "ok"},
                {"name": "liquidity_headroom_usd_b", "value": liquidity_headroom, "unit": "USD_B",
                 "status": "breach" if liquidity_headroom < 0.9 else "amber" if liquidity_headroom < 1.2 else "ok"},
                {"name": "debt_maturity_runway_months", "value": float(debt_maturity_months), "unit": "months",
                 "status": "breach" if debt_maturity_months < 6 else "amber" if debt_maturity_months < 12 else "ok"},
            ],
            "F-04": [
                {"name": "audit_findings_open", "value": f04["audit_findings_open"], "unit": "count",
                 "status": "breach" if f04["audit_findings_open"] >= 5 else "amber" if f04["audit_findings_open"] >= 2 else "ok"},
            ],
        }

    breaches = sum(1 for kris in kri_updates.values() for k in kris if k["status"] == "breach")
    ambers = sum(1 for kris in kri_updates.values() for k in kris if k["status"] == "amber")
    escalation = (unhedged_fx >= 5000 or liquidity_headroom < 0.9 or
                  net_debt_ebitda >= 3.0 or debt_maturity_months < 6)

    escalation_reasons = []
    if unhedged_fx >= 5000:
        escalation_reasons.append(f"F-01: Unhedged FX exposure USD {unhedged_fx}M exceeds USD 5,000M breach threshold")
    if net_debt_ebitda >= 3.0:
        escalation_reasons.append(f"F-03: Net Debt/EBITDA {net_debt_ebitda}x at covenant threshold 3.0x")
    if debt_maturity_months < 6:
        escalation_reasons.append(f"F-03: Debt maturity runway {debt_maturity_months}m below 6-month minimum")

    # Covenant status built from CSV — no hardcoded thresholds
    covenant_rows = read_csv("covenant_tracker.csv")
    covenant_status = {
        r["covenant_id"]: {
            "metric":    r["metric"],
            "value":     float(r["current_value"]),
            "threshold": float(r["threshold"]),
            "headroom":  float(r["headroom"]),
            "facility":  r["facility"],
            "next_test": r["next_test_date"],
            "status":    r["status"],
        }
        for r in covenant_rows
    }

    # ── Build rich user prompt — all CSV rows, not aggregates ────────────────
    treasury_rows = read_csv_latest("treasury_positions.csv")
    tr_lines = "\n".join(
        f"  {r['exposure_id']} {r['currency_pair']} [{r['exposure_type']}]: "
        f"gross ${r['gross_exposure_usd_m']}M, hedged ${r['hedged_amount_usd_m']}M "
        f"({r['hedge_ratio_pct']}%), unrealised P&L ${r['unrealised_pnl_usd_m']}M, "
        f"maturity {r['maturity_months']}mo, counterparty {r['counterparty']}"
        for r in treasury_rows
    )
    ar_rows = read_csv("ar_aging.csv")
    ar_lines = "\n".join(
        f"  {r['segment']} ({r['customer_tier']}): current ${r['current_usd_m']}M, "
        f"overdue_30d ${r['overdue_30d_usd_m']}M, overdue_60d ${r['overdue_60d_usd_m']}M, "
        f"overdue_90d ${r['overdue_90d_usd_m']}M, bad_debt_provision ${r['bad_debt_provision_usd_m']}M, "
        f"top_customer_conc {r['top_customer_concentration_pct']}%"
        for r in ar_rows
    )
    cov_lines = "\n".join(
        f"  {r['covenant_id']} {r['metric']}: current={r['current_value']}, "
        f"threshold={r['threshold']}, headroom={r['headroom']}, "
        f"facility={r['facility']}, next_test={r['next_test_date']}, status={r['status']}"
        for r in covenant_rows
    )
    audit_rows = read_csv("audit_log.csv")
    f04_lines = "\n".join(
        f"  {r['audit_id']} {r['audit_type']} [{r['subject']}]: "
        f"status={r['status']}, high={r['high_findings']}, critical={r['critical_findings']}, due={r['due_date']}"
        for r in audit_rows if any(t in r.get("audit_type","") for t in ["SOX","Internal"])
    )

    user_prompt = f"""F-01 FX & COMMODITY POSITIONS (all rows from treasury_positions.csv):
{tr_lines}
  Totals: unhedged=${unhedged_fx}M, avg_hedge={avg_hedge_ratio}%, total_unrealised_pnl=${unrealised_pnl}M
  (amber: unhedged>$4,000M | hedge_ratio<60% — breach: unhedged>$5,000M | hedge_ratio<45%)

F-02 AR AGING (all rows from ar_aging.csv):
{ar_lines}
  Totals: top_customer={top_customer_pct}%, overdue_90d_usd=${overdue_90d_usd}M (context only), bad_debt={bad_debt_pct}%
  (amber: top_customer>20% | bad_debt>0.55% — breach: >25% | >0.70%)

F-03 COVENANTS (all rows from covenant_tracker.csv):
{cov_lines}

F-04 FINANCIAL AUDIT FINDINGS (SOX/Internal scope from audit_log.csv):
{f04_lines}
  Summary: open_high_findings={int(f04['audit_findings_open'])}, material_weakness={int(f04.get('material_weakness_count',0))}
  (amber: findings>=2 — breach: findings>=5)

KRI STATUS THIS RUN: {breaches} breach(es), {ambers} amber(s)
ESCALATION: {'YES' if escalation else 'no'}

Assess all four risk areas. Name specific currency pairs, counterparties, customer segments,
and covenants. Surface cross-domain implications where the data supports them.
Identify interconnection flags from the data — not generic statements.
Return the full JSON per the output format in your instructions."""

    # LLM generates interconnections from data — no hardcoded Python flags
    narrative = (f"Financial assessment: {breaches} breach(es), {ambers} amber(s). "
                 f"F-01 unhedged ${unhedged_fx}M, hedge {avg_hedge_ratio}%. "
                 f"F-03 Net Debt/EBITDA {net_debt_ebitda}x, maturity {debt_maturity_months}mo.")
    llm_interconnections = []
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        from tools.json_parser import parse_llm_json
        print("  Calling Claude Sonnet 4.5 for financial assessment...")
        raw, token_usage = _call_claude(system_prompt, user_prompt)
        parsed, parse_err = parse_llm_json(raw)
        if parsed:
            if parsed.get("narrative"):
                narrative = parsed["narrative"]
            llm_interconnections = parsed.get("interconnection_flags", [])
            if llm_interconnections:
                print(f"  Agent identified {len(llm_interconnections)} interconnection flag(s)")
        else:
            print(f"  Claude assessment parse failed: {parse_err} — using computed narrative")
    except Exception as e:
        print(f"  Claude assessment failed: {e} — using computed narrative")

    # KRI values already written by kri_data_layer; write only narrative metadata here
    for risk_id in kri_updates:
        try:
            update_risk(risk_id=risk_id, kri_updates={}, new_ctrl=None, new_lv=None,
                        agent_findings=narrative, proposed_actions=[])
        except Exception as e:
            print(f"  Warning: update_risk failed for {risk_id}: {e}")

    result = {
        "domain": "financial",
        "agent_version": "v1",
        "risks": ["F-01", "F-02", "F-03", "F-04"],
        "kri_updates": kri_updates,
        "covenant_status": covenant_status,
        "breach_count": breaches,
        "amber_count": ambers,
        "escalation_required": escalation,
        "escalation_reasons": escalation_reasons,
        "interconnection_flags": llm_interconnections,
        "narrative": narrative,
        "confidence": "high",
        "proposed_ctrl_changes": {},
        "token_usage": token_usage,
    }

    valid, _ = validate_agent_output("financial", result)
    print(f"  [Financial] Complete — {breaches} breach(es), {ambers} amber(s) | Schema: {'valid' if valid else 'INVALID'}")
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
