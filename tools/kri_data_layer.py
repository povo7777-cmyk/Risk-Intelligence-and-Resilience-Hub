"""
tools/kri_data_layer.py
=======================
Single source of truth for ALL KRI values.

ARCHITECTURE
------------
DASHBOARD KRIs  — computed deterministically from CSV source files.
                  Written to risk_store under the domain bucket (e.g. financial_risks).
                  No LLM involvement. Reproducible on every run.

AGENT CONTEXT   — additional metrics computed from the same CSV files but NOT
                  shown as dashboard tiles. Stored separately in state under
                  'agent_context' and passed to the Chief Risk Agent so it can
                  reason about signals that don't have a formal dashboard KRI.

This module runs as its own graph node (kri_data_layer_node) BEFORE domain agents.
Agents read KRI values from state rather than re-computing them from CSVs.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

STORE_PATH = Path(__file__).parent.parent / "api" / "risk_store.json"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_reader import (
    read_csv,
    get_supply_chain_data,
    get_cyber_data,
    get_quality_data,
    get_talent_data,
)


# ── Threshold helpers ─────────────────────────────────────────────────────────

# KRI thresholds loaded once at module import from kri_thresholds.csv.
# Each entry: {kri_name: {amber, breach, direction, unit, description}}
_THRESHOLDS: dict = {}

def _load_thresholds() -> dict:
    """Load KRI thresholds from data/kri_thresholds.csv."""
    csv_path = Path(__file__).parent.parent / "data" / "kri_thresholds.csv"
    result = {}
    if not csv_path.exists():
        return result
    import csv as _csv
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            result[row["kri_name"]] = {
                "amber":     float(row["amber_threshold"]),
                "breach":    float(row["breach_threshold"]),
                "direction": row.get("direction", "higher_worse"),
                "unit":      row.get("unit", ""),
                "description": row.get("description", ""),
                "risk_id":   row.get("risk_id", ""),
            }
    return result

_THRESHOLDS = _load_thresholds()


def _status(value: float, kri_name: str) -> str:
    """
    Return 'breach' / 'amber' / 'ok' using thresholds from kri_thresholds.csv.
    Raises KeyError if kri_name not found — surfaces missing threshold entries early.
    """
    t = _THRESHOLDS[kri_name]
    amber, breach, direction = t["amber"], t["breach"], t["direction"]
    if direction == "higher_worse":
        if value >= breach: return "breach"
        if value >= amber:  return "amber"
        return "ok"
    else:  # lower_worse
        if value <= breach: return "breach"
        if value <= amber:  return "amber"
        return "ok"


def _kri(name: str, value: float, unit: str, status: str) -> dict:
    """Build a KRI dict, including threshold values for CRA context."""
    t = _THRESHOLDS.get(name, {})
    return {
        "name":      name,
        "value":     value,
        "unit":      unit,
        "status":    status,
        "threshold": t.get("breach", "?"),
        "amber_threshold": t.get("amber", "?"),
        "direction": t.get("direction", "higher_worse"),
    }


# ── OPERATIONAL ───────────────────────────────────────────────────────────────

def _compute_operational() -> dict:
    """Returns {risk_id: [kri_dict, ...]} for O-01 → O-04."""
    sc = get_supply_chain_data()
    cy = get_cyber_data()
    qu = get_quality_data()
    ta = get_talent_data()

    conc  = sc["overall_single_source_concentration_pct"]
    inv   = sc["min_inventory_weeks"]
    dist  = float(sc["supplier_distress_flags"])

    mttd   = cy["mttd_days"]
    patch  = cy["patch_compliance_pct"]
    vulns  = float(cy["critical_vulns_open_gt30d"])
    rto    = cy["it_rto_hours"]

    ffr    = qu["max_field_failure_rate_pct"]
    recall = qu["recall_readiness_score_pct"]
    safety = float(qu["safety_incidents_ytd"])

    attr   = ta["tech_attrition_engineering_pct"]
    roles  = float(ta["critical_open_roles_gt60d"])
    succ   = ta["svp_succession_coverage_pct"]

    return {
        "O-01": [
            _kri("single_source_concentration", conc, "%",   _status(conc,  "single_source_concentration")),
            _kri("inventory_cover_weeks",        inv,  "weeks", _status(inv,  "inventory_cover_weeks")),
            _kri("supplier_distress_flags",       dist, "count", _status(dist, "supplier_distress_flags")),
        ],
        "O-02": [
            _kri("mttd_days",                    mttd,  "days",  _status(mttd,  "mttd_days")),
            _kri("patch_compliance_pct",         patch, "%",     _status(patch, "patch_compliance_pct")),
            _kri("critical_vulns_open_gt30d",    vulns, "count", _status(vulns, "critical_vulns_open_gt30d")),
            _kri("it_rto_hours",                 rto,   "hours", _status(rto,   "it_rto_hours")),
        ],
        "O-03": [
            _kri("field_failure_rate_pct",       ffr,    "%",   _status(ffr,    "field_failure_rate_pct")),
            _kri("recall_readiness_score_pct",   recall, "%",   _status(recall, "recall_readiness_score_pct")),
            _kri("safety_incidents_ytd",         safety, "count", _status(safety, "safety_incidents_ytd")),
        ],
        "O-04": [
            _kri("tech_attrition_rate_pct",      attr,  "%",   _status(attr,  "tech_attrition_rate_pct")),
            _kri("critical_open_roles_gt60d",    roles, "count", _status(roles, "critical_open_roles_gt60d")),
            _kri("svp_succession_coverage_pct",  succ,  "%",   _status(succ,  "svp_succession_coverage_pct")),
        ],
    }


# ── FINANCIAL ────────────────────────────────────────────────────────────────

def _compute_financial() -> dict:
    """Returns {risk_id: [kri_dict, ...]} for F-01 → F-04 (dashboard KRIs only)."""
    # F-01 — treasury positions
    tr_rows   = read_csv("treasury_positions.csv")
    gross     = sum(float(r.get("gross_exposure_usd_m", 0)) for r in tr_rows)
    hedged    = sum(float(r.get("hedged_amount_usd_m", 0)) for r in tr_rows)
    unhedged  = round(gross - hedged, 1)
    hedge_pct = round((hedged / gross * 100) if gross > 0 else 0, 1)

    # F-02 — accounts receivable
    ar_rows   = read_csv("ar_aging.csv")
    top_conc  = round(max((float(r.get("top_customer_concentration_pct", 0)) for r in ar_rows), default=0), 1)
    overdue   = sum(float(r.get("overdue_90d_usd_m", 0)) for r in ar_rows)
    curr_ar   = sum(float(r.get("current_usd_m", 0)) for r in ar_rows)
    bad_debt  = sum(float(r.get("bad_debt_provision_usd_m", 0)) for r in ar_rows)
    total_ar  = curr_ar + overdue
    bd_pct    = round((bad_debt / total_ar * 100) if total_ar > 0 else 0, 2)
    ov_pct    = round((overdue / total_ar * 100) if total_ar > 0 else 0, 1)

    # F-03 — covenants
    cv_rows   = read_csv("covenant_tracker.csv")
    cv        = {}
    for r in cv_rows:
        m = r.get("metric", "")
        try:
            v = float(r.get("current_value", 0))
        except Exception:
            v = 0.0
        if "Net_Debt_EBITDA" in m:   cv["net_debt_ebitda_ratio"]     = v
        elif "Liquidity"      in m:  cv["liquidity_headroom_usd_b"]  = v
        elif "maturity_runway" in m: cv["debt_maturity_runway_months"] = int(v)
    nde   = cv.get("net_debt_ebitda_ratio")
    liq   = cv.get("liquidity_headroom_usd_b")
    mat_m = cv.get("debt_maturity_runway_months")
    if any(v is None for v in [nde, liq, mat_m]):
        missing = [k for k, v in {"net_debt_ebitda_ratio": nde,
                                   "liquidity_headroom_usd_b": liq,
                                   "debt_maturity_runway_months": mat_m}.items() if v is None]
        raise ValueError(f"covenant_tracker.csv missing required metrics: {missing}")

    # F-04 — audit log (Internal-SOX scope)
    audit_rows = read_csv("audit_log.csv")
    sox_rows   = [r for r in audit_rows if any(t in r.get("audit_type", "") for t in ["SOX", "Internal"])]
    sox_high   = float(sum(int(r.get("high_findings", 0)) for r in sox_rows))

    return {
        "F-01": [
            _kri("unhedged_fx_exposure_usd_m", unhedged,  "USD_M", _status(unhedged,  "unhedged_fx_exposure_usd_m")),
            _kri("avg_hedge_ratio_pct",        hedge_pct, "%",     _status(hedge_pct, "avg_hedge_ratio_pct")),
        ],
        "F-02": [
            _kri("top_customer_concentration_pct", top_conc, "%",  _status(top_conc, "top_customer_concentration_pct")),
            _kri("bad_debt_provision_pct",          bd_pct,   "%",  _status(bd_pct,   "bad_debt_provision_pct")),
        ],
        "F-03": [
            _kri("net_debt_ebitda_ratio",       nde,          "ratio",  _status(nde,   "net_debt_ebitda_ratio")),
            _kri("liquidity_headroom_usd_b",    liq,          "USD_B",  _status(liq,   "liquidity_headroom_usd_b")),
            _kri("debt_maturity_runway_months", float(mat_m), "months", _status(mat_m, "debt_maturity_runway_months")),
        ],
        "F-04": [
            _kri("audit_findings_open", sox_high, "count", _status(sox_high, "audit_findings_open")),
        ],
    }


# ── STRATEGIC ────────────────────────────────────────────────────────────────

def _compute_strategic() -> dict:
    """Returns {risk_id: [kri_dict, ...]} for S-01, S-02, S-03 — all from CSVs."""
    # S-01 / S-02 — market intelligence signals (market_intelligence.csv)
    signals    = read_csv("market_intelligence.csv")
    s01_count  = float(len([s for s in signals if s.get("risk_id") == "S-01"]))
    s02_count  = float(len([s for s in signals if s.get("risk_id") == "S-02"]))
    high_count = float(len([s for s in signals if s.get("severity") == "high"]))

    # S-03 — synergy delivery from deals currently in active integration (ma_pipeline.csv)
    # Any stage starting with "Integration" counts as active integration
    pipeline          = read_csv("ma_pipeline.csv")
    integration_deals = [d for d in pipeline if d.get("stage", "").startswith("Integration")]
    if not integration_deals:
        raise ValueError(
            "ma_pipeline.csv has no deals in an Integration stage — "
            "cannot compute synergy_delivery_pct for S-03"
        )
    synergy = round(
        sum(float(d["synergy_delivered_pct"]) for d in integration_deals) / len(integration_deals), 1
    )

    return {
        "S-01": [
            _kri("geopolitical_signal_count", s01_count,  "count", _status(s01_count,  "geopolitical_signal_count")),
        ],
        "S-02": [
            _kri("competitive_signals", s02_count, "count", _status(s02_count, "competitive_signals")),
        ],
        "S-03": [
            _kri("synergy_delivery_pct", synergy, "%", _status(synergy, "synergy_delivery_pct")),
        ],
    }


# ── COMPLIANCE ───────────────────────────────────────────────────────────────

def _compute_compliance() -> dict:
    """Returns {risk_id: [kri_dict, ...]} for C-01 → C-03 (dashboard KRIs only)."""
    # C-01 — export screening
    scr_rows  = read_csv("screening_results.csv")
    scr       = {r["metric"]: float(r["value"]) for r in scr_rows}
    cov       = scr.get("export_screening_coverage_pct")
    viols     = scr.get("confirmed_sanctions_violations_ytd")
    if cov is None or viols is None:
        missing = [k for k, v in {"export_screening_coverage_pct": cov,
                                   "confirmed_sanctions_violations_ytd": viols}.items() if v is None]
        raise ValueError(f"screening_results.csv missing required metrics: {missing}")

    # C-02/C-03 — compliance metrics
    cm_rows   = read_csv("compliance_metrics.csv")
    cm        = {r["metric"]: float(r["value"]) for r in cm_rows}
    ai_audit  = cm.get("ai_audit_coverage_pct")
    dsr_rate  = cm.get("gdpr_dsr_resolution_rate_pct")
    abac_cov  = cm.get("third_party_abac_coverage_pct")
    scope3    = cm.get("csrd_scope3_disclosure_pct")
    if any(v is None for v in [ai_audit, dsr_rate, abac_cov, scope3]):
        missing = [k for k, v in {"ai_audit_coverage_pct": ai_audit,
                                   "gdpr_dsr_resolution_rate_pct": dsr_rate,
                                   "third_party_abac_coverage_pct": abac_cov,
                                   "csrd_scope3_disclosure_pct": scope3}.items() if v is None]
        raise ValueError(f"compliance_metrics.csv missing required metrics: {missing}")

    return {
        "C-01": [
            _kri("export_screening_coverage_pct",      cov,   "%",     _status(cov,   "export_screening_coverage_pct")),
            _kri("confirmed_sanctions_violations_ytd", viols, "count", _status(viols, "confirmed_sanctions_violations_ytd")),
        ],
        "C-02": [
            _kri("ai_audit_coverage_pct",         ai_audit, "%", _status(ai_audit, "ai_audit_coverage_pct")),
            _kri("gdpr_dsr_resolution_rate_pct",  dsr_rate, "%", _status(dsr_rate, "gdpr_dsr_resolution_rate_pct")),
        ],
        "C-03": [
            _kri("third_party_abac_coverage_pct", abac_cov, "%", _status(abac_cov, "third_party_abac_coverage_pct")),
            _kri("csrd_scope3_disclosure_pct",    scope3,   "%", _status(scope3,   "csrd_scope3_disclosure_pct")),
        ],
    }


# ── AGENT CONTEXT (additional signals — no dashboard tile) ───────────────────

def _compute_agent_context() -> dict:
    """
    Additional metrics computed from CSVs that the Chief Risk Agent uses
    for context and cross-domain reasoning. NOT written to dashboard.

    Structured as {risk_id: {metric_name: value, ...}}.
    Each entry includes a human-readable label and the computed value.
    """
    ctx = {}

    # F-01: unrealised FX P&L (treasury_positions.csv)
    try:
        tr = read_csv("treasury_positions.csv")
        pnl = round(sum(float(r.get("unrealised_pnl_usd_m", 0)) for r in tr), 1)
        ctx["F-01"] = {"unrealised_pnl_usd_m": pnl}
    except Exception:
        ctx["F-01"] = {}

    # F-02: absolute USD overdue >90 days (overdue_90d_pct is a dashboard KRI in _compute_financial)
    try:
        ar = read_csv("ar_aging.csv")
        overdue = sum(float(r.get("overdue_90d_usd_m", 0)) for r in ar)
        ctx["F-02"] = {"overdue_90d_usd_m": round(overdue, 1)}
    except Exception:
        ctx["F-02"] = {}

    # F-04: material weakness (audit_log.csv — Internal-SOX critical findings)
    try:
        audit = read_csv("audit_log.csv")
        sox   = [r for r in audit if any(t in r.get("audit_type", "") for t in ["SOX", "Internal"])]
        mw    = float(sum(int(r.get("critical_findings", 0)) for r in sox))
        ctx["F-04"] = {"material_weakness_count": mw}
    except Exception:
        ctx["F-04"] = {}

    # S-01: geopolitical signal counts (market_intelligence.csv)
    try:
        signals = read_csv("market_intelligence.csv")
        s01_signals  = [s for s in signals if s.get("risk_id") == "S-01"]
        high_signals = [s for s in signals if s.get("severity") == "high"]
        ctx["S-01"] = {
            "geopolitical_signal_count": len(s01_signals),
            "high_severity_signal_count": len(high_signals),  # context only — not a dashboard KRI
            # Include titles for narrative use
            "signal_titles": [s.get("title", "") for s in s01_signals[:6]],
        }
    except Exception:
        ctx["S-01"] = {}

    # S-02: competitive signal count (market_intelligence.csv)
    try:
        if "signals" not in dir():
            signals = read_csv("market_intelligence.csv")
        s02_signals = [s for s in signals if s.get("risk_id") == "S-02"]
        ctx["S-02"] = {
            "competitive_signals": len(s02_signals),
            "signal_titles": [s.get("title", "") for s in s02_signals[:4]],
        }
    except Exception:
        ctx["S-02"] = {}

    # S-03: deals in active pipeline without controls designed yet (ma_pipeline.csv)
    try:
        pipeline = read_csv("ma_pipeline.csv")
        active   = [d for d in pipeline if d.get("stage") in ["Due-Diligence", "Negotiation"]]
        ctx["S-03"] = {
            "pipeline_deals_active": len(active),
            "deal_names": [d.get("target_name", "") for d in active],
        }
    except Exception:
        ctx["S-03"] = {}

    # C-01: denied party matches pending (screening_results.csv)
    try:
        scr = read_csv("screening_results.csv")
        scr_d = {r["metric"]: float(r["value"]) for r in scr}
        ctx["C-01"] = {"denied_party_matches_pending": scr_d.get("denied_party_matches_pending", 0.0)}
    except Exception:
        ctx["C-01"] = {}

    # C-02: formal regulatory investigation open (audit_log.csv)
    try:
        audit = read_csv("audit_log.csv")
        formal = any("investigation" in r.get("audit_type", "").lower() for r in audit)
        ctx["C-02"] = {"formal_investigation_open": 1 if formal else 0}
    except Exception:
        ctx["C-02"] = {}

    # C-03: whistleblower high findings (audit_log.csv)
    try:
        audit = read_csv("audit_log.csv")
        wb = next((r for r in audit if "Whistleblower" in r.get("audit_type", "")), {})
        ctx["C-03"] = {"whistleblower_high_findings": int(wb.get("high_findings", 0))}
    except Exception:
        ctx["C-03"] = {}

    return ctx


# ── Write helpers ─────────────────────────────────────────────────────────────

_BUCKET_MAP = {
    "O": "operational_risks",
    "S": "strategic_risks",
    "F": "financial_risks",
    "C": "compliance_risks",
}


def _write_to_store(dashboard_kris: dict):
    """
    Write dashboard KRI values directly to risk_store.json.
    Uses file locking (same as risk_writer) to avoid corruption.
    """
    import fcntl
    with open(STORE_PATH, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            store = json.load(f)

            for domain, risks in dashboard_kris.items():
                for risk_id, kri_list in risks.items():
                    bucket = _BUCKET_MAP.get(risk_id[0], "operational_risks")
                    if bucket not in store:
                        store[bucket] = {}
                    if risk_id not in store[bucket]:
                        store[bucket][risk_id] = {"kris": {}}
                    for kri in kri_list:
                        store[bucket][risk_id]["kris"][kri["name"]] = {
                            "value":  kri["value"],
                            "status": kri["status"],
                        }

            store["last_updated"] = datetime.now(timezone.utc).isoformat()
            store["kri_last_computed"] = datetime.now(timezone.utc).isoformat()

            f.seek(0)
            f.truncate()
            json.dump(store, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ── Public entry point ────────────────────────────────────────────────────────

def run() -> dict:
    """
    Compute all KRI values from CSV source files.

    Returns:
        {
            "dashboard_kris": {domain: {risk_id: [kri_dict, ...]}}
                — written to risk_store; updater patches dashboard HTML
            "agent_context": {risk_id: {metric: value, ...}}
                — available to Chief Risk Agent; NOT written to dashboard
            "summary": {breach_count, amber_count, by_domain}
        }
    """
    errors = []

    try:
        op_kris = _compute_operational()
    except Exception as e:
        op_kris = {}
        errors.append(f"Operational KRI computation failed: {e}")

    try:
        fin_kris = _compute_financial()
    except Exception as e:
        fin_kris = {}
        errors.append(f"Financial KRI computation failed: {e}")

    try:
        str_kris = _compute_strategic()
    except Exception as e:
        str_kris = {}
        errors.append(f"Strategic KRI computation failed: {e}")

    try:
        com_kris = _compute_compliance()
    except Exception as e:
        com_kris = {}
        errors.append(f"Compliance KRI computation failed: {e}")

    try:
        agent_ctx = _compute_agent_context()
    except Exception as e:
        agent_ctx = {}
        errors.append(f"Agent context computation failed: {e}")

    # Merge by domain bucket
    dashboard_kris = {
        "operational_risks": op_kris,
        "financial_risks":   fin_kris,
        "strategic_risks":   str_kris,
        "compliance_risks":  com_kris,
    }

    # Write dashboard KRIs to store
    _write_to_store(dashboard_kris)

    # Build summary
    all_kris = [k for domain in dashboard_kris.values()
                  for kri_list in domain.values()
                  for k in kri_list]
    breaches = sum(1 for k in all_kris if k["status"] == "breach")
    ambers   = sum(1 for k in all_kris if k["status"] == "amber")

    by_domain = {}
    for bucket, risks in dashboard_kris.items():
        domain_name = bucket.replace("_risks", "")
        kris_flat = [k for kl in risks.values() for k in kl]
        by_domain[domain_name] = {
            "breach_count": sum(1 for k in kris_flat if k["status"] == "breach"),
            "amber_count":  sum(1 for k in kris_flat if k["status"] == "amber"),
        }

    print(f"  [KRI DATA LAYER] {len(all_kris)} KRIs computed — "
          f"{breaches} breach(es), {ambers} amber(s)")
    if errors:
        for e in errors:
            print(f"  ⚠ {e}")

    return {
        "dashboard_kris": dashboard_kris,
        "agent_context":  agent_ctx,
        "summary": {
            "total_kris":    len(all_kris),
            "breach_count":  breaches,
            "amber_count":   ambers,
            "by_domain":     by_domain,
        },
        "errors": errors,
    }


if __name__ == "__main__":
    import json as _json
    result = run()
    print("\n--- DASHBOARD KRIs ---")
    for bucket, risks in result["dashboard_kris"].items():
        for risk_id, kris in risks.items():
            for k in kris:
                flag = "🔴" if k["status"] == "breach" else "🟡" if k["status"] == "amber" else "🟢"
                print(f"  {flag} {risk_id}.{k['name']:45s} = {k['value']} ({k['status']})")

    print("\n--- AGENT CONTEXT ---")
    for risk_id, ctx in result["agent_context"].items():
        for metric, value in ctx.items():
            if metric != "signal_titles" and metric != "deal_names":
                print(f"  {risk_id}.{metric} = {value}")
