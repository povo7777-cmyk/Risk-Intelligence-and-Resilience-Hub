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
    read_csv_latest,
    read_csv_prior,
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

    # EM-04: Geographic concentration — Taiwan + China combined supply chain spend
    sc_raw = read_csv_latest("erp_supply_chain.csv")
    _total_sc_spend = sum(float(r.get("our_spend_usd_m", 0)) for r in sc_raw)
    _taiwan_china_spend = sum(
        float(r.get("our_spend_usd_m", 0)) for r in sc_raw
        if r.get("country") in ("Taiwan", "China")
    )
    geo_conc = round(_taiwan_china_spend / _total_sc_spend * 100, 1) if _total_sc_spend else 0.0

    mttd         = cy["mttd_days"]
    patch        = cy["patch_compliance_pct"]
    vulns        = float(cy["critical_vulns_open_gt30d"])
    rto          = cy["it_rto_hours"]
    cyber_assess = cy.get("supplier_cyber_resilience_assess_pct", 0.0) or 0.0

    ffr       = qu["max_field_failure_rate_pct"]
    recall    = qu["recall_readiness_score_pct"]
    safety    = float(qu["safety_incidents_ytd"])
    rejection = qu.get("supplier_quality_rejection_rate_pct", 0.0) or 0.0

    attr   = ta["tech_attrition_engineering_pct"]
    roles  = float(ta["critical_open_roles_gt60d"])
    succ   = ta["svp_succession_coverage_pct"]

    return {
        "O-01": [
            _kri("single_source_concentration",          conc,         "%",     _status(conc,         "single_source_concentration")),
            _kri("inventory_cover_weeks",                inv,          "weeks", _status(inv,          "inventory_cover_weeks")),
            _kri("supplier_distress_flags",              dist,         "count", _status(dist,         "supplier_distress_flags")),
            _kri("supplier_cyber_resilience_assess_pct", cyber_assess, "%",     _status(cyber_assess, "supplier_cyber_resilience_assess_pct")),
            _kri("geo_concentration_pct",                geo_conc,     "%",     _status(geo_conc,     "geo_concentration_pct")),
        ],
        "O-02": [
            _kri("mttd_days",                 mttd,  "days",  _status(mttd,  "mttd_days")),
            _kri("patch_compliance_pct",      patch, "%",     _status(patch, "patch_compliance_pct")),
            _kri("critical_vulns_open_gt30d", vulns, "count", _status(vulns, "critical_vulns_open_gt30d")),
            _kri("it_rto_hours",              rto,   "hours", _status(rto,   "it_rto_hours")),
        ],
        "O-03": [
            _kri("field_failure_rate_pct",           ffr,       "%",   _status(ffr,       "field_failure_rate_pct")),
            _kri("recall_readiness_score_pct",       recall,    "%",   _status(recall,    "recall_readiness_score_pct")),
            _kri("safety_incidents_ytd",             safety,    "count", _status(safety,  "safety_incidents_ytd")),
            _kri("supplier_quality_rejection_rate_pct", rejection, "%", _status(rejection, "supplier_quality_rejection_rate_pct")),
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
    # F-01 — treasury positions (current period only)
    # FX positions (Revenue + Cost) and Commodity positions (DRAM/NAND) are separated.
    # unhedged_fx_exposure_usd_m and avg_hedge_ratio_pct cover FX-only.
    # unhedged_commodity_exposure_usd_m covers commodity swaps separately.
    tr_rows    = read_csv_latest("treasury_positions.csv")
    fx_types   = ("Revenue", "Cost")
    fx_rows    = [r for r in tr_rows if r.get("exposure_type", "") in fx_types]
    com_rows   = [r for r in tr_rows if r.get("exposure_type", "") == "Commodity"]
    fx_gross   = sum(float(r.get("gross_exposure_usd_m", 0)) for r in fx_rows)
    fx_hedged  = sum(float(r.get("hedged_amount_usd_m",  0)) for r in fx_rows)
    com_gross  = sum(float(r.get("gross_exposure_usd_m", 0)) for r in com_rows)
    com_hedged = sum(float(r.get("hedged_amount_usd_m",  0)) for r in com_rows)
    unhedged   = round(fx_gross  - fx_hedged,  1)   # FX-only unhedged (F-01 KRI scope)
    com_unhedged = round(com_gross - com_hedged, 1)  # commodity unhedged (new KRI)
    hedge_pct  = round((fx_hedged / fx_gross * 100) if fx_gross > 0 else 0, 1)

    # F-02 — accounts receivable (current period only)
    ar_rows   = read_csv_latest("ar_aging.csv")
    top_conc  = round(max((float(r.get("top_customer_concentration_pct", 0)) for r in ar_rows), default=0), 1)
    overdue   = sum(float(r.get("overdue_90d_usd_m", 0)) for r in ar_rows)
    curr_ar   = sum(float(r.get("current_usd_m", 0)) for r in ar_rows)
    bad_debt  = sum(float(r.get("bad_debt_provision_usd_m", 0)) for r in ar_rows)
    total_ar  = curr_ar + overdue
    bd_pct    = round((bad_debt / total_ar * 100) if total_ar > 0 else 0, 2)
    ov_pct    = round((overdue / total_ar * 100) if total_ar > 0 else 0, 1)

    # F-03 — covenants (current period only)
    cv_rows   = read_csv_latest("covenant_tracker.csv")
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

    # F-04 — audit log (Internal-SOX scope, current period only)
    audit_rows = read_csv_latest("audit_log.csv")
    sox_rows   = [r for r in audit_rows if any(t in r.get("audit_type", "") for t in ["SOX", "Internal"])]
    sox_high   = float(sum(int(r.get("high_findings", 0)) for r in sox_rows))

    return {
        "F-01": [
            _kri("unhedged_fx_exposure_usd_m",        unhedged,     "USD_M", _status(unhedged,     "unhedged_fx_exposure_usd_m")),
            _kri("avg_hedge_ratio_pct",                hedge_pct,    "%",     _status(hedge_pct,    "avg_hedge_ratio_pct")),
            _kri("unhedged_commodity_exposure_usd_m",  com_unhedged, "USD_M", _status(com_unhedged, "unhedged_commodity_exposure_usd_m")),
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
    # S-01 / S-02 — market intelligence signals (current period only)
    signals    = read_csv_latest("market_intelligence.csv")
    s01_count  = float(len([s for s in signals if s.get("risk_id") == "S-01"]))
    s02_count  = float(len([s for s in signals if s.get("risk_id") == "S-02"]))

    # S-01 — ODM/EMS concentration in PRC-jurisdiction (current period)
    sc_rows  = read_csv_latest("erp_supply_chain.csv")
    odm_rows = [r for r in sc_rows if r.get("component_category", "").startswith(("Manufacturing", "ODM"))]
    total_odm = sum(float(r.get("our_spend_usd_m", 0)) for r in odm_rows)
    prc_odm   = sum(float(r.get("our_spend_usd_m", 0)) for r in odm_rows if r.get("country") == "China")
    prc_pct   = round(prc_odm / total_odm * 100, 1) if total_odm > 0 else 0.0

    # S-03 — synergy delivery from deals currently in active integration (current period only)
    # Any stage starting with "Integration" counts as active integration
    pipeline          = read_csv_latest("ma_pipeline.csv")
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
            _kri("geopolitical_signal_count",     s01_count, "count", _status(s01_count, "geopolitical_signal_count")),
            _kri("odm_ems_prc_concentration_pct", prc_pct,   "%",     _status(prc_pct,   "odm_ems_prc_concentration_pct")),
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
    # C-01 — export screening (current period only)
    scr_rows  = read_csv_latest("screening_results.csv")
    scr       = {r["metric"]: float(r["value"]) for r in scr_rows}
    cov       = scr.get("export_screening_coverage_pct")
    viols     = scr.get("confirmed_sanctions_violations_ytd")
    if cov is None or viols is None:
        missing = [k for k, v in {"export_screening_coverage_pct": cov,
                                   "confirmed_sanctions_violations_ytd": viols}.items() if v is None]
        raise ValueError(f"screening_results.csv missing required metrics: {missing}")

    # C-02/C-03 — compliance metrics (current period only)
    cm_rows   = read_csv_latest("compliance_metrics.csv")
    cm        = {r["metric"]: float(r["value"]) for r in cm_rows}
    ai_audit  = cm.get("ai_audit_coverage_pct")
    abac_cov  = cm.get("third_party_abac_coverage_pct")
    if any(v is None for v in [ai_audit, abac_cov]):
        missing = [k for k, v in {"ai_audit_coverage_pct": ai_audit,
                                   "third_party_abac_coverage_pct": abac_cov}.items() if v is None]
        raise ValueError(f"compliance_metrics.csv missing required metrics: {missing}")

    # C-01 — export licence expiry count (optional: present from FY26Q2 onwards)
    # Read from compliance_metrics.csv if the metric exists — no error if absent (backward compat)
    lic_expiring = cm.get("export_licence_expiring_30d")

    c01_kris = [
        _kri("export_screening_coverage_pct",      cov,   "%",     _status(cov,   "export_screening_coverage_pct")),
        _kri("confirmed_sanctions_violations_ytd", viols, "count", _status(viols, "confirmed_sanctions_violations_ytd")),
    ]
    if lic_expiring is not None:
        c01_kris.append(
            _kri("export_licence_expiring_30d", lic_expiring, "count",
                 _status(lic_expiring, "export_licence_expiring_30d"))
        )

    return {
        "C-01": c01_kris,
        "C-02": [
            _kri("ai_audit_coverage_pct", ai_audit, "%", _status(ai_audit, "ai_audit_coverage_pct")),
        ],
        "C-03": [
            _kri("third_party_abac_coverage_pct", abac_cov, "%", _status(abac_cov, "third_party_abac_coverage_pct")),
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

    # F-01: FX P&L + explicit Revenue vs Cost pair breakdown (EM-05 fix)
    try:
        tr = read_csv_latest("treasury_positions.csv")
        pnl = round(sum(float(r.get("unrealised_pnl_usd_m", 0)) for r in tr), 1)
        total_unhedged = round(sum(
            float(r.get("gross_exposure_usd_m", 0)) - float(r.get("hedged_amount_usd_m", 0))
            for r in tr
        ), 0)
        rev_rows = [r for r in tr if r.get("exposure_type") == "Revenue"]
        rev_unhedged = sorted(
            [
                (r["currency_pair"], round(float(r["gross_exposure_usd_m"]) - float(r["hedged_amount_usd_m"]), 0))
                for r in rev_rows
            ],
            key=lambda x: x[1], reverse=True
        )
        total_rev_unhedged = sum(v for _, v in rev_unhedged)
        ctx["F-01"] = {
            "unrealised_pnl_usd_m": pnl,
            "primary_unhedged_revenue_pair": rev_unhedged[0][0] if rev_unhedged else "N/A",
            "primary_unhedged_revenue_usd_m": rev_unhedged[0][1] if rev_unhedged else 0,
            "revenue_unhedged_pairs": rev_unhedged,
            "total_revenue_unhedged_usd_m": total_rev_unhedged,
            "revenue_pct_of_total_unhedged": round(total_rev_unhedged / total_unhedged * 100, 1) if total_unhedged else 0,
            "note": "Revenue pairs only — KRW is a Cost exposure and must NOT be cited as priority Revenue pair",
        }
    except Exception:
        ctx["F-01"] = {}

    # F-02: absolute USD overdue >90 days (current period)
    try:
        ar = read_csv_latest("ar_aging.csv")
        overdue = sum(float(r.get("overdue_90d_usd_m", 0)) for r in ar)
        ctx["F-02"] = {"overdue_90d_usd_m": round(overdue, 1)}
    except Exception:
        ctx["F-02"] = {}

    # F-04: material weakness (audit_log.csv — Internal-SOX critical findings, current period)
    try:
        audit = read_csv_latest("audit_log.csv")
        sox   = [r for r in audit if any(t in r.get("audit_type", "") for t in ["SOX", "Internal"])]
        mw    = float(sum(int(r.get("critical_findings", 0)) for r in sox))
        ctx["F-04"] = {"material_weakness_count": mw}
    except Exception:
        ctx["F-04"] = {}

    # S-01: geopolitical signal counts (current period)
    try:
        signals = read_csv_latest("market_intelligence.csv")
        s01_signals  = [s for s in signals if s.get("risk_id") == "S-01"]
        high_signals = [s for s in s01_signals if s.get("severity") == "high"]
        ctx["S-01"] = {
            "geopolitical_signal_count": len(s01_signals),
            "high_severity_signal_count": len(high_signals),
            "signal_titles": [s.get("title", "") for s in s01_signals[:6]],
        }
    except Exception:
        ctx["S-01"] = {}

    # S-02: competitive signal count (current period)
    try:
        signals_s02 = read_csv_latest("market_intelligence.csv")
        s02_signals = [s for s in signals_s02 if s.get("risk_id") == "S-02"]
        ctx["S-02"] = {
            "competitive_signals": len(s02_signals),
            "signal_titles": [s.get("title", "") for s in s02_signals[:4]],
        }
    except Exception:
        ctx["S-02"] = {}

    # S-03: M&A pipeline counts (current period)
    # active_deals_total = all non-completed deals (integration + pre-close)
    # preclose_deals = Due-Diligence and Negotiation stages only (pre-integration)
    try:
        pipeline = read_csv_latest("ma_pipeline.csv")
        active_all = [d for d in pipeline if d.get("stage") not in ["Completed"]]
        preclose   = [d for d in pipeline if d.get("stage") in ["Due-Diligence", "Negotiation"]]
        ctx["S-03"] = {
            "active_deals_total":    len(active_all),
            "preclose_deals":        len(preclose),
            "deal_names_active":     [d.get("target_name", "") for d in active_all],
            "deal_names_preclose":   [d.get("target_name", "") for d in preclose],
        }
    except Exception:
        ctx["S-03"] = {}

    # C-01: denied party matches pending (current period)
    try:
        scr = read_csv_latest("screening_results.csv")
        scr_d = {r["metric"]: float(r["value"]) for r in scr}
        ctx["C-01"] = {"denied_party_matches_pending": scr_d.get("denied_party_matches_pending", 0.0)}
    except Exception:
        ctx["C-01"] = {}

    # C-02: formal regulatory investigation open (current period)
    try:
        audit = read_csv_latest("audit_log.csv")
        formal = any("investigation" in r.get("audit_type", "").lower() for r in audit)
        ctx["C-02"] = {"formal_investigation_open": 1 if formal else 0}
    except Exception:
        ctx["C-02"] = {}

    # C-03: whistleblower high findings — sum ALL Whistleblower rows (current period)
    try:
        audit_wb = read_csv_latest("audit_log.csv")
        wb_rows = [r for r in audit_wb if "Whistleblower" in r.get("audit_type", "")]
        ctx["C-03"] = {"whistleblower_high_findings": int(sum(int(r.get("high_findings", 0)) for r in wb_rows))}
    except Exception:
        ctx["C-03"] = {}

    return ctx


# ── QoQ delta computation ─────────────────────────────────────────────────────

def _compute_qoq_deltas() -> dict:
    """
    Compute quarter-on-quarter KRI movements by comparing the two most recent
    date slices across all time-series CSVs.

    Returns a structured dict used to inject data-locked facts into the board
    synthesis QoQ section — preventing fabrication of prior-period values.
    """
    # Period labels from financial_summary.csv
    fin_latest = read_csv_latest("financial_summary.csv")
    fin_prior  = read_csv_prior("financial_summary.csv")
    if not fin_prior:
        return {"available": False, "reason": "No prior period data in financial_summary.csv"}

    period_current = fin_latest[-1].get("fiscal_quarter", "Current") if fin_latest else "Current"
    period_prior   = fin_prior[-1].get("fiscal_quarter", "Prior")    if fin_prior  else "Prior"
    date_current   = fin_latest[-1].get("date", "") if fin_latest else ""
    date_prior     = fin_prior[-1].get("date", "")  if fin_prior  else ""

    movements = []

    def _compare(risk_id: str, kri_name: str, prior_val, current_val):
        """Append a movement entry if both values are available."""
        if prior_val is None or current_val is None:
            return
        t = _THRESHOLDS.get(kri_name, {})
        direction = t.get("direction", "higher_worse")
        prior_status   = _status(float(prior_val),   kri_name) if kri_name in _THRESHOLDS else "?"
        current_status = _status(float(current_val), kri_name) if kri_name in _THRESHOLDS else "?"
        # 2% tolerance band for "stable"
        ratio = float(current_val) / float(prior_val) if float(prior_val) != 0 else 1.0
        if direction == "higher_worse":
            trend = "deteriorating" if ratio > 1.02 else ("improving" if ratio < 0.98 else "stable")
        else:
            trend = "deteriorating" if ratio < 0.98 else ("improving" if ratio > 1.02 else "stable")
        movements.append({
            "risk_id":        risk_id,
            "kri_name":       kri_name,
            "prior_value":    round(float(prior_val),   3),
            "current_value":  round(float(current_val), 3),
            "trend":          trend,
            "prior_status":   prior_status,
            "current_status": current_status,
            "status_change":  f"{prior_status}→{current_status}" if prior_status != current_status else "unchanged",
        })

    # ── Cyber ──────────────────────────────────────────────────────────────────
    cy_p = {r["metric"]: float(r["value"]) for r in read_csv_prior("siem_cyber.csv")}
    cy_c = {r["metric"]: float(r["value"]) for r in read_csv_latest("siem_cyber.csv")}
    _compare("O-02", "mttd_days",                        cy_p.get("mean_time_to_detect"),                cy_c.get("mean_time_to_detect"))
    _compare("O-02", "mttr_days",                        cy_p.get("mean_time_to_respond"),               cy_c.get("mean_time_to_respond"))
    _compare("O-02", "patch_compliance_pct",             cy_p.get("patch_compliance_rate"),              cy_c.get("patch_compliance_rate"))
    _compare("O-01", "supplier_cyber_resilience_assess_pct", cy_p.get("supplier_cyber_resilience_assess_pct"), cy_c.get("supplier_cyber_resilience_assess_pct"))

    # ── Talent ─────────────────────────────────────────────────────────────────
    ta_p_eng = {r["metric"]: float(r["value"]) for r in read_csv_prior("hris_talent.csv")  if r.get("department") == "Engineering"}
    ta_c_eng = {r["metric"]: float(r["value"]) for r in read_csv_latest("hris_talent.csv") if r.get("department") == "Engineering"}
    ta_p_all = {r["metric"]: float(r["value"]) for r in read_csv_prior("hris_talent.csv")  if r.get("department") == "All"}
    ta_c_all = {r["metric"]: float(r["value"]) for r in read_csv_latest("hris_talent.csv") if r.get("department") == "All"}
    _compare("O-04", "tech_attrition_rate_pct",     ta_p_eng.get("tech_role_attrition_rate_annualised"), ta_c_eng.get("tech_role_attrition_rate_annualised"))
    _compare("O-04", "critical_open_roles_gt60d",   ta_p_all.get("critical_open_roles_gt60d"),           ta_c_all.get("critical_open_roles_gt60d"))
    _compare("O-04", "svp_succession_coverage_pct", ta_p_all.get("svp_succession_plan_coverage"),        ta_c_all.get("svp_succession_plan_coverage"))

    # ── Quality ────────────────────────────────────────────────────────────────
    qu_p_all = {r["metric"]: float(r["value"]) for r in read_csv_prior("qms_quality.csv")  if r.get("sku_category") == "All"}
    qu_c_all = {r["metric"]: float(r["value"]) for r in read_csv_latest("qms_quality.csv") if r.get("sku_category") == "All"}
    ffr_prior   = max((float(r["value"]) for r in read_csv_prior("qms_quality.csv")  if r["metric"] == "field_failure_rate"), default=None)
    ffr_current = max((float(r["value"]) for r in read_csv_latest("qms_quality.csv") if r["metric"] == "field_failure_rate"), default=None)
    _compare("O-03", "field_failure_rate_pct",     ffr_prior,                                ffr_current)
    _compare("O-03", "recall_readiness_score_pct", qu_p_all.get("recall_readiness_score"),   qu_c_all.get("recall_readiness_score"))

    # ── Supply chain ───────────────────────────────────────────────────────────
    sc_p = read_csv_prior("erp_supply_chain.csv")
    sc_c = read_csv_latest("erp_supply_chain.csv")
    if sc_p and sc_c:
        total_p  = sum(float(r["our_spend_usd_m"]) for r in sc_p)
        single_p = sum(float(r["our_spend_usd_m"]) for r in sc_p if r.get("single_source", "").lower() == "true")
        total_c  = sum(float(r["our_spend_usd_m"]) for r in sc_c)
        single_c = sum(float(r["our_spend_usd_m"]) for r in sc_c if r.get("single_source", "").lower() == "true")
        conc_p = round(single_p / total_p * 100, 1) if total_p else 0
        conc_c = round(single_c / total_c * 100, 1) if total_c else 0
        _compare("O-01", "single_source_concentration", conc_p, conc_c)
        # Min inventory
        inv_p = min((float(r["inventory_weeks"]) for r in sc_p), default=None)
        inv_c = min((float(r["inventory_weeks"]) for r in sc_c), default=None)
        _compare("O-01", "inventory_cover_weeks", inv_p, inv_c)
        # ODM/EMS PRC concentration QoQ
        odm_p = [r for r in sc_p if r.get("component_category", "").startswith(("Manufacturing", "ODM"))]
        odm_c = [r for r in sc_c if r.get("component_category", "").startswith(("Manufacturing", "ODM"))]
        tot_odm_p = sum(float(r["our_spend_usd_m"]) for r in odm_p)
        tot_odm_c = sum(float(r["our_spend_usd_m"]) for r in odm_c)
        prc_p_val = sum(float(r["our_spend_usd_m"]) for r in odm_p if r.get("country") == "China")
        prc_c_val = sum(float(r["our_spend_usd_m"]) for r in odm_c if r.get("country") == "China")
        prc_pct_p = round(prc_p_val / tot_odm_p * 100, 1) if tot_odm_p else 0.0
        prc_pct_c = round(prc_c_val / tot_odm_c * 100, 1) if tot_odm_c else 0.0
        _compare("S-01", "odm_ems_prc_concentration_pct", prc_pct_p, prc_pct_c)

    # ── Financial — covenant ratios ────────────────────────────────────────────
    cv_p_rows = read_csv_prior("covenant_tracker.csv")
    cv_c_rows = read_csv_latest("covenant_tracker.csv")
    cv_p = {r["metric"]: float(r["current_value"]) for r in cv_p_rows}
    cv_c = {r["metric"]: float(r["current_value"]) for r in cv_c_rows}
    nde_p = next((v for m, v in cv_p.items() if "Net_Debt_EBITDA" in m), None)
    nde_c = next((v for m, v in cv_c.items() if "Net_Debt_EBITDA" in m), None)
    _compare("F-03", "net_debt_ebitda_ratio", nde_p, nde_c)

    # ── Financial — bad debt provision ────────────────────────────────────────
    ar_p = read_csv_prior("ar_aging.csv")
    ar_c = read_csv_latest("ar_aging.csv")
    if ar_p and ar_c:
        bd_p  = sum(float(r.get("bad_debt_provision_usd_m", 0)) for r in ar_p)
        tot_p = sum(float(r.get("current_usd_m", 0)) + float(r.get("overdue_90d_usd_m", 0)) for r in ar_p)
        bd_c  = sum(float(r.get("bad_debt_provision_usd_m", 0)) for r in ar_c)
        tot_c = sum(float(r.get("current_usd_m", 0)) + float(r.get("overdue_90d_usd_m", 0)) for r in ar_c)
        bd_pct_p = round(bd_p / tot_p * 100, 2) if tot_p else 0
        bd_pct_c = round(bd_c / tot_c * 100, 2) if tot_c else 0
        _compare("F-02", "bad_debt_provision_pct", bd_pct_p, bd_pct_c)

    # ── Financial — hedge ratio ────────────────────────────────────────────────
    tr_p = read_csv_prior("treasury_positions.csv")
    tr_c = read_csv_latest("treasury_positions.csv")
    if tr_p and tr_c:
        gross_p  = sum(float(r.get("gross_exposure_usd_m", 0)) for r in tr_p)
        hedged_p = sum(float(r.get("hedged_amount_usd_m", 0))  for r in tr_p)
        gross_c  = sum(float(r.get("gross_exposure_usd_m", 0)) for r in tr_c)
        hedged_c = sum(float(r.get("hedged_amount_usd_m", 0))  for r in tr_c)
        hr_p = round(hedged_p / gross_p * 100, 1) if gross_p else 0
        hr_c = round(hedged_c / gross_c * 100, 1) if gross_c else 0
        _compare("F-01", "avg_hedge_ratio_pct", hr_p, hr_c)

    # ── Compliance ────────────────────────────────────────────────────────────
    cm_p = {r["metric"]: float(r["value"]) for r in read_csv_prior("compliance_metrics.csv")}
    cm_c = {r["metric"]: float(r["value"]) for r in read_csv_latest("compliance_metrics.csv")}
    _compare("C-02", "ai_audit_coverage_pct",         cm_p.get("ai_audit_coverage_pct"),         cm_c.get("ai_audit_coverage_pct"))
    _compare("C-03", "third_party_abac_coverage_pct", cm_p.get("third_party_abac_coverage_pct"), cm_c.get("third_party_abac_coverage_pct"))

    # ── Strategic signals ─────────────────────────────────────────────────────
    mi_p = read_csv_prior("market_intelligence.csv")
    mi_c = read_csv_latest("market_intelligence.csv")
    if mi_p and mi_c:
        _compare("S-01", "geopolitical_signal_count",
                 float(len([s for s in mi_p if s.get("risk_id") == "S-01"])),
                 float(len([s for s in mi_c if s.get("risk_id") == "S-01"])))
        _compare("S-02", "competitive_signals",
                 float(len([s for s in mi_p if s.get("risk_id") == "S-02"])),
                 float(len([s for s in mi_c if s.get("risk_id") == "S-02"])))

    # ── M&A synergy ────────────────────────────────────────────────────────────
    try:
        pip_p = read_csv_prior("ma_pipeline.csv")
        pip_c = read_csv_latest("ma_pipeline.csv")
        int_p = [d for d in pip_p if d.get("stage", "").startswith("Integration")]
        int_c = [d for d in pip_c if d.get("stage", "").startswith("Integration")]
        if int_p and int_c:
            syn_p = round(sum(float(d["synergy_delivered_pct"]) for d in int_p) / len(int_p), 1)
            syn_c = round(sum(float(d["synergy_delivered_pct"]) for d in int_c) / len(int_c), 1)
            _compare("S-03", "synergy_delivery_pct", syn_p, syn_c)
    except Exception:
        pass

    # ── Summarise ──────────────────────────────────────────────────────────────
    deteriorating = [m for m in movements if m["trend"] == "deteriorating"]
    improving     = [m for m in movements if m["trend"] == "improving"]
    new_breaches  = [m for m in movements if m["current_status"] == "breach" and m["prior_status"] != "breach"]
    cleared       = [m for m in movements if m["prior_status"] == "breach"   and m["current_status"] != "breach"]

    # Dominant cause: control weakening vs inherent risk growth
    control_kris  = {"patch_compliance_pct", "recall_readiness_score_pct", "avg_hedge_ratio_pct",
                     "supplier_cyber_resilience_assess_pct", "audit_findings_open"}
    det_control   = sum(1 for m in deteriorating if m["kri_name"] in control_kris)
    det_inherent  = len(deteriorating) - det_control
    dominant_cause = "control_weakening" if det_control >= det_inherent else "inherent_risk_growth"

    # Primary decision point: highest-severity deteriorating KRI
    priority = {"breach": 0, "amber": 1, "ok": 2, "?": 3}
    primary = min(deteriorating, key=lambda m: (priority.get(m["current_status"], 3), 0), default=None)

    return {
        "available":       True,
        "period_current":  period_current,
        "period_prior":    period_prior,
        "date_current":    date_current,
        "date_prior":      date_prior,
        "movements":       movements,
        "summary": {
            "total_compared":      len(movements),
            "deteriorating_count": len(deteriorating),
            "improving_count":     len(improving),
            "stable_count":        len(movements) - len(deteriorating) - len(improving),
            "new_breaches":        [(m["risk_id"], m["kri_name"]) for m in new_breaches],
            "cleared_breaches":    [(m["risk_id"], m["kri_name"]) for m in cleared],
        },
        "dominant_cause":  dominant_cause,
        "primary_kri":     primary["kri_name"] if primary else None,
        "deteriorating":   deteriorating,
        "improving":       improving,
    }


# ── Write helpers ─────────────────────────────────────────────────────────────

_BUCKET_MAP = {
    "O": "operational_risks",
    "S": "strategic_risks",
    "F": "financial_risks",
    "C": "compliance_risks",
}

# KRI-status → (likelihood_score, impact_score) deltas applied to base risk rating.
# Aggregation rule: breach KRI → HIGH minimum; 2+ ambers → MEDIUM minimum.
_STATUS_RANK = {"ok": 0, "amber": 1, "breach": 2}
_SCORE_MAP   = {
    # (likelihood, impact) → label
    (1, 1): "low",   (1, 2): "low",   (1, 3): "medium",
    (2, 1): "low",   (2, 2): "medium",(2, 3): "high",
    (3, 1): "medium",(3, 2): "high",  (3, 3): "high",
}


def _recompute_ratings(store: dict) -> None:
    """
    Re-derive each risk's aggregate severity label (lv) and likelihood score (l)
    from the live KRI statuses currently in the store.  Overwrites static values.

    Rules:
      - Any breach KRI       → likelihood ≥ 3 (high); label ≥ "high"
      - ≥ 2 amber KRIs       → likelihood ≥ 2 (medium); label ≥ "medium"
      - All KRIs ok          → likelihood stays at base value from risk_register
    Impact (i) is kept from the existing store value — KRI data alone cannot
    determine consequence magnitude; that requires expert judgment.
    """
    _buckets = [
        "operational_risks", "strategic_risks",
        "financial_risks", "compliance_risks",
    ]
    changed: list[str] = []

    for bucket in _buckets:
        for risk_id, risk_obj in store.get(bucket, {}).items():
            kris = risk_obj.get("kris", {})
            statuses = [
                kri.get("status", "ok")
                for kri in kris.values()
                if isinstance(kri, dict) and "status" in kri
            ]
            if not statuses:
                continue

            breach_count = statuses.count("breach")
            amber_count  = statuses.count("amber")

            # Derive minimum likelihood from KRI state
            if breach_count >= 1:
                min_l = 3
            elif amber_count >= 2:
                min_l = 2
            elif amber_count == 1:
                min_l = max(1, risk_obj.get("l", 1))  # don't downgrade
            else:
                min_l = risk_obj.get("l", 1)          # all ok — preserve

            # Never downgrade existing score (experts may have set it higher)
            current_l = risk_obj.get("l", 1)
            new_l = max(current_l, min_l)

            # Derive label from (l, i) lookup
            current_i = risk_obj.get("i", risk_obj.get("impact", 2))
            new_lv = _SCORE_MAP.get((new_l, current_i), risk_obj.get("lv", "medium"))
            # Hard floor: any breach → at least "high"
            if breach_count >= 1 and new_lv not in ("high", "critical"):
                new_lv = "high"

            if new_l != current_l or new_lv != risk_obj.get("lv"):
                changed.append(
                    f"{risk_id}: l {current_l}→{new_l}, lv '{risk_obj.get('lv')}'→'{new_lv}' "
                    f"({breach_count} breach, {amber_count} amber)"
                )
                store[bucket][risk_id]["l"]  = new_l
                store[bucket][risk_id]["lv"] = new_lv
                store[bucket][risk_id]["rating_auto_updated"] = (
                    datetime.now(timezone.utc).isoformat()
                )

    if changed:
        print(f"  [RATING RECOMPUTE] {len(changed)} risk rating(s) updated from KRI status:")
        for c in changed:
            print(f"    ↳ {c}")


def _write_to_store(dashboard_kris: dict, agent_ctx: dict = None):
    """
    Write dashboard KRI values and agent context to risk_store.json.
    Agent context is persisted so kri_validator can independently recompute
    and diff it — keeping agent context inside the same validation loop as KRIs.
    Uses file locking to avoid corruption.
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

            if agent_ctx:
                store["agent_context"] = agent_ctx

            # Item 4: Auto-recompute parent risk ratings from aggregated KRI status.
            # Prevents static lv/l values diverging from live KRI results.
            _recompute_ratings(store)

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

    try:
        qoq_deltas = _compute_qoq_deltas()
    except Exception as e:
        qoq_deltas = {"available": False, "reason": str(e)}
        errors.append(f"QoQ delta computation failed: {e}")

    # Merge by domain bucket
    dashboard_kris = {
        "operational_risks": op_kris,
        "financial_risks":   fin_kris,
        "strategic_risks":   str_kris,
        "compliance_risks":  com_kris,
    }

    # Write dashboard KRIs and agent context to store
    _write_to_store(dashboard_kris, agent_ctx)

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

    qoq_available = qoq_deltas.get("available", False)
    qoq_msg = (f"QoQ: {qoq_deltas['summary']['deteriorating_count']} deteriorating, "
               f"{qoq_deltas['summary']['improving_count']} improving"
               if qoq_available else "QoQ: no prior period data")
    print(f"  [KRI DATA LAYER] {len(all_kris)} KRIs computed — "
          f"{breaches} breach(es), {ambers} amber(s)  |  {qoq_msg}")
    if errors:
        for e in errors:
            print(f"  ⚠ {e}")

    return {
        "dashboard_kris": dashboard_kris,
        "agent_context":  agent_ctx,
        "qoq_deltas":     qoq_deltas,
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
