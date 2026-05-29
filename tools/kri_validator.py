"""
tools/kri_validator.py
Pure-Python re-computation of KRI values directly from source CSVs.
Compares against agent-written values in risk_store.json and flags
any discrepancy above per-KRI tolerance thresholds.

No LLM calls — deterministic only.
"""

import json
from pathlib import Path

STORE_PATH = Path(__file__).parent.parent / "api" / "risk_store.json"

# Default relative tolerance (5%).  Count KRIs use tolerance=0 (exact match).
DEFAULT_TOLERANCE = 0.05


def _stored(store: dict, bucket: str, risk_id: str, kri_name: str):
    """Return stored KRI value or None if missing."""
    entry = store.get(bucket, {}).get(risk_id, {}).get("kris", {}).get(kri_name, {})
    return entry.get("value")


def _check(results: list, risk_id: str, kri_name: str,
           expected, actual, tolerance=DEFAULT_TOLERANCE):
    if actual is None:
        results.append({
            "risk_id": risk_id, "kri": kri_name,
            "expected": round(float(expected), 4), "actual": None,
            "diff_pct": None, "ok": False,
            "note": "KRI not yet written to store",
        })
        return
    try:
        exp_f, act_f = float(expected), float(actual)
        diff = abs(exp_f - act_f) / abs(exp_f) if exp_f != 0 else abs(act_f)
        ok = diff <= tolerance
        results.append({
            "risk_id": risk_id, "kri": kri_name,
            "expected": round(exp_f, 4), "actual": round(act_f, 4),
            "diff_pct": round(diff * 100, 2), "ok": ok,
        })
    except Exception as e:
        results.append({
            "risk_id": risk_id, "kri": kri_name,
            "expected": expected, "actual": actual,
            "diff_pct": None, "ok": False, "note": str(e),
        })


# ── Operational ───────────────────────────────────────────

def _validate_operational(store: dict) -> list[dict]:
    from tools.data_reader import (
        get_supply_chain_data, get_cyber_data,
        get_quality_data, get_talent_data,
    )
    results = []

    # O-01 — supply chain (4 KRIs: 3 from supply chain data, 1 from cyber data)
    try:
        sc = get_supply_chain_data()
        for key in sc:
            if sc[key] is None:
                sc[key] = 0
        _check(results, "O-01", "single_source_concentration",
               sc["overall_single_source_concentration_pct"],
               _stored(store, "operational_risks", "O-01", "single_source_concentration"))
        _check(results, "O-01", "inventory_cover_weeks",
               sc["min_inventory_weeks"],
               _stored(store, "operational_risks", "O-01", "inventory_cover_weeks"))
        _check(results, "O-01", "supplier_distress_flags",
               float(sc["supplier_distress_flags"]),
               _stored(store, "operational_risks", "O-01", "supplier_distress_flags"),
               tolerance=0)
    except Exception as e:
        results.append({"risk_id": "O-01", "kri": "ALL", "ok": False, "note": str(e)})

    # O-01 supplier_cyber_resilience_assess_pct — sourced from cyber data (SIEM tracks supplier assessments)
    try:
        cy_sc = get_cyber_data()
        _check(results, "O-01", "supplier_cyber_resilience_assess_pct",
               float(cy_sc.get("supplier_cyber_resilience_assess_pct", 0)),
               _stored(store, "operational_risks", "O-01", "supplier_cyber_resilience_assess_pct"))
    except Exception as e:
        results.append({"risk_id": "O-01", "kri": "supplier_cyber_resilience_assess_pct",
                        "ok": False, "note": str(e)})

    # O-02 — cyber
    try:
        cy = get_cyber_data()
        for key in cy:
            if cy[key] is None:
                cy[key] = 0
        _check(results, "O-02", "mttd_days",
               cy["mttd_days"],
               _stored(store, "operational_risks", "O-02", "mttd_days"))
        _check(results, "O-02", "patch_compliance_pct",
               cy["patch_compliance_pct"],
               _stored(store, "operational_risks", "O-02", "patch_compliance_pct"))
        _check(results, "O-02", "critical_vulns_open_gt30d",
               cy["critical_vulns_open_gt30d"],
               _stored(store, "operational_risks", "O-02", "critical_vulns_open_gt30d"),
               tolerance=0)
        _check(results, "O-02", "it_rto_hours",
               cy["it_rto_hours"],
               _stored(store, "operational_risks", "O-02", "it_rto_hours"))
    except Exception as e:
        results.append({"risk_id": "O-02", "kri": "ALL", "ok": False, "note": str(e)})

    # O-03 — quality
    try:
        qu = get_quality_data()
        for key in qu:
            if qu[key] is None:
                qu[key] = 0
        _check(results, "O-03", "field_failure_rate_pct",
               qu["max_field_failure_rate_pct"],
               _stored(store, "operational_risks", "O-03", "field_failure_rate_pct"))
        _check(results, "O-03", "recall_readiness_score_pct",
               qu["recall_readiness_score_pct"],
               _stored(store, "operational_risks", "O-03", "recall_readiness_score_pct"))
        _check(results, "O-03", "safety_incidents_ytd",
               qu["safety_incidents_ytd"],
               _stored(store, "operational_risks", "O-03", "safety_incidents_ytd"),
               tolerance=0)
        _check(results, "O-03", "supplier_quality_rejection_rate_pct",
               qu["supplier_quality_rejection_rate_pct"],
               _stored(store, "operational_risks", "O-03", "supplier_quality_rejection_rate_pct"))
    except Exception as e:
        results.append({"risk_id": "O-03", "kri": "ALL", "ok": False, "note": str(e)})

    # O-04 — talent
    try:
        ta = get_talent_data()
        for key in ta:
            if ta[key] is None:
                ta[key] = 0
        _check(results, "O-04", "tech_attrition_rate_pct",
               ta["tech_attrition_engineering_pct"],
               _stored(store, "operational_risks", "O-04", "tech_attrition_rate_pct"))
        _check(results, "O-04", "critical_open_roles_gt60d",
               ta["critical_open_roles_gt60d"],
               _stored(store, "operational_risks", "O-04", "critical_open_roles_gt60d"),
               tolerance=0)
        _check(results, "O-04", "svp_succession_coverage_pct",
               ta["svp_succession_coverage_pct"],
               _stored(store, "operational_risks", "O-04", "svp_succession_coverage_pct"))
    except Exception as e:
        results.append({"risk_id": "O-04", "kri": "ALL", "ok": False, "note": str(e)})

    return results


# ── Financial ────────────────────────────────────────────

def _validate_financial(store: dict) -> list[dict]:
    from tools.data_reader import read_csv, read_csv_latest
    results = []

    # F-01 — FX / treasury (latest period only — CSV contains multi-period data)
    try:
        rows = read_csv_latest("treasury_positions.csv")
        total_gross = sum(float(r.get("gross_exposure_usd_m", 0)) for r in rows)
        total_hedged = sum(float(r.get("hedged_amount_usd_m", 0)) for r in rows)
        total_pnl = sum(float(r.get("unrealised_pnl_usd_m", 0)) for r in rows)
        unhedged = round(total_gross - total_hedged, 1)
        avg_hedge = round((total_hedged / total_gross * 100) if total_gross > 0 else 0, 1)

        _check(results, "F-01", "unhedged_fx_exposure_usd_m",
               unhedged,
               _stored(store, "financial_risks", "F-01", "unhedged_fx_exposure_usd_m"))
        _check(results, "F-01", "avg_hedge_ratio_pct",
               avg_hedge,
               _stored(store, "financial_risks", "F-01", "avg_hedge_ratio_pct"))
        # unrealised_pnl_usd_m is agent_context only — not a dashboard KRI, not validated here
    except Exception as e:
        results.append({"risk_id": "F-01", "kri": "ALL", "ok": False, "note": str(e)})

    # F-02 — receivables (latest period only)
    try:
        rows = read_csv_latest("ar_aging.csv")
        top_conc = max((float(r.get("top_customer_concentration_pct", 0)) for r in rows), default=0)
        overdue_90 = round(sum(float(r.get("overdue_90d_usd_m", 0)) for r in rows), 1)
        bad_debt = sum(float(r.get("bad_debt_provision_usd_m", 0)) for r in rows)
        total_current = sum(float(r.get("current_usd_m", 0)) for r in rows)
        total_ar = total_current + overdue_90
        bad_debt_pct = round((bad_debt / total_ar * 100) if total_ar > 0 else 0, 2)
        overdue_90_pct = round((overdue_90 / total_ar * 100) if total_ar > 0 else 0, 1)

        _check(results, "F-02", "top_customer_concentration_pct",
               round(top_conc, 1),
               _stored(store, "financial_risks", "F-02", "top_customer_concentration_pct"))
        # overdue_90d_pct — FYI context metric; archived from KRI framework. Skip validation.
        # _check(results, "F-02", "overdue_90d_pct", ...)
        _check(results, "F-02", "bad_debt_provision_pct",
               bad_debt_pct,
               _stored(store, "financial_risks", "F-02", "bad_debt_provision_pct"))
    except Exception as e:
        results.append({"risk_id": "F-02", "kri": "ALL", "ok": False, "note": str(e)})

    # F-03 — covenants (latest period only)
    try:
        rows = read_csv_latest("covenant_tracker.csv")
        cv = {}
        for r in rows:
            metric = r.get("metric", "")
            try:
                val = float(r.get("current_value", 0))
            except Exception:
                val = 0.0
            if "Net_Debt_EBITDA" in metric:
                cv["net_debt_ebitda_ratio"] = val
            elif "Liquidity" in metric:
                cv["liquidity_headroom_usd_b"] = val
            elif "maturity_runway" in metric:
                cv["debt_maturity_runway_months"] = float(int(val))

        for kri_name, expected in cv.items():
            tol = 0 if kri_name == "debt_maturity_runway_months" else DEFAULT_TOLERANCE
            _check(results, "F-03", kri_name, expected,
                   _stored(store, "financial_risks", "F-03", kri_name), tolerance=tol)
    except Exception as e:
        results.append({"risk_id": "F-03", "kri": "ALL", "ok": False, "note": str(e)})

    # F-04 — derived from audit_log.csv (Internal-SOX scope, latest period only)
    try:
        audit_rows = read_csv_latest("audit_log.csv")
        sox_rows = [r for r in audit_rows
                    if any(t in r.get("audit_type", "") for t in ["SOX", "Internal"])]
        expected_findings = float(sum(int(r.get("high_findings", 0)) for r in sox_rows))
        expected_weakness = float(sum(int(r.get("critical_findings", 0)) for r in sox_rows))
        _check(results, "F-04", "audit_findings_open", expected_findings,
               _stored(store, "financial_risks", "F-04", "audit_findings_open"), tolerance=0)
        # material_weakness_count is agent_context only — not a dashboard KRI, not validated here
    except Exception as e:
        results.append({"risk_id": "F-04", "kri": "ALL", "ok": False, "note": str(e)})

    return results


# ── Compliance ───────────────────────────────────────────

def _validate_compliance(store: dict) -> list[dict]:
    from tools.data_reader import read_csv, read_csv_latest
    results = []

    try:
        rows = read_csv_latest("screening_results.csv")
        screening = {}
        for r in rows:
            try:
                screening[r.get("metric", "")] = float(r.get("value", 0))
            except Exception:
                pass

        mapping = [
            ("C-01", "export_screening_coverage_pct",      "export_screening_coverage_pct",      DEFAULT_TOLERANCE),
            ("C-01", "confirmed_sanctions_violations_ytd",  "confirmed_sanctions_violations_ytd",  0),
            # denied_party_matches_pending is agent_context only — not a dashboard KRI, not validated here
        ]
        for risk_id, kri_name, csv_key, tol in mapping:
            if csv_key in screening:
                _check(results, risk_id, kri_name,
                       screening[csv_key],
                       _stored(store, "compliance_risks", risk_id, kri_name),
                       tolerance=tol)
    except Exception as e:
        results.append({"risk_id": "C-01", "kri": "ALL", "ok": False, "note": str(e)})

    # C-02 / C-03 — sourced from compliance_metrics.csv (latest period only)
    try:
        comp_rows = read_csv_latest("compliance_metrics.csv")
        comp_metrics = {r["metric"]: float(r["value"]) for r in comp_rows}

        c02_c03_mapping = [
            ("C-02", "ai_audit_coverage_pct",          "ai_audit_coverage_pct",          DEFAULT_TOLERANCE),
            # gdpr_dsr_resolution_rate_pct — FYI context metric; archived from KRI framework. Skip.
            ("C-03", "third_party_abac_coverage_pct",  "third_party_abac_coverage_pct",  DEFAULT_TOLERANCE),
            # csrd_scope3_disclosure_pct — FYI context metric; archived from KRI framework. Skip.
        ]
        for risk_id, kri_name, csv_key, tol in c02_c03_mapping:
            if csv_key in comp_metrics:
                _check(results, risk_id, kri_name,
                       comp_metrics[csv_key],
                       _stored(store, "compliance_risks", risk_id, kri_name),
                       tolerance=tol)
    except Exception as e:
        results.append({"risk_id": "C-02/C-03", "kri": "ALL", "ok": False, "note": str(e)})

    return results


# ── Strategic ────────────────────────────────────────────

def _validate_strategic(store: dict) -> list[dict]:
    """Validate strategic KRIs by re-computing directly from source CSVs (latest period)."""
    from tools.data_reader import read_csv_latest
    results = []

    try:
        signals = read_csv_latest("market_intelligence.csv")

        # S-01: geopolitical signal count
        s01_count = float(len([s for s in signals if s.get("risk_id") == "S-01"]))
        _check(results, "S-01", "geopolitical_signal_count",
               s01_count,
               _stored(store, "strategic_risks", "S-01", "geopolitical_signal_count"),
               tolerance=0)

        # S-02: competitive signal count
        s02_count = float(len([s for s in signals if s.get("risk_id") == "S-02"]))
        _check(results, "S-02", "competitive_signals",
               s02_count,
               _stored(store, "strategic_risks", "S-02", "competitive_signals"),
               tolerance=0)
    except Exception as e:
        results.append({"risk_id": "S-01/S-02", "kri": "ALL", "ok": False, "note": str(e)})

    # S-01: ODM/EMS concentration in PRC-jurisdiction
    try:
        sc_rows  = read_csv_latest("erp_supply_chain.csv")
        odm_rows = [r for r in sc_rows if r.get("component_category", "").startswith(("Manufacturing", "ODM"))]
        total_odm = sum(float(r.get("our_spend_usd_m", 0)) for r in odm_rows)
        prc_odm   = sum(float(r.get("our_spend_usd_m", 0)) for r in odm_rows if r.get("country") == "China")
        prc_pct   = round(prc_odm / total_odm * 100, 1) if total_odm > 0 else 0.0
        _check(results, "S-01", "odm_ems_prc_concentration_pct",
               prc_pct,
               _stored(store, "strategic_risks", "S-01", "odm_ems_prc_concentration_pct"))
    except Exception as e:
        results.append({"risk_id": "S-01", "kri": "odm_ems_prc_concentration_pct",
                        "ok": False, "note": str(e)})

    # S-03: synergy delivery (Integration-stage deals only)
    try:
        pipeline          = read_csv_latest("ma_pipeline.csv")
        integration_deals = [d for d in pipeline if d.get("stage", "").startswith("Integration")]
        if integration_deals:
            synergy = round(
                sum(float(d["synergy_delivered_pct"]) for d in integration_deals) / len(integration_deals), 1
            )
            _check(results, "S-03", "synergy_delivery_pct",
                   synergy,
                   _stored(store, "strategic_risks", "S-03", "synergy_delivery_pct"))
    except Exception as e:
        results.append({"risk_id": "S-03", "kri": "synergy_delivery_pct",
                        "ok": False, "note": str(e)})

    return results


# ── Public entry point ────────────────────────────────────

def run_kri_validation() -> dict:
    """
    Re-compute all verifiable KRI values from source CSVs and compare
    against agent-written values in risk_store.json.

    Returns a summary dict with discrepancy details.
    """
    store = json.loads(STORE_PATH.read_text())

    all_results = []
    all_results.extend(_validate_strategic(store))
    all_results.extend(_validate_operational(store))
    all_results.extend(_validate_financial(store))
    all_results.extend(_validate_compliance(store))

    discrepancies = [r for r in all_results if not r.get("ok", False)]
    passed = [r for r in all_results if r.get("ok", False)]

    return {
        "total_checked": len(all_results),
        "passed": len(passed),
        "discrepancy_count": len(discrepancies),
        "discrepancy_details": discrepancies,
        "validation_passed": len(discrepancies) == 0,
    }
