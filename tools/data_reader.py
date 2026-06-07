"""
data_reader.py — reads simulated ERP/SIEM/QMS/HRIS CSV extracts
and returns structured dicts for each domain agent.
"""

import csv
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def read_csv(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _dates_in(rows: list[dict]) -> list[str]:
    """Return sorted unique dates from a row list, most recent first."""
    return sorted(set(r.get("date", "") for r in rows if r.get("date")), reverse=True)


def read_csv_latest(filename: str) -> list[dict]:
    """Read only the rows belonging to the most recent date in the CSV."""
    rows = read_csv(filename)
    dates = _dates_in(rows)
    if not dates:
        return rows
    return [r for r in rows if r.get("date") == dates[0]]


def read_csv_prior(filename: str) -> list[dict]:
    """Read only the rows belonging to the second-most-recent date (prior period)."""
    rows = read_csv(filename)
    dates = _dates_in(rows)
    if len(dates) < 2:
        return []
    return [r for r in rows if r.get("date") == dates[1]]


def latest_date(filename: str) -> str | None:
    """Return the most recent date string in a CSV, or None."""
    rows = read_csv(filename)
    dates = _dates_in(rows)
    return dates[0] if dates else None


def prior_date(filename: str) -> str | None:
    """Return the second-most-recent date string in a CSV, or None."""
    rows = read_csv(filename)
    dates = _dates_in(rows)
    return dates[1] if len(dates) >= 2 else None


def get_supply_chain_data() -> dict:
    """
    Reads ERP supply chain extract (current period only).
    Returns concentration metrics, inventory, supplier health.
    """
    rows = read_csv_latest("erp_supply_chain.csv")
    total_spend = sum(float(r["our_spend_usd_m"]) for r in rows)
    single_source_spend = sum(
        float(r["our_spend_usd_m"]) for r in rows if r["single_source"] == "true"
    )
    concentration_pct = round(single_source_spend / total_spend * 100, 1)

    # Per-category concentration
    category_concentration = {}
    categories: dict[str, dict] = {}
    for r in rows:
        cat = r["component_category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "our_spend": 0, "single_source": False}
        categories[cat]["our_spend"] += float(r["our_spend_usd_m"])
        categories[cat]["total"] += float(r["total_category_spend_usd_m"])
        if r["single_source"] == "true":
            categories[cat]["single_source"] = True
    for cat, vals in categories.items():
        pct = round(vals["our_spend"] / vals["total"] * 100, 1)
        category_concentration[cat] = {
            "concentration_pct": pct,
            "single_source": vals["single_source"],
        }

    # Inventory and distress
    inventory_weeks = [float(r["inventory_weeks"]) for r in rows]
    min_inventory = min(inventory_weeks)
    distress_flags = sum(1 for r in rows if float(r["financial_health_score"]) < 65)

    # Geographic concentration
    countries = {}
    for r in rows:
        c = r["country"]
        countries[c] = countries.get(c, 0) + float(r["our_spend_usd_m"])
    geo_concentration = {
        k: round(v / total_spend * 100, 1) for k, v in countries.items()
    }

    return {
        "overall_single_source_concentration_pct": concentration_pct,
        "single_source_spend_usd_m": round(single_source_spend, 0),
        "total_spend_usd_m": round(total_spend, 0),
        "category_concentration": category_concentration,
        "min_inventory_weeks": min_inventory,
        "avg_inventory_weeks": round(sum(inventory_weeks) / len(inventory_weeks), 1),
        "supplier_distress_flags": distress_flags,
        "geographic_concentration_pct": geo_concentration,
        "suppliers_assessed": len(rows),
        "raw_rows": rows,
    }


def get_cyber_data() -> dict:
    """
    Reads SIEM/ITSM extract (current period only).
    Returns all cyber KRI values as a flat dict.
    """
    rows = read_csv_latest("siem_cyber.csv")
    metrics = {r["metric"]: float(r["value"]) for r in rows}
    return {
        "mttd_days": metrics.get("mean_time_to_detect", None),
        "mttr_days": metrics.get("mean_time_to_respond", None),
        "patch_compliance_pct": metrics.get("patch_compliance_rate", None),
        "critical_vulns_open_gt30d": metrics.get("critical_vulnerabilities_open_gt30d", None),
        "high_vulns_open_gt30d": metrics.get("high_vulnerabilities_open_gt30d", None),
        "it_rto_hours": metrics.get("it_rto_oms", metrics.get("it_rto_trading_platform", None)),
        "privileged_access_unreviewed_days": metrics.get("privileged_access_unreviewed_days", None),
        "security_incidents_mtd": metrics.get("security_incidents_mtd", None),
        "mfa_coverage_pct": metrics.get("mfa_coverage", None),
        "third_party_vendor_assessed_pct": metrics.get("third_party_vendor_assessed", None),
        "supplier_cyber_resilience_assess_pct": metrics.get("supplier_cyber_resilience_assess_pct", None),
        "raw_metrics": metrics,
    }


def get_quality_data() -> dict:
    """
    Reads QMS extract (current period only).
    Returns product quality KRI values.
    """
    rows = read_csv_latest("qms_quality.csv")
    # Get the worst-case field failure rate across SKU categories
    failure_rates = [
        float(r["value"])
        for r in rows
        if r["metric"] == "field_failure_rate"
    ]
    metrics = {r["metric"]: float(r["value"]) for r in rows if r["sku_category"] == "All"}
    laptop_failure = next(
        (float(r["value"]) for r in rows
         if r["metric"] == "field_failure_rate" and r["sku_category"] == "Laptops"), None
    )

    rejection_rate = next(
        (float(r["value"]) for r in rows
         if r["metric"] == "supplier_quality_rejection_rate" and r["sku_category"] == "All"), None
    )

    return {
        "max_field_failure_rate_pct": max(failure_rates) if failure_rates else None,
        "laptop_field_failure_rate_pct": laptop_failure,
        "recall_readiness_score_pct": metrics.get("recall_readiness_score", None),
        "safety_incidents_ytd": metrics.get("safety_incidents_ytd", None),
        "near_miss_events_ytd": metrics.get("near_miss_events_ytd", None),
        "recall_simulation_last_run_months_ago": metrics.get("recall_simulation_last_run_months_ago", None),
        "warranty_claims_mtd": metrics.get("warranty_claims_mtd", None),
        "supplier_quality_rejection_rate_pct": rejection_rate,
        "raw_metrics": {r["metric"]: float(r["value"]) for r in rows},
    }


def get_talent_data() -> dict:
    """
    Reads HRIS extract (current period only).
    Returns talent KRI values.
    """
    rows = read_csv_latest("hris_talent.csv")
    # Engineering attrition is the highest-risk segment
    eng_attrition = next(
        (float(r["value"]) for r in rows
         if r["metric"] == "tech_role_attrition_rate_annualised"
         and r["department"] == "Engineering"), None
    )
    metrics_all = {
        r["metric"]: float(r["value"])
        for r in rows if r["department"] == "All"
    }
    metrics_eng = {
        r["metric"]: float(r["value"])
        for r in rows if r["department"] == "Engineering"
    }

    return {
        "tech_attrition_engineering_pct": eng_attrition,
        "critical_open_roles_gt60d": metrics_all.get("critical_open_roles_gt60d", None),
        "svp_succession_coverage_pct": metrics_all.get("svp_succession_plan_coverage", None),
        "flight_risk_flagged": metrics_eng.get("flight_risk_employees_flagged", None),
        "compensation_gap_pct": metrics_eng.get("compensation_gap_vs_market", None),
        "avg_time_to_fill_days": metrics_eng.get("avg_time_to_fill_critical_role", None),
        "engagement_score_pct": metrics_all.get("employee_engagement_score", None),
        "counter_offer_acceptance_pct": metrics_eng.get("counter_offer_acceptance_rate", None),
        "raw_metrics": {r["metric"]: float(r["value"]) for r in rows},
        "raw_rows": rows,
    }
