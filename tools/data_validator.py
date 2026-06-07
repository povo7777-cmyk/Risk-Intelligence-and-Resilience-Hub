"""
tools/data_validator.py — Item 1: Source data integrity checks.

Runs as data_validation_node BEFORE kri_data_layer. Validates:
  1. Required columns present in every CSV
  2. Key numeric fields non-null and parseable
  3. Data freshness: latest date within warning/error windows
  4. Row-count plausibility: sudden drops/spikes detected

Hard-blocks the pipeline on errors; records warnings in state.
"""

from __future__ import annotations
import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Schema registry ────────────────────────────────────────────────────────────
# Each entry: required_cols, numeric_cols, max_age_warning_days, max_age_error_days,
#             min_rows_per_date, max_rows_per_date
CSV_SCHEMAS: dict[str, dict] = {
    "erp_supply_chain.csv": {
        "required_cols": ["date", "supplier_id", "supplier_name", "inventory_weeks",
                          "lead_time_weeks", "our_spend_usd_m"],
        "numeric_cols":  ["inventory_weeks", "lead_time_weeks", "our_spend_usd_m"],
        "max_age_warning_days": 35,
        "max_age_error_days":   60,
        "min_rows_per_date":    4,
        "max_rows_per_date":    30,
    },
    "treasury_positions.csv": {
        "required_cols": ["date", "exposure_id", "currency_pair", "gross_exposure_usd_m",
                          "hedged_amount_usd_m", "hedge_ratio_pct"],
        "numeric_cols":  ["gross_exposure_usd_m", "hedged_amount_usd_m", "hedge_ratio_pct"],
        "max_age_warning_days": 35,
        "max_age_error_days":   60,
        "min_rows_per_date":    4,
        "max_rows_per_date":    20,
    },
    "siem_cyber.csv": {
        "required_cols": ["date", "metric", "value", "unit", "source_system"],
        "numeric_cols":  ["value"],
        "max_age_warning_days": 14,
        "max_age_error_days":   45,
        "min_rows_per_date":    5,
        "max_rows_per_date":    60,
    },
    "qms_quality.csv": {
        "required_cols": ["date", "metric", "value", "unit"],
        "numeric_cols":  ["value"],
        "max_age_warning_days": 35,
        "max_age_error_days":   60,
        "min_rows_per_date":    4,
        "max_rows_per_date":    40,
    },
    "hris_talent.csv": {
        "required_cols": ["date", "metric", "value", "unit"],
        "numeric_cols":  ["value"],
        "max_age_warning_days": 35,
        "max_age_error_days":   60,
        "min_rows_per_date":    4,
        "max_rows_per_date":    60,
    },
    "covenant_tracker.csv": {
        "required_cols": ["date", "covenant_id", "covenant_type", "metric",
                          "current_value", "threshold"],
        "numeric_cols":  ["current_value", "threshold"],
        "max_age_warning_days": 35,
        "max_age_error_days":   60,
        "min_rows_per_date":    4,
        "max_rows_per_date":    20,
    },
    "ar_aging.csv": {
        "required_cols": ["date", "segment", "bad_debt_provision_usd_m",
                          "current_usd_m", "overdue_90d_usd_m"],
        "numeric_cols":  ["bad_debt_provision_usd_m", "current_usd_m", "overdue_90d_usd_m"],
        "max_age_warning_days": 35,
        "max_age_error_days":   60,
        "min_rows_per_date":    2,
        "max_rows_per_date":    20,
    },
    "screening_results.csv": {
        "required_cols": ["date", "metric", "value", "unit"],
        "numeric_cols":  ["value"],
        "max_age_warning_days": 14,
        "max_age_error_days":   45,
        "min_rows_per_date":    2,
        "max_rows_per_date":    30,
    },
    "market_intelligence.csv": {
        "required_cols": ["date", "signal_type", "severity", "risk_id"],
        "numeric_cols":  [],
        "max_age_warning_days": 14,
        "max_age_error_days":   45,
        "min_rows_per_date":    1,
        "max_rows_per_date":    100,
    },
    "compliance_metrics.csv": {
        "required_cols": ["date", "metric", "value", "unit"],
        "numeric_cols":  ["value"],
        "max_age_warning_days": 35,
        "max_age_error_days":   60,
        "min_rows_per_date":    2,
        "max_rows_per_date":    30,
    },
}


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _age_days(date_str: str) -> int | None:
    try:
        d = date.fromisoformat(date_str)
        return (_today() - d).days
    except (ValueError, TypeError):
        return None


def validate_csv(filename: str, schema: dict) -> list[dict]:
    issues: list[dict] = []
    path = DATA_DIR / filename

    if not path.exists():
        issues.append({
            "severity": "ERROR",
            "code": "FILE_MISSING",
            "file": filename,
            "detail": f"Data file {filename} not found in {DATA_DIR}",
        })
        return issues

    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        issues.append({
            "severity": "ERROR",
            "code": "FILE_UNREADABLE",
            "file": filename,
            "detail": f"Cannot read {filename}: {e}",
        })
        return issues

    if not rows:
        issues.append({
            "severity": "ERROR",
            "code": "EMPTY_FILE",
            "file": filename,
            "detail": f"{filename} has no data rows",
        })
        return issues

    actual_cols = set(rows[0].keys())

    # ── Check 1: Required columns ──────────────────────────────────────────────
    for col in schema.get("required_cols", []):
        if col not in actual_cols:
            issues.append({
                "severity": "ERROR",
                "code": "MISSING_COLUMN",
                "file": filename,
                "detail": f"Required column '{col}' missing from {filename}. "
                          f"Found: {sorted(actual_cols)}",
            })

    # ── Check 2: Numeric fields non-null ──────────────────────────────────────
    for col in schema.get("numeric_cols", []):
        if col not in actual_cols:
            continue
        for i, row in enumerate(rows, start=2):
            val = row.get(col, "").strip()
            if val == "" or val is None:
                issues.append({
                    "severity": "ERROR",
                    "code": "NULL_NUMERIC",
                    "file": filename,
                    "detail": f"Row {i}: numeric column '{col}' is empty/null",
                })
                break
            try:
                float(val)
            except ValueError:
                issues.append({
                    "severity": "ERROR",
                    "code": "NON_NUMERIC_VALUE",
                    "file": filename,
                    "detail": f"Row {i}: '{col}' = '{val}' is not a valid number",
                })
                break

    # ── Check 3: Freshness ────────────────────────────────────────────────────
    if "date" in actual_cols:
        dates = sorted(
            set(r.get("date", "").strip() for r in rows if r.get("date", "").strip()),
            reverse=True,
        )
        if dates:
            latest = dates[0]
            age = _age_days(latest)
            if age is None:
                issues.append({
                    "severity": "WARNING",
                    "code": "UNPARSEABLE_DATE",
                    "file": filename,
                    "detail": f"Latest date '{latest}' cannot be parsed as ISO date",
                })
            else:
                warn_days  = schema.get("max_age_warning_days", 35)
                error_days = schema.get("max_age_error_days",   60)
                if age > error_days:
                    issues.append({
                        "severity": "ERROR",
                        "code": "DATA_TOO_STALE",
                        "file": filename,
                        "detail": f"Latest data is {age} days old (latest: {latest}). "
                                  f"Error threshold: {error_days} days. "
                                  f"Pipeline blocked — refresh source data before running.",
                    })
                elif age > warn_days:
                    issues.append({
                        "severity": "WARNING",
                        "code": "DATA_STALE",
                        "file": filename,
                        "detail": f"Latest data is {age} days old (latest: {latest}). "
                                  f"Warning threshold: {warn_days} days. "
                                  f"Recommend refreshing source data.",
                    })

        # ── Check 4: Row count plausibility per latest date ────────────────────
        if dates:
            latest_rows = [r for r in rows if r.get("date", "").strip() == dates[0]]
            n = len(latest_rows)
            mn = schema.get("min_rows_per_date", 1)
            mx = schema.get("max_rows_per_date", 1000)
            if n < mn:
                issues.append({
                    "severity": "WARNING",
                    "code": "ROW_COUNT_LOW",
                    "file": filename,
                    "detail": f"Only {n} row(s) for latest date {dates[0]} "
                              f"(expected ≥ {mn}). Possible missing supplier/position data.",
                })
            elif n > mx:
                issues.append({
                    "severity": "WARNING",
                    "code": "ROW_COUNT_HIGH",
                    "file": filename,
                    "detail": f"{n} row(s) for latest date {dates[0]} "
                              f"(expected ≤ {mx}). Possible duplicate ingestion.",
                })

    return issues


def validate_all() -> dict:
    """Run all CSV schema checks. Returns {errors, warnings, issues, passed}."""
    all_issues: list[dict] = []
    for filename, schema in CSV_SCHEMAS.items():
        all_issues.extend(validate_csv(filename, schema))

    errors   = [i for i in all_issues if i["severity"] == "ERROR"]
    warnings = [i for i in all_issues if i["severity"] == "WARNING"]

    return {
        "passed":        len(errors) == 0,
        "error_count":   len(errors),
        "warning_count": len(warnings),
        "issues":        all_issues,
        "errors":        errors,
        "warnings":      warnings,
    }


def print_report(result: dict) -> None:
    icon = "✓" if result["passed"] else "✗"
    print(f"\n[DATA VALIDATION] {icon} "
          f"{result['error_count']} error(s), {result['warning_count']} warning(s)")
    for issue in result["issues"]:
        sev = issue["severity"]
        marker = "✗" if sev == "ERROR" else "⚠"
        print(f"  {marker} [{issue['code']}] {issue['file']}: {issue['detail']}")


if __name__ == "__main__":
    import sys
    result = validate_all()
    print_report(result)
    sys.exit(0 if result["passed"] else 1)
