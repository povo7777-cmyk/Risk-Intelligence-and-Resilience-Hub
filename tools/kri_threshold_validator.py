"""
kri_threshold_validator.py — Structural integrity checks for kri_thresholds.csv.

Checks:
  1. Non-integer amber/breach thresholds on count metrics (amber=0.5 is a footgun)
  2. Logical inversions: amber >= breach for higher_worse; amber <= breach for lower_worse
  3. KRIs present in risk_store.json but absent from kri_thresholds.csv
  4. KRIs present in kri_thresholds.csv but absent from risk_store.json

Usage:
  python tools/kri_threshold_validator.py
  python tools/kri_threshold_validator.py --fix-log           # write issues to issues.json
  python tools/kri_threshold_validator.py --store api/risk_store.json --thresholds data/kri_thresholds.csv
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
THRESHOLDS_CSV = BASE_DIR / "data" / "kri_thresholds.csv"
RISK_STORE_JSON = BASE_DIR / "api" / "risk_store.json"


def load_thresholds(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_store_kri_keys(path: Path) -> dict[str, set[str]]:
    """Return {risk_id: {kri_name, ...}} from risk_store.json."""
    store = json.loads(path.read_text())
    result: dict[str, set[str]] = {}
    sections = [
        ("operational_risks",  "O"),
        ("strategic_risks",    "S"),
        ("financial_risks",    "F"),
        ("compliance_risks",   "C"),
    ]
    for section_key, _ in sections:
        for risk_id, risk_obj in store.get(section_key, {}).items():
            kris = risk_obj.get("kris", {})
            result.setdefault(risk_id, set()).update(kris.keys())
    return result


def check_thresholds(rows: list[dict]) -> list[dict]:
    issues: list[dict] = []

    for row in rows:
        rid = row["risk_id"]
        kri = row["kri_name"]
        unit = row.get("unit", "")
        direction = row["direction"]
        ref = f"{rid}/{kri}"

        try:
            amber = float(row["amber_threshold"])
            breach = float(row["breach_threshold"])
        except ValueError:
            issues.append({
                "severity": "ERROR",
                "code": "NON_NUMERIC_THRESHOLD",
                "ref": ref,
                "detail": f"amber='{row['amber_threshold']}' breach='{row['breach_threshold']}' — not parseable as float"
            })
            continue

        # Check 1: non-integer thresholds on count metrics
        if unit == "count":
            if amber != int(amber):
                issues.append({
                    "severity": "ERROR",
                    "code": "NON_INTEGER_COUNT_THRESHOLD",
                    "ref": ref,
                    "detail": f"amber_threshold={amber} is non-integer on a count metric — will never be matched exactly by integer KRI values (panel finding EM-06)"
                })
            if breach != int(breach):
                issues.append({
                    "severity": "ERROR",
                    "code": "NON_INTEGER_COUNT_THRESHOLD",
                    "ref": ref,
                    "detail": f"breach_threshold={breach} is non-integer on a count metric — will never be matched exactly by integer KRI values (panel finding EM-06)"
                })

        # Check 2: logical threshold inversion
        if direction == "higher_worse":
            if amber > breach:
                issues.append({
                    "severity": "ERROR",
                    "code": "THRESHOLD_INVERSION",
                    "ref": ref,
                    "detail": f"higher_worse: amber ({amber}) > breach ({breach}) — amber must be ≤ breach; KRI will jump from ok straight to breach with no amber zone"
                })
            elif amber == breach:
                issues.append({
                    "severity": "WARNING",
                    "code": "AMBER_EQUALS_BREACH",
                    "ref": ref,
                    "detail": f"higher_worse: amber ({amber}) == breach ({breach}) — zero-width amber zone; confirm this is intentional (zero-tolerance metric)"
                })
        elif direction == "lower_worse":
            if amber < breach:
                issues.append({
                    "severity": "ERROR",
                    "code": "THRESHOLD_INVERSION",
                    "ref": ref,
                    "detail": f"lower_worse: amber ({amber}) < breach ({breach}) — for lower_worse metrics amber must be ≥ breach (amber fires first)"
                })
            elif amber == breach:
                issues.append({
                    "severity": "WARNING",
                    "code": "AMBER_EQUALS_BREACH",
                    "ref": ref,
                    "detail": f"lower_worse: amber ({amber}) == breach ({breach}) — zero-width amber zone; confirm this is intentional"
                })
        else:
            issues.append({
                "severity": "WARNING",
                "code": "UNKNOWN_DIRECTION",
                "ref": ref,
                "detail": f"direction='{direction}' is not 'higher_worse' or 'lower_worse' — cannot validate threshold ordering"
            })

    return issues


def check_coverage(
    rows: list[dict], store_kris: dict[str, set[str]]
) -> list[dict]:
    """Find KRIs in store missing from CSV and vice versa."""
    issues: list[dict] = []
    csv_set: set[tuple[str, str]] = {(r["risk_id"], r["kri_name"]) for r in rows}

    # Store KRIs not in CSV
    for risk_id, kri_names in store_kris.items():
        for kri in kri_names:
            if (risk_id, kri) not in csv_set:
                issues.append({
                    "severity": "WARNING",
                    "code": "MISSING_THRESHOLD_DEFINITION",
                    "ref": f"{risk_id}/{kri}",
                    "detail": "KRI present in risk_store.json but has no row in kri_thresholds.csv — status evaluation will use store-level thresholds only (no central definition)"
                })

    # CSV KRIs not in store
    store_all: set[tuple[str, str]] = {
        (risk_id, kri)
        for risk_id, kris in store_kris.items()
        for kri in kris
    }
    for rid, kri in csv_set:
        if (rid, kri) not in store_all:
            issues.append({
                "severity": "INFO",
                "code": "THRESHOLD_DEFINED_NO_STORE_VALUE",
                "ref": f"{rid}/{kri}",
                "detail": "Threshold defined in kri_thresholds.csv but no matching KRI value found in risk_store.json — may be a future KRI or a naming mismatch"
            })

    return issues


def print_report(issues: list[dict]) -> None:
    if not issues:
        print("✓ All KRI threshold checks passed — no issues found")
        return

    errors   = [i for i in issues if i["severity"] == "ERROR"]
    warnings = [i for i in issues if i["severity"] == "WARNING"]
    infos    = [i for i in issues if i["severity"] == "INFO"]

    print(f"\nKRI Threshold Validation Report")
    print(f"{'='*60}")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Info:     {len(infos)}")
    print(f"{'='*60}\n")

    for sev, label in [("ERROR", "ERRORS"), ("WARNING", "WARNINGS"), ("INFO", "INFO")]:
        group = [i for i in issues if i["severity"] == sev]
        if not group:
            continue
        print(f"── {label} ──")
        for i in group:
            print(f"  [{i['code']}] {i['ref']}")
            print(f"    {i['detail']}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate kri_thresholds.csv structural integrity")
    parser.add_argument("--thresholds", default=str(THRESHOLDS_CSV), help="Path to kri_thresholds.csv")
    parser.add_argument("--store", default=str(RISK_STORE_JSON), help="Path to risk_store.json")
    parser.add_argument("--fix-log", metavar="FILE", help="Write issues to this JSON file")
    args = parser.parse_args()

    thresh_path = Path(args.thresholds)
    store_path  = Path(args.store)

    if not thresh_path.exists():
        print(f"ERROR: kri_thresholds.csv not found at {thresh_path}", file=sys.stderr)
        return 2
    if not store_path.exists():
        print(f"WARNING: risk_store.json not found at {store_path} — coverage checks skipped", file=sys.stderr)

    rows = load_thresholds(thresh_path)
    issues = check_thresholds(rows)

    if store_path.exists():
        store_kris = load_store_kri_keys(store_path)
        issues += check_coverage(rows, store_kris)

    print_report(issues)

    if args.fix_log:
        with open(args.fix_log, "w") as f:
            json.dump(issues, f, indent=2)
        print(f"Issues written to {args.fix_log}")

    errors = [i for i in issues if i["severity"] == "ERROR"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
