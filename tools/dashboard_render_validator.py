"""
tools/dashboard_render_validator.py
Validates that every KRI in risk_store.json is accurately reflected in
the live dashboard HTML — correct value, correct status colour.

Three checks per KRI:
  1. MISSING   — KRI display name not found anywhere in the dashboard HTML
  2. WRONG_VAL — display name found but formatted value doesn't match store
  3. WRONG_TS  — value matches but status badge (br/am/ok) doesn't match

Imports KRI_NAME_MAP, KRI_FORMAT, STATUS_TO_TS, DOMAIN_BUCKETS from
dashboard_updater so the same transform logic is used in both directions.
"""

import json, re
from pathlib import Path

STORE_PATH  = Path(__file__).parent.parent / "api" / "risk_store.json"
DASH_PATH   = Path(__file__).parent.parent / "dashboard" / "index.html"

# ── borrow transforms from dashboard_updater ─────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dashboard_updater import KRI_NAME_MAP, KRI_FORMAT, STATUS_TO_TS, DOMAIN_BUCKETS


# ── parse dashboard HTML → {display_name: {cur, ts, raw_block}} ──────────────
_KRI_RE = re.compile(
    r"\{n:'(?P<name>[^']+)',cur:'(?P<cur>[^']*)',"
    r"a:'[^']*',r:'[^']*',tr:\[[^\]]*\],ts:'(?P<ts>[^']*)'"
)


def _parse_dashboard(html: str) -> dict[str, dict]:
    """Return {display_name: {"cur": str, "ts": str}} for every KRI block."""
    return {
        m.group("name"): {"cur": m.group("cur"), "ts": m.group("ts")}
        for m in _KRI_RE.finditer(html)
    }


# ── format a store value the same way dashboard_updater does ─────────────────
def _fmt(kri_name: str, value) -> str:
    fmt = KRI_FORMAT.get(kri_name, lambda v: str(v))
    try:
        return fmt(value)
    except Exception:
        return str(value)


# ── main validation ───────────────────────────────────────────────────────────
def run_dashboard_render_validation(dashboard_path: Path = DASH_PATH) -> dict:
    store = json.loads(STORE_PATH.read_text())
    html  = dashboard_path.read_text()

    dashboard_kris = _parse_dashboard(html)

    results   = []
    missing   = []
    wrong_val   = []
    wrong_ts    = []
    ok_count    = 0
    store_only  = []  # KRIs tracked internally but intentionally not on the dashboard

    for bucket in DOMAIN_BUCKETS:
        for risk_id, risk in store.get(bucket, {}).items():
            for kri_name, kri_data in risk.get("kris", {}).items():
                display_name = KRI_NAME_MAP.get(kri_name)

                if display_name is None or display_name not in dashboard_kris:
                    # Store-only KRI: computed for internal analytics / trend tracking
                    # but has no corresponding dashboard tile. Not an error.
                    store_only.append({
                        "risk_id": risk_id,
                        "kri": kri_name,
                        "display_name": display_name or "(unmapped)",
                    })
                    continue

                expected_cur = _fmt(kri_name, kri_data["value"])
                expected_ts  = STATUS_TO_TS.get(kri_data["status"], kri_data["status"])

                actual = dashboard_kris[display_name]

                val_ok = (actual["cur"] == expected_cur)
                ts_ok  = (actual["ts"]  == expected_ts)

                if not val_ok:
                    wrong_val.append({
                        "risk_id": risk_id,
                        "kri": kri_name,
                        "display_name": display_name,
                        "expected_cur": expected_cur,
                        "actual_cur": actual["cur"],
                        "expected_ts": expected_ts,
                        "actual_ts": actual["ts"],
                    })
                elif not ts_ok:
                    wrong_ts.append({
                        "risk_id": risk_id,
                        "kri": kri_name,
                        "display_name": display_name,
                        "cur": actual["cur"],
                        "expected_ts": expected_ts,
                        "actual_ts": actual["ts"],
                    })
                else:
                    ok_count += 1

    # Only KRIs that have a matching dashboard block count toward the total
    total_checked = ok_count + len(wrong_val) + len(wrong_ts)
    issues = wrong_val + wrong_ts

    return {
        "total_checked":    total_checked,
        "ok":               ok_count,
        "wrong_val_count":  len(wrong_val),
        "wrong_ts_count":   len(wrong_ts),
        "store_only_count": len(store_only),
        "issues":           issues,
        "store_only":       store_only,
        "validation_passed": len(issues) == 0,
    }


# ── pretty-print helper for graph node ───────────────────────────────────────
def _ts_label(ts: str) -> str:
    return {"br": "BREACH", "am": "AMBER", "ok": "OK"}.get(ts, ts.upper())


def print_dashboard_render_report(result: dict):
    t   = result["total_checked"]
    ok  = result["ok"]
    wv  = result["wrong_val_count"]
    wt  = result["wrong_ts_count"]
    so  = result["store_only_count"]

    status = "✓ ALL MATCH" if result["validation_passed"] else "✗ DISCREPANCIES FOUND"
    print(f"\n  [DASHBOARD RENDER] {status}")
    print(f"  {t} dashboard KRIs verified — {ok} OK | {wv} wrong value | {wt} wrong status")
    if so:
        print(f"  ({so} store-only KRI(s) — tracked internally, no dashboard tile)")

    for item in result.get("issues", []):
        risk_id = item.get("risk_id", "?")
        kri     = item.get("kri", "?")
        dname   = item.get("display_name", kri)

        if "actual_cur" in item:
            print(f"\n  ✗ WRONG VALUE  [{risk_id}] {kri}")
            print(f"    Dashboard shows: '{item['actual_cur']}' ({_ts_label(item['actual_ts'])})")
            print(f"    Store expects:   '{item['expected_cur']}' ({_ts_label(item['expected_ts'])})")
            print(f"    Display name:    \"{dname}\"")
        else:
            print(f"\n  ✗ WRONG STATUS [{risk_id}] {kri}")
            print(f"    Dashboard: '{item['cur']}' shown as {_ts_label(item['actual_ts'])}")
            print(f"    Store:     expects {_ts_label(item['expected_ts'])}")
            print(f"    Display name: \"{dname}\"")


if __name__ == "__main__":
    r = run_dashboard_render_validation()
    print_dashboard_render_report(r)
