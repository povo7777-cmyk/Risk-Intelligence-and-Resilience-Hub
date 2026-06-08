"""
tools/dashboard_updater.py
Patches the Risk Intelligence and Resilience Hub index.html with
updated KRI values and approved executive recommendation text.
Includes idempotency check and sparkline update.

Systemic consistency guarantees applied on every run:
  - KRI tile thresholds (a: / r:) are synced from kri_thresholds.csv
    so the CSV is always the single authoritative threshold source.
  - Model benchmark figures are verified via consistency_checker after
    all patches are applied, with issues reported before GitHub push.
"""

import csv, json, re, shutil
from datetime import datetime, timezone
from pathlib import Path

THRESHOLDS_PATH   = Path(__file__).parent.parent / "data" / "kri_thresholds.csv"
BENCHMARKS_PATH   = Path(__file__).parent.parent / "data" / "model_benchmarks.json"

STORE_PATH = Path(__file__).parent.parent / "api" / "risk_store.json"
BACKUP_DIR = Path(__file__).parent.parent / "dashboard" / "backups"

STATUS_TO_TS = {"ok": "ok", "amber": "am", "breach": "br"}

DOMAIN_BUCKETS = [
    "operational_risks",
    "financial_risks",
    "strategic_risks",
    "compliance_risks",
]

KRI_NAME_MAP = {
    # Operational — supply chain (O-01)
    "single_source_concentration":        "Single-source concentration",
    "inventory_cover_weeks":              "Inventory cover (weeks)",
    "supplier_distress_flags":            "Supplier financial distress flags",
    "supplier_cyber_resilience_assess_pct": "Single-source supplier cyber resilience assessment coverage",
    "geo_concentration_pct":               "Taiwan+China supply chain concentration (%)",
    # Operational — cyber (O-02)
    "mttd_days":                          "[Step 1 — Detection] Mean time to detect — MTTD (hours)",
    "mttr_days":                          "[Step 2 — Response] Mean time to respond — MTTR (days)",
    "patch_compliance_pct":               "[Enabler — all chain stages] Patch compliance rate",
    "critical_vulns_open_gt30d":          "Critical vulnerabilities open >30 days",
    "it_rto_hours":                       "[Step 3 — Recovery] RTO — order management & fulfilment systems (hours)",
    # Operational — quality (O-03)
    "field_failure_rate_pct":             "Field failure rate",
    "recall_readiness_score_pct":         "Recall readiness score",
    "supplier_quality_rejection_rate_pct": "Supplier quality rejection rate (%)",
    "safety_incidents_ytd":               "Confirmed product safety incidents YTD",
    # Operational — talent (O-04)
    "tech_attrition_rate_pct":            "Tech role attrition rate",
    "critical_open_roles_gt60d":          "Critical open roles >60 days",
    "svp_succession_coverage_pct":        "SVP+ succession coverage (%)",
    # Financial — FX & treasury (F-01)
    "unhedged_fx_exposure_usd_m":         "Unhedged FX exposure (USD M)",
    "avg_hedge_ratio_pct":                "FX hedge ratio (% of exposure hedged)",
    # F-01: unrealised_pnl_usd_m — store-only, no dashboard tile
    # Financial — receivables (F-02)
    "top_customer_concentration_pct":     "Single customer concentration (ISG segment)",
    "overdue_90d_pct":                    "AR overdue >90 days (% of total AR)",
    "bad_debt_provision_pct":             "Bad debt provision % of receivables",
    # Financial — covenants (F-03)
    "net_debt_ebitda_ratio":              "Net-debt/EBITDA ratio",
    "liquidity_headroom_usd_b":           "Available liquidity headroom (USD)",
    "debt_maturity_runway_months":        "Nearest debt maturity runway (months)",
    # Financial — controls (F-04)
    "audit_findings_open":                "Open SOX significant deficiencies",
    # F-04: material_weakness_count — store-only, no dashboard tile
    # Strategic — S-01
    "geopolitical_signal_count":          "Geopolitical escalation signals (YTD)",
    "odm_ems_prc_concentration_pct":      "ODM/EMS concentration in PRC-jurisdiction (%)",
    # Strategic — S-02, S-03
    "competitive_signals":                "Competitive threat signals (active)",
    "synergy_delivery_pct":               "Synergy delivery vs target",
    # Compliance — export & sanctions (C-01)
    "export_screening_coverage_pct":      "Export screening coverage rate",
    "confirmed_sanctions_violations_ytd": "Confirmed sanctions violations YTD",
    # C-01: denied_party_matches_pending — store-only, no dashboard tile
    # Compliance — data & AI (C-02)
    "ai_audit_coverage_pct":             "AI Act audit coverage (%)",
    # gdpr_dsr_resolution_rate_pct has no dashboard tile in the chat HTML
    # C-02: formal_investigation_open — store-only, no dashboard tile
    # Compliance — anti-bribery & ESG (C-03)
    "third_party_abac_coverage_pct":      "Third-party ABAC coverage (%)",
    # csrd_scope3_disclosure_pct has no dashboard tile in the chat HTML
    # C-03: whistleblower_high_findings — store-only, no dashboard tile
}

KRI_FORMAT = {
    # Operational
    "single_source_concentration":        lambda v: f"{v}%",
    "inventory_cover_weeks":              lambda v: f"{v}wk",
    "supplier_distress_flags":            lambda v: str(int(v)),
    "supplier_cyber_resilience_assess_pct": lambda v: f"{v}%",
    "geo_concentration_pct":              lambda v: f"{v}%",
    "mttd_days":                          lambda v: f"{int(v * 24)}h",   # store in days, HTML shows hours
    "mttr_days":                          lambda v: f"{int(v)} days",
    "patch_compliance_pct":               lambda v: f"{v}%",
    "critical_vulns_open_gt30d":          lambda v: str(int(v)),
    "it_rto_hours":                       lambda v: f"{v}h",
    "field_failure_rate_pct":             lambda v: f"{v}%",
    "recall_readiness_score_pct":         lambda v: f"{int(v)}%",
    "supplier_quality_rejection_rate_pct": lambda v: f"{v}%",
    "safety_incidents_ytd":               lambda v: str(int(v)),
    "tech_attrition_rate_pct":            lambda v: f"{v}%",
    "critical_open_roles_gt60d":          lambda v: str(int(v)),
    "svp_succession_coverage_pct":        lambda v: f"{int(v)}%",
    # Financial
    "unhedged_fx_exposure_usd_m":         lambda v: f"USD {int(v)}M",
    "avg_hedge_ratio_pct":                lambda v: f"{int(v)}%",
    "top_customer_concentration_pct":     lambda v: f"{v}%",
    "overdue_90d_pct":                    lambda v: f"{v}%",
    "bad_debt_provision_pct":             lambda v: f"{v}%",
    "net_debt_ebitda_ratio":              lambda v: f"{v}x",
    "liquidity_headroom_usd_b":           lambda v: f"USD {v}B",
    "debt_maturity_runway_months":        lambda v: f"{int(v)} months",
    "audit_findings_open":                lambda v: str(int(v)),
    # Strategic
    "geopolitical_signal_count":          lambda v: str(int(v)),
    "competitive_signals":                lambda v: str(int(v)),
    "synergy_delivery_pct":               lambda v: f"{int(v)}%",
    # Compliance
    "export_screening_coverage_pct":      lambda v: f"{v}%",
    "confirmed_sanctions_violations_ytd": lambda v: str(int(v)),
    "ai_audit_coverage_pct":             lambda v: f"{v}%",
    "gdpr_dsr_resolution_rate_pct":      lambda v: f"{v}%",
    "third_party_abac_coverage_pct":      lambda v: f"{v}%",
    "csrd_scope3_disclosure_pct":         lambda v: f"{v}%",
}

# ── KRI_DATA chip key → CSV kri_name ─────────────────────────────────────────
# These are the JS dict keys in the KRI_DATA block (for status chips/tiles).
# Only entries present in kri_thresholds.csv are listed here.
KRI_DATA_KEY_MAP = {
    "o01_single_source":      "single_source_concentration",
    "o01_inventory":          "inventory_cover_weeks",
    "o01_distress":           "supplier_distress_flags",
    "o01_cyber_assess":       "supplier_cyber_resilience_assess_pct",
    "o01_geo_conc":           "geo_concentration_pct",
    "o02_mttd":               "mttd_days",
    "o02_mttr":               "mttr_days",
    "o02_patch":              "patch_compliance_pct",
    "o02_rto":                "it_rto_hours",
    "o02_vulns":              "critical_vulns_open_gt30d",
    "o03_field_failure":      "field_failure_rate_pct",
    "o03_recall":             "recall_readiness_score_pct",
    "o03_supp_quality":       "supplier_quality_rejection_rate_pct",
    "o04_open_roles":         "critical_open_roles_gt60d",
    "o04_succession":         "svp_succession_coverage_pct",
    "s01_geo_signals":        "geopolitical_signal_count",
    "s02_comp_signals":       "competitive_signals",
    "s03_synergy":            "synergy_delivery_pct",
    "f01_unhedged_fx":        "unhedged_fx_exposure_usd_m",
    "f01_hedge_ratio":        "avg_hedge_ratio_pct",
    "f02_cust_concentration": "top_customer_concentration_pct",
    "f02_bad_debt":           "bad_debt_provision_pct",
    "f03_nd_ebitda":          "net_debt_ebitda_ratio",
    "f03_liquidity":          "liquidity_headroom_usd_b",
    "f03_maturity":           "debt_maturity_runway_months",
    "f04_journal":            "audit_findings_open",
    "c01_screening":          "export_screening_coverage_pct",
    "c02_ai_act":             "ai_audit_coverage_pct",
    "c03_abac":               "third_party_abac_coverage_pct",
    "c03_sb253":              "csrd_scope3_disclosure_pct",
}

# Threshold display formatters — same units as KRI_FORMAT so tile display is consistent
THRESHOLD_FORMAT = {
    "single_source_concentration":        lambda v: f"{v:g}%",
    "inventory_cover_weeks":              lambda v: f"{int(v)}wk",
    "supplier_distress_flags":            lambda v: str(int(v)),
    "supplier_cyber_resilience_assess_pct": lambda v: f"{v:g}%",
    "geo_concentration_pct":              lambda v: f"{v:g}%",
    "mttd_days":                          lambda v: f"{int(v * 24)}h",
    "mttr_days":                          lambda v: f"{int(v)} days",
    "patch_compliance_pct":               lambda v: f"{v:g}%",
    "critical_vulns_open_gt30d":          lambda v: str(int(v)),
    "it_rto_hours":                       lambda v: f"{v:g}h",
    "field_failure_rate_pct":             lambda v: f"{v}%",
    "recall_readiness_score_pct":         lambda v: f"{v:g}%",
    "supplier_quality_rejection_rate_pct": lambda v: f"{v}%",
    "safety_incidents_ytd":               lambda v: str(int(v)),
    "tech_attrition_rate_pct":            lambda v: f"{v:g}%",
    "critical_open_roles_gt60d":          lambda v: str(int(v)),
    "svp_succession_coverage_pct":        lambda v: f"{v:g}%",
    "unhedged_fx_exposure_usd_m":         lambda v: f"USD {int(v)}M",
    "avg_hedge_ratio_pct":                lambda v: f"{v:g}%",
    "top_customer_concentration_pct":     lambda v: f"{v:g}%",
    "bad_debt_provision_pct":             lambda v: f"{v}%",
    "net_debt_ebitda_ratio":              lambda v: f"{v}x",
    "liquidity_headroom_usd_b":           lambda v: f"USD {v}B",
    "debt_maturity_runway_months":        lambda v: f"{int(v)} months",
    "audit_findings_open":                lambda v: str(int(v)),
    "geopolitical_signal_count":          lambda v: str(int(v)),
    "competitive_signals":                lambda v: str(int(v)),
    "synergy_delivery_pct":               lambda v: f"{v:g}%",
    "export_screening_coverage_pct":      lambda v: f"{v:g}%",
    "confirmed_sanctions_violations_ytd": lambda v: str(int(v)),
    "ai_audit_coverage_pct":              lambda v: f"{v:g}%",
    "gdpr_dsr_resolution_rate_pct":       lambda v: f"{v:g}%",
    "third_party_abac_coverage_pct":      lambda v: f"{v:g}%",
    "csrd_scope3_disclosure_pct":         lambda v: f"{v:g}%",
}


def _load_thresholds() -> dict:
    """Load kri_thresholds.csv → {kri_name: {amber, breach, direction}}."""
    if not THRESHOLDS_PATH.exists():
        return {}
    out = {}
    with open(THRESHOLDS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            out[row["kri_name"]] = {
                "amber":     float(row["amber_threshold"]),
                "breach":    float(row["breach_threshold"]),
                "direction": row["direction"],
            }
    return out


def _sync_kri_thresholds_to_html(html: str, thresholds: dict) -> tuple[str, list[str]]:
    """
    Update every KRI threshold display in the HTML to match kri_thresholds.csv.
    Patches both:
      (a) KRI_DATA chip entries:  'js_key': {cur:'...',a:'AMBER',r:'BREACH',ts:'...',risk:'...'}
      (b) Risk register tile entries: {n:'Name',cur:'...',a:'AMBER',r:'BREACH',tr:[...],ts:'...'}

    Called at the end of run_dashboard_update() so the CSV is always authoritative.
    Returns (updated_html, list_of_change_descriptions).
    """
    changes = []

    # ── Part A: KRI_DATA chip entries ─────────────────────────────────────────
    for js_key, kri_name in KRI_DATA_KEY_MAP.items():
        if kri_name not in thresholds:
            continue
        t = thresholds[kri_name]
        fmt = THRESHOLD_FORMAT.get(kri_name)
        if not fmt:
            continue
        amber_str  = fmt(t["amber"])
        breach_str = fmt(t["breach"])

        # Match: 'js_key': {cur:'...',a:'OLD_A',r:'OLD_R',ts:'...',risk:'...'}
        pat = re.compile(
            r"('" + re.escape(js_key) + r"':\s*\{cur:'[^']*',a:')([^']*)(',r:')([^']*)(',ts:'[^']*',risk:'[^']*'\})"
        )
        m = pat.search(html)
        if not m:
            continue
        old_a, old_r = m.group(2), m.group(4)
        if old_a == amber_str and old_r == breach_str:
            continue
        new_frag = m.group(1) + amber_str + m.group(3) + breach_str + m.group(5)
        html = html[:m.start()] + new_frag + html[m.end():]
        changes.append(f"chip:{js_key} a:{old_a}→{amber_str} r:{old_r}→{breach_str}")

    # ── Part B: Risk register tile entries (n: format) ────────────────────────
    for kri_name, js_display_name in KRI_NAME_MAP.items():
        if kri_name not in thresholds:
            continue
        t = thresholds[kri_name]
        fmt = THRESHOLD_FORMAT.get(kri_name)
        if not fmt:
            continue
        amber_str  = fmt(t["amber"])
        breach_str = fmt(t["breach"])

        # Match: {n:'Display Name',cur:'...',a:'OLD_A',r:'OLD_R',tr:[...],ts:'...'}
        pat = re.compile(
            r"(\{n:'" + re.escape(js_display_name) + r"',cur:'[^']*',a:')([^']*)(',r:')([^']*)(',tr:\[[^\]]*\],ts:'[^']*')",
            re.DOTALL
        )
        m = pat.search(html)
        if not m:
            continue
        old_a, old_r = m.group(2), m.group(4)
        if old_a == amber_str and old_r == breach_str:
            continue
        new_frag = m.group(1) + amber_str + m.group(3) + breach_str + m.group(5)
        html = html[:m.start()] + new_frag + html[m.end():]
        changes.append(f"tile:{kri_name} a:{old_a}→{amber_str} r:{old_r}→{breach_str}")

    return html, changes


def _get_bench_value(benchmarks: dict, key_path: tuple):
    obj = benchmarks
    for k in key_path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return float(obj) if obj is not None else None


def _sync_model_benchmarks_to_html(html: str) -> tuple[str, list[str]]:
    """
    Replace stale model benchmark figures in HTML with canonical values from
    data/model_benchmarks.json.  Uses the same regex patterns as consistency_checker
    so every HIGH benchmark issue is auto-corrected before the checker runs.

    Returns (updated_html, list_of_change_descriptions).
    """
    if not BENCHMARKS_PATH.exists():
        return html, []
    try:
        benchmarks = json.loads(BENCHMARKS_PATH.read_text())
    except Exception:
        return html, []

    try:
        from tools.consistency_checker import BENCHMARK_CHECKS
    except ImportError:
        return html, []

    all_changes: list[str] = []

    for chk in BENCHMARK_CHECKS:
        canonical = _get_bench_value(benchmarks, chk["bench_key"])
        if canonical is None:
            continue
        tol     = chk.get("tolerance", 5.0)
        is_raw  = chk.get("is_raw", False)
        pat     = re.compile(chk["pattern"], re.IGNORECASE | re.DOTALL)

        parts: list[str] = []
        last_end = 0
        changed_this = False

        for m in pat.finditer(html):
            old_str = m.group(1)
            old_val = float(old_str.replace(",", ""))
            pct_diff = (abs(old_val - canonical) / canonical * 100
                        if canonical != 0 else abs(old_val))

            if pct_diff <= tol:
                # Within tolerance — keep as-is, continue scanning
                parts.append(html[last_end:m.end()])
                last_end = m.end()
                continue

            # Format canonical value to same style as original
            new_str = str(int(canonical)) if is_raw else f"{int(canonical):,}"

            # Rebuild the full match with only the captured group swapped
            full_match = m.group(0)
            replaced   = full_match.replace(old_str, new_str, 1)
            parts.append(html[last_end:m.start()])
            parts.append(replaced)
            last_end = m.end()
            changed_this = True
            all_changes.append(
                f"{chk['label']}: {int(old_val):,}→{int(canonical):,}"
            )

        parts.append(html[last_end:])
        if changed_this:
            html = "".join(parts)

    return html, all_changes


EXEC_REC_IDS = {
    "bcm":          "ec-bcm",
    "ebitda":       "ec-mc",
    "fx":           "ec-hg",
    "supply_chain": "ec-op",
}


def _trim_to_sentence(text: str, limit: int) -> str:
    """Return text trimmed to complete sentences within limit chars.
    Falls back to the last word boundary if no sentence end is found."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    # Find the last sentence-ending punctuation within the window
    last_end = max(window.rfind('. '), window.rfind('! '), window.rfind('? '))
    if last_end > limit // 2:          # only use it if it's in the second half
        return window[:last_end + 1]   # include the punctuation, drop the trailing space
    # Fall back to last word boundary
    last_space = window.rfind(' ')
    if last_space > 0:
        return window[:last_space] + '…'
    return window + '…'


def _check_js_balance(html: str) -> tuple[bool, str]:
    """Return (ok, message) for JS brace/bracket balance in the main script block."""
    scripts = list(re.finditer(r'<script[^>]*>(.*?)</script>', html, re.DOTALL))
    if len(scripts) < 2:
        return True, "no script block"
    js = scripts[1].group(1)
    diff_b  = js.count('{') - js.count('}')
    diff_sq = js.count('[') - js.count(']')
    if diff_b == 0 and diff_sq == 0:
        return True, "ok"
    return False, f"JS imbalanced: {{}} diff={diff_b}, [] diff={diff_sq}"


def backup_dashboard(html_path):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"index_{ts}.html"
    shutil.copy2(html_path, backup_path)
    # Warn immediately if the file being backed up is already structurally broken
    ok, msg = _check_js_balance(html_path.read_text())
    if not ok:
        print(f"  ⚠ STRUCTURAL WARNING at backup: {msg} — file may cause page freeze")
    return backup_path


def _update_kri(html, kri_name, new_value, new_status):
    js_name = KRI_NAME_MAP.get(kri_name)
    if not js_name:
        return html, False
    fmt = KRI_FORMAT.get(kri_name, lambda v: str(v))
    new_cur = fmt(new_value)
    new_ts = STATUS_TO_TS.get(new_status, "ok")
    pattern = re.compile(
        r"(\{n:'" + re.escape(js_name) + r"',cur:')(.*?)(',a:'[^']*',r:'[^']*',tr:\[[^\]]*\],ts:')(.*?)(')",
        re.DOTALL
    )
    match = pattern.search(html)
    if not match:
        return html, False
    if match.group(2) == new_cur and match.group(4) == new_ts:
        return html, False
    new_frag = match.group(1) + new_cur + match.group(3) + new_ts + match.group(5)
    return html[:match.start()] + new_frag + html[match.end():], True


def _update_sparkline(html, kri_name, new_value):
    js_name = KRI_NAME_MAP.get(kri_name)
    if not js_name:
        return html
    pattern = re.compile(
        r"(n:'" + re.escape(js_name) + r"'.*?tr:\[)([^\]]+)(\])",
        re.DOTALL
    )
    match = pattern.search(html)
    if not match:
        return html
    old_values = [float(x.strip()) for x in match.group(2).split(",")]
    new_values = old_values[1:] + [round(new_value, 3)]
    new_array = ",".join(str(v) for v in new_values)
    return html[:match.start(1)] + match.group(1) + new_array + match.group(3) + html[match.end(3):]


def run_dashboard_update(dashboard_path):
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Dashboard not found: {dashboard_path}")
    store = json.loads(STORE_PATH.read_text())
    html = dashboard_path.read_text()
    backup = backup_dashboard(dashboard_path)
    changes = []
    total_updates = 0
    unmapped = []
    fyi_skipped = []

    for bucket in DOMAIN_BUCKETS:
        for risk_id, risk in store.get(bucket, {}).items():
            kri_changes = 0
            for kri_name, kri_data in risk.get("kris", {}).items():
                # FYI context metrics — archived, not part of the KRI framework.
                # No dashboard tile; skip silently (no warning).
                if kri_data.get("status") == "fyi":
                    fyi_skipped.append(f"{risk_id}.{kri_name}")
                    continue
                if kri_name not in KRI_NAME_MAP:
                    unmapped.append(f"{risk_id}.{kri_name}")
                    continue
                updated_html, changed = _update_kri(
                    html, kri_name, kri_data["value"], kri_data["status"]
                )
                if changed:
                    html = updated_html
                    html = _update_sparkline(html, kri_name, kri_data["value"])
                    kri_changes += 1
                    total_updates += 1
            if kri_changes > 0:
                changes.append(f"{risk_id}: {kri_changes} KRI(s) updated")

    if fyi_skipped:
        print(f"  ℹ {len(fyi_skipped)} FYI context metric(s) skipped (archived, not KRIs): {fyi_skipped}")
    if unmapped:
        print(f"  ⚠ {len(unmapped)} store KRI(s) not in KRI_NAME_MAP — no tile updated: {unmapped}")

    # ── Threshold sync: CSV → HTML (runs every time regardless of KRI updates) ──
    thresholds = _load_thresholds()
    threshold_changes = []
    if thresholds:
        html, threshold_changes = _sync_kri_thresholds_to_html(html, thresholds)
        if threshold_changes:
            print(f"  Threshold sync: {len(threshold_changes)} tile(s) corrected from kri_thresholds.csv")
            for c in threshold_changes:
                print(f"    ↳ {c}")
        else:
            print("  Threshold sync: all tiles already match kri_thresholds.csv ✓")
        total_updates += len(threshold_changes)
    else:
        print("  ⚠ Threshold sync skipped — kri_thresholds.csv not found")

    # ── Benchmark sync: model_benchmarks.json → HTML text ────────────────────
    # Auto-corrects exec rec / risk register / AI prompt text that drifted from
    # the canonical simulation outputs (e.g. after MTBF or EBITDA recalibration).
    html, bench_changes = _sync_model_benchmarks_to_html(html)
    if bench_changes:
        print(f"  Benchmark sync: {len(bench_changes)} figure(s) corrected from model_benchmarks.json")
        for c in bench_changes:
            print(f"    ↳ {c}")
        total_updates += len(bench_changes)
    else:
        print("  Benchmark sync: all figures already match model_benchmarks.json ✓")

    # ── Write HTML if anything changed ───────────────────────────────────────
    if total_updates > 0:
        dashboard_path.write_text(html)

    # ── Consistency check: catch remaining figure contradictions ─────────────
    consistency_report = {}
    try:
        from tools.consistency_checker import run as _cc_run, print_report as _cc_print
        consistency_report = _cc_run(dashboard_path)
        _cc_print(consistency_report)
    except Exception as e:
        print(f"  ⚠ Consistency checker failed: {e}")

    return {
        "dashboard_path":      str(dashboard_path),
        "backup_path":         str(backup),
        "total_kri_updates":   total_updates,
        "risks_updated":       changes,
        "threshold_changes":   threshold_changes,
        "unmapped_kris":       unmapped,
        "consistency":         consistency_report,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }


def update_exec_recommendations(dashboard_path, approved):
    if not dashboard_path.exists():
        return 0
    html = dashboard_path.read_text()
    backup_dashboard(dashboard_path)
    changes = 0

    for section_key, new_text in approved.items():
        if not new_text:
            continue
        ec_id = EXEC_REC_IDS.get(section_key)
        if not ec_id:
            continue
        pattern = re.compile(
            r'(<div class="exec" id="' + re.escape(ec_id) + r'">'
            r'.*?<div class="exec-hd">.*?</div>\s*)'
            r'(<p>.*?</p>)',
            re.DOTALL
        )
        match = pattern.search(html)
        if match:
            html = html[:match.start(2)] + f"<p>{new_text}</p>" + html[match.end(2):]
            changes += 1

    if changes > 0:
        dashboard_path.write_text(html)
    return changes


def _extract_board_alerts(text: str) -> list:
    """
    Extract up to 3 board-level consequence alerts from the CRA synthesis text.
    Each alert answers 'so what?' — consequence first, not raw metrics.
    Returns list of dicts: {title, headline, detail, color}
    """
    import re
    alerts = []
    text_lc = text.lower()

    # Alert 1 — Covenant breach / lender notification
    cov_breach = (
        re.search(r'cov00\d[^.]{0,80}(?:breach|in breach)', text_lc) or
        re.search(r'covenant[^.]{0,60}confirmed[^.]{0,30}breach', text_lc) or
        re.search(r'lender notification[^.]{0,60}required', text_lc)
    )
    if cov_breach:
        deadline_m = re.search(r'lender notification[^.]{0,40}(\d{1,2}\s+\w+\s+\d{4})', text, re.IGNORECASE)
        deadline   = f'Notification due {deadline_m.group(1)}' if deadline_m else 'Notification required immediately'
        alerts.append({
            'title':    'LENDER NOTIFICATION REQUIRED',
            'headline': 'Covenant confirmed in breach — Trade Finance facility',
            'detail':   deadline,
            'color':    '#ef4444',
        })

    # Alert 2 — Near-zero stress headroom
    hm = re.search(r'only USD ([\d,\.]+M)[^.]{0,50}(?:margin|headroom)', text)
    if hm:
        alerts.append({
            'title':    'STRESS BUFFER CRITICAL',
            'headline': f'USD {hm.group(1)} left under combined scenario',
            'detail':   'FX crystallisation + cure write-off near wipes EBITDA headroom',
            'color':    '#f97316',
        })

    # Alert 3 — Systemic / multi-domain contagion
    if re.search(r'systemic risk flag|three or more domains|3\+?\s+domains', text_lc):
        alerts.append({
            'title':    'SYSTEMIC CONTAGION ACTIVE',
            'headline': '3+ domains simultaneously in breach',
            'detail':   'Concurrent failure modes amplify each other — not isolated risk',
            'color':    '#f59e0b',
        })

    # Fallback: always return 3 alerts (pad with defaults if needed)
    defaults = [
        {'title': 'COVENANT ENFORCEMENT RISK', 'headline': 'COV006 in breach',
         'detail': 'Lender notification required immediately', 'color': '#ef4444'},
        {'title': 'STRESS BUFFER CRITICAL', 'headline': 'Near-zero headroom under stress',
         'detail': 'Combined FX and cure scenario consumes EBITDA headroom', 'color': '#f97316'},
        {'title': 'MULTI-DOMAIN RISK ACTIVE', 'headline': 'Systemic amplification in play',
         'detail': 'Financial, Operational and Strategic breaches reinforce each other', 'color': '#f59e0b'},
    ]
    while len(alerts) < 3:
        alerts.append(defaults[len(alerts)])
    return alerts[:3]


def _detect_priority(action_text: str) -> tuple:
    """
    Return (label, color) board-quarterly priority tier.

    Three tiers that make sense at quarterly review cadence:
      IMMEDIATE   — required at this meeting or within 2 weeks
      IN-QUARTER  — complete within this 90-day review period (~30-60 days)
      NEXT QUARTER — planning horizon beyond this quarter (Q+1)

    Default is IN-QUARTER (the natural delivery window for a quarterly board cycle).
    """
    import re
    tl = action_text.lower()
    if re.search(
        r'at this meeting|this session|within\s+(?:7|14)\s+days?|'
        r'immediat|this\s+week|before\s+(?:next|end\s+of)',
        tl
    ):
        return ('IMMEDIATE', '#ef4444')
    if re.search(
        r'next\s+quarter|q\+1|within\s+90|90[- ]day|'
        r'medium[- ]term|following\s+quarter|beyond\s+this\s+quarter',
        tl
    ):
        return ('NEXT QUARTER', '#3b82f6')
    # Default: in-quarter (covers "within 30 days", "within 60 days",
    # "this quarter", or no explicit timing — all land in the current cycle)
    return ('IN-QUARTER', '#f97316')


def _chips_html_dark(domain_keys: list) -> str:
    """Chips styled for dark header backgrounds."""
    if not domain_keys:
        return ''
    chips = []
    for key in domain_keys:
        dm = _DOMAIN_META.get(key, {'color': '#7f8c8d', 'label': key})
        c, lbl = dm['color'], dm['label']
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:3px;'
            f'padding:1px 6px 1px 4px;border-radius:9px;'
            f'background:{c}30;border:1px solid {c}60;margin-left:3px;'
            f'vertical-align:middle;white-space:nowrap">'
            f'<span style="width:4px;height:4px;border-radius:50%;'
            f'background:{c};flex-shrink:0"></span>'
            f'<span style="font-size:7.5px;font-weight:700;color:{c};'
            f'text-transform:uppercase;letter-spacing:.05em">{lbl}</span>'
            f'</span>'
        )
    return ''.join(chips)


def _chips_arrow_html(domain_keys: list) -> str:
    """Chips connected with → arrows — for cross-domain connection cards."""
    if not domain_keys:
        return ''
    parts = []
    for i, key in enumerate(domain_keys):
        dm = _DOMAIN_META.get(key, {'color': '#7f8c8d', 'label': key})
        c, lbl = dm['color'], dm['label']
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:3px;'
            f'padding:2px 8px 2px 6px;border-radius:10px;'
            f'background:{c}22;border:1px solid {c}55;'
            f'vertical-align:middle;white-space:nowrap">'
            f'<span style="width:5px;height:5px;border-radius:50%;'
            f'background:{c};flex-shrink:0"></span>'
            f'<span style="font-size:8.5px;font-weight:700;color:{c};'
            f'text-transform:uppercase;letter-spacing:.05em">{lbl}</span>'
            f'</span>'
        )
        if i < len(domain_keys) - 1:
            parts.append(
                f'<span style="font-size:11px;color:var(--txt-m);'
                f'vertical-align:middle;margin:0 2px">→</span>'
            )
    return ''.join(parts)


def _format_board_summary(text: str) -> str:
    """
    Convert CRA board summary text into styled HTML sections — Command Deck style.

    Recognises only the five canonical CRA synthesis sections:
      RISK POSTURE / KEY RISK DRIVERS / CROSS-DOMAIN CONNECTIONS /
      QUARTER-ON-QUARTER MOVEMENT / RISK COMMITTEE RECOMMENDED ACTIONS

    Domain labels (STRATEGIC / OPERATIONAL / FINANCIAL / COMPLIANCE) inside
    the RISK POSTURE body are rendered as colour-coded domain sub-cards by
    _render_section_body — they are NOT top-level section splits.

    Also tolerates stray markdown (## headers, # title lines, • bullets) that
    the LLM sometimes emits despite being asked for plain text.
    """
    import re

    # ── Step 1: strip any leading markdown title line (# BOARD RISK SUMMARY …) ──
    text = re.sub(r'^#[^\n]*\n?', '', text.strip())

    # ── Step 2: normalise the five CRA section headers to "SECTION NAME:" ──
    # Only these five are treated as top-level splits. Domain labels (STRATEGIC:
    # etc.) inside RISK POSTURE body are handled by _render_section_body.
    SECTION_NAMES = [
        'RISK POSTURE',
        'KEY RISK DRIVERS',
        'CROSS-DOMAIN CONNECTIONS',
        'QUARTER-ON-QUARTER MOVEMENT',
        'RISK COMMITTEE RECOMMENDED ACTIONS',
    ]
    for name in SECTION_NAMES:
        # markdown heading: ## SECTION NAME (optional colon)
        text = re.sub(
            r'##\s*' + re.escape(name) + r'\s*:?\s*',
            name + ':\n',
            text, flags=re.IGNORECASE
        )
        # bare label on its own line followed by newline(s), no colon
        text = re.sub(
            r'(?m)^' + re.escape(name) + r'\s*$',
            name + ':',
            text, flags=re.IGNORECASE
        )

    # ── Step 3: section → colour map (five CRA sections only) ──
    SECTION_CONFIG = [
        # Ordered longest-first to avoid prefix collisions
        ('RISK COMMITTEE RECOMMENDED ACTIONS', 'var(--navy)'),
        ('QUARTER-ON-QUARTER MOVEMENT',        'var(--grn-md)'),
        ('CROSS-DOMAIN CONNECTIONS',           '#7f8c8d'),
        ('KEY RISK DRIVERS',                   'var(--amb-md)'),
        ('RISK POSTURE',                       'var(--red-md)'),
    ]

    segments = []
    for label, color in SECTION_CONFIG:
        for m in re.finditer(re.escape(label) + r'\s*:\s*', text, re.IGNORECASE):
            segments.append((m.start(), m.end(), label, color))

    if not segments:
        # Fallback: strip leftover markdown and return as a plain paragraph
        clean = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE).strip()
        return (
            f'<p style="font-size:12.5px;color:var(--txt);line-height:1.6;margin:0">'
            f'{clean}</p>'
        )

    segments.sort(key=lambda x: x[0])
    # dedupe: if two labels overlap keep the longer one
    deduped: list = []
    for seg in segments:
        if deduped and seg[0] < deduped[-1][1]:
            if len(seg[2]) > len(deduped[-1][2]):
                deduped[-1] = seg
        else:
            deduped.append(seg)
    segments = deduped

    # Any text before the first section header becomes a title paragraph
    overview = text[:segments[0][0]].strip()
    overview = re.sub(r'^#+\s*', '', overview, flags=re.MULTILINE).strip()

    # ── Consequence alert strip (3 board-level "so what" cards) ──────────────
    alerts = _extract_board_alerts(text)
    alert_cards = []
    for a in alerts:
        c = a['color']
        alert_cards.append(
            f'<div style="flex:1;min-width:0;padding:9px 11px 8px;border-radius:6px;'
            f'background:{c}0d;border:1px solid {c}35">'
            f'<div style="font-size:7.5px;font-weight:800;color:{c};'
            f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">'
            f'⚠ {a["title"]}</div>'
            f'<div style="font-size:12px;font-weight:700;color:var(--txt);'
            f'line-height:1.3;margin-bottom:3px">{a["headline"]}</div>'
            f'<div style="font-size:10.5px;color:var(--txt);opacity:0.65;'
            f'line-height:1.4">{a["detail"]}</div>'
            f'</div>'
        )
    parts = [
        f'<div style="display:flex;gap:6px;margin-bottom:0.7rem">'
        + ''.join(alert_cards)
        + f'</div>'
    ]

    if overview:
        parts.append(
            f'<p style="font-size:12px;font-weight:500;color:var(--txt);opacity:0.75;'
            f'margin:0 0 0.6rem;line-height:1.5">{overview}</p>'
        )

    for i, (start, end, label, color) in enumerate(segments):
        body_end = segments[i + 1][0] if i + 1 < len(segments) else len(text)
        body = text[end:body_end].strip()
        body = re.sub(r'^#+\s*', '', body, flags=re.MULTILINE).strip()

        # Domain chips for section header (dark background variant)
        section_domains = _detect_domains(body)
        header_chips = _chips_html_dark(section_domains)

        mb = '0' if i == len(segments) - 1 else '0.55rem'
        label_display = label.title().replace('Risk Committee Recommended', 'Recommended')
        parts.append(
            # ── Command Deck card: dark header + body ────────────────────────
            f'<div style="margin-bottom:{mb};border-radius:7px;overflow:hidden;'
            f'border:1px solid rgba(255,255,255,0.06);box-shadow:0 1px 3px rgba(0,0,0,0.18)">'
            # dark header strip
            f'<div style="background:linear-gradient(90deg,rgba(10,18,35,0.96) 0%,'
            f'rgba(20,30,52,0.94) 100%);padding:8px 12px 7px;'
            f'display:flex;align-items:center;justify-content:space-between;gap:8px">'
            f'<span style="font-size:9.5px;font-weight:800;color:#ffffff;'
            f'text-transform:uppercase;letter-spacing:.13em;flex-shrink:0">{label_display}</span>'
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;'
            f'justify-content:flex-end">{header_chips}</div>'
            f'</div>'
            # body area
            f'<div style="padding:0.65rem 0.75rem 0.6rem">'
            f'{_render_section_body(label, body)}'
            f'</div>'
            f'</div>'
        )

    return ''.join(parts)


# ── Domain detection helpers ──────────────────────────────────────────────────

_DOMAIN_META = {
    'S': {'color': '#e8c547', 'label': 'Strategic'},
    'O': {'color': '#f97316', 'label': 'Operational'},
    'F': {'color': '#3b82f6', 'label': 'Financial'},
    'C': {'color': '#a78bfa', 'label': 'Compliance'},
}

def _detect_domains(text: str) -> list:
    """Return ordered list of domain keys (S/O/F/C) found in text."""
    import re as _re
    _PATTERNS = {
        'S': [r'S-0[1-3]', r'geopolit', r'Taiwan', r'Entity List', r'\bM&A\b',
              r'synergy', r'competitive.signal', r'strategic\b'],
        'O': [r'O-0[1-4]', r'supply.chain', r'TSMC', r'Foxconn', r'Quanta',
              r'single.source', r'inventory.cover', r'\bMTTD\b', r'\bMTTR\b',
              r'patch.compliance', r'attrition', r'talent\b', r'flight.risk',
              r'\bR&D\b', r'\bcyber\b', r'\bBCM\b', r'\bOMS\b', r'operational\b'],
        'F': [r'F-0[1-4]', r'COV0\d+', r'covenant\b', r'\bEBITDA\b', r'\bFX\b',
              r'hedge', r'bad.debt', r'maturity\b', r'revolving.credit',
              r'financial\b', r'lender\b', r'liquidity\b', r'USD\s+\d'],
        'C': [r'C-0[1-3]', r'sanction', r'export.licen', r'\bABAC\b', r'\bGDPR\b',
              r'denied.party', r'\bAUD\d', r'compliance\b'],
    }
    found = []
    for key, patterns in _PATTERNS.items():
        for pat in patterns:
            if _re.search(pat, text, _re.IGNORECASE):
                found.append(key)
                break
    return found


def _chips_html(domain_keys: list) -> str:
    """Return inline HTML chips for the given domain keys."""
    if not domain_keys:
        return ''
    chips = []
    for key in domain_keys:
        dm = _DOMAIN_META.get(key, {'color': '#7f8c8d', 'label': key})
        c, lbl = dm['color'], dm['label']
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:3px;'
            f'padding:1px 7px 1px 5px;border-radius:10px;'
            f'background:{c}22;border:1px solid {c}55;margin-right:3px;'
            f'vertical-align:middle;white-space:nowrap">'
            f'<span style="width:5px;height:5px;border-radius:50%;'
            f'background:{c};flex-shrink:0"></span>'
            f'<span style="font-size:8.5px;font-weight:700;color:{c};'
            f'text-transform:uppercase;letter-spacing:.05em">{lbl}</span>'
            f'</span>'
        )
    return ''.join(chips)


# ── CRO-style card title lookup for Key Risk Driver cards ─────────────────────
# Ordered: more specific patterns first.  First match wins.
_KRD_TITLE_RULES = [
    # Specific patterns first — most distinctive signals
    (r'covenant.*confirmed.*breach|confirmed.*breach.*covenant|trade finance.*breach',
                                                                    'Covenant Breach Exposure'),
    (r'EBITDA headroom|headroom.*(?:USD|million)|revolving credit.*covenant',
                                                                    'EBITDA Headroom Pressure'),
    (r'stress scenario|combined.*charge|crystallis.*covenant|trigger.*covenant|not remote',
                                                                    'Compound Stress Scenario'),
    (r'covenant|COV00\d|cure.*write|write.*cure|lender.*notif|notif.*lender',
                                                                    'Covenant & Lender Risk'),
    (r'hedge.*ratio|FX.*unrealised|unrealised.*FX|portfolio.*loss|commodity.*deriv',
                                                                    'Portfolio Loss Exposure'),
    (r'bad.debt|debt.*provision|receivable|overdue|credit.quality', 'Receivables Deterioration'),
    (r'cyber|MTTD|mean.time.to.detect|detection.gap|SIEM|patch.compliance',
                                                                    'Cyber Detection Gap'),
    (r'supply.chain|single.source|Quanta|TSMC|Foxconn|inventory.cover|supplier.distress',
                                                                    'Supply Chain Concentration'),
    (r'talent|attrition|flight.risk|succession|open.roles|R&D.departure',
                                                                    'Talent & Succession Risk'),
    (r'geopolit|Taiwan|Entity List|export.*restrict|PRC.*concentrat',
                                                                    'Geopolitical Exposure'),
    (r'sanction|ABAC|denied.party|export.*screen|whistleblower',    'Sanctions & ABAC Risk'),
    (r'audit.*coverage|ai.audit|GDPR|data.*privacy',               'Compliance Coverage Gap'),
    (r'M&A|synergy|competitive|AI.native|product.*roadmap',        'Strategic Position Risk'),
]


def _krd_card_title(sentence: str) -> str:
    """Return a CRO-quality card title by matching the sentence against _KRD_TITLE_RULES."""
    import re as _re
    for pattern, title in _KRD_TITLE_RULES:
        if _re.search(pattern, sentence, _re.IGNORECASE):
            return title
    # Fallback: extract noun phrase from "The X is/are/has..."
    m = _re.match(
        r'(?:The |A |That |Against this backdrop, the )(\w[\w\s]{3,40}?)\s+'
        r'(?:is\b|are\b|has\b|presents\b|remains\b|stands\b|compounds\b)',
        sentence, _re.IGNORECASE
    )
    if m:
        return m.group(1).strip().title()
    # Last resort: first 4-5 content words
    stop = {'the', 'a', 'an', 'that', 'this', 'is', 'are', 'has', 'against', 'backdrop'}
    words = [w for w in sentence.split()[:9] if w.lower().rstrip('.,') not in stop][:5]
    return ' '.join(words).rstrip('.,').title()


def _board_brief(text: str, max_sents: int = 2) -> str:
    """
    Return the first N sentences — enough board context without a wall of text.
    Strips any trailing executive-attribution sentence ("The CFO holds accountability...").
    """
    import re as _re
    sents = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    # Drop pure accountability/ownership sentences at the end
    filtered = [s for s in sents if not _re.match(
        r'The\s+(?:Chief|CFO|COO|CISO|CEO|CRO)\b.*(?:accountab|responsib|owner)',
        s, _re.IGNORECASE
    )]
    selected = filtered[:max_sents]
    out = ' '.join(s.rstrip('.') for s in selected)
    return out + ('.' if out and out[-1] not in '.!?' else '')


def _render_section_body(label: str, body: str) -> str:
    """
    Command Deck renderers — each section has a distinct, impactful visual treatment.
    Every element answers "so what?" — consequence first, metrics second.

    RISK POSTURE:              compact domain status table, one-liner per row
    KEY RISK DRIVERS:          3-column consequence grid + italic lead + footer
    CROSS-DOMAIN CONNECTIONS:  arrow-connector chips → bold headline cards
    QUARTER-ON-QUARTER:        trend banner + two-column domain rows
    RECOMMENDED ACTIONS:       priority-tiered cards with domain chips
    """
    import re

    lbl_up = label.upper()

    def _split_sentences(text: str) -> list:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text.strip())
        return [s.strip() for s in parts if s.strip()]

    def _oneliner(text: str, max_chars: int = 130) -> str:
        """First sentence, capped at max_chars."""
        m = re.match(r'([^.!?]+[.!?])', text.strip())
        s = m.group(1) if m else text.strip()
        return s[:max_chars] + ('…' if len(s) > max_chars else '')

    # ═══════════════════════════════════════════════════════════════════════════
    # RISK POSTURE — compact domain status table
    # ═══════════════════════════════════════════════════════════════════════════
    if 'RISK POSTURE' in lbl_up:
        import re as _re
        DOMAIN_META_RP = {
            'STRATEGIC':   '#e8c547',
            'OPERATIONAL': '#f97316',
            'FINANCIAL':   '#3b82f6',
            'COMPLIANCE':  '#a78bfa',
        }
        domain_pattern = r'(?i)\b(STRATEGIC|OPERATIONAL|FINANCIAL|COMPLIANCE)\s*:'
        if _re.search(domain_pattern, body):
            chunks = _re.split(domain_pattern, body)
            overview = _re.sub(r'[🔴🟡🟢🟠⚫⚪]', '', chunks[0]).strip()
            html_parts = []
            if overview:
                # Overview: bold consequence statement
                html_parts.append(
                    f'<div style="font-size:12.5px;font-weight:600;color:var(--txt);'
                    f'line-height:1.55;margin-bottom:0.6rem;padding-bottom:0.5rem;'
                    f'border-bottom:1px solid rgba(255,255,255,0.07)">{overview}</div>'
                )
            # Domain rows — compact table style
            domain_rows = []
            for i in range(1, len(chunks) - 1, 2):
                dname  = chunks[i].upper()
                raw    = chunks[i + 1] if i + 1 < len(chunks) else ''
                dtext  = _re.sub(r'[🔴🟡🟢🟠⚫⚪]', '', raw).strip()
                oneliner = _board_brief(dtext, 2)
                dcolor = DOMAIN_META_RP.get(dname, '#7f8c8d')
                is_last = (i + 2 >= len(chunks) - 1)
                domain_rows.append(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;'
                    f'{"" if is_last else "margin-bottom:0;padding-bottom:0.45rem;border-bottom:1px solid rgba(255,255,255,0.055);"}'
                    f'{"padding-top:0.45rem;" if i > 1 else ""}'
                    f'">'
                    # Left: dot + name + live status badge
                    f'<div style="flex-shrink:0;width:108px;display:flex;'
                    f'align-items:center;gap:5px;padding-top:2px">'
                    f'<span id="bs-dot-{dname.lower()}" style="display:inline-block;'
                    f'width:7px;height:7px;border-radius:50%;background:{dcolor};'
                    f'flex-shrink:0"></span>'
                    f'<span style="font-size:9.5px;font-weight:800;color:{dcolor};'
                    f'text-transform:uppercase;letter-spacing:.07em">{dname.title()}</span>'
                    f'</div>'
                    # Middle: live status badge (JS-populated)
                    f'<div style="flex-shrink:0;width:54px;padding-top:1px">'
                    f'<span id="bs-status-{dname.lower()}" style="font-size:8px;'
                    f'font-weight:700;padding:1px 5px;border-radius:3px;'
                    f'text-transform:uppercase;letter-spacing:.05em"></span>'
                    f'</div>'
                    # Right: one-liner business consequence
                    f'<div style="flex:1;font-size:12px;color:var(--txt);'
                    f'line-height:1.5">{oneliner}</div>'
                    f'</div>'
                )
            html_parts.append(''.join(domain_rows))
            return ''.join(html_parts)
        else:
            sents = _split_sentences(body)
            rows = []
            for idx, s in enumerate(sents):
                if s and s[-1] not in '.!?':
                    s += '.'
                chips = _chips_html(_detect_domains(s))
                is_last = idx == len(sents) - 1
                rows.append(
                    f'<div style="font-size:12px;color:var(--txt);line-height:1.5;'
                    f'{"" if is_last else "margin-bottom:0.35rem;padding-bottom:0.35rem;border-bottom:1px solid rgba(255,255,255,0.06);"}'
                    f'">'
                    + (f'<div style="margin-bottom:3px">{chips}</div>' if chips else '')
                    + f'{s}</div>'
                )
            return ''.join(rows)

    # ═══════════════════════════════════════════════════════════════════════════
    # KEY RISK DRIVERS — italic lead + 3-column consequence grid + footer
    # ═══════════════════════════════════════════════════════════════════════════
    elif 'KEY RISK DRIVERS' in lbl_up or 'KEY RISK' in lbl_up:
        sents = _split_sentences(body)
        if len(sents) <= 2:
            rows = []
            for idx, s in enumerate(sents):
                chips = _chips_html(_detect_domains(s))
                is_last = idx == len(sents) - 1
                rows.append(
                    f'<div style="font-size:12px;color:var(--txt);line-height:1.5;'
                    f'{"" if is_last else "margin-bottom:0.35rem;padding-bottom:0.35rem;border-bottom:1px solid rgba(255,255,255,0.06);"}'
                    f'">'
                    + (f'<div style="margin-bottom:3px">{chips}</div>' if chips else '')
                    + f'{s}</div>'
                )
            return ''.join(rows)

        lead = sents[0]
        if lead and lead[-1] not in '.!?':
            lead += '.'
        html_parts = [
            f'<div style="font-size:12px;font-style:italic;color:var(--txt);'
            f'opacity:0.8;line-height:1.55;margin-bottom:0.55rem">{lead}</div>'
        ]

        # 3-column flex grid — first 3 middle sentences as prominent cards
        middle = sents[1:-1] if len(sents) > 2 else sents[1:]
        grid_cards = middle[:3]
        extra_cards = middle[3:]

        grid_html = []
        for s in grid_cards:
            if s and s[-1] not in '.!?':
                s += '.'
            doms = _detect_domains(s)
            chips = _chips_arrow_html(doms) if len(doms) > 1 else _chips_html(doms)
            # CRO-quality title from keyword lookup (never parsed from sentence fragments)
            headline = _krd_card_title(s)
            grid_html.append(
                f'<div style="flex:1 1 calc(33% - 4px);min-width:140px;'
                f'padding:0.6rem 0.65rem;border-radius:6px;'
                f'background:rgba(255,255,255,0.035);'
                f'border:1px solid rgba(255,255,255,0.08)">'
                + (f'<div style="margin-bottom:6px;line-height:1">{chips}</div>' if chips else '')
                + f'<div style="font-size:11.5px;font-weight:800;color:var(--txt);'
                f'text-transform:uppercase;letter-spacing:.04em;'
                f'line-height:1.3;margin-bottom:6px">{headline}</div>'
                + f'<div style="font-size:11px;color:var(--txt);opacity:0.75;line-height:1.45">{s}</div>'
                + f'</div>'
            )
        html_parts.append(
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:{"0.45rem" if extra_cards or len(sents) > 2 else "0"}">'
            + ''.join(grid_html)
            + f'</div>'
        )

        # Any overflow middle sentences as compact list
        for s in extra_cards:
            if s and s[-1] not in '.!?':
                s += '.'
            chips = _chips_html(_detect_domains(s))
            html_parts.append(
                f'<div style="font-size:11.5px;color:var(--txt);opacity:0.8;line-height:1.45;'
                f'margin-bottom:0.3rem">'
                + (f'<span style="vertical-align:middle">{chips}</span> ' if chips else '')
                + f'{s}</div>'
            )

        # Last sentence → footer
        if len(sents) > 2:
            footer = sents[-1]
            if footer and footer[-1] not in '.!?':
                footer += '.'
            footer_chips = _chips_html(_detect_domains(footer))
            html_parts.append(
                f'<div style="margin-top:0.4rem;padding-top:0.4rem;'
                f'border-top:1px solid rgba(255,255,255,0.07)">'
                + (f'<span style="vertical-align:middle">{footer_chips}</span> ' if footer_chips else '')
                + f'<span style="font-size:11px;color:var(--txt);opacity:0.7;line-height:1.45">{footer}</span>'
                f'</div>'
            )
        return ''.join(html_parts)

    # ═══════════════════════════════════════════════════════════════════════════
    # CROSS-DOMAIN CONNECTIONS — arrow-connector chips + bold headline cards
    # ═══════════════════════════════════════════════════════════════════════════
    elif 'CROSS-DOMAIN' in lbl_up or 'CROSS DOMAIN' in lbl_up:
        import re as _re2
        split_pat = (
            r'(?<=[.!?])\s+'
            r'(?=A second|A third|The compliance|The financial|The Quanta|'
            r'Additionally,|Furthermore,|Critically,|A further|Another|'
            r'The operational|The strategic|Second,|Third,)'
        )
        segments = _re2.split(split_pat, body.strip())
        segments = [s.strip() for s in segments if s.strip()]
        if not segments:
            segments = _split_sentences(body)

        cards = []
        for idx, seg in enumerate(segments):
            if seg and seg[-1] not in '.!?':
                seg += '.'
            doms = _detect_domains(seg)
            # Arrow connector chips when 2+ domains detected
            chip_html = _chips_arrow_html(doms) if len(doms) >= 2 else _chips_html(doms)

            seg_sents = _split_sentences(seg)
            if len(seg_sents) > 1:
                headline = seg_sents[0]
                if headline and headline[-1] not in '.!?':
                    headline += '.'
                rest = ' '.join(seg_sents[1:])
                body_html = (
                    f'<div style="font-size:12.5px;font-weight:700;color:var(--txt);'
                    f'line-height:1.45;margin-bottom:5px">{headline}</div>'
                    f'<div style="font-size:11.5px;color:var(--txt);opacity:0.8;line-height:1.5">{rest}</div>'
                )
            else:
                body_html = (
                    f'<div style="font-size:12.5px;font-weight:700;color:var(--txt);'
                    f'line-height:1.5">{seg}</div>'
                )
            is_last = (idx == len(segments) - 1)
            cards.append(
                f'<div style="{"" if is_last else "margin-bottom:0.5rem;"}'
                f'padding:0.6rem 0.75rem;border-radius:6px;'
                f'background:rgba(255,255,255,0.03);'
                f'border:1px solid rgba(255,255,255,0.07)">'
                + (f'<div style="margin-bottom:7px;line-height:1">{chip_html}</div>' if chip_html else '')
                + body_html
                + f'</div>'
            )
        return ''.join(cards)

    # ═══════════════════════════════════════════════════════════════════════════
    # QUARTER-ON-QUARTER MOVEMENT — trend banner + 2-col domain rows
    # ═══════════════════════════════════════════════════════════════════════════
    elif 'QUARTER' in lbl_up:
        text_lc = body.lower()
        if any(w in text_lc for w in ['deteriorat', 'worsen', 'increased breach', 'new breach', 'escalat', 'crossed into breach']):
            trend_label = 'DETERIORATING'
            trend_color = '#ef4444'
            trend_bg    = 'rgba(239,68,68,0.1)'
            trend_border = 'rgba(239,68,68,0.3)'
            trend_icon  = '▲'
        elif any(w in text_lc for w in ['improv', 'reduc', 'resolv', 'closed breach', 'recover']):
            trend_label = 'IMPROVING'
            trend_color = '#22c55e'
            trend_bg    = 'rgba(34,197,94,0.1)'
            trend_border = 'rgba(34,197,94,0.3)'
            trend_icon  = '▼'
        else:
            trend_label = 'STABLE'
            trend_color = '#f59e0b'
            trend_bg    = 'rgba(245,158,11,0.1)'
            trend_border = 'rgba(245,158,11,0.3)'
            trend_icon  = '→'

        banner = (
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'padding:8px 12px;border-radius:6px;margin-bottom:0.6rem;'
            f'background:{trend_bg};border:1px solid {trend_border}">'
            f'<span style="font-size:20px;color:{trend_color};line-height:1;font-weight:700">'
            f'{trend_icon}</span>'
            f'<div>'
            f'<div style="font-size:11px;font-weight:800;color:{trend_color};'
            f'text-transform:uppercase;letter-spacing:.1em">{trend_label}</div>'
            f'<div style="font-size:10px;color:var(--txt);opacity:0.65;margin-top:1px">'
            f'Quarter-on-quarter risk position</div>'
            f'</div>'
            f'</div>'
        )

        sents = _split_sentences(body)
        rows = [banner]
        for idx, s in enumerate(sents):
            if s and s[-1] not in '.!?':
                s += '.'
            doms = _detect_domains(s)
            chips = _chips_html(doms)
            is_last = (idx == len(sents) - 1)
            # 2-column: chip label col | text col
            rows.append(
                f'<div style="display:flex;align-items:baseline;gap:8px;'
                f'{"" if is_last else "margin-bottom:0.35rem;padding-bottom:0.35rem;border-bottom:1px solid rgba(255,255,255,0.055);"}'
                f'">'
                f'<div style="flex-shrink:0;min-width:80px;padding-top:1px">'
                f'{chips if chips else "<span></span>"}</div>'
                f'<div style="flex:1;font-size:12px;color:var(--txt);line-height:1.5">{s}</div>'
                f'</div>'
            )
        return ''.join(rows)

    # ═══════════════════════════════════════════════════════════════════════════
    # RECOMMENDED ACTIONS — priority-tiered cards with domain chips
    # ═══════════════════════════════════════════════════════════════════════════
    elif 'RECOMMENDED ACTIONS' in lbl_up:
        bullets = [b.strip() for b in re.split(r'\s*•\s*', body) if b.strip()]
        if not bullets:
            return f'<div style="font-size:12px;color:var(--txt);line-height:1.6">{body}</div>'
        cards = []
        for idx, bullet in enumerate(bullets):
            is_last = (idx == len(bullets) - 1)
            p_label, p_color = _detect_priority(bullet)
            chips = _chips_html(_detect_domains(bullet))
            cards.append(
                f'<div style="{"" if is_last else "margin-bottom:0.45rem;"}'
                f'border-radius:5px;overflow:hidden;'
                f'border:1px solid rgba(255,255,255,0.07)">'
                # Priority tier header strip
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'padding:4px 10px;background:{p_color}18;'
                f'border-bottom:1px solid {p_color}30">'
                f'<span style="display:inline-block;width:3px;height:14px;'
                f'border-radius:2px;background:{p_color};flex-shrink:0"></span>'
                f'<span style="font-size:8.5px;font-weight:800;color:{p_color};'
                f'text-transform:uppercase;letter-spacing:.1em">{p_label}</span>'
                + (f'<div style="margin-left:4px">{chips}</div>' if chips else '')
                + f'</div>'
                # Action body
                f'<div style="padding:0.45rem 0.65rem;">'
                f'<div style="font-size:12px;color:var(--txt);line-height:1.55">{bullet}</div>'
                f'</div>'
                f'</div>'
            )
        return ''.join(cards)

    # ═══════════════════════════════════════════════════════════════════════════
    # DEFAULT — 2-col domain rows
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        sentences = _split_sentences(body)
        if len(sentences) <= 1:
            return f'<div style="font-size:12px;color:var(--txt);line-height:1.55">{body}</div>'
        rows = []
        for idx, sentence in enumerate(sentences):
            if sentence and sentence[-1] not in '.!?':
                sentence += '.'
            chips = _chips_html(_detect_domains(sentence))
            is_last = (idx == len(sentences) - 1)
            rows.append(
                f'<div style="display:flex;align-items:baseline;gap:8px;'
                f'font-size:12px;color:var(--txt);line-height:1.5;'
                f'{"" if is_last else "margin-bottom:0.35rem;padding-bottom:0.35rem;border-bottom:1px solid rgba(255,255,255,0.055);"}'
                f'">'
                f'<div style="flex-shrink:0;min-width:70px">{chips}</div>'
                f'<div style="flex:1">{sentence}</div>'
                f'</div>'
            )
        return ''.join(rows)


def update_board_summary(dashboard_path, summary_text: str, run_id: str) -> bool:
    """Write Risk Committee Summary into the dashboard panel as colour-coded domain sections."""
    import re
    from datetime import datetime, timezone
    html = Path(dashboard_path).read_text()
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    formatted = _format_board_summary(summary_text)

    # Update run/timestamp meta (handles both <div> and <p> closing tags)
    html = re.sub(
        r'id="board-summary-meta"[^>]*>[^<]*</(?:div|p)>',
        f'id="board-summary-meta" style="font-size:11px;color:var(--txt-m);margin-top:2px">'
        f'Run {run_id[:8]} · {timestamp}</div>',
        html
    )

    # Update content block — sentinel pattern for <div> structure
    new_html = re.sub(
        r'id="board-summary-text"[^>]*>.*?</div><!-- /bst -->',
        f'id="board-summary-text">{formatted}</div><!-- /bst -->',
        html, flags=re.DOTALL
    )
    if new_html == html:
        # Fallback: legacy <p> structure
        new_html = re.sub(
            r'id="board-summary-text"[^>]*>.*?</p>',
            f'id="board-summary-text">{formatted}</p>',
            html, flags=re.DOTALL
        )
    html = new_html

    Path(dashboard_path).write_text(html)
    return True

def update_signals_panel(dashboard_path, regulatory: dict, emerging: dict, run_id: str) -> bool:
    """Write regulatory and emerging signals into the dashboard signals panel."""
    import re
    from datetime import datetime, timezone
    html = Path(dashboard_path).read_text()
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    items = []

    # Regulatory deadline alerts
    for d in regulatory.get('deadline_alerts', []):
        urgency = d.get('urgency', 'watch')
        color = '#c0392b' if urgency == 'immediate' else '#e67e22' if urgency == '90_days' else '#7f8c8d'
        items.append(
            f'<div style="margin-bottom:0.7rem;padding-bottom:0.7rem;border-bottom:1px solid var(--bdr)">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:2px">'
            f'<span style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase">{urgency.replace("_"," ")}</span>'
            f'<span style="font-size:11px;font-weight:600">{d.get("regulation","")} — {d.get("jurisdiction","")}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:var(--txt)">{d.get("description","")}</div>'
            f'<div style="font-size:11px;color:var(--txt-m)">Deadline: {d.get("deadline","")} · Affects: {d.get("affects_risk","")}</div>'
            f'</div>'
        )

    # New regulatory signals
    for s in regulatory.get('new_signals', []):
        items.append(
            f'<div style="margin-bottom:0.7rem;padding-bottom:0.7rem;border-bottom:1px solid var(--bdr)">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:2px">'
            f'<span style="font-size:10px;font-weight:700;color:#2980b9;text-transform:uppercase">NEW SIGNAL</span>'
            f'<span style="font-size:11px;font-weight:600">{s.get("title","")}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:var(--txt)">{s.get("summary","")}</div>'
            f'<div style="font-size:11px;color:var(--txt-m)">{s.get("regulation","")} · {s.get("jurisdiction","")} · Confidence: {s.get("confidence","")}</div>'
            f'</div>'
        )

    # Emerging risk candidates
    for c in emerging.get('risk_candidates', []):
        l = c.get('initial_L', 0)
        i = c.get('initial_I', 0)
        action = c.get('recommended_action', 'watch_list')
        color = '#c0392b' if action == 'immediate_board_attention' else '#e67e22' if action == 'assess_for_register' else '#7f8c8d'
        # Strip citation tags then trim to a clean sentence boundary
        rationale = re.sub(r'<cite[^>]*>.*?</cite>', '', c.get('rationale', ''), flags=re.DOTALL).strip()
        rationale = re.sub(r'\s+', ' ', rationale)
        rationale = _trim_to_sentence(rationale, 200)
        items.append(
            f'<div style="margin-bottom:0.7rem;padding-bottom:0.7rem;border-bottom:1px solid var(--bdr)">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:2px">'
            f'<span style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase">{action.replace("_"," ")}</span>'
            f'<span style="font-size:11px;font-weight:600">Emerging: {c.get("proposed_id","")} ({c.get("proposed_domain","")})</span>'
            f'</div>'
            f'<div style="font-size:12px;color:var(--txt)">{c.get("signal","")}</div>'
            f'<div style="font-size:11px;color:var(--txt-m)">L={l} I={i} · Horizon: {c.get("horizon","")} · {rationale}</div>'
            f'</div>'
        )

    if not items:
        items = ['<div style="color:var(--txt-m);font-style:italic">No new signals this run.</div>']

    signals_html = ''.join(items)

    # Update meta
    html = re.sub(
        r'id="signals-meta"[^>]*>[^<]*</div>',
        f'id="signals-meta" style="font-size:11px;color:var(--txt-m);margin-top:2px">Run {run_id[:8]} · {timestamp}</div>',
        html
    )
    # Update signals list
    html = re.sub(
        r'id="signals-list"[^>]*>.*?</div>(?=\s*</div>\s*</div>\s*</div>)',
        f'id="signals-list" style="font-size:12.5px;color:var(--txt)">{signals_html}',
        html, flags=re.DOTALL
    )
    Path(dashboard_path).write_text(html)
    return True
