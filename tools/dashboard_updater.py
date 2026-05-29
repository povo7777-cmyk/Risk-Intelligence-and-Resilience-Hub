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


def _format_board_summary(text: str) -> str:
    """
    Convert CRA board summary text into styled HTML sections.

    Handles two layouts:
      • Current (cra_synthesis_v1): four plain-text headers
        RISK POSTURE / KEY RISK DRIVERS / CROSS-DOMAIN CONNECTIONS / QUARTER-ON-QUARTER MOVEMENT
        followed by RISK COMMITTEE RECOMMENDED ACTIONS appended by the graph.
      • Legacy: domain-based headers  STRATEGIC / OPERATIONAL / FINANCIAL / COMPLIANCE

    Also tolerates stray markdown (## headers, # title lines, • bullets) that the
    LLM sometimes emits despite being asked for plain text.

    The KRI STATUS ROLL-UP TABLE is appended as sentinel-wrapped HTML by
    _build_kri_rollup_table(). Extract it before section processing and re-append
    as a properly styled section at the end — never let it bleed into prose sections.
    """
    import re

    # ── Step 1: strip any leading markdown title line (# BOARD RISK SUMMARY …) ──
    text = re.sub(r'^#[^\n]*\n?', '', text.strip())

    # ── Step 2: normalise section headers to "SECTION NAME:" ──
    # Handles three patterns the LLM might emit:
    #   "## RISK POSTURE"   — markdown heading (with or without colon)
    #   "RISK POSTURE\n\n"  — bare label on its own line (plain text, no colon)
    #   "RISK POSTURE:"     — already has colon (already fine)
    SECTION_NAMES = [
        'RISK POSTURE',
        'KEY RISK DRIVERS',
        'CROSS-DOMAIN CONNECTIONS',
        'QUARTER-ON-QUARTER MOVEMENT',
        'RISK COMMITTEE RECOMMENDED ACTIONS',
        'COMPOUND SCENARIOS',
        'COMPOUND',
        'STRATEGIC',
        'OPERATIONAL',
        'FINANCIAL',
        'COMPLIANCE',
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

    # ── Step 3: section → colour map ──
    SECTION_CONFIG = [
        # Current synthesis sections (ordered longest-first to avoid prefix collisions)
        ('RISK COMMITTEE RECOMMENDED ACTIONS', 'var(--navy)'),
        ('QUARTER-ON-QUARTER MOVEMENT',        'var(--grn-md)'),
        ('CROSS-DOMAIN CONNECTIONS',           '#7f8c8d'),
        ('KEY RISK DRIVERS',                   'var(--amb-md)'),
        ('RISK POSTURE',                       'var(--red-md)'),
        # Legacy domain sections
        ('COMPOUND SCENARIOS',                 '#7f8c8d'),
        ('COMPOUND',                           '#7f8c8d'),
        ('STRATEGIC',                          'var(--red-md)'),
        ('OPERATIONAL',                        'var(--amb-md)'),
        ('FINANCIAL',                          'var(--grn-md)'),
        ('COMPLIANCE',                         'var(--pur-md)'),
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

    parts = []
    if overview:
        parts.append(
            f'<p style="font-size:12.5px;font-weight:600;color:var(--navy);'
            f'margin:0 0 0.75rem;line-height:1.5">{overview}</p>'
        )

    for i, (start, end, label, color) in enumerate(segments):
        body_end = segments[i + 1][0] if i + 1 < len(segments) else len(text)
        body = text[end:body_end].strip()
        # Strip any residual markdown headers from body text
        body = re.sub(r'^#+\s*', '', body, flags=re.MULTILINE).strip()
        mb = '0' if i == len(segments) - 1 else '0.55rem'
        parts.append(
            f'<div style="margin-bottom:{mb};padding:0.4rem 0.6rem 0.4rem 0.7rem;'
            f'border-left:3px solid {color};background:rgba(0,0,0,0.02)">'
            f'<div style="font-size:10px;font-weight:700;color:{color};'
            f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">'
            f'{label.title()}</div>'
            f'{_render_section_body(label, body)}'
            f'</div>'
        )

    return ''.join(parts)


def _render_section_body(label: str, body: str) -> str:
    """
    Render a section body for board-level readability.

    Recommended Actions: each bullet becomes its own numbered card.
    Prose sections:       body is split into sentence-level rows so the reader
                          can scan one point at a time instead of a wall of text.
    """
    import re

    if 'RECOMMENDED ACTIONS' in label.upper():
        # ── Numbered action cards ──────────────────────────────────────────────
        bullets = [b.strip() for b in re.split(r'\s*•\s*', body) if b.strip()]
        if not bullets:
            return (
                f'<div style="font-size:12px;color:var(--txt);line-height:1.6">{body}</div>'
            )
        cards = []
        for idx, bullet in enumerate(bullets):
            num = f'{idx + 1:02d}'
            is_last = (idx == len(bullets) - 1)
            cards.append(
                f'<div style="display:flex;gap:0.55rem;'
                f'{"" if is_last else "margin-bottom:0.4rem;"}'
                f'padding:0.45rem 0.55rem;background:rgba(0,0,80,0.03);border-radius:3px">'
                f'<span style="font-size:10px;font-weight:700;color:var(--navy);'
                f'min-width:20px;padding-top:2px;flex-shrink:0;line-height:1">{num}</span>'
                f'<span style="font-size:12px;color:var(--txt);line-height:1.55">{bullet}</span>'
                f'</div>'
            )
        return ''.join(cards)

    elif 'RISK POSTURE' in label.upper():
        # ── Risk Posture: single dense paragraph ──────────────────────────────
        # This section is deliberately compact — it's the opening verdict and
        # reads well as one block.
        return (
            f'<div style="font-size:12.5px;color:var(--txt);line-height:1.6">{body}</div>'
        )

    else:
        # ── Sentence-level rows ────────────────────────────────────────────────
        # Split on sentence-ending punctuation followed by a space and capital letter.
        # This handles board prose well; abbreviations like "USD 4,940M" and KRI
        # codes like "O-02" don't trigger false splits because they aren't followed
        # by ". Capital".
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', body.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return (
                f'<div style="font-size:12.5px;color:var(--txt);line-height:1.6">{body}</div>'
            )

        rows = []
        for idx, sentence in enumerate(sentences):
            # Ensure trailing punctuation
            if sentence and sentence[-1] not in '.!?':
                sentence += '.'
            is_last = (idx == len(sentences) - 1)
            rows.append(
                f'<div style="font-size:12.5px;color:var(--txt);line-height:1.55;'
                f'{"" if is_last else "margin-bottom:0.38rem;padding-bottom:0.38rem;border-bottom:1px solid rgba(0,0,0,0.05);"}'
                f'">{sentence}</div>'
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
