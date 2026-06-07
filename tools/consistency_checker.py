"""
tools/consistency_checker.py
Post-update consistency validator. Runs after dashboard_updater patches the HTML.

Checks:
  1. KRI tile thresholds (a: / r: in KRI_DATA chips and risk register) vs kri_thresholds.csv
  2. Quantitative model figures in static HTML text vs data/model_benchmarks.json
     — catches exec rec / risk register / AI prompt text drifting from the canonical values

Returns a structured report dict. Issues are warnings only — they do not block the push
but are printed prominently so the team can act before the next release.
"""

import csv, json, re
from pathlib import Path

ROOT         = Path(__file__).parent.parent
THRESHOLDS   = ROOT / "data" / "kri_thresholds.csv"
BENCHMARKS   = ROOT / "data" / "model_benchmarks.json"
DASHBOARD    = ROOT / "dashboard" / "index.html"


# ── KRI_DATA chip key → CSV kri_name ─────────────────────────────────────────
KRI_DATA_KEY_MAP = {
    "o01_single_source":      "single_source_concentration",
    "o01_inventory":          "inventory_cover_weeks",
    "o01_distress":           "supplier_distress_flags",
    "o01_cyber_assess":       "supplier_cyber_resilience_assess_pct",
    "o02_mttd":               "mttd_days",
    "o02_mttr":               "mttr_days",
    "o02_patch":              "patch_compliance_pct",
    "o02_rto":                "it_rto_hours",
    "o03_field_failure":      "field_failure_rate_pct",
    "o03_recall":             "recall_readiness_score_pct",
    "o04_open_roles":         "critical_open_roles_gt60d",
    "o04_succession":         "svp_succession_coverage_pct",
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

# Threshold value formatters — must match dashboard_updater.THRESHOLD_FORMAT exactly
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
    "synergy_delivery_pct":               lambda v: f"{v:g}%",
    "export_screening_coverage_pct":      lambda v: f"{v:g}%",
    "confirmed_sanctions_violations_ytd": lambda v: str(int(v)),
    "ai_audit_coverage_pct":              lambda v: f"{v:g}%",
    "gdpr_dsr_resolution_rate_pct":       lambda v: f"{v:g}%",
    "third_party_abac_coverage_pct":      lambda v: f"{v:g}%",
    "csrd_scope3_disclosure_pct":         lambda v: f"{v:g}%",
}


# ── Model benchmark checks: (description, regex_pattern, benchmark_key_path, tolerance_pct) ──
# Each entry defines a pattern to search in the HTML, the benchmark JSON path to compare against,
# and a tolerance (% difference) before flagging as an issue.
BENCHMARK_CHECKS = [
    # ── Supply chain: dual-source VaR saving ─────────────────────────────────
    {
        # Plain text: "dual-source saves USD 253M VaR" or "by USD 253M VaR"
        "label":     "Supply chain dual-source VaR saving",
        "pattern":   r"dual.source[^.!?]*?(?:saves?\s+USD|by\s+USD)\s*([\d,]+)M\s*VaR",
        "bench_key": ("supply_chain", "dual_source_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # HTML-tagged in exec rec: saves <strong>USD 253M</strong> VaR
        # Use [^<]*? (lazy, no < allowed) so decimal points in "14.4M/yr" don't break the match
        "label":     "Supply chain dual-source VaR saving (HTML tagged)",
        "pattern":   r"[Dd]ual.source[^<]*?<strong>USD\s*([\d,]+)M</strong>\s*VaR",
        "bench_key": ("supply_chain", "dual_source_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # Risk register "reduces VaR [95%] by [~]USD NM" — no "VaR" after the number.
        # Use .{0,80}? (not [^.!?]*) so decimal points in "14.4M/yr" don't break the match.
        "label":     "Supply chain dual-source VaR saving (reduces by)",
        "pattern":   r"[Dd]ual.source.{0,80}?reduces\s+VaR\s+(?:95%\s+)?by\s+~?USD\s*([\d,]+)M",
        "bench_key": ("supply_chain", "dual_source_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # Risk register: "shows USD 253M VaR improvement"
        "label":     "Supply chain dual-source VaR saving (shows improvement)",
        "pattern":   r"shows\s+USD\s*([\d,]+)M\s+VaR\s+improvement",
        "bench_key": ("supply_chain", "dual_source_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # Risk register: "validated VaR saving of USD 253M"
        "label":     "Supply chain dual-source VaR saving (saving of)",
        "pattern":   r"VaR\s+saving\s+of\s+USD\s*([\d,]+)M",
        "bench_key": ("supply_chain", "dual_source_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # Geo-diversification note: "reduces VaR 95% by ~USD 253M" (not anchored on dual-source)
        "label":     "Supply chain dual-source VaR saving (geo diversification)",
        "pattern":   r"reduces\s+VaR\s+95%\s+by\s+~USD\s*([\d,]+)M",
        "bench_key": ("supply_chain", "dual_source_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # AI advisor prompt: "reduces VaR by ~USD 253M (" — parenthesised ROI follows.
        # Use .{0,80}? so decimal points in "14.4M/yr" don't break the match.
        "label":     "Supply chain dual-source VaR saving (AI prompt)",
        "pattern":   r"[Dd]ual.source.{0,80}?reduces\s+VaR\s+by\s+~USD\s*([\d,]+)M\s*\(",
        "bench_key": ("supply_chain", "dual_source_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    # ── Supply chain: dual-source ROI multiplier ──────────────────────────────
    {
        # Plain text: "a 17× return" — anchor on "a " to prevent greedy capture
        "label":     "Supply chain dual-source ROI multiplier",
        "pattern":   r"a\s+(\d{1,3})[×x×]\s*return",
        "bench_key": ("supply_chain", "dual_source_roi_x"),
        "tolerance": 0.0,
        "is_raw":    True,
    },
    {
        # AI advisor prompt: "(17× ROI)"
        "label":     "Supply chain dual-source ROI multiplier (AI prompt)",
        "pattern":   r"\((\d{1,3})[×x×]\s*ROI\)",
        "bench_key": ("supply_chain", "dual_source_roi_x"),
        "tolerance": 0.0,
        "is_raw":    True,
    },
    # ── Supply chain: baseline VaR 95% ────────────────────────────────────────
    {
        # Match: "Baseline VaR 95%: USD 2,049M" or with HTML tags
        "label":     "Supply chain baseline VaR 95%",
        "pattern":   r"[Bb]aseline\s+VaR\s+95%[^U]{0,30}USD\s*([\d,]+)M",
        "bench_key": ("supply_chain", "baseline_var_95_usd_m"),
        "tolerance": 5.0,
    },
    # ── Supply chain: inventory buffer VaR saving ─────────────────────────────
    {
        # Plain text: "inventory buffer ... saves USD 147M"
        "label":     "Supply chain inventory buffer VaR saving",
        "pattern":   r"[Ii]nventory\s+buffer[^.!?]*?saves?\s+USD\s*([\d,]+)M\b",
        "bench_key": ("supply_chain", "inventory_buffer_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # HTML-tagged: inventory buffer saves <strong>USD 147M</strong>
        # Use [^<]*? so decimal points in "7.7M/yr" don't break the match
        "label":     "Supply chain inventory buffer VaR saving (HTML tagged)",
        "pattern":   r"[Ii]nventory\s+buffer[^<]*?<strong>USD\s*([\d,]+)M</strong>",
        "bench_key": ("supply_chain", "inventory_buffer_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # AI advisor prompt: "reduces VaR by ~USD 147M." — period follows (not paren).
        # Use .{0,80}? so decimal points in "7.7M/yr" don't break the match.
        "label":     "Supply chain inventory buffer VaR saving (AI prompt)",
        "pattern":   r"[Ii]nventory\s+buffer.{0,80}?reduces\s+VaR\s+by\s+~USD\s*([\d,]+)M\.",
        "bench_key": ("supply_chain", "inventory_buffer_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    # ── Supply chain: combined VaR saving ─────────────────────────────────────
    {
        # Plain text: "Both together: saves USD 369M"
        "label":     "Supply chain combined VaR saving",
        "pattern":   r"[Bb]oth\s+together[^.!?]{0,20}saves?\s+USD\s*([\d,]+)M",
        "bench_key": ("supply_chain", "combined_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # HTML-tagged: Both together ... saves <strong>USD 369M</strong>
        "label":     "Supply chain combined VaR saving (HTML tagged)",
        "pattern":   r"[Bb]oth\s+together[^<.!?]*<strong>USD\s*([\d,]+)M</strong>",
        "bench_key": ("supply_chain", "combined_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # AI advisor prompt: "~USD 369M saving"
        "label":     "Supply chain combined VaR saving (AI prompt)",
        "pattern":   r"~USD\s*([\d,]+)M\s+saving",
        "bench_key": ("supply_chain", "combined_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    {
        # Simulator comment: "saves USD 369M VaR combined"
        "label":     "Supply chain combined VaR saving (comment)",
        "pattern":   r"saves\s+USD\s*([\d,]+)M\s+VaR\s+combined",
        "bench_key": ("supply_chain", "combined_var_saving_usd_m"),
        "tolerance": 5.0,
    },
    # ── EBITDA ────────────────────────────────────────────────────────────────
    {
        "label":     "EBITDA covenant breach probability",
        "pattern":   r"covenant\s+breach\s+probability\s*[=:]\s*([\d.]+)%",
        "bench_key": ("ebitda", "covenant_breach_probability_pct"),
        "tolerance": 1.0,
        "is_raw":    True,
    },
]


def _load_thresholds() -> dict:
    if not THRESHOLDS.exists():
        return {}
    out = {}
    with open(THRESHOLDS, newline="") as f:
        for row in csv.DictReader(f):
            out[row["kri_name"]] = {
                "amber":     float(row["amber_threshold"]),
                "breach":    float(row["breach_threshold"]),
                "direction": row["direction"],
            }
    return out


def _load_benchmarks() -> dict:
    if not BENCHMARKS.exists():
        return {}
    with open(BENCHMARKS) as f:
        return json.load(f)


def _get_bench_value(benchmarks: dict, key_path: tuple) -> float | None:
    obj = benchmarks
    for k in key_path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return float(obj) if obj is not None else None


# ── Check 1: KRI tile threshold alignment ────────────────────────────────────

def check_threshold_alignment(html: str, thresholds: dict) -> list[dict]:
    """Compare KRI_DATA chip a:/r: values in HTML against kri_thresholds.csv."""
    issues = []
    for js_key, kri_name in KRI_DATA_KEY_MAP.items():
        if kri_name not in thresholds:
            continue
        t = thresholds[kri_name]
        fmt = THRESHOLD_FORMAT.get(kri_name)
        if not fmt:
            continue
        expected_a = fmt(t["amber"])
        expected_r = fmt(t["breach"])

        pat = re.compile(
            r"'" + re.escape(js_key) + r"':\s*\{cur:'[^']*',a:'([^']*)',r:'([^']*)'"
        )
        m = pat.search(html)
        if not m:
            continue
        html_a, html_r = m.group(1), m.group(2)
        if html_a != expected_a or html_r != expected_r:
            issues.append({
                "severity":  "CRITICAL",
                "check":     "threshold_alignment",
                "kri":       kri_name,
                "js_key":    js_key,
                "html_amber":   html_a,
                "csv_amber":    expected_a,
                "html_breach":  html_r,
                "csv_breach":   expected_r,
                "message": (
                    f"{js_key}: HTML amber={html_a} (CSV={expected_a}), "
                    f"HTML breach={html_r} (CSV={expected_r})"
                ),
            })
    return issues


# ── Check 2: Model benchmark figure consistency ───────────────────────────────

def check_benchmark_figures(html: str, benchmarks: dict) -> list[dict]:
    """
    For each BENCHMARK_CHECK entry, find ALL instances of the pattern in the HTML.
    If any instance differs from the canonical benchmark value beyond tolerance, flag it.
    """
    issues = []
    for chk in BENCHMARK_CHECKS:
        canonical = _get_bench_value(benchmarks, chk["bench_key"])
        if canonical is None:
            continue
        pat = re.compile(chk["pattern"], re.IGNORECASE | re.DOTALL)
        found_values = []
        for m in pat.finditer(html):
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            found_values.append(val)

        if not found_values:
            continue  # pattern not found — not flagged (section may not exist)

        for val in found_values:
            if canonical == 0:
                pct_diff = abs(val - canonical)
            else:
                pct_diff = abs(val - canonical) / canonical * 100
            if pct_diff > chk["tolerance"]:
                issues.append({
                    "severity": "HIGH",
                    "check":    "benchmark_figure",
                    "label":    chk["label"],
                    "found":    val,
                    "canonical": canonical,
                    "pct_diff": round(pct_diff, 1),
                    "message": (
                        f"{chk['label']}: found {val} in HTML but "
                        f"model_benchmarks.json says {canonical} "
                        f"({pct_diff:.1f}% difference)"
                    ),
                })
    return issues


# ── Check 3: Cross-section numeric contradiction ──────────────────────────────

def check_cross_section_contradiction(html: str) -> list[dict]:
    """
    For select high-value figures, confirm the same number is used consistently
    across exec rec text, risk register ra/rr fields, and AI advisor prompts.
    Looks for the same label appearing with different USD values across sections.
    """
    issues = []

    # Pairs: (label, regex returning a numeric group)
    targets = [
        ("dual-source VaR saving",    r"dual.source.{0,80}USD\s*([\d]+)M\s*VaR"),
        ("dual-source ROI",           r"dual.source.{0,80}(\d+)[×x]\s*return"),
        ("supply chain baseline VaR", r"[Bb]aseline VaR 95%.{0,20}USD\s*([\d,]+)M"),
    ]

    for label, pattern in targets:
        pat = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        vals = set()
        for m in pat.finditer(html):
            raw = m.group(1).replace(",", "")
            try:
                vals.add(float(raw))
            except ValueError:
                pass
        if len(vals) > 1:
            issues.append({
                "severity": "CRITICAL",
                "check":    "cross_section_contradiction",
                "label":    label,
                "values":   sorted(vals),
                "message": (
                    f"Contradictory values for '{label}' found in HTML: "
                    f"{sorted(vals)} — all occurrences must use the same figure."
                ),
            })
    return issues


# ── Main entry point ──────────────────────────────────────────────────────────

def run(html_path: Path | None = None) -> dict:
    html_path = html_path or DASHBOARD
    if not html_path.exists():
        return {"error": f"Dashboard not found: {html_path}", "total_issues": 0}

    html       = html_path.read_text()
    thresholds = _load_thresholds()
    benchmarks = _load_benchmarks()

    threshold_issues    = check_threshold_alignment(html, thresholds)
    benchmark_issues    = check_benchmark_figures(html, benchmarks)
    contradiction_issues = check_cross_section_contradiction(html)

    all_issues  = threshold_issues + benchmark_issues + contradiction_issues
    critical    = [i for i in all_issues if i["severity"] == "CRITICAL"]
    high        = [i for i in all_issues if i["severity"] == "HIGH"]

    status = "pass" if not all_issues else ("critical" if critical else "warn")

    return {
        "status":                status,
        "total_issues":          len(all_issues),
        "critical_count":        len(critical),
        "high_count":            len(high),
        "threshold_issues":      threshold_issues,
        "benchmark_issues":      benchmark_issues,
        "contradiction_issues":  contradiction_issues,
    }


def print_report(report: dict) -> None:
    total = report.get("total_issues", 0)
    if total == 0:
        print("  ✓ Consistency check: no issues found")
        return

    crit = report.get("critical_count", 0)
    high = report.get("high_count", 0)
    print(f"\n  {'⛔' if crit else '⚠'} Consistency check: {total} issue(s) — "
          f"{crit} CRITICAL, {high} HIGH")

    for issue in report.get("threshold_issues", []):
        print(f"    [CRITICAL] {issue['message']}")
    for issue in report.get("contradiction_issues", []):
        print(f"    [CRITICAL] {issue['message']}")
    for issue in report.get("benchmark_issues", []):
        print(f"    [HIGH]     {issue['message']}")

    if crit:
        print(f"\n  ⛔ CRITICAL issues above will cause Risk Panel failures.")
        print(f"     Fix in data/kri_thresholds.csv or data/model_benchmarks.json then re-run.")
    print()


if __name__ == "__main__":
    report = run()
    print_report(report)
    if report["total_issues"]:
        import json as _json
        print(_json.dumps(report, indent=2))
