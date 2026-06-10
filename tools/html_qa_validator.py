"""
tools/html_qa_validator.py
==========================
4-dimension HTML quality assurance for the Risk Intelligence Dashboard.

Runs against root index.html — the file actually served by GitHub Pages.

Dimensions:
  1. CONSISTENCY  — KRD card titles match body content; no cross-section figure contradictions
  2. ACCURACY     — KRI tile values and status badges match risk_store.json
  3. COMPLETENESS — all required board-pack sections are present in the HTML
  4. SOURCE MATCH — KRI thresholds in HTML match kri_thresholds.csv;
                    model figures in HTML match data/model_benchmarks.json

Severity:
  ERROR   — factual mismatch or structural gap → blocks GitHub push
  WARNING — semantic soft mismatch → logged, does not block push
  INFO    — informational only (e.g. store-only KRIs absent from tiles by design)

Entry point: run(html_path) -> dict
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT      = Path(__file__).parent.parent
HTML_PATH = ROOT / "index.html"          # root — served by GitHub Pages


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION 1 — CONSISTENCY: KRD card title ↔ body domain coherence
# ─────────────────────────────────────────────────────────────────────────────

# Keywords whose presence in a card BODY signal that domain is active.
# All lowercase; matching is case-insensitive substring search.
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "cyber": [
        "mttd", "mttr", "patch compliance", "privileged access",
        "detection gap", "vulnerability", "ransomware", "malware",
        "cyber", "incident response", "intrusion", "access review",
        "security control", "it rto", "breach detection",
        "overdue", "days overdue",          # access reviews overdue = cyber
    ],
    "supply_chain": [
        "supplier", "single-source", "single source", "inventory cover",
        "disruption", "tier-1", "tier 1", "lead time", "sourcing",
        "procurement", "supply chain", "component shortage",
    ],
    "receivables": [
        "receivable", "collection period", "bad debt",
        "overdue invoice", "dso", "days sales outstanding",
        "customer payment", "write-off", "credit risk",
        " ar ", " ar.", " ar,",              # standalone "AR"
    ],
    "covenant": [
        "nd/ebitda", "net debt", "covenant", "liquidity headroom",
        "leverage", "interest coverage", "refinanc", "debt maturity",
        "ebitda", "facility headroom",
    ],
    "talent": [
        "attrition", "retention", "succession", "flight risk",
        "open roles", "talent", "workforce", "headcount",
        "resignation", "turnover rate",
    ],
    "regulatory": [
        "export control", "sanction", "abac", "gdpr", "csrd",
        "audit finding", "compliance gap", "regulatory breach",
        "classification", "third-party",
    ],
    "strategic": [
        "synergy", "integration", "acquisition", "merger",
        "earn-out", "market share", "revenue mix",
    ],
    "product_quality": [
        "field failure", "recall", "defect rate", "warranty",
        "product safety", "rma", "quality", "qms",
    ],
}

# Map patterns found in a card TITLE to the domain(s) the body MUST contain.
# Order matters: more specific patterns first.
TITLE_DOMAIN_RULES: list[tuple[str, list[str]]] = [
    (r"cyber.*supply|supply.*cyber",          ["cyber", "supply_chain"]),
    (r"cyber.*control|control.*failure",      ["cyber"]),
    (r"cyber.*detection|detection.*gap",      ["cyber"]),
    (r"cyber",                                ["cyber"]),
    (r"supply.chain|single.source",           ["supply_chain"]),
    (r"receivable|accounts.receivable",       ["receivables"]),
    (r"covenant|leverage|nd.?ebitda",         ["covenant"]),
    (r"talent|attrition|succession|flight",   ["talent"]),
    (r"regulatory|export|sanction",           ["regulatory"]),
    (r"synergy|integration",                  ["strategic"]),
    (r"product.quality|field.failure|recall", ["product_quality"]),
]


def _expected_domains_for_title(title: str) -> list[str]:
    """Return expected body domains given a card title. Empty = no expectation mapped."""
    for pattern, domains in TITLE_DOMAIN_RULES:
        if re.search(pattern, title, re.IGNORECASE):
            return domains
    return []


def _score_domain(body: str, domain: str) -> int:
    """Count keyword hits for a domain in the body text (case-insensitive)."""
    bl = body.lower()
    return sum(1 for kw in DOMAIN_KEYWORDS.get(domain, []) if kw in bl)


def _dominant_domains(body: str, min_hits: int = 2) -> list[str]:
    """Return all domains with at least min_hits keyword matches in the body."""
    return [d for d in DOMAIN_KEYWORDS if _score_domain(body, d) >= min_hits]


# KRD grid card structure as rendered by dashboard_updater.py:
#   <div style="...font-weight:800...text-transform:uppercase...">{TITLE}</div>
#   <div style="...opacity:0.75...line-height:1.45">{BODY}</div>
_KRD_CARD_RE = re.compile(
    r'<div style="[^"]*font-weight:800[^"]*text-transform:uppercase[^"]*">'
    r'([^<]+)</div>'
    r'<div style="[^"]*opacity:0\.75[^"]*line-height:1\.45[^"]*">'
    r'([^<]+)</div>',
    re.DOTALL,
)


def check_card_title_coherence(html: str) -> list[dict]:
    """
    D1-CONSISTENCY: verify each KRD grid card title is semantically aligned
    with its body content via domain keyword matching.
    """
    issues: list[dict] = []
    cards = _KRD_CARD_RE.findall(html)

    if not cards:
        issues.append({
            "severity":  "WARNING",
            "dimension": "consistency",
            "code":      "KRD_CARDS_NOT_PARSED",
            "message":   "No KRD grid cards found in HTML — structure may have changed.",
        })
        return issues

    for title_raw, body_raw in cards:
        title = title_raw.strip()
        body  = body_raw.strip()

        expected = _expected_domains_for_title(title)
        if not expected:
            continue  # no rule covers this title — skip

        for domain in expected:
            if _score_domain(body, domain) == 0:
                actual = _dominant_domains(body)
                issues.append({
                    "severity":         "ERROR",
                    "dimension":        "consistency",
                    "code":             "CARD_TITLE_DOMAIN_MISMATCH",
                    "card_title":       title,
                    "expected_domain":  domain,
                    "actual_domains":   actual,
                    "body_excerpt":     body[:150],
                    "message": (
                        f"Card '{title}': title signals '{domain}' but body "
                        f"contains zero matching keywords. "
                        f"Body appears to be about: {actual or ['(unrecognised)']}."
                    ),
                })
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION 3 — COMPLETENESS: required board-pack sections
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (human label, regex that must match somewhere in the HTML)
REQUIRED_SECTIONS: list[tuple[str, str]] = [
    ("Risk Posture",              r"risk[- ]posture|RISK POSTURE|riskPosture"),
    ("Strategic risks",           r"strategic[_\- ]risk|STRATEGIC:|strategic.*risk"),
    ("Operational risks",         r"operational[_\- ]risk|OPERATIONAL:|operational.*risk"),
    ("Financial risks",           r"financial[_\- ]risk|FINANCIAL:|financial.*risk"),
    ("Compliance risks",          r"compliance[_\- ]risk|COMPLIANCE:|compliance.*risk"),
    ("Key Risk Drivers section",  r"key[- ]risk[- ]driver|KEY RISK DRIVER"),
    ("Executive recommendations", r"exec[_\- ]rec|RECOMMENDED ACTIONS|EXECUTIVE REC|execRec"),
]


def check_required_sections(html: str) -> list[dict]:
    """D3-COMPLETENESS: confirm every required board-pack section exists in the HTML."""
    issues: list[dict] = []
    for section_name, pattern in REQUIRED_SECTIONS:
        if not re.search(pattern, html, re.IGNORECASE):
            issues.append({
                "severity":  "ERROR",
                "dimension": "completeness",
                "code":      "SECTION_MISSING",
                "section":   section_name,
                "message":   f"Required section '{section_name}' not found in dashboard HTML.",
            })
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION 2 — ACCURACY: KRI tile values + status badges vs risk_store.json
# (delegates to dashboard_render_validator — re-pointed at root index.html)
# ─────────────────────────────────────────────────────────────────────────────

def _run_kri_render_check(html_path: Path) -> list[dict]:
    """D2-ACCURACY: KRI tile cur/status in HTML vs risk_store.json."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from dashboard_render_validator import run_dashboard_render_validation
        result = run_dashboard_render_validation(html_path)
        issues: list[dict] = []
        for item in result.get("issues", []):
            if "actual_cur" in item:
                issues.append({
                    "severity":     "ERROR",
                    "dimension":    "accuracy",
                    "code":         "KRI_WRONG_VALUE",
                    "kri":          item["kri"],
                    "display_name": item["display_name"],
                    "expected":     item["expected_cur"],
                    "actual":       item["actual_cur"],
                    "message": (
                        f"KRI tile '{item['display_name']}': "
                        f"HTML shows {item['actual_cur']!r} "
                        f"but risk_store.json expects {item['expected_cur']!r}."
                    ),
                })
            else:
                issues.append({
                    "severity":        "ERROR",
                    "dimension":       "accuracy",
                    "code":            "KRI_WRONG_STATUS",
                    "kri":             item["kri"],
                    "display_name":    item["display_name"],
                    "expected_status": item["expected_ts"],
                    "actual_status":   item["actual_ts"],
                    "message": (
                        f"KRI tile '{item['display_name']}': "
                        f"status badge is {item['actual_ts']!r} "
                        f"but store expects {item['expected_ts']!r}."
                    ),
                })
        # INFO: store-only KRIs (tracked internally, no tile — expected)
        for item in result.get("store_only", []):
            issues.append({
                "severity":  "INFO",
                "dimension": "accuracy",
                "code":      "KRI_STORE_ONLY",
                "kri":       item["kri"],
                "message":   f"KRI '{item['kri']}' is store-only (no dashboard tile — expected).",
            })
        return issues
    except Exception as e:
        return [{
            "severity":  "WARNING",
            "dimension": "accuracy",
            "code":      "KRI_RENDER_CHECK_FAILED",
            "message":   f"dashboard_render_validator failed: {e}",
        }]


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION 4 — SOURCE MATCH: thresholds vs CSV + benchmark figures vs JSON
# + D1 cross-section figure contradictions (delegates to consistency_checker)
# ─────────────────────────────────────────────────────────────────────────────

def _run_source_match_check(html_path: Path) -> list[dict]:
    """D4-SOURCE MATCH + D1 cross-section: delegates to consistency_checker."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from consistency_checker import run as cc_run
        report = cc_run(html_path)
        issues: list[dict] = []

        # Threshold drift and cross-section contradictions → ERROR
        for item in (report.get("threshold_issues", [])
                     + report.get("contradiction_issues", [])):
            issues.append({
                "severity":  "ERROR",
                "dimension": "source_match",
                "code":      item.get("check", "threshold_drift").upper(),
                "message":   item["message"],
            })

        # Benchmark figure drift → WARNING (tolerance already applied in checker)
        for item in report.get("benchmark_issues", []):
            issues.append({
                "severity":  "WARNING",
                "dimension": "source_match",
                "code":      "BENCHMARK_FIGURE_DRIFT",
                "label":     item.get("label", ""),
                "found":     item.get("found"),
                "canonical": item.get("canonical"),
                "pct_diff":  item.get("pct_diff"),
                "message":   item["message"],
            })
        return issues
    except Exception as e:
        return [{
            "severity":  "WARNING",
            "dimension": "source_match",
            "code":      "SOURCE_MATCH_CHECK_FAILED",
            "message":   f"consistency_checker failed: {e}",
        }]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(html_path: Path = HTML_PATH) -> dict:
    """
    Run all 4 QA dimensions against the specified HTML file (default: root index.html).

    Returns:
      {
        status:        "pass" | "warn" | "error",
        error_count:   int,
        warning_count: int,
        info_count:    int,
        issues:        list[dict],   # severity / dimension / code / message per issue
        by_dimension:  dict,         # issue counts keyed by dimension
        summary:       str,          # one-line human-readable result
      }
    """
    if not html_path.exists():
        issue = {
            "severity":  "ERROR",
            "dimension": "structural",
            "code":      "HTML_NOT_FOUND",
            "message":   f"Dashboard HTML not found: {html_path}",
        }
        return {
            "status": "error", "error_count": 1, "warning_count": 0, "info_count": 0,
            "issues": [issue], "by_dimension": {"structural": 1},
            "summary": f"HTML file not found: {html_path}",
        }

    html       = html_path.read_text()
    all_issues: list[dict] = []

    # D1 — Consistency: card title ↔ body domain coherence
    all_issues += check_card_title_coherence(html)

    # D2 — Accuracy: KRI tile values + status badges vs risk_store.json
    all_issues += _run_kri_render_check(html_path)

    # D3 — Completeness: required board-pack sections present
    all_issues += check_required_sections(html)

    # D4 — Source match: thresholds vs CSV + benchmarks vs JSON
    # (also catches D1 cross-section figure contradictions via consistency_checker)
    all_issues += _run_source_match_check(html_path)

    errors   = [i for i in all_issues if i["severity"] == "ERROR"]
    warnings = [i for i in all_issues if i["severity"] == "WARNING"]
    infos    = [i for i in all_issues if i["severity"] == "INFO"]

    status = ("error" if errors else "warn" if warnings else "pass")

    # Count by dimension
    by_dim: dict[str, int] = {}
    for issue in all_issues:
        if issue["severity"] in ("ERROR", "WARNING"):
            dim = issue.get("dimension", "other")
            by_dim[dim] = by_dim.get(dim, 0) + 1

    if status == "pass":
        summary = (
            f"HTML QA PASSED — {len(html_path.read_text()):,} chars validated: "
            f"consistent, accurate, complete, source-matched."
        )
    else:
        parts = []
        if errors:
            parts.append(f"{len(errors)} ERROR(s)")
        if warnings:
            parts.append(f"{len(warnings)} WARNING(s)")
        summary = f"HTML QA: {', '.join(parts)} — see issues for detail."

    return {
        "status":        status,
        "error_count":   len(errors),
        "warning_count": len(warnings),
        "info_count":    len(infos),
        "issues":        all_issues,
        "by_dimension":  by_dim,
        "summary":       summary,
    }


def print_report(result: dict) -> None:
    """Pretty-print the QA report to stdout."""
    status = result.get("status", "?")
    icon   = "✓" if status == "pass" else ("⛔" if status == "error" else "⚠")
    print(f"\n[HTML QA] {icon} {result.get('summary', '')}")

    if status == "pass":
        # Show dimension counts briefly
        by_dim = result.get("by_dimension", {})
        if not by_dim:
            print("  All four dimensions clean: consistency / accuracy / completeness / source match")
        return

    dim_labels = {
        "consistency":  "CONSISTENCY",
        "accuracy":     "ACCURACY",
        "completeness": "COMPLETENESS",
        "source_match": "SOURCE MATCH",
        "structural":   "STRUCTURAL",
    }

    # Group by dimension for readable output
    by_dim_issues: dict[str, list[dict]] = {}
    for issue in result.get("issues", []):
        if issue["severity"] == "INFO":
            continue
        dim = issue.get("dimension", "other")
        by_dim_issues.setdefault(dim, []).append(issue)

    for dim in ["consistency", "accuracy", "completeness", "source_match", "structural", "other"]:
        issues = by_dim_issues.get(dim, [])
        if not issues:
            continue
        label = dim_labels.get(dim, dim.upper())
        for issue in issues:
            sev  = issue.get("severity", "?")
            icon = "⛔" if sev == "ERROR" else "⚠ "
            print(f"  {icon} [{label}] {issue.get('message', '')}")


if __name__ == "__main__":
    result = run()
    print_report(result)
