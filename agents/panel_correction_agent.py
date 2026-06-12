"""
Panel Correction Agent
======================
Validates each panel finding against source data, then proposes a specific
board-text correction.  Designed to feed the HITL correction gate.

Pipeline position:
  risk_panel → panel_correction_agent → hitl_correction_gate → apply_corrections
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import os

import anthropic

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL    = "claude-haiku-4-5"          # cheap + fast for per-finding calls

# ---------------------------------------------------------------------------
# System prompt for the validator / proposer
# ---------------------------------------------------------------------------

VALIDATOR_SYSTEM = """You are a board pack quality-assurance specialist with access to
source financial and risk data.

For each panel finding you will:
1. VALIDATE — check whether the finding is accurate by comparing the finding's claim
   against the source data excerpts provided.
2. PROPOSE — if the finding is confirmed (in full or in part), propose the minimum
   specific change to the board pack text that would resolve it.

VALIDATION VERDICTS:
  CONFIRMED     — the finding is accurate; the board text has an error or gap that
                  matches the finding's description.
  FALSE_POSITIVE — the finding is inaccurate; the board text is correct and the panel
                   has misread the data.  Explain precisely why.
  PARTIAL       — the finding is partly correct; the board text is partly wrong.
                  Propose a correction for the confirmed part only.

CORRECTION PRINCIPLES:
  • A correction must be the MINIMUM change that resolves the finding.
  • Never change facts that are correct.
  • Never introduce a new figure that is not in the source data.
  • If the finding is a MISSING disclosure, propose adding the missing text.
  • If the finding is a WRONG figure, propose replacing only that figure.
  • If the finding is a MISSING KRI threshold or framework gap, note that the
    correction is a data/config change (not a board-text change) and describe it.
  • current_text_excerpt must be a verbatim quote from the board text (or
    "(not in board text)" if the issue is an omission).

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanation:
{
  "finding_id": "<id from input>",
  "severity":   "<severity from input>",
  "verdict":    "CONFIRMED" | "FALSE_POSITIVE" | "PARTIAL",
  "verdict_rationale": "<1–2 sentences: why the finding is confirmed or rejected>",
  "board_section": "<which section is affected: risk_posture | strategic | operational |
                    financial | compliance | key_risk_drivers | cross_domain |
                    quarter_on_quarter | exec_recs | committee_actions | config_change>",
  "current_text_excerpt": "<verbatim quote from board text, or '(not in board text)'>",
  "proposed_correction":  "<replacement text, or description of config change>",
  "correction_rationale": "<why this is the right fix; cite specific data>",
  "data_sources": ["<source1>", "<source2>"]
}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_text(board_summary: str, section: str) -> str:
    """Extract a named section from the flat board_summary string."""
    markers = {
        "risk_posture":        "RISK POSTURE",
        "strategic":           "STRATEGIC:",
        "operational":         "OPERATIONAL:",
        "financial":           "FINANCIAL:",
        "compliance":          "COMPLIANCE:",
        "key_risk_drivers":    "KEY RISK DRIVERS",
        "cross_domain":        "CROSS-DOMAIN CONNECTIONS",
        "quarter_on_quarter":  "QUARTER-ON-QUARTER",
        "exec_recs":           "EXECUTIVE RECOMMENDATIONS",
        "committee_actions":   "RISK COMMITTEE RECOMMENDED ACTIONS",
    }
    start_token = markers.get(section.lower())
    if not start_token:
        return board_summary[:3000]        # fallback: first 3000 chars

    idx = board_summary.find(start_token)
    if idx == -1:
        return board_summary[:3000]

    # find the next section header to bound the excerpt
    next_idx = len(board_summary)
    for tok in markers.values():
        if tok == start_token:
            continue
        pos = board_summary.find(tok, idx + len(start_token))
        if 0 < pos < next_idx:
            next_idx = pos

    return board_summary[idx:next_idx].strip()


def _source_digest(raw_csvs: dict, model_params: dict) -> str:
    """Compact digest of source data to pass to the validator LLM."""
    lines: list[str] = []

    # KRI thresholds (always useful)
    thr = raw_csvs.get("kri_thresholds", "")
    if thr:
        lines.append("=== KRI THRESHOLDS (kri_thresholds.csv) ===")
        lines.append(thr[:4000])

    # Treasury positions (for FX / hedge findings)
    tr = raw_csvs.get("treasury", "")
    if tr:
        lines.append("=== TREASURY POSITIONS (treasury_positions.csv) ===")
        lines.append(tr[:2000])

    # Supply chain (for O-01 findings)
    sc = raw_csvs.get("supply_chain", "")
    if sc:
        lines.append("=== SUPPLY CHAIN (erp_supply_chain.csv — recent rows) ===")
        lines.append(sc[:2000])

    # Key model parameters
    if model_params:
        hp = model_params.get("hedge_params", {})
        ep = model_params.get("ebitda_params", {})
        sp = model_params.get("supply_params", {})
        lines.append("=== KEY MODEL PARAMETERS ===")
        if hp:
            lines.append(f"Hedge Analyser: {json.dumps({k: v for k, v in hp.items() if not isinstance(v, dict)}, indent=2)[:1500]}")
        if ep:
            lines.append(f"EBITDA Model: {json.dumps({k: v for k, v in ep.items() if not isinstance(v, dict)}, indent=2)[:1500]}")
        if sp:
            lines.append(f"Supply Chain Model: {json.dumps({k: v for k, v in sp.items() if not isinstance(v, dict)}, indent=2)[:1000]}")

    return "\n".join(lines)


def _repair_json(text: str) -> str:
    """Best-effort JSON repair for truncated LLM output."""
    # Remove code fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text)
    # Count open brackets; close if needed
    if text.count('{') > text.count('}'):
        text += '}' * (text.count('{') - text.count('}'))
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    findings: list[dict],
    board_summary: str,
    raw_csvs: dict,
    model_params: dict,
    exec_recs: dict | None = None,
) -> list[dict]:
    """
    Validate each finding and produce correction proposals.

    Returns a list of proposal dicts (one per finding), each containing:
      finding_id, severity, title, verdict, verdict_rationale,
      board_section, current_text_excerpt, proposed_correction,
      correction_rationale, data_sources
    """
    if not findings:
        return []

    client        = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    source_digest = _source_digest(raw_csvs, model_params)
    proposals: list[dict] = []

    print(f"  [Panel Correction] Validating {len(findings)} finding(s)…")

    for i, finding in enumerate(findings, 1):
        fid      = finding.get("finding_id", f"F-{i:02d}")
        severity = finding.get("severity", "?")
        title    = finding.get("title",    "")
        detail   = finding.get("detail",   "")
        category = finding.get("category", "")
        rec      = finding.get("recommendation", "")

        # Best-guess section from category / title keywords
        guess_section = "financial"
        for kw, sec in [
            ("O-0", "operational"), ("O-01", "operational"), ("O-02", "operational"),
            ("O-03", "operational"), ("O-04", "operational"),
            ("S-0", "strategic"),
            ("F-0", "financial"), ("COV", "financial"), ("hedge", "financial"),
            ("C-0", "compliance"), ("export", "compliance"), ("ABAC", "compliance"),
            ("KRI", "risk_posture"), ("coverage_gap", "risk_posture"),
            ("exec_rec", "exec_recs"), ("committee", "committee_actions"),
        ]:
            if kw.lower() in (fid + title + category + detail).lower():
                guess_section = sec
                break

        section_text = _section_text(board_summary, guess_section)
        # Also include executive recs if relevant
        exec_recs_text = ""
        if exec_recs:
            exec_recs_text = "\n=== EXEC RECOMMENDATIONS ===\n" + json.dumps(exec_recs, indent=2)[:2000]

        prompt = f"""You are reviewing the following panel finding against the board pack and source data.

=== FINDING ===
ID: {fid}
Severity: {severity}
Category: {category}
Title: {title}
Detail: {detail}
Recommendation: {rec}

=== BOARD PACK — {guess_section.upper()} SECTION ===
{section_text[:4000]}
{exec_recs_text}

=== SOURCE DATA ===
{source_digest[:6000]}

Validate this finding and propose a correction. Return JSON only."""

        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=VALIDATOR_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            raw = _repair_json(raw)
            proposal = json.loads(raw)
            # Preserve fields from original finding
            proposal["title"]    = title
            proposal["panelist"] = finding.get("panelist", "")
            proposals.append(proposal)
            verdict  = proposal.get("verdict", "?")
            icon     = "✓" if verdict == "FALSE_POSITIVE" else ("⚠" if verdict == "PARTIAL" else "✗")
            print(f"    {icon} [{severity}] {fid} → {verdict}")

        except Exception as e:
            print(f"    ⚠ [{severity}] {fid} — validation failed: {e}")
            # Passthrough: show finding to human even if validation failed
            proposals.append({
                "finding_id":           fid,
                "severity":             severity,
                "title":                title,
                "panelist":             finding.get("panelist", ""),
                "verdict":              "VALIDATION_FAILED",
                "verdict_rationale":    str(e),
                "board_section":        guess_section,
                "current_text_excerpt": "(validation failed — see finding detail)",
                "proposed_correction":  detail,
                "correction_rationale": rec,
                "data_sources":         [],
            })

    confirmed = sum(1 for p in proposals if p.get("verdict") not in ("FALSE_POSITIVE",))
    fp        = sum(1 for p in proposals if p.get("verdict") == "FALSE_POSITIVE")
    print(f"  [Panel Correction] {confirmed} confirmed/partial, {fp} false positive(s)")
    return proposals
