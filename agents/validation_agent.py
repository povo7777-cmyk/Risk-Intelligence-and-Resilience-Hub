"""
agents/validation_agent.py
Validates board summary and executive recommendations against KRI data.
Uses Claude Sonnet 4.6 — annotates rather than blocks.

Called after chief_risk_synthesis and kri_validation, before HITL gate.
"""

import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import anthropic
from tools.agent_findings_builder import build_findings_summary
from tools.json_parser import parse_llm_json

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a Risk Assurance Validator. Your job is to verify that the Chief \
Risk Agent's outputs are grounded in the actual agent findings and KRI data provided, \
and flag anything that cannot be traced to those inputs.

The Chief Risk Officer's role is to professionally interpret what the agents found — not to \
supplement it. Every conclusion must have a thread back to something an agent actually reported.

You are given four tiers of verified input data:
1. KRI GROUND TRUTH — all dashboard KRI values and statuses (deterministic, from source CSVs).
   All risk domains including S-01, S-02, S-03 have dashboard KRIs — validate all of them here.
2. AGENT CONTEXT — supplementary verified metrics not on the dashboard (signal counts, \
   unrealised P&L, investigation flags, whistleblower findings, denied-party matches, etc.).
   This includes list-type data such as deal names, signal titles, and screening results.
3. DOMAIN AGENT NARRATIVES — the full narrative and interconnection flags from each domain agent.
   Board synthesis claims must trace to these narratives or to Tier 1/2 data above.
4. REGULATORY / EMERGING SIGNALS — findings from web-search agents (cited sources where available).

BOARD SUMMARY STRUCTURE:
The board summary is a synthesised document with four sections:
  RISK POSTURE — aggregate KRI picture (breaches, ambers, escalation status)
  KEY RISK DRIVERS — CRO judgment on the most consequential 2-3 findings
  CROSS-DOMAIN CONNECTIONS — where risks in one domain amplify another
  MANAGEMENT RESPONSE — references exec rec drafts; what management is directed to do
Validate each section against the tier data above. MANAGEMENT RESPONSE is grounded if it
references the exec rec drafts provided — do not flag it as ungrounded if it matches them.

WHAT TO FLAG:
1. Ungrounded conclusions: a claim with no traceable basis in any tier above, including \
   both invented numbers AND invented interpretations with no agent narrative source.
2. Unsupported factual claims: specific values, counts, entity names, or events not present \
   in any verified tier.
3. Breach coverage gaps: a domain with breach-level KRIs described as "stable", "manageable", \
   or "within appetite" without qualification; or a breach completely absent from the summary.
4. Recommendation contradictions: a recommendation that directly contradicts KRI severity \
   (e.g., "maintain current hedge ratio" when hedge ratio is in breach).
5. Internal inconsistencies: conflicting claims about the same KRI or domain.
6. KRI status misrepresentation: a KRI whose Tier 1 status is AMBER described as "breach", \
   "in breach", or "breached" in the board narrative; or a BREACH KRI softened to "amber" \
   or "approaching". Check every KRI name mentioned in the content against Tier 1 ground truth. \
   Example: if svp_succession_coverage_pct = 72% → AMBER, flagging it as "breach" is an error.
7. Component-vs-aggregate confusion: citing a product-line or sub-segment value (e.g. \
   Workstations failure rate 0.019%) as if it were the KRI aggregate (0.048%). \
   The KRI value in Tier 1 is always the aggregate — flag any lower component value \
   used as if it were the KRI status determinant.

WHAT NOT TO FLAG:
- Cross-domain synthesis grounded in the findings: connecting O-01 amber + F-03 amber into \
  a compound exposure is valid CRO judgment — both inputs are real.
- Significance and urgency judgments: calling a finding "material" or "structurally significant" \
  is interpretation, not invention — do not flag if the underlying KRI status supports it.
- Board-level framing: ordering, emphasis, or contextualisation for a board audience is \
  professional discretion.
- Recommended actions that go further than any single agent proposed — provided the risk \
  rationale traces to the findings.
- Deal names, signal titles, or named items that appear in Tier 2 agent context or Tier 3 \
  domain narratives — these are sourced data, not fabrications.

CONFIDENCE SCORING:
  Start at 100.
  Deduct 20-25 per conclusion or claim with no traceable basis in any verified tier.
  Deduct 10-15 per unsupported specific factual value (number, count, status, entity name).
  Deduct 5-10 per breach-level KRI completely absent from the content.
  Deduct 5 per recommendation that contradicts KRI severity or per internal inconsistency.
  Do NOT deduct for professional interpretation, synthesis, or conclusions consistent with \
  — even if they go beyond — what the raw data shows.

FLAGS FORMAT — every entry in the "flags" array must start with one of two prefixes:

  "CONFIRMED: <what was checked and found correct>"
    Use this when you verify a claim, value, or interconnection is properly grounded.
    Example: "CONFIRMED: F-01 unhedged exposure USD 4940M (threshold 500M) correctly stated."

  "FLAG: <the specific problem, with KRI name/value>"
    Use this only for genuine issues: ungrounded claims, wrong values, missing breaches,
    internal inconsistencies, or recommendation contradictions.
    Example: "FLAG: Board states 8 breaches but agent data shows 10 — recount required."

Do NOT use any other prefix. Every CONFIRMED entry tells the human reviewer what was
verified. Every FLAG entry tells them exactly what needs attention. This separation
lets the reviewer scan the output at a glance without reading every line.

FLAGS LIMIT: board_summary max 12 flags; each exec_rec section max 6 flags. \
Prioritise FLAGS over CONFIRMED entries — omit lower-priority CONFIRMEDs if near the limit. \
Keep flag text concise (max 25 words each). This limit prevents output truncation.

Return ONLY valid JSON — no markdown, no preamble:
{
  "board_summary": {
    "confidence": <0-100>,
    "flags": ["CONFIRMED: ...", "FLAG: ...", ...],
    "verdict": "<one sentence>"
  },
  "exec_recs": {
    "bcm":          {"confidence": <0-100>, "flags": ["CONFIRMED: ...", ...], "verdict": "..."},
    "ebitda":       {"confidence": <0-100>, "flags": ["CONFIRMED: ...", ...], "verdict": "..."},
    "fx":           {"confidence": <0-100>, "flags": ["CONFIRMED: ...", ...], "verdict": "..."},
    "supply_chain": {"confidence": <0-100>, "flags": ["CONFIRMED: ...", ...], "verdict": "..."}
  }
}"""



def run(state: dict) -> dict:
    """
    Validate board summary and exec rec drafts against the complete agent findings.

    The validator receives exactly the same agent data the CRA used to generate
    the board summary — via the shared build_findings_summary() utility. No manual
    field selection, no curated tiers, no maintained label maps. If the CRA saw it,
    the validator sees it.
    """
    print("\n[VALIDATION AGENT] Validating board summary and exec recs against agent findings...")

    board_summary = state.get("board_summary", "")
    exec_recs     = state.get("exec_rec_drafts", {})

    # Content to validate
    content_sections = []
    if board_summary:
        content_sections.append(f"=== BOARD SUMMARY (to validate) ===\n{board_summary}")
    for section_key, text in exec_recs.items():
        if text:
            content_sections.append(f"=== EXEC REC: {section_key.upper()} (to validate) ===\n{text}")

    if not content_sections:
        print("  Nothing to validate — no board summary or exec recs found")
        return {
            "board_summary": {
                "confidence": 0,
                "flags": ["No content provided to validate"],
                "verdict": "Nothing to validate",
            },
            "exec_recs": {},
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    # Source data — same complete view the CRA used; validator cannot flag what CRA could see
    agent_findings = build_findings_summary(state)

    user_prompt = (
        "=== AGENT FINDINGS (source data — validate the content below against this) ===\n\n"
        + agent_findings
        + "\n\n"
        + "─" * 60
        + "\n\n"
        + "\n\n".join(content_sections)
        + "\n\nValidate the board summary and exec recs above against the agent findings. "
          "Return JSON only."
    )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    token_usage = {"input_tokens": 0, "output_tokens": 0}

    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        token_usage = {
            "input_tokens":  msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        raw = msg.content[0].text.strip()
        result, parse_err = parse_llm_json(raw)
        if result is None:
            raise ValueError(f"JSON parse failed: {parse_err}")
        result["token_usage"] = token_usage

        # ── Normalise flag prefixes ──────────────────────────────────────────
        # The prompt requires CONFIRMED: or FLAG:, but the LLM sometimes emits
        # legacy "⚠ ..." or bare text.  Convert them to FLAG: so downstream
        # logic (CI gate, HITL display) always sees canonical prefixes.
        def _normalise_flags(flags: list) -> list:
            out = []
            for f in flags:
                s = str(f).strip()
                if not s:
                    continue
                if s.startswith("CONFIRMED:") or s.startswith("FLAG:"):
                    out.append(s)
                elif s.startswith("⚠"):
                    # If the LLM explicitly says "no issue" or "grounded, no issue"
                    # after the ⚠, reclassify as CONFIRMED to avoid false-positive FLAGs
                    rest = s[1:].strip()
                    _low = rest.lower()
                    if ("no issue" in _low or "not an issue" in _low
                            or "grounded, no" in _low or "grounded — no" in _low):
                        out.append("CONFIRMED:" + rest)
                    else:
                        out.append("FLAG:" + rest)
                elif s.startswith("WARNING:") or s.startswith("ISSUE:") or s.startswith("ERROR:"):
                    out.append("FLAG: " + s)
                else:
                    # Unknown prefix — treat as FLAG to be safe
                    out.append("FLAG: " + s)
            return out

        # Apply normalisation to all flag lists in the result
        if isinstance(result.get("board_summary"), dict):
            result["board_summary"]["flags"] = _normalise_flags(
                result["board_summary"].get("flags", [])
            )
        for sec_val in result.get("exec_recs", {}).values():
            if isinstance(sec_val, dict):
                sec_val["flags"] = _normalise_flags(sec_val.get("flags", []))

        def _tally(flags: list) -> str:
            # After normalisation all flags are FLAG: or CONFIRMED:
            n_flag      = sum(1 for f in flags if str(f).startswith("FLAG:"))
            n_confirmed = sum(1 for f in flags if str(f).startswith("CONFIRMED:"))
            n_other     = len(flags) - n_flag - n_confirmed  # shouldn't be >0 after normalisation
            parts = []
            if n_flag + n_other: parts.append(f"⚠ {n_flag + n_other} issue(s)")
            if n_confirmed:      parts.append(f"✓ {n_confirmed} confirmed")
            return "  ".join(parts) if parts else "no annotations"

        # Print summary
        bs = result.get("board_summary", {})
        print(f"  Board summary  — confidence: {bs.get('confidence','?')}%  "
              f"{_tally(bs.get('flags', []))}")
        for sec, val in result.get("exec_recs", {}).items():
            print(f"  Exec rec [{sec:12s}] — confidence: {val.get('confidence','?')}%  "
                  f"{_tally(val.get('flags', []))}")
        return result

    except Exception as e:
        print(f"  Validation agent failed: {e} — continuing without validation")
        return {
            "board_summary": {
                "confidence": 0,
                "flags": [f"Validation agent error: {e}"],
                "verdict": "Validation could not complete",
            },
            "exec_recs": {},
            "token_usage": token_usage,
        }
