"""
HITL Correction Gate
====================
CLI diff view for panel correction proposals.
Called after panel_correction_agent; before apply_corrections.

Each proposal is displayed as:
  • Finding metadata (severity, id, title)
  • Validation verdict + rationale
  • Colored diff: current text (red) vs proposed correction (green)
  • Prompt: [A]pprove / [R]eject / [E]dit / [S]kip

In CI mode (non-interactive), all CONFIRMED findings are auto-approved
and FALSE_POSITIVES auto-rejected.
"""
from __future__ import annotations

import difflib
import os
import sys
import textwrap
from typing import Any


# ANSI colour codes (disabled on non-TTY)
_TTY  = sys.stdout.isatty()
RED   = "\033[91m" if _TTY else ""
GREEN = "\033[92m" if _TTY else ""
CYAN  = "\033[96m" if _TTY else ""
BOLD  = "\033[1m"  if _TTY else ""
DIM   = "\033[2m"  if _TTY else ""
RESET = "\033[0m"  if _TTY else ""

SEVERITY_COLOUR = {
    "CRITICAL": "\033[91m" if _TTY else "",   # red
    "HIGH":     "\033[93m" if _TTY else "",   # yellow
    "MEDIUM":   "\033[94m" if _TTY else "",   # blue
    "LOW":      "\033[2m"  if _TTY else "",   # dim
}


def _wrap(text: str, width: int = 100, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _show_diff(current: str, proposed: str) -> None:
    """Print a unified-style diff of current vs proposed."""
    cur_lines  = current.splitlines(keepends=True)  if current  else ["(empty)\n"]
    prop_lines = proposed.splitlines(keepends=True) if proposed else ["(empty)\n"]

    diff = list(difflib.unified_diff(
        cur_lines, prop_lines,
        fromfile="CURRENT", tofile="PROPOSED", lineterm=""
    ))

    if not diff:
        print(f"  {DIM}(no textual difference){RESET}")
        return

    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            print(f"{BOLD}{line}{RESET}")
        elif line.startswith("-"):
            print(f"{RED}{line}{RESET}")
        elif line.startswith("+"):
            print(f"{GREEN}{line}{RESET}")
        elif line.startswith("@@"):
            print(f"{CYAN}{line}{RESET}")
        else:
            print(f"  {line}")


def _prompt_user(prompt_str: str) -> str:
    """Read a single keypress (or line in dumb terminal)."""
    try:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print(ch)          # echo
        return ch
    except Exception:
        # Fallback for non-TTY
        return input(prompt_str).strip().lower()[:1]


def run(
    proposals: list[dict],
    ci_mode: bool = False,
) -> dict[str, Any]:
    """
    Show each correction proposal to the human and collect decisions.

    Returns:
        {
          "approved":  [proposal, ...],    # human said yes (or CI auto-approved)
          "rejected":  [proposal, ...],    # human said no (or CI auto-rejected FP)
          "edited":    [proposal, ...],    # human provided custom text
          "skipped":   [proposal, ...],    # deferred
          "decisions": { finding_id: "approved"|"rejected"|"edited"|"skipped" }
        }
    """
    if not proposals:
        return {"approved": [], "rejected": [], "edited": [], "skipped": [],
                "decisions": {}}

    approved:  list[dict] = []
    rejected:  list[dict] = []
    edited:    list[dict] = []
    skipped:   list[dict] = []
    decisions: dict[str, str] = {}

    total = len(proposals)

    print(f"\n{'═'*72}")
    print(f"{BOLD}  PANEL CORRECTION REVIEW{RESET} — {total} finding(s)")
    if ci_mode:
        print(f"  {DIM}CI mode: CONFIRMED → auto-approve · FALSE_POSITIVE → auto-reject{RESET}")
    print(f"{'═'*72}")

    for idx, prop in enumerate(proposals, 1):
        fid       = prop.get("finding_id", f"?-{idx:02d}")
        severity  = prop.get("severity", "?").upper()
        title     = prop.get("title", "")
        panelist  = prop.get("panelist", "")
        verdict   = prop.get("verdict", "?").upper()
        rationale = prop.get("verdict_rationale", "")
        section   = prop.get("board_section", "?")
        current   = prop.get("current_text_excerpt", "")
        proposed  = prop.get("proposed_correction", "")
        corr_rat  = prop.get("correction_rationale", "")
        sources   = prop.get("data_sources", [])

        sev_col = SEVERITY_COLOUR.get(severity, "")

        print(f"\n{'─'*72}")
        print(f"{BOLD}[{idx}/{total}]{RESET}  "
              f"{sev_col}{BOLD}[{severity}]{RESET}  "
              f"{BOLD}{fid}{RESET} — {title}")
        if panelist:
            print(f"  {DIM}Panelist: {panelist}{RESET}")
        print(f"  Section : {section}")

        # Verdict banner
        if verdict == "FALSE_POSITIVE":
            print(f"  {GREEN}Verdict : FALSE POSITIVE{RESET}")
        elif verdict == "CONFIRMED":
            print(f"  {RED}Verdict : CONFIRMED{RESET}")
        elif verdict == "PARTIAL":
            print(f"  {CYAN}Verdict : PARTIAL{RESET}")
        else:
            print(f"  Verdict : {verdict}")
        if rationale:
            print(_wrap(rationale, indent="  "))

        # Diff
        print(f"\n  {BOLD}── DIFF ──────────────────────────────────────────────{RESET}")
        _show_diff(current, proposed)

        # Rationale + sources
        if corr_rat:
            print(f"\n  {DIM}Rationale: {corr_rat[:200]}{RESET}")
        if sources:
            print(f"  {DIM}Sources:   {', '.join(sources[:3])}{RESET}")

        # CI auto-decision
        if ci_mode:
            if verdict == "FALSE_POSITIVE":
                decision = "rejected"
                print(f"  {DIM}[CI] Auto-rejected (false positive){RESET}")
            elif verdict in ("CONFIRMED", "PARTIAL", "VALIDATION_FAILED"):
                decision = "approved"
                print(f"  {DIM}[CI] Auto-approved{RESET}")
            else:
                decision = "skipped"
                print(f"  {DIM}[CI] Skipped{RESET}")
        else:
            # Interactive
            print(f"\n  {BOLD}[A]pprove  [R]eject  [E]dit  [S]kip{RESET}  ({idx}/{total}): ", end="", flush=True)
            ch = _prompt_user("")
            if ch == "a":
                decision = "approved"
            elif ch == "r":
                decision = "rejected"
            elif ch == "e":
                print(f"  Enter corrected text (blank line to finish):")
                lines: list[str] = []
                while True:
                    line = input("  > ")
                    if line == "":
                        break
                    lines.append(line)
                prop = dict(prop)
                prop["proposed_correction"] = "\n".join(lines)
                decision = "edited"
            else:
                decision = "skipped"
            print(f"  → {decision.upper()}")

        decisions[fid] = decision
        if decision == "approved":
            approved.append(prop)
        elif decision == "rejected":
            rejected.append(prop)
        elif decision == "edited":
            edited.append(prop)
        else:
            skipped.append(prop)

    print(f"\n{'═'*72}")
    print(f"  REVIEW COMPLETE — "
          f"{len(approved)} approved · {len(rejected)} rejected · "
          f"{len(edited)} edited · {len(skipped)} skipped")
    print(f"{'═'*72}\n")

    return {
        "approved":  approved,
        "rejected":  rejected,
        "edited":    edited,
        "skipped":   skipped,
        "decisions": decisions,
    }
