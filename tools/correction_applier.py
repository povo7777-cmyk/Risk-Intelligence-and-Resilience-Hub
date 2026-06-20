"""
Correction Applier
==================
Applies approved correction proposals to the board_summary string
stored in risk_store.json.

Each approved proposal provides:
  current_text_excerpt  — verbatim text to find in board_summary
  proposed_correction   — replacement text

For config_change proposals (no board_summary change), records the
action in the audit log and returns it for downstream handling.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


API_DIR = Path(__file__).parent.parent / "api"


def _apply_text_replacement(
    board_summary: str,
    current: str,
    proposed: str,
) -> tuple[str, bool]:
    """
    Replace `current` with `proposed` in board_summary.

    Falls back to fuzzy matching if verbatim search fails.
    Returns (updated_text, was_applied).
    """
    if not current or current == "(not in board text)":
        # Omission finding — append to the relevant section or skip
        return board_summary, False

    # 1. Exact match
    if current in board_summary:
        return board_summary.replace(current, proposed, 1), True

    # 2. Normalised whitespace match
    def norm(s: str) -> str:
        return re.sub(r'\s+', ' ', s.strip())

    norm_board   = norm(board_summary)
    norm_current = norm(current)
    if norm_current in norm_board:
        # Rebuild with single normalised replace
        start = norm_board.index(norm_current)
        # Map back to original positions (approximate)
        head = board_summary[:start]
        tail = board_summary[start + len(norm_current):]
        return head + proposed + tail, True

    # 3. First 100 chars as anchor
    anchor = norm(current[:100])
    idx    = norm_board.find(anchor)
    if idx >= 0:
        head = board_summary[:idx]
        tail = board_summary[idx + len(current):]
        return head + proposed + tail, True

    return board_summary, False


def run(
    approved_proposals: list[dict],
    store_path: Path | None = None,
) -> dict[str, Any]:
    """
    Apply approved proposals to board_summary in risk_store.json.

    Returns a summary: {applied, skipped, config_changes, board_summary}
    """
    store_path = store_path or (API_DIR / "risk_store.json")

    with open(store_path) as f:
        store = json.load(f)

    board_summary: str = store.get("board_summary", "")
    original_summary   = board_summary

    applied:        list[str] = []
    skipped:        list[str] = []
    config_changes: list[str] = []

    for prop in approved_proposals:
        fid     = prop.get("finding_id", "?")
        section = prop.get("board_section", "")
        current = prop.get("current_text_excerpt", "")
        proposed = prop.get("proposed_correction", "")

        # Config-change proposals (KRI registration, data fixes)
        if section == "config_change" or current == "(config change — not a board text correction)":
            config_changes.append(f"{fid}: {proposed[:120]}")
            print(f"  [Apply] {fid} → config change noted (no board text edit)")
            continue

        if not proposed or proposed == current:
            skipped.append(fid)
            print(f"  [Apply] {fid} → skipped (no change)")
            continue

        updated, ok = _apply_text_replacement(board_summary, current, proposed)
        if ok:
            board_summary = updated
            applied.append(fid)
            print(f"  [Apply] {fid} → applied ✓")
        else:
            skipped.append(fid)
            print(f"  [Apply] {fid} → skipped (anchor not found in board text)")

    # Write back if anything changed
    if board_summary != original_summary:
        store["board_summary"] = board_summary
        with open(store_path, "w") as f:
            json.dump(store, f, indent=2)
        print(f"  [Apply] {len(applied)} correction(s) written to risk_store.json")
    else:
        print(f"  [Apply] No board text changes.")

    return {
        "applied":        applied,
        "skipped":        skipped,
        "config_changes": config_changes,
        "board_summary":  board_summary,
    }
