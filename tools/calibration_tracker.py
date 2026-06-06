"""
tools/calibration_tracker.py
Tracks panel calibration findings across rib runs.

Each finding is keyed by its panel ID (e.g. "MO-01", "EM-02").
Status labels:
  🆕 NEW  — first time this finding has appeared
  🔁 OPEN — seen in prior runs; underlying issue not yet resolved

A finding moves to "resolved" only when it stops appearing in the panel
report (i.e. the panel no longer flags it). At that point it can be
manually closed via mark_resolved(), or left as stale open.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

TRACKER_PATH = Path(__file__).parent.parent / "api" / "calibration_tracker.json"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_tracker() -> dict:
    """Load tracker from disk. Returns empty structure if file missing or corrupt."""
    if not TRACKER_PATH.exists():
        return {"open": {}, "resolved": {}}
    try:
        data = json.loads(TRACKER_PATH.read_text())
        data.setdefault("open", {})
        data.setdefault("resolved", {})
        return data
    except Exception:
        return {"open": {}, "resolved": {}}


def save_tracker(tracker: dict) -> None:
    """Persist tracker to disk."""
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(tracker, indent=2))


def get_status(tracker: dict, fid: str) -> tuple:
    """
    Return ("NEW", {}) or ("OPEN", entry_dict) for a finding ID.
    Does NOT modify the tracker — call record_seen() separately.
    """
    if fid and fid in tracker["open"]:
        return "OPEN", tracker["open"][fid]
    return "NEW", {}


def record_seen(tracker: dict, finding: dict, run_id: str) -> None:
    """
    Record that this finding appeared in this run.
    Creates a new entry for NEW findings; increments times_seen for OPEN ones.
    Call AFTER get_status() so the displayed count reflects prior runs only.
    """
    fid = finding.get("id", "")
    if not fid:
        return
    today = _today()
    if fid in tracker["open"]:
        entry = tracker["open"][fid]
        entry["last_seen_run"]  = run_id[:8]
        entry["last_seen_date"] = today
        entry["times_seen"]     = entry.get("times_seen", 1) + 1
        # Refresh mutable fields in case panel wording shifted
        entry["title"]    = finding.get("title",    entry.get("title", ""))
        entry["owner"]    = finding.get("owner",    entry.get("owner", ""))
        entry["severity"] = finding.get("severity", entry.get("severity", "high")).upper()
    else:
        tracker["open"][fid] = {
            "title":            finding.get("title", ""),
            "severity":         finding.get("severity", "high").upper(),
            "owner":            finding.get("owner", ""),
            "first_seen_run":   run_id[:8],
            "first_seen_date":  today,
            "last_seen_run":    run_id[:8],
            "last_seen_date":   today,
            "times_seen":       1,
        }


def mark_resolved(tracker: dict, fid: str, run_id: str) -> bool:
    """
    Move a finding from open → resolved (panel stopped flagging it).
    Returns True if the entry existed.
    """
    if fid in tracker["open"]:
        entry = tracker["open"].pop(fid)
        entry["resolved_run"]  = run_id[:8]
        entry["resolved_date"] = _today()
        tracker["resolved"][fid] = entry
        return True
    return False


def open_count(tracker: dict) -> int:
    return len(tracker["open"])


def summary_line(tracker: dict) -> str:
    """One-line human-readable summary for terminal output."""
    n_open     = len(tracker["open"])
    n_resolved = len(tracker["resolved"])
    return f"{n_open} open calibration item(s) tracked | {n_resolved} resolved lifetime"
