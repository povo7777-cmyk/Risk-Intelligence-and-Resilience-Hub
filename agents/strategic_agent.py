"""
strategic_agent.py — Strategic Risk Agent (Claude Sonnet 4.6)
Owns: S-01 Geopolitical & trade, S-02 AI disruption, S-03 M&A integration
Data: market_intelligence.csv, ma_pipeline.csv
"""

import csv, json, os, sys
from pathlib import Path
from datetime import datetime, timezone
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.risk_writer import update_risk
from schemas.agent_outputs import validate_agent_output

DATA_DIR = Path(__file__).parent.parent / "data"
STORE_PATH = Path(__file__).parent.parent / "api" / "risk_store.json"

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_risk_register() -> dict:
    """Load risk parameters from risk_register.csv. Returns {risk_id: row_dict}."""
    import csv as _csv
    path = DATA_DIR / "risk_register.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["risk_id"]: r for r in _csv.DictReader(f)}


def read_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_csv_latest(filename):
    """Read only the rows belonging to the most recent date in the CSV."""
    rows = read_csv(filename)
    dates = sorted(set(r.get("date", "") for r in rows if r.get("date")), reverse=True)
    if not dates:
        return rows
    return [r for r in rows if r.get("date") == dates[0]]


def eval_kri(value, amber, red, direction="higher_worse"):
    if direction == "higher_worse":
        if value >= red:
            return "breach"
        if value >= amber:
            return "amber"
        return "ok"
    else:
        if value <= red:
            return "breach"
        if value <= amber:
            return "amber"
        return "ok"


def run(kri_data: dict | None = None) -> dict:
    """
    kri_data: pre-computed KRI values from kri_data_layer (when running in graph).
    If None, computes KRI values from CSVs directly (standalone / test mode).
    """
    print(f"\n{'='*60}")
    print("[Strategic Agent] Starting — Claude Sonnet 4.6")
    print(f"{'='*60}")

    # Always read CSVs for narrative context (signal titles, pipeline details)
    signals  = read_csv_latest("market_intelligence.csv")
    pipeline = read_csv_latest("ma_pipeline.csv")

    high_signals = [s for s in signals if s.get("severity") == "high"]
    s01_signals  = [s for s in signals if s.get("risk_id") == "S-01"]
    s02_signals  = [s for s in signals if s.get("risk_id") == "S-02"]
    active_deals = [d for d in pipeline if d.get("stage") not in ["Completed"]]
    pipeline_deals = [d for d in pipeline if d.get("stage") in ["Due-Diligence", "Negotiation"]]
    # active_deals = all non-completed deals (integration + pipeline)
    # pipeline_deals = pre-close stages only (Due-Diligence, Negotiation)
    # pl_count must match active_deals for consistency with the header count shown in the prompt
    s03_synergy  = next((float(d["synergy_delivered_pct"]) for d in pipeline
                         if d["deal_id"] == "ACQ001"), 84.0)

    # All KRI values and statuses come from the data layer — no local computation
    if not (kri_data and kri_data.get("dashboard_kris")):
        raise RuntimeError(
            "Strategic agent requires pre-computed KRI data from kri_data_layer. "
            "Run via the graph, not standalone."
        )
    kri_updates  = kri_data["dashboard_kris"]
    s01_kris     = kri_updates.get("S-01", [])
    s02_kris     = kri_updates.get("S-02", [])
    s03_kri_list = kri_updates.get("S-03", [])
    s01_count    = int(next((k["value"] for k in s01_kris if k["name"] == "geopolitical_signal_count"), len(s01_signals)))
    hs_count     = len(high_signals)  # high_severity_signals KRI removed; use raw count for context
    s02_count    = int(next((k["value"] for k in s02_kris if k["name"] == "competitive_signals"),        len(s02_signals)))
    pl_count     = len(active_deals)   # all non-completed deals — consistent with prompt header
    s03_synergy  = next((k["value"] for k in s03_kri_list if k["name"] == "synergy_delivery_pct"), s03_synergy)
    s01_kri      = next((k["status"] for k in s01_kris     if k["name"] == "geopolitical_signal_count"), "ok")
    s02_kri      = next((k["status"] for k in s02_kris     if k["name"] == "competitive_signals"),        "ok")
    s03_kri      = next((k["status"] for k in s03_kri_list if k["name"] == "synergy_delivery_pct"),       "ok")

    all_kris = [kri for lst in kri_updates.values() for kri in lst]
    breaches = sum(1 for k in all_kris if k["status"] == "breach")
    ambers   = sum(1 for k in all_kris if k["status"] == "amber")

    signals_text  = "\n".join([f"  [{s['severity'].upper()}] {s['title']} ({s['region']})" for s in signals[:6]])
    pipeline_text = "\n".join([f"  {d['deal_id']}: {d['target_name']} — {d['stage']} — USD {d['deal_value_usd_m']}M" for d in active_deals])

    prompt = f"""You are the Strategic Risk Agent for an Enterprise Risk Management system.
Report findings for S-01, S-02, and S-03 based on the data provided below.

DATA INTEGRITY RULE: Your narrative must report only what this data shows.
Do not add residual risk scores, control effectiveness percentages, likelihood or
impact scores, market statistics, or any value not present in the data below.
If you only know a count, state the count. If you know signal titles, cite them.

CURRENT MARKET SIGNALS ({len(signals)} total, {hs_count} high severity):
{signals_text}

M&A PIPELINE ({len(active_deals)} active deals — includes integration-phase and pre-close):
{pipeline_text}

KRI STATUS (computed from source data this run):
- S-01 Geopolitical: {s01_count} signals detected, {hs_count} high-severity, status={s01_kri.upper()}
- S-02 AI Disruption: {s02_count} competitive signals detected, status={s02_kri.upper()}
- S-03 M&A Integration: synergy delivery={s03_synergy}% against target, {pl_count} active deals (includes integration and pre-close stages), status={s03_kri.upper()}

Provide a structured assessment:
1. DOMAIN STATUS (1 sentence overall verdict)
2. S-01 FINDING (2-3 sentences — cite signal titles and count only)
3. S-02 FINDING (2-3 sentences — cite signal titles and count only)
4. S-03 FINDING (2-3 sentences — cite deal names and synergy % only)
5. TOP ACTION (single most important action for each risk)
6. ESCALATION (yes/no — only yes if a KRI status is breach)
Direct language."""

    print("  Calling Claude Sonnet 4.6...")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    narrative = msg.content[0].text
    print(f"  Response: {len(narrative)} chars")

    token_usage = {
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }

    # No truncation — schema max_length removed

    # KRI values already written by kri_data_layer; write only narrative metadata here
    for risk_id in kri_updates:
        try:
            update_risk(risk_id=risk_id, kri_updates={}, new_ctrl=None, new_lv=None,
                        agent_findings=narrative, proposed_actions=[])
        except Exception as e:
            print(f"  Warning: update_risk failed for {risk_id}: {e}")

    timestamp = datetime.now(timezone.utc).isoformat()

    # Interconnections derived from actual data — no hardcoded entity names
    interconnections = []
    if s01_kri in ("breach", "amber"):
        interconnections.append(
            f"S-01+F-01: {s01_count} geopolitical signal(s) detected — "
            "elevated FX exposure warrants review against hedging position"
        )
    if pl_count > 0:
        deal_names_str = ", ".join(d.get("target_name", "") for d in active_deals if d.get("target_name"))
        interconnections.append(
            f"S-03+F-02: {pl_count} active deal(s) "
            + (f"({deal_names_str}) " if deal_names_str else "")
            + "may shift customer revenue concentration"
        )

    result = {
        "domain": "strategic",
        "agent_version": "v1",
        "timestamp": timestamp,
        "risks": ["S-01", "S-02", "S-03"],
        "kri_updates": kri_updates,
        "breach_count": breaches,
        "amber_count": ambers,
        "narrative": narrative,
        "escalation_required": breaches > 0,
        "escalation_reasons": [f"S-01 breach: {s01_count} geopolitical signals"] if breaches > 0 else [],
        "interconnection_flags": interconnections,
        "confidence": "high",
        "proposed_ctrl_changes": {},
        "token_usage": token_usage,
    }
    valid, _ = validate_agent_output("strategic", result)
    print(f"  [Strategic] Complete — {breaches} breach(es), {ambers} amber(s) | Schema: {'valid' if valid else 'INVALID'}")
    return result


if __name__ == "__main__":
    result = run()
    print("\n--- STRATEGIC AGENT OUTPUT ---")
    print(json.dumps(result, indent=2))
