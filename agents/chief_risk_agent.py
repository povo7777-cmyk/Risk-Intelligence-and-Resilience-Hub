"""
chief_risk_agent.py — Orchestrator for Operational Risk domain
Runs all four domain agents, synthesises findings, updates the dashboard.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.supply_chain_agent import run as run_supply_chain
from agents.cyber_agent import run as run_cyber
from agents.quality_agent import run as run_quality
from agents.talent_agent import run as run_talent
from tools.risk_writer import load_store
from tools.dashboard_updater import run_dashboard_update
from pathlib import Path as P


DOMAIN = "Operational Risk"
DASHBOARD_PATH = P(__file__).parent.parent / "dashboard" / "index.html"


def synthesise_findings(all_results: list[dict], client: anthropic.Anthropic) -> str:
    """
    Chief Risk Agent calls Claude to synthesise all domain findings
    into a board-level operational risk summary.
    """
    total_breaches = sum(r["breach_count"] for r in all_results)
    total_ambers = sum(r["amber_count"] for r in all_results)
    escalations = [r["risk_id"] for r in all_results if r["escalation_required"]]

    summaries = []
    for r in all_results:
        breach_str = f"{r['breach_count']} breach(es), {r['amber_count']} amber(s)"
        summaries.append(
            f"[{r['risk_id']}] {r['risk_name']}\n"
            f"  KRI status: {breach_str}\n"
            f"  Residual: {r['residual_score']} ({r['arithmetic_label']}, expert: {r['expert_label']})\n"
            f"  Escalation required: {r['escalation_required']}\n"
            f"  Agent narrative (abridged):\n  {r['narrative'][:400]}..."
        )

    domain_summary = "\n\n".join(summaries)

    prompt = f"""You are the Chief Risk Agent for the Risk Intelligence and Resilience Hub.
You have just received findings from all four Operational Risk domain agents.
Your role: synthesise into a board-level operational risk summary.

PROFESSIONAL JUDGMENT STANDARD:
Professional judgment is the skill of extracting accurate, insightful conclusions from real
inputs. Fabricating or inventing any detail is a failure of professional judgment, not an
exercise of it. Everything you write must be traceable to the agent findings below.
Do not add specific counts, dates, regulatory article numbers, currency pairs, company names,
causal mechanisms, or residual risk scores that no agent explicitly provided.
Cite findings at the level of detail agents provided — do not add precision that was not there.

OPERATIONAL DOMAIN APPETITE:
Low appetite for supply chain disruption and product quality failures.
Zero tolerance for major safety incidents.
Low tolerance for cyber incidents (MTTD <5 days, patch compliance >95%).

DOMAIN AGENT FINDINGS:
{domain_summary}

AGGREGATE METRICS:
- Total KRI breaches across operational domain: {total_breaches}
- Total KRI amber warnings: {total_ambers}
- Risks requiring escalation: {', '.join(escalations) if escalations else 'None'}

Produce a BOARD-LEVEL OPERATIONAL RISK SUMMARY with exactly these sections:

1. OPERATIONAL DOMAIN STATUS (1 sentence verdict: overall RAG status and primary concern)

2. CRITICAL ISSUES REQUIRING BOARD ATTENTION (only genuine breaches — be specific, cite numbers)

3. CROSS-RISK INTERCONNECTIONS (identify compound risks where two or more risks amplify each other — be specific about which risks and why)

4. CONTROL ENVIRONMENT ASSESSMENT (overall control effectiveness across the domain — what is working, what is failing)

5. RECOMMENDED BOARD ACTIONS (top 3 actions the Board must approve or direct — not management actions)

6. MANAGEMENT ACTIONS IN PROGRESS (summary of what management is already doing or should be doing without Board approval)

Be direct. Board-quality language. Cite specific KRI values. """

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def run_orchestration(dashboard_path: Path | None = None) -> None:
    """
    Main orchestration loop. Runs all domain agents, synthesises, updates dashboard.
    """
    start_time = datetime.now(timezone.utc)
    print(f"\n{'#'*65}")
    print(f"  CHIEF RISK AGENT — {DOMAIN}")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'#'*65}")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # ── Run all four domain agents ──
    print(f"\n[ORCHESTRATOR] Dispatching domain agents...")
    results = []

    # Run sequentially (simpler for pilot — parallel available if needed)
    for agent_fn, label in [
        (run_supply_chain, "Supply Chain"),
        (run_cyber, "Cyber"),
        (run_quality, "Quality"),
        (run_talent, "Talent"),
    ]:
        try:
            result = agent_fn()
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {label} agent failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[ORCHESTRATOR] All domain agents complete.")
    print(f"  Results received: {len(results)}/4")

    # ── Synthesise ──
    print(f"\n[ORCHESTRATOR] Synthesising board-level summary...")
    board_summary = synthesise_findings(results, client)

    # ── Update dashboard ──
    dash_path = dashboard_path or DASHBOARD_PATH
    if dash_path.exists():
        print(f"\n[ORCHESTRATOR] Updating dashboard at {dash_path}...")
        try:
            update_result = run_dashboard_update(dash_path)
            print(f"  Dashboard updated: {update_result['total_kri_updates']} KRI values patched")
            print(f"  Backup saved: {update_result['backup_path']}")
            for change in update_result.get("risks_updated", []):
                print(f"    {change}")
        except Exception as e:
            print(f"  [WARNING] Dashboard update failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[ORCHESTRATOR] Dashboard not found at {dash_path} — skipping HTML update")
        print(f"  Place your index.html at: {dash_path}")

    # ── Print board summary ──
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    total_breaches = sum(r["breach_count"] for r in results)
    total_ambers = sum(r["amber_count"] for r in results)
    escalations = [r["risk_id"] for r in results if r["escalation_required"]]

    print(f"\n{'='*65}")
    print(f"  BOARD-LEVEL OPERATIONAL RISK SUMMARY")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"{'='*65}")
    print(f"\n  AGGREGATE: {total_breaches} KRI breaches | {total_ambers} amber warnings")
    print(f"  ESCALATION REQUIRED: {', '.join(escalations) if escalations else 'None'}")
    print(f"\n{board_summary}")
    print(f"\n{'='*65}")

    # ── Save board summary ──
    output_path = P(__file__).parent.parent / "board_summary.txt"
    with open(output_path, "w") as f:
        f.write(f"BOARD-LEVEL OPERATIONAL RISK SUMMARY\n")
        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"{'='*65}\n\n")
        f.write(board_summary)
        f.write(f"\n\n{'='*65}\n")
        f.write(f"DOMAIN AGENT FINDINGS\n{'='*65}\n")
        for r in results:
            f.write(f"\n[{r['risk_id']}] {r['risk_name']}\n")
            f.write(f"Breaches: {r['breach_count']} | Ambers: {r['amber_count']}\n")
            f.write(f"Residual: {r['residual_score']} ({r['arithmetic_label']}, expert: {r['expert_label']})\n")
            f.write(f"\n{r['narrative']}\n")
            f.write("-"*40 + "\n")

    print(f"\n[ORCHESTRATOR] Board summary saved to: {output_path}")


if __name__ == "__main__":
    # Allow passing dashboard path as argument
    dash = P(sys.argv[1]) if len(sys.argv) > 1 else None
    run_orchestration(dash)
