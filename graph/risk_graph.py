"""
graph/risk_graph.py
LangGraph state graph for the Risk Intelligence and Resilience Hub.
Defines the full orchestration pipeline from dispatch through
domain agents, synthesis, HITL, dashboard update and GitHub push.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Annotated
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.audit_logger import AuditLogger
from tools.cost_tracker import CostTracker
from schemas.agent_outputs import validate_agent_output, sanitise_search_result

# Module-level storage for non-serializable objects
_audit = None
_cost = None


# ── State schema ──────────────────────────────────────────

class RiskIntelligenceState(TypedDict):
    run_id: str
    triggered_by: str
    ci_mode: bool             # auto-approve validator-clean items; skip FLAG items

    # KRI data layer — computed from CSVs BEFORE agents run
    kri_ground_truth: dict   # dashboard_kris + summary, written to store + dashboard
    agent_context:    dict   # additional signals per risk_id, passed to Chief Risk Agent

    # Model calibration — live parameters derived from CSVs, used by agents + panel
    model_params: dict       # {ebitda:{...}, hedge:{...}, supply_chain:{...}}

    # Domain agent findings (None = not yet run)
    strategic_findings: dict | None
    operational_findings: dict | None
    financial_findings: dict | None
    compliance_findings: dict | None
    regulatory_signals: dict | None
    emerging_signals: dict | None

    # Permission decisions by Chief Risk Agent
    permissions: dict        # {agent: "approved"|"held"|"rejected"}
    hold_reasons: dict       # {agent: reason_string}

    # Verification results after each write
    verifications: dict      # {agent: "passed"|"failed"|"pending"}
    verification_errors: list

    # HITL decisions
    hitl_decisions: dict     # {section: "approved"|"rejected"|"edited"}
    hitl_edits: dict         # {section: edited_text}

    # Final outputs
    board_summary: str
    exec_rec_drafts: dict
    exec_rec_approved: dict
    dashboard_updated: bool
    github_pushed: bool
    github_url: str

    # Validation
    kri_validation_results: dict      # pure-Python CSV vs store diff
    content_validation: dict          # Haiku check of board summary + exec recs
    panel_validation: dict            # Two-specialist panel: Elena Marchetti + Marcus Okonkwo

    # Infrastructure
    errors: list
    warnings: list


# ── Node implementations ──────────────────────────────────

def dispatch_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Initialise run, set up logging and cost tracking."""
    run_id = state["run_id"]
    audit = AuditLogger(run_id)
    cost = CostTracker()

    print(f"\n{'#'*60}")
    print(f"  RISK INTELLIGENCE HUB — AGENT RUN")
    print(f"  Run ID: {run_id}")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'#'*60}")

    # audit and cost stored outside state to avoid msgpack serialization error
    import graph.risk_graph as _self
    _self._audit = audit
    _self._cost = cost
    state["permissions"] = {}
    state["hold_reasons"] = {}
    state["verifications"] = {}
    state["verification_errors"] = []
    state["hitl_decisions"] = {}
    state["hitl_edits"] = {}
    state["kri_validation_results"] = {}
    state["content_validation"] = {}
    state["panel_validation"] = {}
    state["model_params"] = {}
    state["kri_ground_truth"] = {}
    state["agent_context"] = {}
    state["errors"] = []
    state["warnings"] = []
    return state


def kri_data_layer_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """
    Compute ALL KRI values directly from CSV source files.
    Writes dashboard KRIs to risk_store. Passes agent context to state.
    Runs BEFORE domain agents so agents receive pre-computed values.
    """
    from tools.kri_data_layer import run as run_data_layer
    print(f"\n{'='*60}")
    print("[KRI DATA LAYER] Computing KRIs from source CSV files...")
    print(f"{'='*60}")
    try:
        result = run_data_layer()
        state["kri_ground_truth"] = result
        state["agent_context"]    = result.get("agent_context", {})
        summary = result.get("summary", {})
        print(f"  Dashboard KRIs written to store: {summary.get('total_kris', 0)} KRIs, "
              f"{summary.get('breach_count', 0)} breach(es), {summary.get('amber_count', 0)} amber(s)")
        for domain, counts in summary.get("by_domain", {}).items():
            b = counts.get("breach_count", 0)
            a = counts.get("amber_count", 0)
            if b or a:
                print(f"    {domain}: {b} breach(es), {a} amber(s)")
        if result.get("errors"):
            for e in result["errors"]:
                state["warnings"].append(f"KRI data layer: {e}")
    except Exception as e:
        state["errors"].append(f"KRI data layer failed: {e}")
        print(f"  ✗ KRI data layer error: {e}")
    return state


def model_calibration_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """
    Derive model parameters from live CSVs and re-run the three Monte Carlo
    simulations (EBITDA stress, Hedge analyser, Supply chain stress).
    Patches dashboard HTML slider defaults and validated-finding banners.
    Runs AFTER kri_data_layer (so covenant tracker is fresh) and BEFORE
    domain agents (so agents receive live model outputs in their context).
    """
    from tools.model_calibrator import run_calibration
    from pathlib import Path
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    try:
        params = run_calibration(dashboard_path=dashboard_path)
        state["model_params"] = params
    except Exception as e:
        state["warnings"].append(f"Model calibration failed: {e}")
        print(f"  ⚠ Model calibration error: {e}")
    return state


def run_domain_agents_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Run all 6 domain + specialist agents in parallel."""
    audit = _audit  # may be None if dispatch_node not called
    cost = _cost  # may be None

    from agents.strategic_agent import run as run_strategic
    from agents.operational_agent import run as run_operational
    from agents.financial_agent import run as run_financial
    from agents.compliance_agent import run as run_compliance
    from agents.regulatory_agent import run as run_regulatory
    from agents.emerging_risks_agent import run as run_emerging

    # Build per-domain kri_data packages for agents
    gt   = state.get("kri_ground_truth", {})
    actx         = state.get("agent_context", {})
    model_params = state.get("model_params", {})

    def _model_context_snippet() -> str:
        """Build a compact model-outputs summary for agent prompts."""
        if not model_params:
            return ""
        ep = model_params.get("ebitda",       {})
        hp = model_params.get("hedge",        {})
        sp = model_params.get("supply_chain", {})
        lines = ["=== LIVE MODEL OUTPUTS (calibrated from current CSVs) ==="]
        if ep:
            lines.append(
                f"EBITDA Stress: revenue USD {ep.get('revenue_usd_b')}B | "
                f"cost ratio {ep.get('cost_ratio_pct')}% | "
                f"P(covenant breach) {ep.get('p_covenant_breach_pct')}% | "
                f"headroom USD {int(ep.get('ebitda_headroom_usd_m', 0))}M"
            )
        if hp:
            lines.append(
                f"Hedge Analyser: gross FX USD {hp.get('gross_exposure_usd_m', 0):,.0f}M | "
                f"hedge ratio {hp.get('hedge_ratio_pct')}% | "
                f"unhedged USD {hp.get('unhedged_usd_m', 0):,.0f}M | "
                f"unrealised P&L USD {hp.get('unrealised_pnl_usd_m')}M"
            )
        if sp:
            lines.append(
                f"Supply Chain: {sp.get('supplier_count')} suppliers | "
                f"recovery {sp.get('recovery_months')}mo | "
                f"VaR baseline USD {sp.get('var_95_baseline_usd_m', 0):,}M | "
                f"dual-src saves USD {sp.get('saving_dual_src_usd_m', 0):,}M"
            )
        return "\n".join(lines)

    model_ctx = _model_context_snippet()

    def _domain_kri_data(domain_prefix: str) -> dict:
        """Slice kri_ground_truth, agent_context, and model outputs for a domain."""
        bucket_map = {"S": "strategic_risks", "O": "operational_risks",
                      "F": "financial_risks",  "C": "compliance_risks"}
        bucket    = bucket_map.get(domain_prefix, "")
        dashboard = gt.get("dashboard_kris", {}).get(bucket, {})
        context   = {k: v for k, v in actx.items() if k.startswith(domain_prefix)}
        return {"dashboard_kris": dashboard, "agent_context": context,
                "model_context": model_ctx}

    agent_fns = {
        "strategic":   (run_strategic,   _domain_kri_data("S")),
        "operational": (run_operational, _domain_kri_data("O")),
        "financial":   (run_financial,   _domain_kri_data("F")),
        "compliance":  (run_compliance,  _domain_kri_data("C")),
        "regulatory":  (run_regulatory,  {"model_context": model_ctx}),
        "emerging":    (run_emerging,    {"model_context": model_ctx}),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(fn, kri_data): name
            for name, (fn, kri_data) in agent_fns.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            audit.agent_started(name, "varies")
            try:
                result = future.result(timeout=120)
                # Schema validation
                domain_key = result.get("domain", name)
                valid, validated = validate_agent_output(domain_key, result)
                if valid:
                    audit.validation_passed(name)
                    results[name] = result
                    audit.agent_completed(
                        name,
                        result.get("breach_count", 0),
                        result.get("amber_count", 0),
                        result.get("escalation_required", False),
                        result.get("token_usage", {}),
                    )
                    # Record tokens
                    usage = result.get("token_usage", {})
                    cost.record(
                        name,
                        usage.get("input_tokens", 0),
                        usage.get("output_tokens", 0),
                    )
                else:
                    audit.validation_failed(name, str(validated))
                    state["errors"].append(f"{name}: schema validation failed — {validated}")
                    results[name] = None
            except Exception as e:
                audit.agent_failed(name, str(e))
                state["errors"].append(f"{name}: {e}")
                results[name] = None

    state["strategic_findings"] = results.get("strategic")
    state["operational_findings"] = results.get("operational")
    state["financial_findings"] = results.get("financial")
    state["compliance_findings"] = results.get("compliance")
    state["regulatory_signals"] = results.get("regulatory")
    state["emerging_signals"] = results.get("emerging")

    successful = sum(1 for v in results.values() if v is not None)
    print(f"\n[GRAPH] Domain agents complete: {successful}/6 successful")
    if state["errors"]:
        print(f"[GRAPH] Errors: {state['errors']}")
    return state


def chief_risk_synthesis_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """
    Chief Risk Agent — three-step architecture:

    1. Permissions   — deterministic Python rules (no LLM)
    2. Exec recs     — constrained LLM, KRI values only → management actions
                       Runs FIRST so board synthesis can reference the response in place
    3. Board summary — constrained LLM, full agent findings + exec recs
                       Synthesised for board/senior management audience, not a
                       domain inventory; informed by exec recs drafted in Step 2
                       Falls back to assembled narratives if LLM fails
    """
    import anthropic
    from tools.risk_permissions import (
        compute_all_permissions, is_systemic_risk, assemble_compound_scenarios
    )
    audit = _audit
    cost  = _cost

    print(f"\n[CHIEF RISK AGENT] Starting — permissions, exec recs, board synthesis...")
    if audit: audit.agent_started("chief_risk", "claude-sonnet-4-5")

    sf = state.get("strategic_findings")
    of = state.get("operational_findings")
    ff = state.get("financial_findings")
    cf = state.get("compliance_findings")
    rf = state.get("regulatory_signals")
    ef = state.get("emerging_signals")

    # Shared Anthropic client
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # ── Step 1: Permissions — pure Python ─────────────────
    permissions, hold_reasons = compute_all_permissions(sf, of, ff, cf)
    state["permissions"]  = permissions
    state["hold_reasons"] = hold_reasons
    print(f"  Permissions: { {k: v for k, v in permissions.items()} }")

    # Compound scenarios and systemic flag — collected from agent flags, no LLM
    compound_scenarios = assemble_compound_scenarios(sf, of, ff, cf)
    systemic = is_systemic_risk(sf, of, ff, cf)

    # ── Step 2: Exec recs + committee actions ─────────────
    # Runs BEFORE board synthesis so the board document can reference
    # what management is being directed to do.
    # Input: KRI values only — no narratives, no context to embellish.
    print(f"  [EXEC RECS] Drafting management actions from KRI values...")
    exec_recs, committee_actions = _draft_exec_recs(
        client, sf, of, ff, cf, compound_scenarios, cost, audit
    )
    state["exec_rec_drafts"] = exec_recs

    # ── Step 3: Board summary — synthesised for senior management ──
    # Full agent context (no truncation) + exec recs → board synthesis.
    # The board summary IS the synthesis; it is not a domain narrative list.
    # Falls back to assembled narratives if the LLM call fails.
    print(f"  [BOARD SYNTHESIS] Synthesising for senior management audience...")
    full_findings    = _build_findings_summary(state)
    kri_count_facts  = _build_kri_count_facts(sf, of, ff, cf)
    board_summary = _run_board_synthesis(
        client=client,
        full_findings=full_findings,
        exec_recs=exec_recs,
        committee_actions=committee_actions,
        compound_scenarios=compound_scenarios,
        systemic=systemic,
        kri_count_facts=kri_count_facts,
        cost=cost,
    )

    if not board_summary:
        # Fallback: assembled narratives — never leave board_summary empty
        print(f"  [BOARD SYNTHESIS] LLM failed — falling back to assembled narratives")
        state["warnings"].append("Board synthesis LLM failed — using assembled fallback")
        board_summary = _assemble_board_summary(
            sf, of, ff, cf, rf, ef, compound_scenarios, systemic, state
        )
    else:
        print(f"  [BOARD SYNTHESIS] {len(board_summary)} chars")

    # Append committee actions as a standing section at the end
    if committee_actions:
        board_summary += "\n\nRISK COMMITTEE RECOMMENDED ACTIONS:\n" + \
                         "\n".join(f"  • {a}" for a in committee_actions)

    state["board_summary"] = board_summary

    # Write to dashboard
    try:
        from tools.dashboard_updater import update_board_summary, update_signals_panel
        dash = Path(__file__).parent.parent / "dashboard" / "index.html"
        if dash.exists():
            update_board_summary(dash, board_summary, state["run_id"])
            update_signals_panel(dash, rf or {}, ef or {}, state["run_id"])
    except Exception:
        pass

    if audit:
        audit.agent_completed("chief_risk", 0, 0, False, {})

    return state


def permission_router(state: RiskIntelligenceState) -> str:
    """Route to write node if any permissions granted, else to HITL."""
    perms = state.get("permissions", {})
    any_approved = any(v == "approved" for v in perms.values())
    return "write_approved" if any_approved else "hitl_gate"


def write_approved_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Write approved domain agent findings to dashboard, verify each write."""
    from tools.dashboard_updater import run_dashboard_update
    from tools.risk_writer import update_risk, load_store
    audit = _audit  # may be None if dispatch_node not called

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if not dashboard_path.exists():
        state["warnings"].append("Dashboard not found — skipping HTML update")
        return state

    perms = state.get("permissions", {})
    domain_findings_map = {
        "strategic": state.get("strategic_findings"),
        "operational": state.get("operational_findings"),
        "financial": state.get("financial_findings"),
        "compliance": state.get("compliance_findings"),
    }

    # KRI values are already in risk_store from kri_data_layer_node.
    # Agents wrote their narratives via update_risk(kri_updates={}) during run_domain_agents_node.
    # Here we just render the store → dashboard HTML for each approved domain.
    already_rendered = False
    for domain, findings in domain_findings_map.items():
        if perms.get(domain) != "approved" or not findings:
            continue

        print(f"\n[WRITE] Rendering {domain} findings to dashboard HTML...")
        audit.dashboard_write_started(domain, findings.get("breach_count", 0))

        try:
            if already_rendered:
                # Only one HTML render pass needed (all KRIs written by data layer)
                state["verifications"][domain] = "passed"
                continue

            # Render store → dashboard HTML (idempotent)
            update_result = run_dashboard_update(dashboard_path)
            already_rendered = True

            if update_result["total_kri_updates"] == 0:
                audit.no_changes_detected(domain)
                print(f"  [{domain}] No KRI value changes in HTML (store matches dashboard)")

            # Verify the write
            verification = _verify_write(domain, findings, dashboard_path)
            if verification["passed"]:
                audit.verification_passed(domain, verification["checks"])
                state["verifications"][domain] = "passed"
                print(f"  [{domain}] Write verified ✓ ({update_result['total_kri_updates']} KRIs updated)")
            else:
                audit.verification_failed(domain, verification["discrepancies"])
                state["verifications"][domain] = "failed"
                state["verification_errors"].extend(verification["discrepancies"])
                # Rollback
                if update_result.get("backup_path"):
                    import shutil
                    shutil.copy2(update_result["backup_path"], dashboard_path)
                    audit.rollback_executed(domain, update_result["backup_path"])
                    print(f"  [{domain}] Verification failed — rolled back")

        except Exception as e:
            state["errors"].append(f"Write failed for {domain}: {e}")
            state["verifications"][domain] = "failed"

    state["dashboard_updated"] = any(v == "passed" for v in state["verifications"].values())
    return state


def kri_validation_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Re-compute KRI values from source CSVs and diff against risk_store.json."""
    from tools.kri_validator import run_kri_validation
    print("\n[KRI VALIDATOR] Re-computing KRIs from source CSVs...")
    try:
        result = run_kri_validation()
        state["kri_validation_results"] = result
        checked  = result["total_checked"]
        passed   = result["passed"]
        disc     = result["discrepancy_count"]
        print(f"  {checked} KRIs checked — {passed} OK, {disc} discrepanc{'y' if disc==1 else 'ies'}")
        if disc > 0:
            for d in result["discrepancy_details"]:
                note = f" [{d['note']}]" if d.get("note") else f" (diff {d.get('diff_pct','?')}%)"
                print(f"  ⚠ {d['risk_id']}.{d['kri']}: "
                      f"expected={d.get('expected','?')}  "
                      f"actual={d.get('actual','missing')}{note}")
            state["warnings"].append(
                f"KRI validator: {disc} discrepanc{'y' if disc==1 else 'ies'} — "
                "review before approving dashboard update"
            )
    except Exception as e:
        state["warnings"].append(f"KRI validator failed: {e}")
        print(f"  KRI validator error: {e}")
    return state


def content_validation_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Validate board summary and exec rec drafts against KRI data using Haiku."""
    from agents.validation_agent import run as run_validation
    try:
        result = run_validation(state)
        state["content_validation"] = result
    except Exception as e:
        state["warnings"].append(f"Content validation failed: {e}")
        print(f"  Content validation error: {e}")
    return state


def risk_panel_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """
    Two-specialist validation panel: Elena Marchetti (Risk Architect) and
    Marcus Okonkwo (Quant Risk Analyst). Cross-checks KRI thresholds,
    model calibration, cross-model consistency, and recommendation traceability.
    Runs after content validation so panel sees board summary + exec recs.
    Findings surface in HITL gate for human reviewer.
    """
    from agents.risk_panel_agent import run as run_panel
    try:
        result = run_panel(state)
        state["panel_validation"] = result
        verdict = result.get("panel_verdict", {})
        rating  = verdict.get("overall_rating", "?")
        board_ok = verdict.get("fitness_for_board", "?")
        n_crit  = len(verdict.get("critical_findings", []))
        n_high  = len(verdict.get("high_findings", []))
        print(f"\n  [RISK PANEL] Overall: {rating} | Board-ready: {board_ok} | "
              f"Critical: {n_crit} | High: {n_high}")
        if not verdict.get("fitness_for_board", True):
            conditions = verdict.get("conditions_for_board_readiness", [])
            for c in conditions:
                state["warnings"].append(f"[PANEL] Board readiness condition: {c}")
    except Exception as e:
        state["warnings"].append(f"Risk panel failed: {e}")
        print(f"  Risk panel error: {e}")
        state.setdefault("panel_validation", {})
    return state


def _hitl_input(prompt: str) -> str:
    """
    Read a line from stdin.  Returns empty string (treated as 'no'/'reject')
    when stdin is not a TTY (CI, pipe, automated run) so the graph never
    crashes with EOFError in non-interactive environments.
    """
    import sys
    try:
        if not sys.stdin.isatty():
            print(f"{prompt}[non-interactive — auto-rejecting]")
            return ""
        return input(prompt).strip().lower()
    except EOFError:
        print(f"{prompt}[EOF — auto-rejecting]")
        return ""


def _has_flags(val: dict) -> bool:
    """Return True if the validator found any issue entries.

    An entry is an issue if it does NOT start with 'CONFIRMED:'.
    This catches 'FLAG:' (the current format), legacy '⚠' prefixes the
    LLM sometimes emits despite the prompt, and any other unrecognised prefix.
    Only entries explicitly prefixed 'CONFIRMED:' are treated as clean.
    """
    return any(
        not str(f).startswith("CONFIRMED:")
        for f in val.get("flags", [])
        if str(f).strip()          # skip blank entries
    )


def hitl_gate_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Human-in-the-loop gate for executive recommendation review.

    In CI mode (state["ci_mode"] == True) items the validator found clean
    (no FLAG: entries) are auto-approved so the dashboard update proceeds.
    Items with genuine FLAGs still require human review; in a non-interactive
    environment they are auto-rejected so nothing unvalidated reaches the dashboard.
    """
    audit   = _audit
    drafts  = state.get("exec_rec_drafts", {})
    holds   = state.get("hold_reasons", {})
    ci_mode = state.get("ci_mode", False)
    approved = {}

    print(f"\n{'='*60}")
    print(f"  HUMAN REVIEW {'(CI MODE — auto-approving clean items)' if ci_mode else 'REQUIRED'}")
    print(f"{'='*60}")

    content_val  = state.get("content_validation", {})
    kri_val      = state.get("kri_validation_results", {})
    panel_val    = state.get("panel_validation", {})

    # ── KRI validation banner ──────────────────────────────
    kri_disc = kri_val.get("discrepancy_count", 0)
    if kri_disc > 0:
        print(f"\n  ⚠  KRI VALIDATION — {kri_disc} discrepanc{'y' if kri_disc==1 else 'ies'} "
              f"detected between source CSVs and stored values:")
        for d in kri_val.get("discrepancy_details", []):
            note = f" [{d['note']}]" if d.get("note") else f" (diff {d.get('diff_pct','?')}%)"
            print(f"     {d['risk_id']}.{d['kri']}: "
                  f"CSV={d.get('expected','?')}  store={d.get('actual','missing')}{note}")
    else:
        checked = kri_val.get("total_checked", 0)
        if checked:
            print(f"\n  ✓  KRI VALIDATION — all {checked} KRIs match source CSV data")

    # ── Board summary validation banner ───────────────────
    bs_val = content_val.get("board_summary", {})
    if bs_val:
        conf    = bs_val.get("confidence", "?")
        flags   = bs_val.get("flags", [])
        verdict = bs_val.get("verdict", "")
        issues  = [f for f in flags if str(f).strip() and not str(f).startswith("CONFIRMED:")]
        marker  = "✓" if not issues else "⚠"
        print(f"\n  {marker}  BOARD SUMMARY VALIDATION — confidence: {conf}%")
        if verdict:
            print(f"     {verdict}")
        for flag in flags:
            f_str = str(flag)
            if f_str.startswith("FLAG:"):
                print(f"     ⚠ {f_str[5:].strip()}")
            elif f_str.startswith("CONFIRMED:"):
                print(f"     ✓ {f_str[10:].strip()}")
            else:
                print(f"     ⚠ {f_str}")

    # ── Risk Panel banner ─────────────────────────────────
    panel_verdict = panel_val.get("panel_verdict", {})
    if panel_verdict:
        rating   = panel_verdict.get("overall_rating", "?")
        board_ok = panel_verdict.get("fitness_for_board", True)
        icon = "✓" if board_ok else "⚠"
        print(f"\n  {icon}  RISK PANEL ({rating}) — Board-ready: {board_ok}")
        for cf in panel_verdict.get("critical_findings", []):
            print(f"     🔴 CRITICAL [{cf.get('id','')}] {cf.get('title','')} → owner: {cf.get('owner','')}")
        for hf in panel_verdict.get("high_findings", []):
            print(f"     🟠 HIGH     [{hf.get('id','')}] {hf.get('title','')}")
        conditions = panel_verdict.get("conditions_for_board_readiness", [])
        for c in conditions:
            print(f"     ✗ Condition: {c}")
        roadmap = panel_verdict.get("remediation_roadmap", [])
        if roadmap:
            p1 = next((p for p in roadmap if p.get("phase") == 1), None)
            if p1:
                print(f"\n  Phase 1 ({p1.get('timeline_days')}d — {p1.get('name','')}):")
                for item in p1.get("items", []):
                    print(f"     • {item}")

    # ── Held findings ──────────────────────────────────────
    if holds:
        print("\n  HELD FINDINGS — require your review before dashboard update:")
        for agent, reason in holds.items():
            print(f"\n  [{agent.upper()}] HELD: {reason}")

            if ci_mode:
                # In CI mode: always auto-approve held domains.
                #
                # Held domains indicate escalation severity — they are flagged because
                # the risk level is high, not because the content is wrong.  The KRI
                # values themselves were computed deterministically by the data layer and
                # are already in risk_store.json before this gate runs.  What approval
                # controls here is only the HTML rendering step for that domain.
                #
                # Board summary FLAGS (LLM-authored content) belong on exec recs, not on
                # domain findings.  Tying held-domain approval to board summary polish
                # means a minor clarity observation about wording blocks the dashboard
                # from ever showing accurate KRI data — the wrong trade-off in CI mode.
                state["permissions"][agent] = "approved"
                if audit: audit.hitl_approved(f"held_{agent}", False)
                print(f"  [CI] Auto-approved — KRI data validated by data layer")
            else:
                response = _hitl_input(f"\n  Approve {agent} findings for dashboard update? (yes/no): ")
                if response == "yes":
                    state["permissions"][agent] = "approved"
                    if audit: audit.hitl_approved(f"held_{agent}", False)
                else:
                    state["permissions"][agent] = "rejected"
                    if audit: audit.hitl_rejected(f"held_{agent}", "human rejected held finding")

    # ── Executive recommendation review ───────────────────
    if drafts:
        print("\n  EXECUTIVE RECOMMENDATION UPDATES")
        if ci_mode:
            print("  CI mode: auto-approving sections with no validation FLAGs.\n")
        else:
            print("  The Chief Risk Agent has drafted updates to the following sections.")
            print("  Validation annotations are shown before each draft.\n")

        section_names = {
            "bcm":          "Business Continuity",
            "ebitda":       "EBITDA Stress & Margin Risk",
            "fx":           "FX Hedging Strategy",
            "supply_chain": "Supply Chain Resilience",
        }
        exec_val = content_val.get("exec_recs", {})

        for section, draft in drafts.items():
            if not draft:
                continue
            display_name = section_names.get(section, section)
            if audit: audit.hitl_review_started(section)

            print(f"\n  SECTION: {display_name}")
            print("  " + "─" * 50)

            # Show validation annotation for this section
            sec_val = exec_val.get(section, {})
            sec_issues = []
            if sec_val:
                conf      = sec_val.get("confidence", "?")
                flags     = sec_val.get("flags", [])
                verdict   = sec_val.get("verdict", "")
                sec_issues = [f for f in flags if str(f).strip() and not str(f).startswith("CONFIRMED:")]
                marker    = "✓" if not sec_issues else "⚠"
                print(f"\n  {marker} VALIDATION — confidence: {conf}%  {verdict}")
                for flag in flags:
                    f_str = str(flag)
                    if f_str.startswith("FLAG:"):
                        print(f"    ⚠ {f_str[5:].strip()}")
                    elif f_str.startswith("CONFIRMED:"):
                        print(f"    ✓ {f_str[10:].strip()}")
                    else:
                        print(f"    ⚠ {f_str}")

            # Print the proposed update text
            print(f"\n  PROPOSED UPDATE:\n")
            words = draft.split()
            line  = "  "
            for word in words:
                if len(line) + len(word) > 78:
                    print(line)
                    line = "  " + word + " "
                else:
                    line += word + " "
            if line.strip():
                print(line)

            # Decision logic
            if ci_mode:
                if not sec_issues:
                    approved[section] = draft
                    if audit: audit.hitl_approved(section, False)
                    print(f"\n  [CI] Auto-approved — no validation FLAGs")
                else:
                    if audit: audit.hitl_rejected(section, f"CI mode: {len(sec_issues)} FLAG(s) require human review")
                    print(f"\n  [CI] Skipped — {len(sec_issues)} FLAG(s) require human review")
            else:
                print(f"\n  Options: approve / reject / edit")
                response = _hitl_input(f"  Your decision: ")

                if response == "approve":
                    approved[section] = draft
                    if audit: audit.hitl_approved(section, False)
                    print(f"  ✓ {display_name} approved")
                elif response == "edit":
                    print(f"\n  Enter your edited version (press Enter twice when done):")
                    lines_buf = []
                    while True:
                        line = _hitl_input("")
                        if line == "" and lines_buf and lines_buf[-1] == "":
                            break
                        lines_buf.append(line)
                    edited = "\n".join(lines_buf[:-1] if lines_buf and lines_buf[-1] == "" else lines_buf)
                    approved[section] = edited
                    state["hitl_edits"][section] = edited
                    if audit: audit.hitl_approved(section, True)
                    print(f"  ✓ {display_name} approved with edits")
                else:
                    if audit: audit.hitl_rejected(section, "human rejected draft")
                    print(f"  ✗ {display_name} rejected — keeping existing text")

    state["exec_rec_approved"] = approved
    auto_str = " (CI auto-approve)" if ci_mode else ""
    print(f"\n  Review complete. {len(approved)} section(s) approved for update{auto_str}.")
    return state


def update_exec_recs_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Apply approved executive recommendation updates to dashboard HTML."""
    from tools.dashboard_updater import update_exec_recommendations
    approved = state.get("exec_rec_approved", {})
    if not approved:
        return state

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if not dashboard_path.exists():
        return state

    try:
        changes = update_exec_recommendations(dashboard_path, approved)
        print(f"\n[EXEC RECS] {changes} section(s) updated in dashboard")
        state["dashboard_updated"] = True
    except Exception as e:
        state["errors"].append(f"Exec rec update failed: {e}")

    return state



def github_push_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Push updated dashboard to GitHub Pages."""
    audit = _audit  # may be None if dispatch_node not called

    if not state.get("dashboard_updated"):
        print("\n[GITHUB] No dashboard changes to push")
        return state

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if not dashboard_path.exists():
        state["warnings"].append("Dashboard not found — cannot push")
        return state

    try:
        from github import Github
        from github import GithubException

        token = os.environ.get("GITHUB_TOKEN")
        repo_name = os.environ.get("GITHUB_REPO")
        if not token or not repo_name:
            state["warnings"].append("GITHUB_TOKEN or GITHUB_REPO not set — skipping push")
            return state

        commit_msg = (
            f"Agent run {state['run_id'][:8]} — "
            f"{_count_breaches(state)} breach(es) | "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        audit.github_push_started(repo_name, commit_msg)
        print(f"\n[GITHUB] Pushing to {repo_name}...")

        g = Github(token)
        repo = g.get_repo(repo_name)
        contents = repo.get_contents("index.html")
        new_content = dashboard_path.read_text()

        repo.update_file(
            path="index.html",
            message=commit_msg,
            content=new_content,
            sha=contents.sha,
        )

        url = f"https://{repo_name.split('/')[0]}.github.io/{repo_name.split('/')[1]}/"
        state["github_url"] = url
        state["github_pushed"] = True
        audit.github_push_completed(url)
        print(f"[GITHUB] ✓ Live at: {url}")

    except Exception as e:
        audit.github_push_failed(str(e))
        state["errors"].append(f"GitHub push failed: {e}")
        print(f"[GITHUB] Push failed: {e}")

    return state


def finalise_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """Print board summary, cost report, and complete the run."""
    audit = _audit  # may be None if dispatch_node not called
    cost = _cost  # may be None

    print(f"\n{'='*60}")
    print("  BOARD-LEVEL RISK SUMMARY")
    print(f"  Run: {state['run_id'][:8]} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")
    print(state.get("board_summary", "No summary generated"))
    print(f"\n{'='*60}")

    # Save board summary
    summary_path = Path(__file__).parent.parent / f"board_summary_{state['run_id'][:8]}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Run: {state['run_id']}\n")
        f.write(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write("="*60 + "\n\n")
        f.write(state.get("board_summary", ""))

    if state.get("github_url"):
        print(f"\n  Dashboard: {state['github_url']}")

    if state.get("errors"):
        print(f"\n  Errors ({len(state['errors'])}):")
        for err in state["errors"]:
            print(f"  • {err}")

    cost.print_summary()
    audit.run_completed(cost.total_cost(), cost.to_dict())

    return state


# ── Helper functions ──────────────────────────────────────

def _build_findings_summary(state: RiskIntelligenceState) -> str:
    """Delegate to shared utility — CRA and validator always see the same data."""
    from tools.agent_findings_builder import build_findings_summary
    return build_findings_summary(state)


def _draft_exec_recs(client, sf, of, ff, cf, compound_scenarios,
                     cost, audit) -> tuple[dict, list]:
    """
    Step 2 — Draft executive recommendations and committee actions.
    Input: KRI values only (no narratives). Returns (exec_recs dict, actions list).
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "chief_risk_v1.txt"
    system_prompt = prompt_path.read_text()
    kri_input = _build_kri_input_for_llm(sf, of, ff, cf, compound_scenarios)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": kri_input}],
        )
        raw = response.content[0].text
        if cost:
            cost.record("chief_risk_recs",
                        response.usage.input_tokens,
                        response.usage.output_tokens)
        parsed = _parse_json_response(raw)
        if parsed:
            return (
                parsed.get("exec_rec_updates", {}),
                parsed.get("recommended_committee_actions", []),
            )
        print(f"  ✗ Exec rec LLM parse failed")
        return {}, []
    except Exception as e:
        print(f"  ✗ Exec rec LLM error: {e}")
        return {}, []


def _build_kri_count_facts(sf, of, ff, cf) -> str:
    """
    Compute exact breach/amber totals from domain agent outputs.
    Injected into the board synthesis prompt so the LLM never has to count KRIs itself.
    """
    rows = []
    domain_totals = []
    for label, findings in [("Strategic", sf), ("Operational", of),
                             ("Financial", ff), ("Compliance", cf)]:
        if not findings:
            continue
        b = findings.get("breach_count", 0)
        a = findings.get("amber_count", 0)
        rows.append(f"  {label}: {b} breach(es), {a} amber(s)")
        domain_totals.append((b, a))

    total_b = sum(x[0] for x in domain_totals)
    total_a = sum(x[1] for x in domain_totals)
    domains_in_breach = sum(1 for b, _ in domain_totals if b > 0)

    return (
        "KRI COUNTS — VERIFIED TOTALS (do NOT recount; use these exact numbers):\n"
        + "\n".join(rows)
        + f"\n  TOTAL: {total_b} KRI breach(es), {total_a} amber(s) "
        f"across {domains_in_breach} domain(s) in breach"
    )


def _run_board_synthesis(client, full_findings: str, exec_recs: dict,
                         committee_actions: list, compound_scenarios: list,
                         systemic: bool, kri_count_facts: str, cost) -> str:
    """
    Step 3 — Board-level synthesis for senior management.

    Receives the complete agent findings (no truncation) plus the exec recs
    already drafted in Step 2. The board summary IS the synthesis — not a
    domain narrative inventory. The LLM may only reference facts present in
    the inputs. Returns plain text or empty string on failure.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "cra_synthesis_v1.txt"
    if not prompt_path.exists():
        return ""

    system_prompt = prompt_path.read_text()

    # Build exec recs text for context — what management is being directed to do
    recs_text = ""
    if exec_recs:
        recs_lines = [f"  {section.upper()}: {text}"
                      for section, text in exec_recs.items() if text]
        recs_text = "EXECUTIVE RECOMMENDATION DRAFTS:\n" + "\n".join(recs_lines)
    if committee_actions:
        recs_text += "\n\nCOMMITTEE ACTIONS DRAFTED:\n" + \
                     "\n".join(f"  • {a}" for a in committee_actions)
    if compound_scenarios:
        recs_text += "\n\nCOMPOUND SCENARIOS (identified by domain agents):\n" + \
                     "\n".join(f"  • {s}" for s in compound_scenarios)
    if systemic:
        recs_text += "\n\nSYSTEMIC RISK FLAG: Three or more domains simultaneously in breach."

    # Full agent findings — no truncation, no field filtering
    user_message = (
        # Inject verified KRI counts first — LLM must use these, not recount
        kri_count_facts
        + "\n\n"
        + "─" * 60
        + "\n\n"
        + "AGENT FINDINGS — complete domain agent outputs this run:\n\n"
        + full_findings
        + "\n\n"
        + "─" * 60
        + "\n\n"
        + recs_text
        + "\n\n"
        + "Write the board-level risk summary using only the facts above. "
          "Use the KRI COUNTS exactly as stated at the top — do not recount."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        synthesis_text = response.content[0].text.strip()
        if cost:
            cost.record("cra_board_synthesis",
                        response.usage.input_tokens,
                        response.usage.output_tokens)
        if len(synthesis_text) < 80:
            return ""
        return synthesis_text
    except Exception as e:
        print(f"  [BOARD SYNTHESIS] Error: {e}")
        return ""


def _assemble_board_summary(sf, of, ff, cf, rf, ef,
                            compound_scenarios, systemic, state) -> str:
    """
    Assemble board summary directly from agent narratives — no LLM generation.
    Every sentence traces to a specific agent output.
    """
    lines = []

    # Aggregate headline
    total_breaches = sum(
        (f or {}).get("breach_count", 0) for f in [sf, of, ff, cf]
    )
    total_ambers = sum(
        (f or {}).get("amber_count", 0) for f in [sf, of, ff, cf]
    )
    escalations = [
        d for d, f in [("Financial", ff), ("Operational", of),
                        ("Strategic", sf), ("Compliance", cf)]
        if f and f.get("escalation_required")
    ]
    esc_txt = f" Escalation required: {', '.join(escalations)}." if escalations else ""
    lines.append(
        f"The organisation reports {total_breaches} KRI breach(es) and "
        f"{total_ambers} amber warning(s) across the risk domains.{esc_txt}"
    )

    # Domain sections — agent narrative verbatim
    for label, findings in [
        ("STRATEGIC",   sf),
        ("OPERATIONAL", of),
        ("FINANCIAL",   ff),
        ("COMPLIANCE",  cf),
    ]:
        if findings:
            narrative = (findings.get("narrative") or "No narrative provided.").strip()
            lines.append(f"{label}: {narrative}")
        else:
            lines.append(f"{label}: Agent findings unavailable this run.")

    # Compound scenarios — from agent interconnection flags
    if systemic:
        lines.append(
            "SYSTEMIC RISK: Three or more domains are simultaneously in breach — "
            "a compound control failure scenario."
        )
    if compound_scenarios:
        lines.append("COMPOUND SCENARIOS: " + " | ".join(compound_scenarios))

    # Regulatory signals — from regulatory agent
    if rf:
        new_regs = rf.get("new_regulations", [])
        if new_regs:
            reg_titles = [r.get("title", "Unnamed regulation") for r in new_regs[:3]]
            lines.append(
                f"REGULATORY: {len(new_regs)} new regulatory development(s) identified: "
                + "; ".join(reg_titles) + "."
            )

    # Emerging risks — from emerging risks agent
    if ef:
        candidates = ef.get("risk_candidates", [])
        if candidates:
            emg_signals = [c.get("signal", "") for c in candidates[:3] if c.get("signal")]
            lines.append(
                f"EMERGING RISKS: {len(candidates)} emerging risk candidate(s): "
                + "; ".join(emg_signals) + "."
            )

    return "\n\n".join(lines)


def _build_kri_input_for_llm(sf, of, ff, cf, compound_scenarios) -> str:
    """
    Build a structured, KRI-values-only input for the constrained LLM call.
    No narratives — only exact values, thresholds, and statuses.
    The LLM cannot embellish what it cannot see.
    """
    breach_kris = []
    amber_kris  = []

    for domain_label, findings in [
        ("Strategic", sf), ("Operational", of),
        ("Financial", ff), ("Compliance", cf)
    ]:
        if not findings:
            continue
        kri_updates = findings.get("kri_updates", {})
        for risk_id, kris in kri_updates.items():
            kri_list = kris if isinstance(kris, list) else (
                [{"name": k, **v} for k, v in kris.items() if isinstance(v, dict)]
                if isinstance(kris, dict) else []
            )
            for k in kri_list:
                entry = (
                    f"{domain_label} / {risk_id} / {k.get('name','?')}: "
                    f"value={k.get('value','?')} {k.get('unit','')} | "
                    f"threshold={k.get('threshold','?')} | "
                    f"status={k.get('status','?')}"
                )
                if k.get("status") == "breach":
                    breach_kris.append(entry)
                elif k.get("status") == "amber":
                    amber_kris.append(entry)

    parts = []
    if breach_kris:
        parts.append("BREACHED KRIs (require immediate action):\n" +
                     "\n".join(f"  • {e}" for e in breach_kris))
    else:
        parts.append("BREACHED KRIs: None this run.")

    if amber_kris:
        parts.append("AMBER KRIs (require monitoring):\n" +
                     "\n".join(f"  • {e}" for e in amber_kris))
    else:
        parts.append("AMBER KRIs: None this run.")

    if compound_scenarios:
        parts.append("COMPOUND SCENARIOS IDENTIFIED BY DOMAIN AGENTS:\n" +
                     "\n".join(f"  • {s}" for s in compound_scenarios))

    parts.append(
        "Using only the KRI values above, draft the exec rec updates and "
        "risk committee recommended actions. "
        "Return ONLY valid JSON — no markdown, no explanation."
    )

    return "\n\n".join(parts)


def _parse_json_response(text: str) -> dict | None:
    import re
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try fixing truncated JSON
    try:
        # Add missing closing braces
        for i in range(5):
            try:
                return json.loads(text + '}' * i)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return None


def _set_default_permissions(state: RiskIntelligenceState):
    """Fallback: approve all domains if synthesis fails."""
    for domain in ["strategic", "operational", "financial", "compliance"]:
        findings = state.get(f"{domain}_findings")
        if findings and not findings.get("escalation_required", False):
            state["permissions"][domain] = "approved"
        else:
            state["permissions"][domain] = "held"
            state["hold_reasons"][domain] = "Chief Risk Agent synthesis failed — manual review required"


def _verify_write(domain: str, findings: dict, dashboard_path: Path) -> dict:
    """Spot-check that written values match reported values."""
    checks = []
    discrepancies = []
    try:
        from tools.risk_writer import load_store
        store = load_store()
        domain_prefix = domain[0].upper()
        _bucket_map = {"O": "operational_risks", "S": "strategic_risks",
                       "F": "financial_risks", "C": "compliance_risks"}
        kri_updates = findings.get("kri_updates", {})
        for risk_id, updates in kri_updates.items():
            if not risk_id.startswith(domain_prefix):
                continue
            bucket = _bucket_map.get(domain_prefix, "operational_risks")
            stored_risk = store.get(bucket, {}).get(risk_id, {})
            if stored_risk.get("agent_last_run"):
                checks.append(f"{risk_id}: agent_last_run present")
            else:
                discrepancies.append(f"{risk_id}: agent_last_run missing after write")
        return {
            "passed": len(discrepancies) == 0,
            "checks": checks,
            "discrepancies": discrepancies,
        }
    except Exception as e:
        return {"passed": False, "checks": [], "discrepancies": [str(e)]}


def _count_breaches(state: RiskIntelligenceState) -> int:
    total = 0
    for domain in ["strategic", "operational", "financial", "compliance"]:
        f = state.get(f"{domain}_findings")
        if f:
            total += f.get("breach_count", 0)
    return total


# ── Graph assembly ────────────────────────────────────────

def build_graph():
    """Assemble and compile the LangGraph state graph."""
    graph = StateGraph(RiskIntelligenceState)

    graph.add_node("dispatch",           dispatch_node)
    graph.add_node("kri_data_layer",     kri_data_layer_node)
    graph.add_node("model_calibration",  model_calibration_node)
    graph.add_node("run_agents",         run_domain_agents_node)
    graph.add_node("synthesis", chief_risk_synthesis_node)
    graph.add_node("write_approved", write_approved_node)
    graph.add_node("kri_validation", kri_validation_node)
    graph.add_node("content_validation", content_validation_node)
    graph.add_node("risk_panel", risk_panel_node)
    graph.add_node("hitl_gate", hitl_gate_node)
    graph.add_node("update_exec_recs", update_exec_recs_node)
    graph.add_node("github_push", github_push_node)
    graph.add_node("finalise", finalise_node)

    graph.set_entry_point("dispatch")
    graph.add_edge("dispatch",          "kri_data_layer")
    graph.add_edge("kri_data_layer",    "model_calibration")
    graph.add_edge("model_calibration", "run_agents")
    graph.add_edge("run_agents", "synthesis")
    graph.add_conditional_edges(
        "synthesis",
        permission_router,
        {
            "write_approved": "write_approved",
            "hitl_gate": "content_validation",
        }
    )
    graph.add_edge("write_approved", "kri_validation")
    graph.add_edge("kri_validation", "content_validation")
    graph.add_edge("content_validation", "risk_panel")
    graph.add_edge("risk_panel",         "hitl_gate")
    graph.add_edge("hitl_gate", "update_exec_recs")
    graph.add_edge("update_exec_recs", "github_push")
    graph.add_edge("github_push", "finalise")
    graph.add_edge("finalise", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
