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
    qoq_deltas:       dict   # quarter-on-quarter KRI movements (current vs prior period)

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
    panel_remediation: dict           # Auto-fix results: {auto_fixed, calibration_findings, blocked}
    board_summary_corrections: dict   # {applied, flags, reason} — Loop 1 feedback
    panel_action_items: list          # calibration findings user elected to add to board actions — Loop 2

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
    state["panel_remediation"] = {}
    state["board_summary_corrections"] = {}
    state["panel_action_items"] = []
    state["model_params"] = {}
    state["kri_ground_truth"] = {}
    state["agent_context"] = {}
    state["qoq_deltas"] = {}
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
        state["qoq_deltas"]       = result.get("qoq_deltas", {})
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
        """Build a compact model-outputs summary for agent prompts.
        Labels use plain business language — no statistical notation — so agents
        and the CRA write in terms board directors can act on directly."""
        if not model_params:
            return ""
        ep = model_params.get("ebitda",       {})
        hp = model_params.get("hedge",        {})
        sp = model_params.get("supply_chain", {})
        lines = ["=== LIVE MODEL OUTPUTS (calibrated from current CSVs) ===",
                 "NOTE: Express these as business consequences in your output — "
                 "do not reproduce statistical notation (no percentile labels, "
                 "no probability notation, no simulation parameters)."]
        if ep:
            p_breach = ep.get('p_covenant_breach_pct', 0)
            # Translate probability to plain-language severity
            if p_breach >= 60:
                breach_language = f"likely to breach covenant ({p_breach}% probability — majority of stress scenarios)"
            elif p_breach >= 50:
                breach_language = f"more likely than not to breach covenant ({p_breach}% — odds are against the company)"
            elif p_breach >= 35:
                breach_language = f"material covenant breach risk ({p_breach}% — roughly 1-in-3 scenarios)"
            else:
                breach_language = f"elevated covenant breach risk ({p_breach}% — watchlist)"
            lines.append(
                f"EBITDA / Covenant: revenue USD {ep.get('revenue_usd_b')}B | "
                f"cost ratio {ep.get('cost_ratio_pct')}% | "
                f"covenant breach assessment: {breach_language} | "
                f"headroom to covenant floor: USD {int(ep.get('ebitda_headroom_usd_m', 0))}M"
            )
        if hp:
            lines.append(
                f"FX Exposure: total currency exposure USD {hp.get('gross_exposure_usd_m', 0):,.0f}M | "
                f"currently hedged: {hp.get('hedge_ratio_pct')}% | "
                f"unhedged (at risk): USD {hp.get('unhedged_usd_m', 0):,.0f}M | "
                f"unrealised currency loss: USD {hp.get('unrealised_pnl_usd_m')}M"
            )
        if sp:
            var_m = sp.get('var_95_baseline_usd_m', 0)
            save_m = sp.get('saving_dual_src_usd_m', 0)
            inv_m  = sp.get('saving_inv_buffer_usd_m', 0)
            cost_m = sp.get('urgent_dual_cost_usd_m', 0)
            lines.append(
                f"Supply Chain: {sp.get('supplier_count')} critical suppliers | "
                f"worst-case disruption cost (stress scenario): USD {var_m:,}M | "
                f"qualifying alternative suppliers would reduce that exposure by USD {save_m:,}M "
                f"at a cost of USD {cost_m}M/yr | "
                f"inventory buffer saves additional USD {inv_m:,}M"
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
    # Architecture: RISK POSTURE is generated deterministically (no LLM);
    # KEY RISK DRIVERS, CROSS-DOMAIN CONNECTIONS, and QUARTER-ON-QUARTER
    # MOVEMENT are LLM-synthesised from agent findings + exec recs.
    # This eliminates the primary hallucination surface (wrong counts, status upgrades).
    print(f"  [BOARD SYNTHESIS] Synthesising for senior management audience...")
    full_findings      = _build_findings_summary(state)
    kri_count_facts    = _build_kri_count_facts(sf, of, ff, cf)
    kri_gt             = state.get("kri_ground_truth", {})
    kri_status_truth   = _build_kri_status_ground_truth(kri_gt)
    qoq_deltas         = state.get("qoq_deltas", {})
    qoq_fact_block     = _build_qoq_fact_block(qoq_deltas)
    risk_posture_facts = _build_risk_posture_facts(
        sf, of, ff, cf, compound_scenarios, systemic
    )
    risk_posture_text  = _render_risk_posture(client, risk_posture_facts, cost)
    synthesis_sections = _run_board_synthesis(
        client=client,
        full_findings=full_findings,
        exec_recs=exec_recs,
        committee_actions=committee_actions,
        compound_scenarios=compound_scenarios,
        systemic=systemic,
        kri_count_facts=kri_count_facts,
        kri_status_truth=kri_status_truth,
        qoq_fact_block=qoq_fact_block,
        cost=cost,
    )

    if not synthesis_sections:
        # Fallback: assembled narratives — never leave board_summary empty
        print(f"  [BOARD SYNTHESIS] LLM failed — falling back to assembled narratives")
        state["warnings"].append("Board synthesis LLM failed — using assembled fallback")
        board_summary = _assemble_board_summary(
            sf, of, ff, cf, rf, ef, compound_scenarios, systemic, state
        )
    else:
        # Deterministic RISK POSTURE + LLM synthesis sections
        board_summary = risk_posture_text + "\n\n" + synthesis_sections
        print(f"  [BOARD SYNTHESIS] {len(board_summary)} chars "
              f"(RISK POSTURE: deterministic, synthesis: LLM)")

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


def panel_remediation_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """
    Tier-1 auto-remediation: runs immediately after the Risk Panel.

    STRUCTURAL findings (threshold drift, figure contradictions) are fixed
    automatically by re-running the threshold sync and consistency checker.
    No human involvement required — these are deterministic mechanical fixes.

    CALIBRATION findings (model assumptions, threshold level policy, missing
    KRI coverage) cannot be auto-resolved and are passed to the HITL gate
    for human review with clear recommendations from the panel.

    After remediation, a final consistency check gates the GitHub push:
    if CRITICAL structural issues survive remediation the push is blocked and
    the run is flagged for manual intervention.
    """
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    remediation = {
        "auto_fixed":            [],   # structural fixes applied this run
        "calibration_findings":  [],   # panel findings requiring human judgment
        "blocked":               False,
        "block_reasons":         [],
        "consistency_status":    "skipped",
    }

    # ── Step 1: Classify panel findings ──────────────────────────────────────
    panel_verdict = state.get("panel_validation", {}).get("panel_verdict", {})
    all_findings  = (
        panel_verdict.get("critical_findings", []) +
        panel_verdict.get("high_findings", [])
    )

    # Structural finding keywords — issues our tooling can detect and fix
    STRUCTURAL_KEYWORDS = {
        "threshold", "diverge", "mismatch", "csv", "html", "tile",
        "contradiction", "inconsisten", "figure", "var saving", "663",
        "benchmark", "status badge", "amber threshold", "breach threshold",
    }

    for f in all_findings:
        title  = (f.get("title",  "") + " " + f.get("detail", "")).lower()
        is_structural = any(kw in title for kw in STRUCTURAL_KEYWORDS)
        if not is_structural:
            remediation["calibration_findings"].append(f)
        # structural findings get processed below — no need to store separately

    calibration_count = len(remediation["calibration_findings"])
    structural_count  = len(all_findings) - calibration_count
    print(f"\n[PANEL REMEDIATION] {len(all_findings)} finding(s): "
          f"{structural_count} structural → auto-fix | "
          f"{calibration_count} calibration → HITL")

    # ── Step 2: Re-run threshold sync ────────────────────────────────────────
    # This catches any threshold drift the panel found, even if consistency
    # checker already ran in run_dashboard_update (belt-and-suspenders).
    if dashboard_path.exists():
        try:
            from tools.dashboard_updater import _load_thresholds, _sync_kri_thresholds_to_html
            html       = dashboard_path.read_text()
            thresholds = _load_thresholds()
            if thresholds:
                updated_html, sync_changes = _sync_kri_thresholds_to_html(html, thresholds)
                if sync_changes:
                    dashboard_path.write_text(updated_html)
                    remediation["auto_fixed"].extend(
                        [f"threshold_sync:{c}" for c in sync_changes]
                    )
                    print(f"  Threshold sync applied: {len(sync_changes)} correction(s)")
                    for c in sync_changes:
                        print(f"    ↳ {c}")
                else:
                    print("  Threshold sync: no drift detected ✓")
        except Exception as e:
            state["warnings"].append(f"Panel remediation threshold sync failed: {e}")
            print(f"  ⚠ Threshold sync failed: {e}")

    # ── Step 3: Run consistency checker ──────────────────────────────────────
    try:
        from tools.consistency_checker import run as cc_run, print_report as cc_print
        cc_report = cc_run(dashboard_path)
        remediation["consistency_status"] = cc_report.get("status", "unknown")
        cc_print(cc_report)

        # Block push if critical structural issues survive remediation
        if cc_report.get("critical_count", 0) > 0:
            remediation["blocked"] = True
            for issue in (
                cc_report.get("threshold_issues", []) +
                cc_report.get("contradiction_issues", [])
            ):
                msg = issue.get("message", str(issue))
                remediation["block_reasons"].append(msg)
                state["warnings"].append(f"[BLOCKED] {msg}")
            print(f"\n  ⛔ PUSH BLOCKED — {cc_report['critical_count']} critical structural "
                  f"issue(s) survived remediation. Fix in kri_thresholds.csv or "
                  f"model_benchmarks.json and re-run.")
            state["dashboard_updated"] = False  # prevents github_push_node from firing

        elif cc_report.get("high_count", 0) > 0:
            # High (non-critical) issues: warn but don't block
            print(f"  ⚠ {cc_report['high_count']} HIGH issue(s) — review model_benchmarks.json")

    except Exception as e:
        state["warnings"].append(f"Consistency checker failed in remediation: {e}")
        print(f"  ⚠ Consistency check failed: {e}")
        remediation["consistency_status"] = "error"

    # ── Step 4: Surface summary ───────────────────────────────────────────────
    if remediation["auto_fixed"]:
        print(f"\n  Auto-fixed {len(remediation['auto_fixed'])} structural issue(s) "
              f"without human intervention ✓")
    if remediation["calibration_findings"]:
        print(f"  {len(remediation['calibration_findings'])} calibration finding(s) "
              f"surfaced to HITL gate for human review →")

    state["panel_remediation"] = remediation
    return state


def board_summary_correction_node(state: RiskIntelligenceState) -> RiskIntelligenceState:
    """
    Loop 1 — Validation feedback: corrects factual errors in the board summary.
    Loop 2 (content accuracy only) — Panel feedback: fixes board summary text errors
    identified as HIGH/CRITICAL content-accuracy findings by the Risk Panel.

    Runs after panel_remediation, before hitl_gate, so the human sees the already-
    corrected summary when they reach the HITL review. Calibration findings (model
    methodology, threshold policy, VaR assumptions) are NOT handled here — those
    stay in hitl_gate as before.

    One LLM call per run, max. HITL approval required before any change is written.
    """
    import anthropic

    board_summary = state.get("board_summary", "")
    content_val   = state.get("content_validation", {})
    panel_val     = state.get("panel_validation", {})
    ci_mode       = state.get("ci_mode", False)

    # ── 1. Validation agent FLAG entries (board_summary section) ─────────────
    validation_flags = [
        str(f)[5:].strip()                          # strip "FLAG:" prefix
        for f in content_val.get("board_summary", {}).get("flags", [])
        if str(f).strip().startswith("FLAG:")
    ]

    # ── 2. Panel content-accuracy findings ───────────────────────────────────
    # "Content accuracy" = panel identified specific text in the board summary
    # that contradicts the underlying data.  Calibration/model findings stay in
    # hitl_gate and are not routed here.
    CONTENT_ACCURACY_KW = {
        "board states", "board claims", "summary states", "summary claims",
        "states '", "claims '", 'states "', 'claims "',
        "recount", "overstate", "understate", "misattribut",
        "component-vs-aggregate", "confusion flagged",
        "incorrect count", "miscounts", "wrong count",
        "primary driver", "sub-component", "aggregate confusion",
    }
    panel_verdict = panel_val.get("panel_verdict", {})
    all_panel = (
        panel_verdict.get("critical_findings", []) +
        panel_verdict.get("high_findings", [])
    )
    panel_content_flags = []
    for f in all_panel:
        combined = (f.get("title", "") + " " + f.get("detail", "")).lower()
        if any(kw in combined for kw in CONTENT_ACCURACY_KW):
            panel_content_flags.append(
                f"{f.get('title', '')} — {f.get('detail', '')}"
            )

    all_flags = validation_flags + panel_content_flags

    if not all_flags:
        print("\n[BOARD SUMMARY CORRECTION] No content flags — skipping ✓")
        state["board_summary_corrections"] = {
            "applied": False, "flags": [], "reason": "no_flags"
        }
        return state

    print(f"\n[BOARD SUMMARY CORRECTION] {len(all_flags)} content flag(s) queued for correction:")
    for i, flag in enumerate(all_flags):
        print(f"  {i+1}. {flag[:120]}{'...' if len(flag) > 120 else ''}")

    # ── 3. LLM targeted correction pass ──────────────────────────────────────
    flags_numbered = "\n".join(f"{i+1}. {flag}" for i, flag in enumerate(all_flags))
    prompt = (
        "You are correcting specific factual errors in a board risk summary.\n"
        "Fix ONLY the numbered errors listed below — change NOTHING else.\n"
        "Do not rephrase, restructure, or improve any other part of the text.\n"
        "Do not add new content. Return ONLY the corrected summary; no preamble.\n\n"
        f"ERRORS TO FIX:\n{flags_numbered}\n\n"
        f"CURRENT BOARD SUMMARY:\n{board_summary}"
    )

    corrected_summary = None
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        corrected_summary = msg.content[0].text.strip()
        if _cost:
            _cost.record(
                "board_summary_correction",
                msg.usage.input_tokens, msg.usage.output_tokens,
            )
        print(f"  Correction draft ready ({len(corrected_summary)} chars)")
    except Exception as e:
        state["warnings"].append(f"Board summary correction LLM failed: {e}")
        print(f"  ⚠ Correction LLM failed: {e} — skipping")
        state["board_summary_corrections"] = {
            "applied": False, "flags": all_flags, "reason": f"llm_error: {e}"
        }
        return state

    # ── 4. HITL approval ──────────────────────────────────────────────────────
    if ci_mode:
        approved = True
        print("  [CI] Auto-approving board summary correction")
    else:
        print(f"\n  Apply corrections to board summary? (yes/no): ", end="", flush=True)
        response = _hitl_input("")
        approved = (response == "yes")

    if approved:
        state["board_summary"] = corrected_summary
        state["board_summary_corrections"] = {
            "applied": True, "flags": all_flags, "reason": "approved"
        }
        try:
            from tools.dashboard_updater import update_board_summary
            dash = Path(__file__).parent.parent / "dashboard" / "index.html"
            if dash.exists():
                update_board_summary(dash, corrected_summary, state["run_id"])
                print("  ✓ Board summary corrected and dashboard re-written")
        except Exception as e:
            state["warnings"].append(f"Board summary correction dashboard write failed: {e}")
            print(f"  ⚠ State corrected but dashboard write failed: {e}")
    else:
        state["board_summary_corrections"] = {
            "applied": False, "flags": all_flags, "reason": "rejected_by_user"
        }
        print("  Board summary correction declined — original retained, flags noted in HITL gate")

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

    # ── Risk Panel + Remediation banner ───────────────────────
    panel_verdict  = panel_val.get("panel_verdict", {})
    remediation    = state.get("panel_remediation", {})
    auto_fixed     = remediation.get("auto_fixed", [])
    calib_findings = remediation.get("calibration_findings", [])
    blocked        = remediation.get("blocked", False)

    if panel_verdict:
        rating   = panel_verdict.get("overall_rating", "?")
        board_ok = panel_verdict.get("fitness_for_board", True)
        icon = "✓" if (board_ok and not blocked) else "⚠"
        print(f"\n  {icon}  RISK PANEL ({rating}) — Board-ready: {board_ok}")

        if auto_fixed:
            print(f"\n  ✅ AUTO-REMEDIATED ({len(auto_fixed)} structural fix(es) applied automatically):")
            for fix in auto_fixed:
                print(f"     ↳ {fix.replace('threshold_sync:','').replace('consistency:','')}")

        if blocked:
            print(f"\n  ⛔ PUSH BLOCKED — critical structural issues survive auto-fix.")
            for reason in remediation.get("block_reasons", []):
                print(f"     🔴 {reason}")
            print(f"     Fix in kri_thresholds.csv or model_benchmarks.json then re-run.")

        if calib_findings:
            print(f"\n  📋 CALIBRATION FINDINGS — require your judgment ({len(calib_findings)}):")
            for f in calib_findings:
                sev   = f.get("severity", "high").upper()
                fid   = f.get("id", "")
                title = f.get("title", "")
                owner = f.get("owner", "")
                rec   = f.get("recommendation", "")
                icon_f = "🔴" if sev == "CRITICAL" else "🟠"
                print(f"     {icon_f} [{sev}] [{fid}] {title}")
                if rec:
                    print(f"          Recommendation: {rec}")
                if owner:
                    print(f"          Owner: {owner}")

                # Loop 2 — offer to add finding to board Risk Committee actions
                if rec and not ci_mode:
                    resp = _hitl_input(
                        f"          Add [{fid}] recommendation to board actions? (yes/no): "
                    )
                    if resp == "yes":
                        action_text = f"[{fid}] {title}: {rec}"
                        state["panel_action_items"].append(action_text)
                        print(f"          ✓ Queued for board summary — will append after exec rec update")
                elif rec and ci_mode:
                    # CI: do not auto-add calibration findings; they require human judgment
                    pass
        elif not blocked:
            print(f"\n  ✓ No calibration findings require human review this run.")

        conditions = panel_verdict.get("conditions_for_board_readiness", [])
        for c in conditions:
            print(f"     ✗ Condition: {c}")

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
    """
    Apply approved executive recommendation updates to dashboard HTML.
    Also appends any panel calibration action items (Loop 2) that the user elected
    to add during the HITL gate to the board summary's Risk Committee Recommended
    Actions section.
    """
    from tools.dashboard_updater import update_exec_recommendations, update_board_summary
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if not dashboard_path.exists():
        return state

    # ── Exec rec updates (existing) ───────────────────────────────────────────
    approved = state.get("exec_rec_approved", {})
    if approved:
        try:
            changes = update_exec_recommendations(dashboard_path, approved)
            print(f"\n[EXEC RECS] {changes} section(s) updated in dashboard")
            state["dashboard_updated"] = True
        except Exception as e:
            state["errors"].append(f"Exec rec update failed: {e}")

    # ── Panel action items → board summary (Loop 2) ───────────────────────────
    panel_actions = state.get("panel_action_items", [])
    if panel_actions:
        # ── Rewrite raw panel findings as board-quality prose ─────────────────
        # Raw items look like "[PANEL-HIGH-02] Cyber response time...: Escalate to CISO..."
        # We rewrite each as a clean, direct board action sentence before appending.
        try:
            import anthropic as _anthropic
            _client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            raw_list = "\n".join(f"{i+1}. {a}" for i, a in enumerate(panel_actions))
            prose_prompt = (
                "You are rewriting raw risk panel findings as clean board-level action items.\n"
                "Each input is a panel finding reference followed by a recommendation.\n"
                "Rewrite each as one concise, direct board action in plain business English.\n"
                "Rules:\n"
                "(1) Remove the panel ID prefix (e.g. [PANEL-HIGH-02]).\n"
                "(2) Start each action with a strong imperative verb directed at an executive "
                "(e.g. 'Direct CISO to...', 'Mandate CFO to...', 'Assign Board to...').\n"
                "(3) One sentence per action — no sub-clauses, no markdown.\n"
                "(4) Return ONLY the rewritten actions, one per line, in the same order.\n"
                "(5) Do not number the lines.\n\n"
                f"RAW ACTIONS:\n{raw_list}"
            )
            _resp = _client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prose_prompt}],
            )
            rewritten_raw = _resp.content[0].text.strip()
            if _cost:
                _cost.record("panel_action_prose",
                             _resp.usage.input_tokens, _resp.usage.output_tokens)
            # Parse — one action per line; strip any residual numbering the model adds
            rewritten = [
                line.strip().lstrip("0123456789.) ").strip()
                for line in rewritten_raw.splitlines()
                if line.strip()
            ]
            # Safety: if LLM returned fewer lines than inputs, pad with originals
            if len(rewritten) < len(panel_actions):
                rewritten += panel_actions[len(rewritten):]
            panel_actions_final = rewritten[:len(panel_actions)]
            print(f"  Panel actions rewritten as board prose ✓")
        except Exception as _e:
            state["warnings"].append(f"Panel action prose rewrite failed — using raw: {_e}")
            print(f"  ⚠ Panel action rewrite failed ({_e}) — appending raw text")
            panel_actions_final = panel_actions

        board_summary = state.get("board_summary", "")
        MARKER = "\n\nRISK COMMITTEE RECOMMENDED ACTIONS:"
        if MARKER in board_summary:
            # Append new board-quality actions after the existing ones
            additions = "\n" + "\n".join(f"  • {a}" for a in panel_actions_final)
            board_summary = board_summary + additions
        else:
            # No existing actions section — create one
            board_summary += (
                MARKER + "\n" +
                "\n".join(f"  • {a}" for a in panel_actions_final)
            )
        state["board_summary"] = board_summary
        try:
            update_board_summary(dashboard_path, board_summary, state["run_id"])
            print(f"\n[EXEC RECS] {len(panel_actions_final)} panel action(s) added to board summary")
            state["dashboard_updated"] = True
        except Exception as e:
            state["warnings"].append(f"Panel action board summary write failed: {e}")
            print(f"  ⚠ Panel actions written to state but dashboard write failed: {e}")

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


def _build_kri_status_ground_truth(kri_ground_truth: dict) -> str:
    """
    Build a flat, verbatim list of every KRI with its exact status from the data layer.
    Injected into board synthesis input — the LLM must not contradict these statuses.
    Source: kri_ground_truth["dashboard_kris"] written deterministically from CSVs.
    """
    lines = [
        "KRI STATUS GROUND TRUTH — verbatim statuses from source data "
        "(do NOT upgrade, downgrade, or misreport these in the narrative):"
    ]
    domain_data = kri_ground_truth.get("dashboard_kris", {})
    domain_order = [
        ("Strategic",   "strategic_risks"),
        ("Operational", "operational_risks"),
        ("Financial",   "financial_risks"),
        ("Compliance",  "compliance_risks"),
    ]
    for domain_label, domain_key in domain_order:
        risk_groups = domain_data.get(domain_key, {})
        if not risk_groups:
            continue
        for risk_id, kris in risk_groups.items():
            for k in kris:
                name = k.get("name", "")
                val  = k.get("value")
                unit = k.get("unit", "")
                st   = k.get("status", "ok").upper()
                val_str = f"{val}{unit}" if val is not None else "N/A"
                lines.append(f"  {risk_id} / {name} = {val_str} → {st}")
    if len(lines) == 1:
        lines.append("  (kri_ground_truth unavailable — run full pipeline)")
    return "\n".join(lines)


# Domain-level executive owners for deterministic RISK POSTURE
_DOMAIN_OWNERS = {
    "Strategic":   "Chief Executive Officer",
    "Operational": "COO / CISO / CHRO (by sub-domain)",
    "Financial":   "Chief Financial Officer",
    "Compliance":  "Chief Compliance Officer / General Counsel",
}


def _build_risk_posture_facts(sf, of, ff, cf, compound_scenarios, systemic) -> dict:
    """
    Compute all RISK POSTURE facts deterministically from agent outputs.
    Returns a structured dict — the single source of truth for RISK POSTURE content.
    No LLM involved; these facts are locked before any rendering step.
    """
    domain_data = [
        ("Strategic",   sf),
        ("Operational", of),
        ("Financial",   ff),
        ("Compliance",  cf),
    ]
    total_b = sum((f or {}).get("breach_count", 0) for _, f in domain_data)
    total_a = sum((f or {}).get("amber_count",  0) for _, f in domain_data)
    domains_in_breach = [
        {
            "name":      lbl,
            "breaches":  f.get("breach_count", 0),
            "ambers":    f.get("amber_count",  0),
            "owner":     _DOMAIN_OWNERS.get(lbl, "Executive management"),
        }
        for lbl, f in domain_data if f and f.get("breach_count", 0) > 0
    ]
    amber_only_domains = [
        lbl for lbl, f in domain_data
        if f and f.get("breach_count", 0) == 0 and f.get("amber_count", 0) > 0
    ]
    escalated = [lbl for lbl, f in domain_data if f and f.get("escalation_required")]
    # Collect escalation reasons from each agent — used to prevent fabricated explanations
    escalation_reasons = {}
    for lbl, f in domain_data:
        if f and f.get("escalation_required"):
            reasons = f.get("escalation_reasons", [])
            if reasons:
                escalation_reasons[lbl] = reasons[:2]  # cap at 2 per domain

    return {
        "total_breaches":       total_b,
        "total_ambers":         total_a,
        "domains_in_breach":    domains_in_breach,
        "amber_only_domains":   amber_only_domains,
        "escalated_domains":    escalated,
        "escalation_reasons":   escalation_reasons,
        "systemic_flag":        systemic,
        "compound_scenarios":   compound_scenarios or [],
    }


def _render_risk_posture_fallback(facts: dict) -> str:
    """
    Deterministic plain-text rendering of RISK POSTURE facts.
    Used when the Haiku LLM call fails — guarantees the section is never empty.
    """
    lines = []
    total_b = facts["total_breaches"]
    total_a = facts["total_ambers"]
    domains_in_breach = facts["domains_in_breach"]
    n_breach_domains = len(domains_in_breach)

    if total_b == 0:
        lines.append(
            f"The organisation records no KRI breaches this period "
            f"and {total_a} amber warning(s)."
        )
    elif n_breach_domains == 1:
        d = domains_in_breach[0]
        lines.append(
            f"The organisation records {total_b} KRI breach(es) in the "
            f"{d['name']} domain and {total_a} amber warning(s) across all domains."
        )
    else:
        lines.append(
            f"The organisation records {total_b} KRI breach(es) across "
            f"{n_breach_domains} domain(s) and {total_a} amber warning(s) this period."
        )

    for d in domains_in_breach:
        lines.append(
            f"{d['name']}: {d['breaches']} breach(es), {d['ambers']} amber(s) "
            f"— accountable executive: {d['owner']}."
        )

    if facts["amber_only_domains"]:
        lines.append(
            "Amber warnings only (no breach): "
            + ", ".join(facts["amber_only_domains"]) + "."
        )

    if facts["escalated_domains"]:
        lines.append("Escalation required: " + ", ".join(facts["escalated_domains"]) + ".")
    else:
        lines.append("No domain escalation required this period.")

    if facts["systemic_flag"]:
        lines.append(
            "Systemic risk flag: three or more domains are simultaneously in breach."
        )

    if facts["compound_scenarios"]:
        lines.append(
            "Active compound scenario(s): "
            + "; ".join(facts["compound_scenarios"]) + "."
        )

    return "RISK POSTURE\n\n" + " ".join(lines)


def _render_risk_posture(client, facts: dict, cost) -> str:
    """
    Render RISK POSTURE as board-quality prose using Claude Haiku.

    Architecture: data-locked LLM rendering.
    - Facts are pre-computed deterministically (counts, domains, escalation status).
    - The LLM's only job is to express those exact facts in natural board language.
    - It cannot add, remove, or change any fact — the input is the entire fact set.
    - max_tokens=500 (one paragraph).
    - Falls back to _render_risk_posture_fallback() if the call fails.
    """
    # Build the locked fact sheet the LLM must render verbatim
    total_b = facts["total_breaches"]
    total_a = facts["total_ambers"]
    domains_in_breach = facts["domains_in_breach"]
    amber_only = facts["amber_only_domains"]
    escalated  = facts["escalated_domains"]

    fact_lines = [
        f"- Total KRI breaches: {total_b}",
        f"- Total amber warnings: {total_a}",
        f"- Number of domains in breach: {len(domains_in_breach)}",
    ]
    for d in domains_in_breach:
        fact_lines.append(
            f"- {d['name']} domain: {d['breaches']} breach(es), "
            f"{d['ambers']} amber(s) — accountable executive: {d['owner']}"
        )
    if amber_only:
        fact_lines.append(
            f"- Domains with amber warnings only (no breach): {', '.join(amber_only)}"
        )
    escalation_reasons = facts.get("escalation_reasons", {})
    if escalated:
        for dom in escalated:
            reasons = escalation_reasons.get(dom, [])
            reason_txt = "; ".join(reasons) if reasons else "KRI breach threshold exceeded"
            fact_lines.append(f"- Escalation required — {dom}: {reason_txt}")
    else:
        fact_lines.append("- Escalation required: none this period")
    if facts["systemic_flag"]:
        fact_lines.append(
            "- Systemic risk flag: YES — three or more domains simultaneously in breach"
        )
    for cs in facts["compound_scenarios"]:
        fact_lines.append(f"- Active compound scenario: {cs}")

    fact_sheet = "\n".join(fact_lines)

    system = (
        "You are the Chief Risk Officer writing the opening paragraph of a board-level "
        "risk summary. You receive a locked fact sheet. Your only job is to render those "
        "exact facts as a single, fluent paragraph of board-quality prose. "
        "Rules: (1) Every fact in the sheet must appear in your paragraph — do not omit any. "
        "(2) Do not introduce any fact not in the sheet — no KRI names, no specific values, "
        "no domain commentary beyond what is listed. "
        "(3) Use authoritative, declarative language appropriate for board directors. "
        "(4) No bullet points, no headers, no markdown. One paragraph only. "
        "(5) Begin with the breach count headline. End with the escalation or systemic status."
    )

    user = (
        "Render this fact sheet as one board-quality paragraph:\n\n"
        + fact_sheet
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text.strip()
        if cost:
            cost.record("cra_risk_posture",
                        response.usage.input_tokens,
                        response.usage.output_tokens)
        if len(text) < 60:
            raise ValueError("Response too short — falling back")
        return "RISK POSTURE\n\n" + text
    except Exception as e:
        print(f"  [RISK POSTURE] Haiku render failed ({e}) — using deterministic fallback")
        return _render_risk_posture_fallback(facts)


def _build_qoq_fact_block(qoq_deltas: dict) -> str:
    """
    Build a data-locked fact sheet for the QUARTER-ON-QUARTER MOVEMENT section.
    The LLM must base its QoQ narrative ONLY on these verified deltas —
    no invented prior-period values, no fabricated comparisons.

    If no prior period data exists, returns a sentinel that tells the LLM to
    limit the QoQ section to current-period trend signals only.
    """
    if not qoq_deltas or not qoq_deltas.get("available"):
        reason = qoq_deltas.get("reason", "prior period data not available") if qoq_deltas else "QoQ not computed"
        return (
            "QoQ MOVEMENT FACT SHEET — DATA-LOCKED (do not add values not shown here):\n"
            f"  Status: {reason}\n"
            "  Instruction: No prior-period comparison is available. Write the QUARTER-ON-QUARTER\n"
            "  MOVEMENT paragraph using ONLY the trend_direction signals from domain agents\n"
            "  (deteriorating/improving/flat). Do NOT invent specific prior-period values."
        )

    period_cur = qoq_deltas.get("period_current", "Current")
    period_pri = qoq_deltas.get("period_prior",   "Prior")
    summary    = qoq_deltas.get("summary", {})
    dom_cause  = qoq_deltas.get("dominant_cause", "unknown")
    primary    = qoq_deltas.get("primary_kri")
    det        = qoq_deltas.get("deteriorating", [])
    imp        = qoq_deltas.get("improving",     [])
    new_br     = summary.get("new_breaches", [])
    cleared    = summary.get("cleared_breaches", [])

    lines = [
        f"QoQ MOVEMENT FACT SHEET — DATA-LOCKED ({period_pri} → {period_cur})",
        f"  Comparison period: {period_pri} (prior) → {period_cur} (current)",
        f"  KRIs compared: {summary.get('total_compared', 0)} | "
        f"Deteriorating: {summary.get('deteriorating_count', 0)} | "
        f"Improving: {summary.get('improving_count', 0)} | "
        f"Stable: {summary.get('stable_count', 0)}",
        f"  Dominant cause of movement: {dom_cause.replace('_', ' ').upper()}",
        f"  Primary decision point (if reversed, most reduces residual risk): {primary or 'none identified'}",
    ]

    if new_br:
        lines.append(f"  New breaches this period: " +
                     ", ".join(f"{r}/{k}" for r, k in new_br))
    if cleared:
        lines.append(f"  Breaches cleared this period: " +
                     ", ".join(f"{r}/{k}" for r, k in cleared))

    if det:
        lines.append("  DETERIORATING KRIs (use these exact values — no others):")
        for m in det:
            lines.append(
                f"    {m['risk_id']}/{m['kri_name']}: "
                f"{m['prior_value']} → {m['current_value']} "
                f"({m['prior_status']} → {m['current_status']})"
            )

    if imp:
        lines.append("  IMPROVING KRIs:")
        for m in imp:
            lines.append(
                f"    {m['risk_id']}/{m['kri_name']}: "
                f"{m['prior_value']} → {m['current_value']} "
                f"({m['prior_status']} → {m['current_status']})"
            )

    lines += [
        "",
        "  INSTRUCTION: Your QUARTER-ON-QUARTER MOVEMENT paragraph MUST:",
        "  (1) Use ONLY the values shown above — do not introduce any other prior-period numbers.",
        "  (2) State whether aggregate residual risk rose, held flat, or fell — based on the "
        "counts above.",
        "  (3) Name the dominant cause (CONTROL WEAKENING or INHERENT RISK GROWTH) using the "
        "label above.",
        "  (4) Name the primary decision point KRI shown above as the board's primary lever.",
        "  (5) Translate all values into plain business language — no statistical terms.",
    ]

    return "\n".join(lines)


def _run_board_synthesis(client, full_findings: str, exec_recs: dict,
                         committee_actions: list, compound_scenarios: list,
                         systemic: bool, kri_count_facts: str,
                         kri_status_truth: str, qoq_fact_block: str, cost) -> str:
    """
    Step 3 — Board-level synthesis for senior management.

    The LLM generates ONLY the three synthesis sections:
      KEY RISK DRIVERS, CROSS-DOMAIN CONNECTIONS, QUARTER-ON-QUARTER MOVEMENT.

    RISK POSTURE is generated deterministically by _build_risk_posture_deterministic()
    and prepended by the caller. This eliminates the single biggest source of
    hallucination (wrong counts, upgraded KRI statuses).

    Returns the three synthesis sections as plain text, or empty string on failure.
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
        # KRI counts and verbatim status ground truth injected first —
        # LLM must reference these, not recount or override them.
        kri_count_facts
        + "\n\n"
        + kri_status_truth
        + "\n\n"
        + "─" * 60
        + "\n\n"
        # Data-locked QoQ fact sheet — prevents fabrication of prior-period values
        + qoq_fact_block
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
        + "NOTE: RISK POSTURE has already been written deterministically and will be "
          "prepended to your output. Do NOT write a RISK POSTURE section. "
          "Begin your output directly with KEY RISK DRIVERS."
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

    graph.add_node("dispatch",                  dispatch_node)
    graph.add_node("kri_data_layer",            kri_data_layer_node)
    graph.add_node("model_calibration",         model_calibration_node)
    graph.add_node("run_agents",                run_domain_agents_node)
    graph.add_node("synthesis",                 chief_risk_synthesis_node)
    graph.add_node("write_approved",            write_approved_node)
    graph.add_node("kri_validation",            kri_validation_node)
    graph.add_node("content_validation",        content_validation_node)
    graph.add_node("risk_panel",                risk_panel_node)
    graph.add_node("panel_remediation",         panel_remediation_node)
    graph.add_node("board_summary_correction",  board_summary_correction_node)  # Loop 1+2
    graph.add_node("hitl_gate",                 hitl_gate_node)
    graph.add_node("update_exec_recs",          update_exec_recs_node)
    graph.add_node("github_push",               github_push_node)
    graph.add_node("finalise",                  finalise_node)

    graph.set_entry_point("dispatch")
    graph.add_edge("dispatch",                 "kri_data_layer")
    graph.add_edge("kri_data_layer",           "model_calibration")
    graph.add_edge("model_calibration",        "run_agents")
    graph.add_edge("run_agents",               "synthesis")
    graph.add_conditional_edges(
        "synthesis",
        permission_router,
        {
            "write_approved": "write_approved",
            "hitl_gate":      "content_validation",
        }
    )
    graph.add_edge("write_approved",           "kri_validation")
    graph.add_edge("kri_validation",           "content_validation")
    graph.add_edge("content_validation",       "risk_panel")
    graph.add_edge("risk_panel",               "panel_remediation")
    graph.add_edge("panel_remediation",        "board_summary_correction")  # ← Loop 1+2
    graph.add_edge("board_summary_correction", "hitl_gate")
    graph.add_edge("hitl_gate",                "update_exec_recs")
    graph.add_edge("update_exec_recs",         "github_push")
    graph.add_edge("github_push",              "finalise")
    graph.add_edge("finalise",                 END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
