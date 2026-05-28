"""
run.py — Single entry point for the Risk Intelligence and Resilience Hub agents.

Usage:
  python run.py                    # run all agents
  python run.py --dry-run          # validate setup without calling APIs
  python run.py --dashboard /path  # specify custom dashboard path

Before running:
  1. Docker Desktop must be running
  2. .env file must exist in this directory
  3. Ollama models must be pulled (handled automatically on first run)
"""

import argparse
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
    print("✓ Environment loaded from .env")
except ImportError:
    print("⚠ python-dotenv not installed — reading environment variables directly")


def check_prerequisites() -> list[str]:
    """Validate all prerequisites before running agents."""
    errors = []

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        errors.append("ANTHROPIC_API_KEY not set in .env")
    elif not os.environ["ANTHROPIC_API_KEY"].startswith("sk-ant-"):
        errors.append("ANTHROPIC_API_KEY does not look valid (should start with sk-ant-)")

    # Check GitHub token
    if not os.environ.get("GITHUB_TOKEN"):
        errors.append("GITHUB_TOKEN not set in .env")
    if not os.environ.get("GITHUB_REPO"):
        errors.append("GITHUB_REPO not set in .env")

    # Check data files
    data_dir = Path(__file__).parent / "data"
    required_files = [
        "erp_supply_chain.csv",
        "siem_cyber.csv",
        "qms_quality.csv",
        "hris_talent.csv",
        "market_intelligence.csv",
        "ma_pipeline.csv",
        "treasury_positions.csv",
        "covenant_tracker.csv",
        "ar_aging.csv",
        "screening_results.csv",
        "compliance_metrics.csv",
        "regulatory_horizon.csv",
        "audit_log.csv",
    ]
    for f in required_files:
        if not (data_dir / f).exists():
            errors.append(f"Missing data file: data/{f}")

    # Check risk store
    store_path = Path(__file__).parent / "api" / "risk_store.json"
    if not store_path.exists():
        errors.append("Missing api/risk_store.json")

    # Check prompts
    prompts_dir = Path(__file__).parent / "prompts"
    required_prompts = [
        "strategic_v1.txt", "operational_v1.txt", "financial_v1.txt",
        "compliance_v1.txt", "regulatory_v1.txt", "emerging_risks_v1.txt",
        "chief_risk_v1.txt",
    ]
    for p in required_prompts:
        if not (prompts_dir / p).exists():
            errors.append(f"Missing prompt file: prompts/{p}")

    # Check dashboard
    dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
    if not dashboard_path.exists():
        errors.append(
            "Dashboard not found at dashboard/index.html\n"
            "  Copy your index.html from ~/rib-agents/dashboard/ "
            "or run: cp /path/to/index.html ~/rib-agents/dashboard/"
        )

    return errors


def dry_run():
    """Validate setup without calling any APIs."""
    print("\n" + "="*55)
    print("  DRY RUN — validating setup")
    print("="*55)
    errors = check_prerequisites()
    if errors:
        print(f"\n  ✗ {len(errors)} issue(s) found:\n")
        for e in errors:
            print(f"  • {e}")
        print("\n  Fix the above issues then run: python run.py")
    else:
        print("\n  ✓ All prerequisites satisfied")
        print("  ✓ Ready to run: python run.py")
    print()


def main():
    parser = argparse.ArgumentParser(description="Risk Intelligence Hub — Agent Runner")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup only")
    parser.add_argument("--dashboard", type=str, help="Path to index.html dashboard")
    parser.add_argument(
        "--ci", action="store_true",
        help=(
            "CI mode: auto-approve exec rec sections and held domains that pass "
            "validator with no FLAG: entries. Sections with genuine issues are "
            "skipped rather than blocking the run. Allows unattended dashboard "
            "updates when all content is validator-clean."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    # Prerequisites check
    print("\nChecking prerequisites...")
    errors = check_prerequisites()
    if errors:
        print(f"\n✗ Cannot run — {len(errors)} issue(s):\n")
        for e in errors:
            print(f"  • {e}")
        print("\nRun 'python run.py --dry-run' for details")
        sys.exit(1)
    print("✓ All prerequisites satisfied\n")

    # Install required packages if missing
    try:
        import langgraph
        import pydantic
        import PyGithub
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install langgraph pydantic PyGithub langchain-anthropic python-dotenv --break-system-packages -q")

    # Override dashboard path if provided
    if args.dashboard:
        dashboard_path = Path(args.dashboard)
    else:
        dashboard_path = Path(__file__).parent / "dashboard" / "index.html"

    # Pull latest dashboard from GitHub before running
    print("Pulling latest dashboard from GitHub...")
    try:
        from github import Github, Auth
        g = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))
        repo = g.get_repo(os.environ["GITHUB_REPO"])
        contents = repo.get_contents("index.html")
        dashboard_path.parent.mkdir(exist_ok=True)
        dashboard_path.write_bytes(contents.decoded_content)
        print(f"✓ Dashboard pulled from GitHub ({len(contents.decoded_content):,} bytes)")
    except Exception as e:
        print(f"⚠ Could not pull from GitHub: {e}")
        if not dashboard_path.exists():
            print("  No local dashboard found either — cannot continue")
            sys.exit(1)
        print("  Using existing local dashboard")

    # Build and run the graph
    sys.path.insert(0, str(Path(__file__).parent))
    from graph.risk_graph import build_graph

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    if args.ci:
        print("⚙  CI mode enabled — validator-clean items will be auto-approved\n")
    initial_state = {
        "run_id": run_id,
        "triggered_by": "ci" if args.ci else "manual",
        "ci_mode": args.ci,
        # KRI data layer (populated by kri_data_layer_node before agents run)
        "kri_ground_truth": {},
        "agent_context": {},
        # Domain agent findings
        "strategic_findings": None,
        "operational_findings": None,
        "financial_findings": None,
        "compliance_findings": None,
        "regulatory_signals": None,
        "emerging_signals": None,
        # Permissions & HITL
        "permissions": {},
        "hold_reasons": {},
        "verifications": {},
        "verification_errors": [],
        "hitl_decisions": {},
        "hitl_edits": {},
        # Outputs
        "board_summary": "",
        "exec_rec_drafts": {},
        "exec_rec_approved": {},
        "dashboard_updated": False,
        "github_pushed": False,
        "github_url": "",
        # Validation
        "kri_validation_results": {},
        "content_validation": {},
        "panel_remediation": {},
        # Infrastructure
        "errors": [],
        "warnings": [],
    }

    graph = build_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        final_state = graph.invoke(initial_state, config=config)
        if final_state.get("github_url"):
            print(f"\n✓ Live dashboard: {final_state['github_url']}")
    except KeyboardInterrupt:
        print("\n\nRun interrupted by user")
    except Exception as e:
        print(f"\n✗ Run failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
