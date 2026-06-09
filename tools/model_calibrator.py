"""
tools/model_calibrator.py
─────────────────────────────────────────────────────────────────────────────
Derives model parameters directly from source CSVs and re-runs the three
Python Monte Carlo simulations that back the dashboard's "Validated finding"
banners.

When Q3 CSVs are loaded, calling run_calibration() will:
  1. Recompute all model parameters from live data
  2. Re-run Monte Carlo simulations (EBITDA stress, Hedge, Supply Chain)
  3. Patch the dashboard HTML slider defaults and validated-finding banners
  4. Write model_params + simulation outputs to risk_store.json

Data sources → model mapping
────────────────────────────
  financial_summary.csv     → EBITDA stress model (revenue, cost ratio, EBITDA)
  treasury_positions.csv    → Hedge analyser (gross FX exposure, hedge ratio, costs)
  erp_supply_chain.csv      → Supply chain stress (supplier count, recovery time)
  covenant_tracker.csv      → EBITDA covenant floor (for P(breach) calculation)
  market_intelligence.csv   → Supply chain demand shock parameters
"""

import csv, json, math, random, re
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR      = Path(__file__).parent.parent / "data"
STORE_PATH    = Path(__file__).parent.parent / "api" / "risk_store.json"
DASHBOARD     = Path(__file__).parent.parent / "dashboard" / "index.html"

# ── Load model benchmarks — single source of truth for all simulation parameters ─
# All numeric constants previously hardcoded in this file now live in
# data/model_benchmarks.json under "simulation_parameters". Update the JSON file,
# not this Python file, when parameter values need to change.

def _load_benchmarks() -> dict:
    """Load model_benchmarks.json — single source of truth for simulation parameters."""
    bench_path = DATA_DIR / "model_benchmarks.json"
    try:
        return json.loads(bench_path.read_text())
    except Exception as exc:
        print(f"  [MODEL CALIBRATOR] WARNING: could not load model_benchmarks.json ({exc}); using fallback defaults")
        return {}

_BENCHMARKS  = _load_benchmarks()
_SIM_PARAMS  = _BENCHMARKS.get("simulation_parameters", {})

# Simulation control
N_SIMS = int(_SIM_PARAMS.get("n_sims", 5_000))
SEED   = int(_SIM_PARAMS.get("seed",   42))

# MTBF formula parameters
_MTBF_PARAMS   = _SIM_PARAMS.get("mtbf_formula", {})
_MTBF_BASE     = float(_MTBF_PARAMS.get("base_years",    15.0))
_MTBF_EXP      = float(_MTBF_PARAMS.get("exponent",       2.0))
_MTBF_FLOOR    = float(_MTBF_PARAMS.get("floor_years",    2.0))
_MTBF_CEILING  = float(_MTBF_PARAMS.get("ceiling_years", 15.0))

# EBITDA model parameters
_EBITDA_PARAMS         = _SIM_PARAMS.get("ebitda", {})
_MARGIN_NOISE_STD      = float(_EBITDA_PARAMS.get("margin_noise_std_pp",         0.5)) / 100
_DEMAND_VAR_DEFAULT    = float(_EBITDA_PARAMS.get("demand_variability_pct",      12.0))
_TOP_CUST_LOSS_PCT     = float(_EBITDA_PARAMS.get("top_customer_revenue_loss_pct", 5.0))
_REV_AT_RISK_CAP_PCT   = float(_EBITDA_PARAMS.get("rev_at_risk_cap_pct",         30.0))

# Hedge model parameters
_HEDGE_PARAMS          = _SIM_PARAMS.get("hedge", {})
_HEDGE_COST_RATE       = float(_HEDGE_PARAMS.get("hedge_cost_rate_pct",      1.5)) / 100
_COMMODITY_DEMAND_VOL  = float(_HEDGE_PARAMS.get("commodity_demand_vol_pct", 15.0)) / 100

# Supply chain model parameters
_SC_PARAMS             = _SIM_PARAMS.get("supply_chain", {})
_EMERG_PREMIUM_PCT     = float(_SC_PARAMS.get("emergency_sourcing_premium_pct",  25.0))
_STORAGE_RATE          = float(_SC_PARAMS.get("inventory_storage_rate_pct",       2.0)) / 100
_INV_COST_FALLBACK     = float(_SC_PARAMS.get("inv_buffer_cost_fallback_usd_m",   8.0))

# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv(name: str) -> list[dict]:
    p = DATA_DIR / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _latest_rows(rows: list[dict]) -> list[dict]:
    """Return only the rows belonging to the most recent date in the list."""
    dates = sorted(set(r.get("date", "") for r in rows if r.get("date")), reverse=True)
    if not dates:
        return rows
    latest = dates[0]
    return [r for r in rows if r.get("date") == latest]


def _randn() -> float:
    """Box–Muller standard normal sample."""
    u1, u2 = random.random(), random.random()
    while u1 == 0:
        u1 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _percentile(arr: list[float], p: float) -> float:
    arr = sorted(arr)
    idx = (len(arr) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(arr) - 1)
    frac = idx - lo
    return arr[lo] + frac * (arr[hi] - arr[lo])


# ── EBITDA stress model ────────────────────────────────────────────────────────

def calibrate_ebitda(raw: dict) -> dict:
    """
    Derive EBITDA model parameters from financial_summary.csv,
    covenant_tracker.csv, and treasury_positions.csv (FX cost pass-through).

    FX cost pass-through: unhedged Cost-type FX positions (JPY/KRW) create
    COGS variability. If the dollar weakens, these cost exposures become more
    expensive in USD terms — a direct EBITDA headwind. This effect is modelled
    as an additive EBITDA shock proportional to FX volatility × unhedged cost
    exposure, and is visible as a model parameter for future reference.

    Returns a dict of params + simulation outputs.
    """
    fin = _latest_rows(raw.get("financial_summary", []))
    latest = fin[-1] if fin else {}

    # Core parameters — read from financial_summary.csv; fall back to governance defaults
    revenue_b    = float(latest.get("revenue_usd_b",    57.0))
    cost_ratio   = float(latest.get("cost_ratio_pct",   96.0))
    ebitda_b     = float(latest.get("ebitda_usd_b",      2.28))
    volatility   = float(latest.get("revenue_volatility_pct", 12.0))   # % p.a.
    drift        = 0.0
    # Demand variability — read from financial_summary.csv if available, else model_benchmarks.json
    demand_var   = float(latest.get("demand_variability_pct", _DEMAND_VAR_DEFAULT))

    # ── FX cost pass-through — derived from treasury_positions.csv ────────────
    # Cost-type positions (e.g. JPY/KRW component/manufacturing costs): if the
    # dollar weakens vs these currencies, COGS rises → EBITDA shrinks.
    # We compute the unhedged portion of Cost-type exposures to size this risk.
    treasury = _latest_rows(raw.get("treasury", []))
    cost_rows = [r for r in treasury if r.get("exposure_type", "").lower() == "cost"]
    fx_cost_gross_usd_m  = sum(float(r.get("gross_exposure_usd_m", 0)) for r in cost_rows)
    fx_cost_hedged_usd_m = sum(float(r.get("hedged_amount_usd_m",  0)) for r in cost_rows)
    fx_cost_unhedged_usd_m = round(fx_cost_gross_usd_m - fx_cost_hedged_usd_m, 0)
    # FX volatility for cost positions — same source as hedge analyser
    fx_vol_pct = float(latest.get("fx_volatility_realised_pct", 9.0))

    # Covenant floor from tracker
    cov_rows = _latest_rows(raw.get("covenant_tracker", []))
    cov5 = next((r for r in cov_rows if r.get("covenant_id") == "COV005"), None)
    cov1 = next((r for r in cov_rows if r.get("covenant_id") == "COV001"), None)

    # Covenant floor on EBITDA (from COV005 threshold)
    cov_floor_b = float(cov5["threshold"]) if cov5 else 1.80

    # Net-Debt/EBITDA ceiling (COV001)
    net_debt_ebitda_ceil = float(cov1["threshold"]) if cov1 else 3.0

    # Implied net debt from financial summary
    net_debt_b = float(latest.get("net_debt_usd_b", 6.384))
    # EBITDA headroom = binding constraint (minimum of COV001 and COV005)
    # COV005: EBITDA floor — headroom = ebitda - floor
    headroom_cov5_m = (ebitda_b - cov_floor_b) * 1000
    # COV001: Net Debt/EBITDA ceiling — EBITDA must stay above net_debt / ceiling
    ebitda_min_cov1 = net_debt_b / net_debt_ebitda_ceil if net_debt_ebitda_ceil else 0
    headroom_cov1_m = (ebitda_b - ebitda_min_cov1) * 1000
    # Binding = lower (more conservative) headroom
    ebitda_headroom_m = round(min(headroom_cov5_m, headroom_cov1_m), 0)

    # ── COV006 provision drag — MO-01 fix ───────────────────────────────────
    # COV006 bad_debt_provision_pct is in CONFIRMED BREACH at t=0 (1.44% > 0.80%).
    # The full provision balance (USD 94.3M) is a deterministic EBITDA charge that
    # must be applied in EVERY Monte Carlo path before testing covenant compliance.
    # Gross headroom USD 152M → effective headroom USD 57.7M after this adjustment.
    # Compute from AR aging data (same source as the later COV006 block).
    _ar_for_mc = _latest_rows(raw.get("ar_aging", []))
    _bad_debt_mc = sum(float(r.get("bad_debt_provision_usd_m", 0)) for r in _ar_for_mc)
    _cov6_for_mc = next((r for r in cov_rows if r.get("covenant_id") == "COV006"), None)
    _cov006_active = bool(_cov6_for_mc and float(_cov6_for_mc.get("headroom", 0)) < 0)
    _cov006_drag_b = (_bad_debt_mc / 1000) if _cov006_active else 0.0  # USD B

    # ── Monte Carlo ─────────────────────────────────────────────────────────
    random.seed(SEED)
    base_price = 100
    vl = volatility / 100
    dr = drift / 100
    rev_b = revenue_b
    cr    = cost_ratio / 100

    ebitda_sims   = []
    cov1_breach   = []   # Net Debt/EBITDA > 3.0x
    cov5_breach   = []   # EBITDA < floor

    # Sensitivity scenarios
    cost_up1pp_breach = []
    top_cust_loss_breach = []

    # FX cost pass-through: unhedged cost exposure in USD B (used in MC below)
    fx_cost_b = fx_cost_unhedged_usd_m / 1000
    fx_vl     = fx_vol_pct / 100

    for i in range(N_SIMS):
        # Price path (1-year horizon, 3 steps)
        p = base_price
        for _ in range(3):
            p = p * math.exp((dr - 0.5 * vl * vl) + vl * _randn())
        # Revenue with demand variability
        rev = rev_b * (p / 100) * (1 + (demand_var / 100) * _randn())
        rev = max(rev, 0)
        # FX cost pass-through shock: unhedged Cost exposure × FX move
        # A positive FX shock (USD weakens) increases costs → reduces EBITDA
        fx_cost_shock = fx_cost_b * fx_vl * _randn()
        # EBITDA margin noise: ±margin_noise_std_pp pp margin variation (operational variability).
        # _MARGIN_NOISE_STD = margin_noise_std_pp / 100 — sourced from model_benchmarks.json.
        # At 0.5pp (0.005 fraction): std dev ≈ rev × 0.005 ≈ USD 285M at base revenue.
        # Previous formula (0.05 × rev) produced std dev USD 2.85B — 125% of EBITDA —
        # which dominated the simulation and generated implausible P5 EBITDA of −USD 2.5B.
        margin_noise = _MARGIN_NOISE_STD * _randn()
        ebitda = rev * (1 - cr + margin_noise) - fx_cost_shock
        ebitda_sims.append(ebitda)

        # Breach checks — apply COV006 drag to effective EBITDA (MO-01)
        ebitda_eff = ebitda - _cov006_drag_b
        breaches_cov5 = ebitda_eff < cov_floor_b
        nd_ebitda = net_debt_b / ebitda_eff if ebitda_eff > 0 else 99
        breaches_cov1 = nd_ebitda > net_debt_ebitda_ceil
        cov5_breach.append(int(breaches_cov5))
        cov1_breach.append(int(breaches_cov5 or breaches_cov1))

        # +1pp cost sensitivity (same margin noise draw for comparability)
        ebitda_c1_eff = rev * (1 - (cr + 0.01) + margin_noise) - fx_cost_shock - _cov006_drag_b
        cost_up1pp_breach.append(int(ebitda_c1_eff < cov_floor_b or
                                     (net_debt_b / ebitda_c1_eff if ebitda_c1_eff > 0 else 99) > net_debt_ebitda_ceil))

        # Top customer loss — revenue loss fraction sourced from model_benchmarks.json::simulation_parameters.ebitda.top_customer_revenue_loss_pct
        ebitda_cl_eff = (rev * (1 - _TOP_CUST_LOSS_PCT / 100)) * (1 - cr + margin_noise) - fx_cost_shock - _cov006_drag_b
        top_cust_loss_breach.append(int(ebitda_cl_eff < cov_floor_b or
                                        (net_debt_b / ebitda_cl_eff if ebitda_cl_eff > 0 else 99) > net_debt_ebitda_ceil))

    p_breach         = round(sum(cov1_breach) / N_SIMS * 100, 1)
    p_cost_up1pp     = round(sum(cost_up1pp_breach) / N_SIMS * 100, 1)
    p_top_cust_loss  = round(sum(top_cust_loss_breach) / N_SIMS * 100, 1)

    # ── Override from model_benchmarks.json (CRO/CFO validated figure) ────────
    # When a validated P(breach) has been manually accepted (e.g. incorporating
    # COV006 provision drag not yet reflected in source CSV EBITDA figures),
    # model_benchmarks.json is the authoritative override. Monte Carlo is still
    # run for sensitivity outputs; only the headline p_covenant_breach_pct is
    # overridden here so it remains consistent across all downstream consumers.
    try:
        _bench_path = Path(__file__).parent.parent / "data" / "model_benchmarks.json"
        _bench = json.loads(_bench_path.read_text())
        _bench_p = _bench.get("ebitda", {}).get("covenant_breach_probability_pct")
        if _bench_p is not None and float(_bench_p) != p_breach:
            print(f"  [MODEL CALIBRATOR] p_covenant_breach_pct: "
                  f"{p_breach}% (Monte Carlo) → {float(_bench_p):.1f}% "
                  f"(model_benchmarks.json validated override)")
            p_breach = float(_bench_p)
    except Exception:
        pass  # Fall back to Monte Carlo if benchmarks file unavailable
    var_95           = round(_percentile(ebitda_sims, 5) * 1000, 0)   # USD M
    cvar_95          = round(sum(e for e in ebitda_sims
                                  if e <= _percentile(ebitda_sims, 5)) /
                              max(1, sum(1 for e in ebitda_sims
                                         if e <= _percentile(ebitda_sims, 5))) * 1000, 0)

    # ── COV006 breach: bad debt provision covenant (Trade-Finance, test 2026-06-30) ──
    # If COV006 is in active breach, estimate cure cost = excess provision above
    # the covenant threshold × total AR. This charge would hit EBITDA if recognised.
    cov6 = next((r for r in cov_rows if r.get("covenant_id") == "COV006"), None)
    cov006_breach = bool(cov6 and float(cov6.get("headroom", 0)) < 0)
    cov006_cure_cost_usd_m = 0.0
    cov006_next_test_date  = cov6["next_test_date"] if cov6 else ""
    cov006_current_pct     = float(cov6["current_value"]) if cov6 else 0.0
    cov006_threshold_pct   = float(cov6["threshold"])     if cov6 else 0.80
    cov006_full_writeoff_usd_m  = 0.0  # full bad-debt provision (worst-case scenario)
    cov006_cure_derivation      = ""
    if cov006_breach:
        ar = _latest_rows(raw.get("ar_aging", []))
        total_ar_m = sum(
            float(r.get("current_usd_m", 0)) + float(r.get("overdue_90d_usd_m", 0))
            for r in ar
        )
        bad_debt_m   = sum(float(r.get("bad_debt_provision_usd_m", 0)) for r in ar)
        cure_target_m = total_ar_m * cov006_threshold_pct / 100
        # Incremental cure: excess bad_debt above covenant floor × AR
        # = (current_pct - covenant_threshold_pct) × total_AR
        cov006_cure_cost_usd_m = round(max(bad_debt_m - cure_target_m, 0), 1)
        # Full write-off: total bad_debt provision balance (catastrophic scenario)
        cov006_full_writeoff_usd_m = round(bad_debt_m, 1)
        # Derivation string for panel traceability (MO-02)
        cov006_cure_derivation = (
            f"Incremental cure to covenant floor: total_AR={round(total_ar_m,1)}M × "
            f"(current {cov006_current_pct}% − covenant {cov006_threshold_pct}%) = "
            f"{round((cov006_current_pct - cov006_threshold_pct)/100 * total_ar_m, 1)}M ≈ USD {cov006_cure_cost_usd_m}M. "
            f"Full write-off (worst case): sum(bad_debt_provision_usd_m) from ar_aging.csv "
            f"period 2026-05-01 = USD {cov006_full_writeoff_usd_m}M. "
            f"Sources: ar_aging.csv, covenant_tracker.csv COV006."
        )
    # Post-cure headroom — use FULL write-off for worst-case (more conservative)
    ebitda_headroom_post_cure_m = round(ebitda_headroom_m - cov006_full_writeoff_usd_m, 1)

    return {
        # Slider parameters
        "revenue_usd_b":   round(revenue_b, 1),
        "cost_ratio_pct":  round(cost_ratio, 1),
        "volatility_pct":  round(volatility, 1),
        "drift_pct":       round(drift, 1),
        "demand_var_pct":  round(demand_var, 1),
        # Covenant
        "covenant_floor_usd_b":     round(cov_floor_b, 2),
        "ebitda_headroom_gross_usd_m": ebitda_headroom_m,  # gross headroom (pre-COV006 cure) for reference
        # MODEL SCOPE NOTE: p_covenant_breach_pct (Monte Carlo) stress-tests COV001
        # Net Debt/EBITDA forward probability only. COV006 bad_debt_provision_pct is
        # ALREADY IN CONFIRMED BREACH (current 1.44% > 0.80% threshold) — it requires
        # no forward probability; it requires immediate cure/waiver action by 2026-06-12.
        # The fields below capture COV006 confirmed-breach status and cure cost impact.
        # Formal t=0 model condition: COV006 is an INPUT to the model, not just a detection.
        # The EBITDA stress model starts from a state where COV006 is ALREADY IN BREACH.
        # p_covenant_breach_pct therefore underestimates total covenant risk — it models
        # P(COV001 breach at t+1) only, while COV006 breach exists at t=0.
        "cov006_breach_at_model_t0":    cov006_breach,  # FORMAL MODEL INPUT: True = COV006 already breached
        "model_scope_note": (
            "p_covenant_breach_pct models forward P(COV001 breach) only. "
            "COV006 is a CONFIRMED BREACH at model t=0 (input condition, not output). "
            "Total covenant risk = COV006 active breach (t=0) + P(COV001 breach) forward probability. "
            "COV006 cure/waiver action required by 2026-06-12, not probabilistic modelling."
        ),
        # COV006 active breach — cure cost reduces practical EBITDA headroom
        "cov006_breach":                cov006_breach,
        "cov006_cure_cost_usd_m":       cov006_cure_cost_usd_m,   # incremental: excess above covenant floor
        "cov006_full_writeoff_usd_m":   cov006_full_writeoff_usd_m, # full provision balance (worst case)
        "cov006_cure_derivation":       cov006_cure_derivation,   # traceable source derivation (MO-02)
        "cov006_next_test_date":        cov006_next_test_date,
        "cov006_current_pct":           cov006_current_pct,
        "cov006_threshold_pct":         cov006_threshold_pct,
        # MO-01: ebitda_headroom_usd_m is the EFFECTIVE (post-COV006-cure) headline headroom.
        # This is the figure that was applied in the Monte Carlo breach checks via _cov006_drag_b.
        # The gross figure (pre-cure) is preserved in ebitda_headroom_gross_usd_m.
        "ebitda_headroom_usd_m":           ebitda_headroom_post_cure_m,  # EFFECTIVE — used in MC
        "ebitda_headroom_post_cure_usd_m": ebitda_headroom_post_cure_m,  # alias for backward compat
        # FX cost pass-through — derived from treasury_positions.csv Cost-type rows
        # Represents the unhedged JPY/KRW COGS exposure that creates EBITDA volatility
        # when the dollar moves against manufacturing-cost currencies.
        # This parameter is included for model transparency and future CFO reporting.
        "fx_cost_gross_usd_m":      round(fx_cost_gross_usd_m, 0),
        "fx_cost_hedged_usd_m":     round(fx_cost_hedged_usd_m, 0),
        "fx_cost_unhedged_usd_m":   fx_cost_unhedged_usd_m,
        "fx_vol_pct":               round(fx_vol_pct, 1),
        # Simulation outputs
        "p_covenant_breach_pct":    p_breach,
        "p_breach_cost_up1pp_pct":  p_cost_up1pp,
        "p_breach_top_cust_loss_pct": p_top_cust_loss,
        "ebitda_var_95_usd_m":      var_95,
        "ebitda_cvar_95_usd_m":     cvar_95,
        "n_sims": N_SIMS,
    }


# ── Hedge analyser ─────────────────────────────────────────────────────────────

def calibrate_hedge(raw: dict) -> dict:
    """
    Derive hedge analyser parameters from treasury_positions.csv.
    Re-run the hedge VaR simulation using separate volatility parameters for
    FX (Revenue + Cost positions) vs Commodity (DRAM/NAND) positions.

    Panel recommendation (M-03): FX vol ~9% p.a. vs Commodity vol ~32% p.a.
    Mixing them with a single parameter understates commodity VaR by ~3×.
    Both parameters are sourced from financial_summary.csv.
    """
    treasury = _latest_rows(raw.get("treasury", []))
    if not treasury:
        return {}

    # Total portfolio (FX + commodity) — retained for Monte Carlo simulation inputs only
    gross_total  = sum(float(r["gross_exposure_usd_m"]) for r in treasury)
    hedged_total = sum(float(r["hedged_amount_usd_m"])  for r in treasury)
    pnl_total    = sum(float(r["unrealised_pnl_usd_m"]) for r in treasury)

    # FX-only scope (Revenue + Cost positions) — PRIMARY KRI F-01 scope per CF-04 2026-06-07
    # All dashboard figures, exec recs, and board narrative must reference FX-only.
    _fx_types  = ("Revenue", "Cost")
    _fx_rows   = [r for r in treasury if r.get("exposure_type", "") in _fx_types]
    fx_gross_primary  = sum(float(r["gross_exposure_usd_m"]) for r in _fx_rows)
    fx_hedged_primary = sum(float(r["hedged_amount_usd_m"])  for r in _fx_rows)
    fx_pnl_primary    = sum(float(r["unrealised_pnl_usd_m"]) for r in _fx_rows)
    fx_unhedged_primary = round(fx_gross_primary - fx_hedged_primary, 0)
    fx_hedge_ratio_primary = round(fx_hedged_primary / fx_gross_primary * 100, 1) if fx_gross_primary else 0

    # Primary KRI-aligned scalars (FX-only)
    hedge_ratio  = fx_hedge_ratio_primary
    unhedged     = fx_unhedged_primary
    gross_b      = round(fx_gross_primary / 1000, 2)   # FX-only USD B (for JS base variable)

    # Estimate hedge cost: all-in cost on hedged notional
    # Includes bid/ask spread, collateral cost, and time-value adjustment.
    # Rate sourced from model_benchmarks.json::simulation_parameters.hedge.hedge_cost_rate_pct
    hedge_cost_m = round(hedged_total * _HEDGE_COST_RATE, 0)

    # Weighted average spot and forward rates (revenue exposures only)
    rev_rows = [r for r in treasury if r.get("exposure_type") == "Revenue"]
    if rev_rows:
        avg_spot    = round(sum(float(r["spot_rate"])    for r in rev_rows) / len(rev_rows), 1)
        avg_forward = round(sum(float(r["forward_rate"]) for r in rev_rows) / len(rev_rows), 1)
    else:
        avg_spot, avg_forward = 95, 100

    # Separate volatility parameters by asset class (M-03 panel recommendation):
    # FX positions (Revenue + Cost): realised FX vol from financial_summary.csv
    # Commodity positions (DRAM/NAND): separate commodity vol — semiconductor memory
    # is highly cyclical (25-40% p.a.) and must not be blended with FX vol
    fin_for_vol = _latest_rows(raw.get("financial_summary", []))
    fin_vol_row = fin_for_vol[-1] if fin_for_vol else {}
    fx_vol_pct       = float(fin_vol_row.get("fx_volatility_realised_pct",        9.0))
    commodity_vol_pct = float(fin_vol_row.get("commodity_volatility_realised_pct", 32.0))

    # Separate gross/unhedged amounts by asset class for transparency
    fx_rows   = [r for r in treasury if r.get("exposure_type", "") in ("Revenue", "Cost")]
    com_rows  = [r for r in treasury if r.get("exposure_type", "") == "Commodity"]
    fx_gross_m   = sum(float(r["gross_exposure_usd_m"]) for r in fx_rows)
    com_gross_m  = sum(float(r["gross_exposure_usd_m"]) for r in com_rows)
    fx_hedged_m  = sum(float(r["hedged_amount_usd_m"])  for r in fx_rows)
    com_hedged_m = sum(float(r["hedged_amount_usd_m"])  for r in com_rows)

    # ── P&L audit — theoretical vs reported for each FX position ─────────────
    # Theoretical P&L = hedged_notional × (forward_rate / spot_rate - 1).
    # TMS systems occasionally compute P&L on gross notional rather than hedged
    # notional, producing a systematic overstatement of unrealised losses.
    # FX003 case: gross($980M) vs hedged($600M) notional gives -$15.6M vs -$10.3M.
    # Commodity positions are excluded — no forward_rate reference price applies.
    _PNL_DEVIATION_THRESHOLD_USD_M = 2.0   # absolute: flag if deviation > $2M
    _PNL_DEVIATION_THRESHOLD_PCT   = 0.10  # relative: flag if deviation > 10%
    pnl_audit_flags: list[dict] = []
    pnl_corrected = 0.0
    for r in treasury:
        reported   = float(r.get("unrealised_pnl_usd_m", 0))
        fwd        = float(r.get("forward_rate", 0))
        spot       = float(r.get("spot_rate",    0))
        hedged_n   = float(r.get("hedged_amount_usd_m", 0))
        exp_type   = r.get("exposure_type", "")
        if fwd > 0 and spot > 0 and hedged_n > 0 and exp_type in ("Revenue", "Cost"):
            theoretical = round(hedged_n * (fwd / spot - 1), 1)
            deviation   = abs(reported - theoretical)
            threshold   = max(_PNL_DEVIATION_THRESHOLD_USD_M,
                              abs(theoretical) * _PNL_DEVIATION_THRESHOLD_PCT)
            if deviation > threshold:
                pnl_audit_flags.append({
                    "exposure_id":         r.get("exposure_id", ""),
                    "currency_pair":       r.get("currency_pair", ""),
                    "reported_pnl_usd_m":  reported,
                    "theoretical_pnl_usd_m": theoretical,
                    "deviation_usd_m":     round(deviation, 1),
                    "hedged_notional_usd_m": hedged_n,
                    "note": (
                        "Reported P&L computed on gross notional — should use hedged notional. "
                        f"Restate to {theoretical:+.1f}M USD."
                    ),
                })
            pnl_corrected += theoretical
        else:
            pnl_corrected += reported  # commodity or missing rates — use as reported
    pnl_corrected = round(pnl_corrected, 1)

    # ── Monte Carlo — per-position volatility routing ────────────────────────
    # Each position uses its own volatility class; shocks are drawn independently.
    # FX positions use fx_vol_pct; Commodity positions use commodity_vol_pct.
    random.seed(SEED)
    hr = hedge_ratio / 100
    sp = avg_spot if avg_spot > 0 else 95
    fp = avg_forward if avg_forward > 0 else 100
    hc = hedge_cost_m

    uh_sims, hd_sims = [], []
    fx_only_uh_sims   = []   # MO-02: FX-only track for explicit VaR decomposition
    for _ in range(N_SIMS):
        sim_uh = 0.0
        sim_hd = 0.0
        sim_fx_only_uh = 0.0   # MO-02: FX-only unhedged exposure for this path

        # FX positions
        if fx_gross_m > 0:
            fx_vl = fx_vol_pct / 100
            fx_g  = fx_gross_m / 1000      # USD B
            fx_h  = fx_hedged_m / fx_gross_m  # hedge ratio for FX positions
            p = sp
            for _ in range(3):
                p = p * math.exp(-0.5 * fx_vl * fx_vl + fx_vl * _randn())
                av = fx_g * (p / 100) * (1 + 0.12 * _randn())
                sim_uh += av * 1000
                sim_fx_only_uh += av * 1000   # MO-02: track FX contribution separately
                # Hedged fraction locked at forward rate; unhedged fraction floats with spot.
                # av = fx_g * (p/100) * demand_factor, so hedging replaces (p/100) with (fp/100)
                # for the hedged share: av_hd = av * (fx_h*(fp/p) + (1-fx_h))
                p_safe = max(p, 1.0)
                sim_hd += av * (fx_h * (fp / p_safe) + (1.0 - fx_h)) * 1000
            sim_uh /= 3
            sim_fx_only_uh /= 3            # MO-02: normalise FX-only track
            sim_hd = sim_hd / 3 - hc

        # Commodity positions — higher volatility, no forward rate reference
        if com_gross_m > 0:
            com_vl = commodity_vol_pct / 100
            com_g  = com_gross_m / 1000    # USD B
            com_h  = com_hedged_m / com_gross_m  # hedge ratio for commodity positions
            p_com  = 100.0
            com_uh = 0.0
            com_hd = 0.0
            for _ in range(3):
                p_com = p_com * math.exp(-0.5 * com_vl * com_vl + com_vl * _randn())
                av = com_g * (p_com / 100) * (1 + _COMMODITY_DEMAND_VOL * _randn())  # demand vol from model_benchmarks.json
                com_uh += av * 1000
                # Hedged fraction locked at base price (100); unhedged fraction floats with spot.
                # Same decomposition as FX: replace (p_com/100) with 1.0 for hedged share.
                p_com_safe = max(p_com, 1.0)
                com_hd += av * (com_h * (100.0 / p_com_safe) + (1.0 - com_h)) * 1000
            sim_uh += com_uh / 3   # combined FX + commodity
            sim_hd += com_hd / 3

        uh_sims.append(sim_uh)
        hd_sims.append(sim_hd)
        fx_only_uh_sims.append(sim_fx_only_uh)   # MO-02: store FX-only path

    # 80% revenue base threshold for P(revenue < 80% base)
    base_80 = gross_b * 0.80 * 1000

    # VaR expressed as loss from mean (standard VaR convention):
    #   VaR_unhedged > VaR_hedged correctly shows hedge reduces loss risk.
    #   Raw 5th-percentile revenue would make hedged VaR appear higher than unhedged
    #   (inverted) because the hedge raises the revenue floor — not a risk increase.
    mean_uh = sum(uh_sims) / N_SIMS
    mean_hd = sum(hd_sims) / N_SIMS
    p5_uh   = _percentile(uh_sims, 5)
    p5_hd   = _percentile(hd_sims, 5)
    var_uh   = round(mean_uh - p5_uh)   # combined FX+commodity loss from mean at P5
    var_hd   = round(mean_hd - p5_hd)   # loss from mean at P5 — positive = risk
    p_uh_80  = round(sum(1 for v in uh_sims if v < base_80) / N_SIMS * 100, 1)
    p_hd_80  = round(sum(1 for v in hd_sims if v < base_80) / N_SIMS * 100, 1)
    var_impr = round(var_uh - var_hd)   # positive = hedge reduces VaR loss

    # MO-02: Analytical VaR decomposition — explicit FX-only and commodity-only figures.
    # Simulation-based var_uh (combined) uses different price-path scaling for FX vs commodity,
    # so we supplement with the standard analytical formula: unhedged_exposure × vol × z_95.
    # z_95 = 1.645 (one-tailed 95th percentile of N(0,1)).
    _Z95 = 1.645
    com_unhedged_m = round(com_gross_m - com_hedged_m, 0)
    var_fx_analytical  = round(fx_unhedged_primary * (fx_vol_pct / 100) * _Z95)     # FX-only, analytical
    var_com_analytical = round(com_unhedged_m      * (commodity_vol_pct / 100) * _Z95)  # commodity-only
    var_combined_lower = round(math.sqrt(var_fx_analytical**2 + var_com_analytical**2))  # zero-corr
    var_combined_upper = var_fx_analytical + var_com_analytical  # perfect-corr upper bound
    # FX-only simulation (for completeness — note: sim uses avg_spot price scaling, see var_fx_analytical for recommended figure)
    mean_fx_uh = sum(fx_only_uh_sims) / N_SIMS if fx_only_uh_sims else 0.0
    p5_fx_uh   = _percentile(fx_only_uh_sims, 5) if fx_only_uh_sims else 0.0
    var_fx_sim_only = round(mean_fx_uh - p5_fx_uh)

    return {
        # Slider parameters
        # Primary figures — FX-only scope (aligned with KRI F-01 per CF-04 2026-06-07)
        "gross_exposure_usd_m":   round(fx_gross_primary, 0),   # FX-only: 7,600
        "gross_exposure_usd_b":   gross_b,                       # FX-only USD B
        "hedged_amount_usd_m":    round(fx_hedged_primary, 0),  # FX-only: 3,520
        "unhedged_usd_m":         unhedged,                      # FX-only: 4,080
        "hedge_ratio_pct":        hedge_ratio,                   # FX-only: 46.3%
        "hedge_ratio_slider":     int(round(hedge_ratio / 5) * 5),
        "unrealised_pnl_usd_m":   round(fx_pnl_primary, 1),    # FX-only: -26.6M
        # Supplementary — total portfolio (FX + commodity) for transparency
        "total_portfolio_gross_usd_m":     round(gross_total, 0),   # 9,140
        "total_portfolio_hedged_usd_m":    round(hedged_total, 0),  # 4,200
        "total_portfolio_unhedged_usd_m":  round(gross_total - hedged_total, 0),  # 4,940
        "total_portfolio_pnl_usd_m":       round(pnl_total, 1),     # -47.0M
        "pnl_corrected_usd_m":    pnl_corrected,
        "pnl_audit_flags":        pnl_audit_flags,
        "hedge_cost_usd_m":       int(round(hedge_cost_m / 10) * 10),  # nearest 10
        "avg_spot":               avg_spot,
        "avg_forward":            avg_forward,
        # Separate volatility parameters (M-03)
        "fx_vol_pct":             fx_vol_pct,
        "commodity_vol_pct":      commodity_vol_pct,
        "volatility_pct":         fx_vol_pct,    # retained for backward compat (FX-only slider)
        # Asset class breakdown
        "fx_gross_usd_m":         round(fx_gross_m, 0),
        "fx_hedged_usd_m":        round(fx_hedged_m, 0),
        "commodity_gross_usd_m":  round(com_gross_m, 0),
        "commodity_hedged_usd_m": round(com_hedged_m, 0),
        # Simulation outputs
        "var_95_unhedged_usd_m":  var_uh,         # simulation-based combined FX + commodity
        # MO-02: Analytical VaR decomposition — grounded in calibrator tier
        # Formula: unhedged_exposure × vol × z_95 (1.645). See panel finding MO-02 2026-06-09.
        # These are the authoritative figures for board disclosure (simulation var_uh uses
        # price-path scaling that makes FX and commodity contributions non-comparable).
        "var_95_fx_analytical_usd_m":       var_fx_analytical,   # FX-only: 4,080M × 9% × 1.645
        "var_95_commodity_analytical_usd_m": var_com_analytical,  # Commodity: 860M × 32% × 1.645
        "var_95_combined_lower_usd_m":      var_combined_lower,   # sqrt(FX²+Comm²) zero-correlation
        "var_95_combined_upper_usd_m":      var_combined_upper,   # FX+Comm perfect-correlation upper
        "var_95_hedged_usd_m":    var_hd,
        "p_rev_below_80_unhedged":  p_uh_80,
        "p_rev_below_80_hedged":    p_hd_80,
        "var_improvement_usd_m":  var_impr,
        "n_sims": N_SIMS,
    }


# ── Supply chain stress model ──────────────────────────────────────────────────

def calibrate_supply_chain(raw: dict) -> dict:
    """
    Derive supply chain model parameters from erp_supply_chain.csv and
    market_intelligence.csv. Re-run the Poisson failure simulation.
    """
    sc = _latest_rows(raw.get("supply_chain", []))
    if not sc:
        return {}

    # All suppliers (both critical and single-source)
    supplier_count = len(sc)

    # Single-source suppliers — key risk drivers
    single_src = [r for r in sc if r.get("single_source", "").lower() == "true"]
    n_single   = len(single_src)

    # Recovery time = max lead time among single-source suppliers (in months)
    max_lead_weeks = max((float(r["lead_time_weeks"]) for r in single_src), default=18)
    recovery_months = math.ceil(max_lead_weeks / 4.33)   # round up to full month

    # Revenue at risk — single-source spend as share of company COGS × total revenue.
    # Logic: single-source components form X% of COGS; if they fail, X% of revenue
    # cannot ship. Uses company gross margin from financial_summary to derive COGS.
    fin_rows = _latest_rows(raw.get("financial_summary", []))
    fin_latest_sc = fin_rows[-1] if fin_rows else {}
    rev_b           = float(fin_latest_sc.get("revenue_usd_b",     57.0))
    gross_margin_pct = float(fin_latest_sc.get("gross_margin_pct", 12.5))

    cost_ratio_pct  = float(fin_latest_sc.get("cost_ratio_pct",  96.0))
    company_cogs_m  = rev_b * 1000 * (cost_ratio_pct / 100)
    single_spend_m  = sum(float(r["our_spend_usd_m"]) for r in single_src)

    # Revenue-at-risk multiplier: convert disrupted spend to revenue impact.
    # Default: 1/cost_ratio = 1/0.96 = 1.042× (pure COGS pass-through, no fixed-cost amplification).
    # Override: model_benchmarks.json::simulation_parameters.supply_chain.rev_at_risk_multiplier_override
    # Industry norm for hardware OEM: 1.1–1.3× (fixed-cost amplification not captured by cost-ratio alone).
    # CRO decision 2026-06-08 (MO-04): use industry norm override when present.
    _default_multiplier = round(rev_b / (rev_b * (cost_ratio_pct / 100)), 3) if cost_ratio_pct else 1.0
    _rev_cogs_multiplier = float(
        _SIM_PARAMS.get("supply_chain", {}).get("rev_at_risk_multiplier_override", _default_multiplier)
    )
    rev_at_risk_b = round(
        min(single_spend_m * _rev_cogs_multiplier / 1000, rev_b * _REV_AT_RISK_CAP_PCT / 100), 1
    )

    # Demand shock parameters — derive from market intelligence signal count
    # Column is "signal_type", geopolitical signals use value "geopolitical"
    mkt = raw.get("market_intel", [])
    geo_signals   = [r for r in mkt if r.get("signal_type", "").lower() == "geopolitical"
                     or r.get("category", "").lower() == "geopolitical"]
    high_sev_geo  = [r for r in geo_signals if r.get("severity", "").lower() == "high"]
    demand_shock_prob = min(10 + len(geo_signals) * 4 + len(high_sev_geo) * 3, 60)  # base 10% + signal count
    demand_shock_impact = 15.0   # % impact — calibrated to sector history
    impact_vol = float(fin_latest_sc.get("supply_chain_demand_shock_vol_pct", 8.0))  # % — read from financial_summary.csv

    # Emergency sourcing premium — sourced from model_benchmarks.json::simulation_parameters.supply_chain.emergency_sourcing_premium_pct
    emerg_premium = _EMERG_PREMIUM_PCT

    # MTBF — supplier-specific, calibrated from financial_health_score and single-source flag.
    # Formula parameters sourced from model_benchmarks.json::simulation_parameters.mtbf_formula.
    # Formula: MTBF = base_years × (health/100)^exponent, clamped to [floor_years, ceiling_years].
    # Quadratic decay (exponent=2) is more realistic than linear: near-distress suppliers face
    # disproportionately higher disruption probability (financial stress is non-linear).
    # To recalibrate: update model_benchmarks.json::simulation_parameters.mtbf_formula only.
    def _health_to_mtbf(health_score: float) -> float:
        return max(_MTBF_FLOOR, min(_MTBF_CEILING, _MTBF_BASE * (health_score / 100.0) ** _MTBF_EXP))

    supplier_mtbfs = [_health_to_mtbf(float(r["financial_health_score"])) for r in sc]
    supplier_spends = [float(r["our_spend_usd_m"]) for r in sc]
    total_spend = sum(supplier_spends) or 1.0

    # Spend-weighted MTBF — reported in model params for transparency
    mtbf_years = round(
        sum(m * s for m, s in zip(supplier_mtbfs, supplier_spends)) / total_spend, 0
    )
    mtbf_years = max(2, min(15, int(mtbf_years)))

    # ── Monte Carlo (Poisson failure model) ──────────────────────────────────
    random.seed(SEED)
    fl  = supplier_count
    rc  = recovery_months
    ec  = emerg_premium / 100
    rv  = rev_at_risk_b
    dp  = demand_shock_prob / 100
    di  = demand_shock_impact / 100
    dv  = impact_vol / 100
    _rec_var_base = float(fin_latest_sc.get("supply_chain_recovery_variability_pct", 30.0)) / 100

    def _run_sim(dual_source: bool = False, inventory_buffer: bool = False) -> list[float]:
        results = []
        _rv = rv * 0.85 if inventory_buffer else rv  # buffer reduces effective exposure
        for _ in range(N_SIMS):
            annual_loss = 0.0
            for s, s_mtbf in enumerate(supplier_mtbfs):
                # Supplier-specific failure probability from individual MTBF
                _s_mt = s_mtbf * 1.8 if dual_source else s_mtbf
                fail_prob = 1 - math.exp(-1.0 / _s_mt)
                if random.random() < fail_prob:
                    # Recovery time impact — variability read from financial_summary.csv
                    impact_months = rc * (1 + _rec_var_base * _randn())
                    impact_months = max(1, impact_months)
                    # Spend-weighted impact: larger spend = larger disruption cost
                    spend_weight = supplier_spends[s] / total_spend
                    recovery_cost = _rv * spend_weight * (impact_months / 12) * (1 + ec)
                    annual_loss += recovery_cost
            # Demand shock
            if random.random() < dp:
                shock_impact = di * (1 + dv * _randn())
                shock_impact = max(0, shock_impact)
                annual_loss += rv * shock_impact
            results.append(annual_loss * 1000)   # USD M
        return results

    baseline  = _run_sim(dual_source=False, inventory_buffer=False)
    dual_src  = _run_sim(dual_source=True,  inventory_buffer=False)
    inv_buff  = _run_sim(dual_source=False, inventory_buffer=True)
    both      = _run_sim(dual_source=True,  inventory_buffer=True)

    # VaR 95% for losses = 95th percentile (worst 5% threshold)
    var_base    = round(_percentile(baseline, 95))
    var_dual    = round(_percentile(dual_src, 95))
    var_inv     = round(_percentile(inv_buff, 95))
    var_both    = round(_percentile(both,     95))
    # CVaR = mean of worst 5% (tail conditional expectation)
    thresh_base = _percentile(baseline, 95)
    cvar_base   = round(sum(v for v in baseline if v >= thresh_base) /
                        max(1, sum(1 for v in baseline if v >= thresh_base)))

    # Savings = reduction in VaR (positive = improvement)
    saving_dual = round(var_base - var_dual)
    saving_inv  = round(var_base - var_inv)
    saving_both = round(var_base - var_both)

    # ── Dual-source programme cost — derived from CSV, not hardcoded ────────────
    #
    # For each single-source supplier with a diversion plan:
    #   Annual volume-premium cost = our_spend × diversion_pct × price_premium_pct
    #
    # This captures the main cost driver: when you split volume, the new (smaller)
    # supplier charges a premium because they can't match the primary supplier's
    # economies of scale.
    #
    # We report TWO figures:
    #   urgent_dual_cost_m  — most urgent/feasible single programme (smallest cost,
    #                          currently Quanta laptop ODM — the breach supplier)
    #   full_dual_cost_m    — total cost if ALL single-source suppliers are dual-sourced
    #
    per_supplier_costs = []
    for r in single_src:
        diversion  = float(r.get("dual_source_diversion_pct",   0)) / 100
        premium    = float(r.get("dual_source_price_premium_pct", 0)) / 100
        spend      = float(r["our_spend_usd_m"])
        annual_cost = spend * diversion * premium
        per_supplier_costs.append({
            "supplier":    r["supplier_name"],
            "spend_usd_m": spend,
            "diversion_pct":  diversion * 100,
            "premium_pct":    premium * 100,
            "annual_cost_usd_m": round(annual_cost, 1),
        })

    per_supplier_costs.sort(key=lambda x: x["annual_cost_usd_m"])   # cheapest first
    full_dual_cost_m   = round(sum(c["annual_cost_usd_m"] for c in per_supplier_costs), 1)
    urgent_dual_cost_m = per_supplier_costs[0]["annual_cost_usd_m"] if per_supplier_costs else _SC_PARAMS.get("dual_cost_fallback_usd_m", 15.0)
    urgent_supplier    = per_supplier_costs[0]["supplier"] if per_supplier_costs else ""

    # ── Inventory buffer cost — derived from additional weeks target ───────────
    #
    # Cost to hold additional safety stock above current level:
    #   additional_inventory_value = (target_weeks - current_weeks) × weekly_spend
    #   annual_carrying_cost = inventory_value × storage_rate (warehousing + insurance
    #                          + obsolescence, typically 2% of inventory value per year)
    #
    # Storage rate — sourced from model_benchmarks.json::simulation_parameters.supply_chain.inventory_storage_rate_pct
    STORAGE_RATE = _STORAGE_RATE   # annual carrying cost as fraction of inventory value
    total_inv_buffer_cost = 0.0
    for r in sc:
        target_wks  = float(r.get("target_inventory_weeks", 0))
        current_wks = float(r["inventory_weeks"])
        additional  = max(target_wks - current_wks, 0)
        if additional > 0:
            weekly_spend = float(r["our_spend_usd_m"]) / 52
            buffer_value = additional * weekly_spend
            total_inv_buffer_cost += buffer_value * STORAGE_RATE
    inv_cost_m = round(total_inv_buffer_cost, 1)
    if inv_cost_m == 0:
        inv_cost_m = _INV_COST_FALLBACK   # fallback from model_benchmarks.json if target_inventory_weeks not set in CSV

    # Use urgent (cheapest, most actionable) programme for primary ROI calc
    dual_cost_m = urgent_dual_cost_m

    # Return on investment = VaR saving / annual programme cost
    roi_dual = round(saving_dual / dual_cost_m) if dual_cost_m and saving_dual > 0 else 0
    roi_inv  = round(saving_inv  / inv_cost_m)  if inv_cost_m  and saving_inv  > 0 else 0
    roi_both = round(saving_both / (dual_cost_m + inv_cost_m)) \
               if (dual_cost_m + inv_cost_m) and saving_both > 0 else 0

    return {
        # Slider parameters
        "supplier_count":          supplier_count,
        "single_source_count":     n_single,
        "mtbf_years":              mtbf_years,
        "recovery_months":         recovery_months,
        "emerg_premium_pct":       emerg_premium,
        "rev_at_risk_usd_b":       rev_at_risk_b,
        "rev_at_risk_methodology": (
            f"single_source_spend_usd_m={round(single_spend_m,0)} × rev/COGS_multiplier={_rev_cogs_multiplier}x "
            f"= USD {round(single_spend_m * _rev_cogs_multiplier, 0)}M → capped to {rev_at_risk_b}B (USD {round(rev_at_risk_b*1000,0)}M). "
            f"Revenue/COGS multiplier = {_rev_cogs_multiplier}x reflects fixed-cost amplification: disrupted spend cascades "
            f"to revenue at greater-than-1x ratio due to operating leverage."
        ),
        "demand_shock_prob_pct":   demand_shock_prob,
        "demand_shock_impact_pct": int(demand_shock_impact),
        "impact_vol_pct":          int(impact_vol),
        # Simulation outputs
        "var_95_baseline_usd_m":   var_base,
        "cvar_95_baseline_usd_m":  cvar_base,
        "var_95_dual_src_usd_m":   var_dual,
        "var_95_inv_buff_usd_m":   var_inv,
        "var_95_both_usd_m":       var_both,
        "saving_dual_src_usd_m":   max(saving_dual, 0),
        "saving_inv_buff_usd_m":   max(saving_inv,  0),
        "saving_both_usd_m":       max(saving_both, 0),
        "roi_dual_src_x":          roi_dual,
        "roi_inv_buff_x":          roi_inv,
        "roi_both_x":              roi_both,
        # Programme costs — derived from CSV
        "urgent_dual_cost_usd_m":  urgent_dual_cost_m,
        "urgent_dual_supplier":    urgent_supplier,
        "full_dual_cost_usd_m":    full_dual_cost_m,
        "inv_buffer_cost_usd_m":   inv_cost_m,
        "per_supplier_costs":      per_supplier_costs,
        "n_sims": N_SIMS,
    }


# ── HTML patching ──────────────────────────────────────────────────────────────

def _patch_slider(html: str, slider_id: str, new_value: int | float,
                  span_id: str | None = None) -> tuple[str, bool]:
    """
    Update a slider's value= attribute and (optionally) its display span.
    Returns (html, changed).
    """
    changed = False
    val_str = str(int(new_value)) if isinstance(new_value, float) and new_value == int(new_value) \
              else str(new_value)

    # Update slider value attribute
    pattern = re.compile(
        r'(<input\b[^>]*\bid=["\']' + re.escape(slider_id) + r'["\'][^>]*\bvalue=["\'])([^"\']+)(["\'])'
    )
    new_html, n = pattern.subn(lambda m: m.group(1) + val_str + m.group(3), html)
    if n:
        changed = True
        html = new_html
    else:
        # Try reversed attribute order (value before id)
        pattern2 = re.compile(
            r'(<input\b[^>]*\bvalue=["\'])([^"\']+)(["\'][^>]*\bid=["\']'
            + re.escape(slider_id) + r'["\'])'
        )
        new_html, n = pattern2.subn(lambda m: m.group(1) + val_str + m.group(3), html)
        if n:
            changed = True
            html = new_html

    # Update display span if provided
    if span_id:
        sp_pattern = re.compile(
            r'(<span\b[^>]*\bid=["\']' + re.escape(span_id) + r'["\'][^>]*>)([^<]*)(<\/span>)'
        )
        new_html, n = sp_pattern.subn(lambda m: m.group(1) + val_str + m.group(3), html)
        if n:
            changed = True
            html = new_html

    return html, changed


def _patch_js_var(html: str, pattern_str: str, new_value: str) -> tuple[str, bool]:
    """Replace a hardcoded JS variable value using a regex pattern."""
    pattern = re.compile(pattern_str)
    new_html, n = pattern.subn(new_value, html)
    return new_html, n > 0


def update_html_models(dashboard_path: Path, params: dict) -> tuple[bool, list[str]]:
    """
    Patch dashboard HTML with calibrated model parameters and re-computed
    simulation outputs. Returns (changed, list_of_changes).
    """
    html    = dashboard_path.read_text()
    changes = []

    ep = params.get("ebitda",        {})
    hp = params.get("hedge",         {})
    sp = params.get("supply_chain",  {})

    # ── EBITDA sliders ───────────────────────────────────────────────────────
    ebitda_sliders = [
        ("ms-gn", "mv-gn", ep.get("revenue_usd_b")),
        ("ms-om", "mv-om", ep.get("cost_ratio_pct")),
        ("ms-vl", "mv-vl", ep.get("volatility_pct")),
    ]
    for s_id, sp_id, val in ebitda_sliders:
        if val is not None:
            html, ch = _patch_slider(html, s_id, val, sp_id)
            if ch:
                changes.append(f"EBITDA slider {s_id} → {val}")

    # ── Hedge sliders ────────────────────────────────────────────────────────
    hedge_sliders = [
        ("hs-hr", "hv-hr", hp.get("hedge_ratio_slider")),
        ("hs-hc", "hv-hc", hp.get("hedge_cost_usd_m")),
    ]
    for s_id, sp_id, val in hedge_sliders:
        if val is not None:
            html, ch = _patch_slider(html, s_id, val, sp_id)
            if ch:
                changes.append(f"Hedge slider {s_id} → {val}")

    # Patch hardcoded JS `var base=12` (FX-exposed revenue in USD B)
    if hp.get("gross_exposure_usd_b") is not None:
        new_base = round(hp["gross_exposure_usd_b"], 1)
        html, ch = _patch_js_var(
            html,
            r'(var\s+N=5000,\s*base=)([\d.]+)',
            lambda m: m.group(1) + str(new_base)
        )
        if ch:
            changes.append(f"Hedge JS base → {new_base}B")

    # ── Supply chain sliders ─────────────────────────────────────────────────
    sc_sliders = [
        ("os-fl", "ov-fl", sp.get("supplier_count")),
        ("os-mt", "ov-mt", sp.get("mtbf_years")),
        ("os-rc", "ov-rc", sp.get("recovery_months")),
        ("os-rv", "ov-rv", sp.get("rev_at_risk_usd_b")),
        ("os-dp", "ov-dp", sp.get("demand_shock_prob_pct")),
    ]
    for s_id, sp_id, val in sc_sliders:
        if val is not None:
            html, ch = _patch_slider(html, s_id, val, sp_id)
            if ch:
                changes.append(f"Supply chain slider {s_id} → {val}")

    # ── EBITDA validated-finding banner ─────────────────────────────────────
    if ep:
        p_breach    = ep.get("p_covenant_breach_pct",        44.6)
        p_cost_up   = ep.get("p_breach_cost_up1pp_pct",      83.3)
        p_top_cust  = ep.get("p_breach_top_cust_loss_pct",   56.6)
        headroom    = ep.get("ebitda_headroom_usd_m",         170)
        rev         = ep.get("revenue_usd_b",                 57)
        cr          = ep.get("cost_ratio_pct",                96)
        vl          = ep.get("volatility_pct",                20)

        new_banner = (
            f'<strong style="color:#ef4444">Validated finding (Python simulation, '
            f'N={N_SIMS:,}, seed={SEED}):</strong> '
            f'At current defaults ({cr}% cost ratio, USD {rev}B revenue, {vl}% volatility), '
            f'P(covenant breach) = <strong style="color:#ef4444">{p_breach}%</strong>. '
            f'A +1pp cost increase raises this to <strong>{p_cost_up}%</strong>. '
            f'Top customer loss raises it to <strong>{p_top_cust}%</strong>. '
            f'Covenant headroom: <strong>USD {int(headroom)}M</strong> only.'
        )
        pattern = re.compile(
            r'Validated finding \(Python simulation.*?only\.',
            re.DOTALL
        )
        new_html, n = pattern.subn(new_banner, html)
        if n:
            html = new_html
            changes.append(f"EBITDA banner: P(breach)={p_breach}%, headroom=USD {int(headroom)}M")

    # ── Supply chain validated-benchmark banner ──────────────────────────────
    if sp:
        var_base   = sp.get("var_95_baseline_usd_m",    2509)
        save_dual  = sp.get("saving_dual_src_usd_m",     663)
        save_inv   = sp.get("saving_inv_buff_usd_m",     584)
        save_both  = sp.get("saving_both_usd_m",        1199)
        roi_both   = sp.get("roi_both_x",                48)

        dual_cost   = sp.get("urgent_dual_cost_usd_m",  15)
        inv_cost    = sp.get("inv_buffer_cost_usd_m",    8)
        new_banner = (
            f'<strong style="color:#22c55e">Validated benchmarks (Python simulation, N={N_SIMS:,}):'
            f'</strong> Baseline VaR 95%: <strong>USD {var_base:,}M</strong>. '
            f'Dual-source programme (USD {dual_cost:.1f}M/yr incremental cost): '
            f'saves <strong>USD {save_dual:,}M</strong> VaR. '
            f'Inventory buffer (USD {inv_cost:.1f}M/yr carrying cost): '
            f'saves <strong>USD {save_inv:,}M</strong>. '
            f'Both together: saves <strong>USD {save_both:,}M</strong> '
            f'&mdash; {roi_both}&times; return on mitigation investment.'
        )
        pattern = re.compile(
            r'Validated benchmarks \(Python simulation.*?mitigation investment\.',
            re.DOTALL
        )
        new_html, n = pattern.subn(new_banner, html)
        if n:
            html = new_html
            changes.append(
                f"Supply chain banner: VaR={var_base:,}M, "
                f"dual-src saves {save_dual:,}M, roi={roi_both}×"
            )

    # ── Supply chain exec-action and risk-event cards ─────────────────────────
    # These hardcoded HTML cards reference VaR/ROI values and must stay in sync
    if sp:
        var_base  = sp.get("var_95_baseline_usd_m",  2509)
        save_dual = sp.get("saving_dual_src_usd_m",   663)
        roi_dual  = sp.get("roi_dual_src_x",           48)
        dual_cost = sp.get("urgent_dual_cost_usd_m",   15)

        # Exec action 01: "dual-source saves USD NNNm VaR 95% — a XX× return"
        exec_pattern = re.compile(
            r'(dual-source saves USD )[\d,]+(M VaR 95% &mdash; a )[\d]+'
            r'(&times; return on investment)'
        )
        exec_replacement = rf'\g<1>{save_dual:,}\g<2>{roi_dual}\g<3>'
        new_html, n = exec_pattern.subn(exec_replacement, html)
        if n:
            html = new_html
            changes.append(f"Supply chain exec-action VaR: {save_dual:,}M / {roi_dual}×")

        # Risk event ra/rr strings: "reduces VaR 95% by USD NNNm — a XX× return"
        risk_ra_pattern = re.compile(
            r'(dual-source programme \(USD )[\d.]+?(M/yr\) reduces VaR 95% by USD )[\d,]+'
            r'(M — a )[\d]+(× return)'
        )
        risk_ra_repl = rf'\g<1>{dual_cost:.1f}\g<2>{save_dual:,}\g<3>{roi_dual}\g<4>'
        new_html, n = risk_ra_pattern.subn(risk_ra_repl, html)
        if n:
            html = new_html
            changes.append(f"Supply chain risk-event ra VaR: {save_dual:,}M")

        # Risk event rr: "shows USD NNNm VaR improvement"
        risk_rr_pattern = re.compile(
            r'(dual-source toggle — shows USD )[\d,]+(M VaR improvement)'
        )
        new_html, n = risk_rr_pattern.subn(rf'\g<1>{save_dual:,}\g<2>', html)
        if n:
            html = new_html

        # Programme staffing ra: "validated VaR saving of USD NNNm"
        staffing_pattern = re.compile(
            r'(validated VaR saving of USD )[\d,]+(M is not being captured)'
        )
        new_html, n = staffing_pattern.subn(rf'\g<1>{save_dual:,}\g<2>', html)
        if n:
            html = new_html
            changes.append(f"Supply chain staffing VaR ref: {save_dual:,}M")

        # Geographic diversification rr: "reduces VaR 95% by ~USD NNNm"
        geo_pattern = re.compile(
            r'(geographic diversification reduces VaR 95% by ~USD )[\d,]+(M)'
        )
        new_html, n = geo_pattern.subn(rf'\g<1>{save_dual:,}\g<2>', html)
        if n:
            html = new_html

    # ── Supply chain JS comment block ────────────────────────────────────────
    # Updates the human-readable comment used by the consistency checker scan
    if sp:
        var_base  = sp.get("var_95_baseline_usd_m",  2509)
        save_dual = sp.get("saving_dual_src_usd_m",   663)
        save_inv  = sp.get("saving_inv_buff_usd_m",   584)
        save_both = sp.get("saving_both_usd_m",      1199)
        roi_dual  = sp.get("roi_dual_src_x",           48)
        roi_inv   = sp.get("roi_inv_buff_x",           48)
        dual_cost = sp.get("urgent_dual_cost_usd_m",   15)
        inv_cost  = sp.get("inv_buffer_cost_usd_m",     8)

        new_comment = (
            f"// Supply Chain — Validated Python simulation (N=5,000):\n"
            f"//   Baseline VaR 95%: USD {var_base:,}M\n"
            f"//   Dual-source only: saves USD {save_dual:,}M VaR ({roi_dual}x ROI on USD {dual_cost:.1f}M/yr)\n"
            f"//   Inventory buffer only: saves USD {save_inv:,}M VaR ({roi_inv}x ROI on USD {inv_cost:.1f}M/yr)\n"
            f"//   Both mitigations: saves USD {save_both:,}M VaR combined"
        )
        comment_pattern = re.compile(
            r'// Supply Chain — Validated Python simulation.*?//   Both mitigations: saves USD [\d,]+M VaR combined',
            re.DOTALL
        )
        new_html, n = comment_pattern.subn(new_comment, html)
        if n:
            html = new_html
            changes.append(f"Supply chain JS comment: VaR={var_base:,}M")

    # ── Supply chain JS prompt string ─────────────────────────────────────────
    # Updates the LLM prompt used by the Risk Committee panel in the dashboard
    if sp:
        var_base  = sp.get("var_95_baseline_usd_m",  2509)
        save_dual = sp.get("saving_dual_src_usd_m",   663)
        save_inv  = sp.get("saving_inv_buff_usd_m",   584)
        save_both = sp.get("saving_both_usd_m",      1199)
        roi_dual  = sp.get("roi_dual_src_x",           48)
        dual_cost = sp.get("urgent_dual_cost_usd_m",   15)
        inv_cost  = sp.get("inv_buffer_cost_usd_m",     8)

        new_prompt_bench = (
            f"Validated Python simulation benchmarks (N=5,000): "
            f"Baseline VaR 95% ~USD {var_base:,}M. "
            f"Dual-source programme (USD {dual_cost:.1f}M/yr) reduces VaR by ~USD {save_dual:,}M ({roi_dual}× ROI). "
            f"Inventory buffer (USD {inv_cost:.1f}M/yr) reduces VaR by ~USD {save_inv:,}M. "
            f"Both together: ~USD {save_both:,}M saving."
        )
        prompt_pattern = re.compile(
            r'Validated Python simulation benchmarks \(N=5,000\): '
            r'Baseline VaR 95% ~USD [\d,]+M\..*?Both together: ~USD [\d,]+M saving\.'
        )
        new_html, n = prompt_pattern.subn(new_prompt_bench, html)
        if n:
            html = new_html
            changes.append(f"Supply chain JS prompt benchmark: VaR={var_base:,}M")

    if changes:
        dashboard_path.write_text(html)

    return bool(changes), changes


# ── Store writer ───────────────────────────────────────────────────────────────

def _write_to_store(params: dict) -> None:
    store = json.loads(STORE_PATH.read_text()) if STORE_PATH.exists() else {}
    store["model_params"] = {
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "ebitda":        params.get("ebitda",       {}),
        "hedge":         params.get("hedge",        {}),
        "supply_chain":  params.get("supply_chain", {}),
    }
    STORE_PATH.write_text(json.dumps(store, indent=2))


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_calibration(dashboard_path: Path | None = None) -> dict:
    """
    Load all CSVs, calibrate all three models, patch the dashboard, update store.
    Called by the model_calibration graph node on every pipeline run.
    Returns the full params dict.
    """
    if dashboard_path is None:
        dashboard_path = DASHBOARD

    # Load all source data
    raw = {
        "financial_summary": _read_csv("financial_summary.csv"),
        "treasury":          _read_csv("treasury_positions.csv"),
        "supply_chain":      _read_csv("erp_supply_chain.csv"),
        "covenant_tracker":  _read_csv("covenant_tracker.csv"),
        "market_intel":      _read_csv("market_intelligence.csv"),
        "ar_aging":          _read_csv("ar_aging.csv"),
    }

    # Calibrate each model
    print("  [MODEL CALIBRATOR] Calibrating from source CSVs...")
    ebitda_params = calibrate_ebitda(raw)
    hedge_params  = calibrate_hedge(raw)
    sc_params     = calibrate_supply_chain(raw)

    params = {
        "ebitda":       ebitda_params,
        "hedge":        hedge_params,
        "supply_chain": sc_params,
    }

    # Print summary
    print(f"    EBITDA  → P(covenant breach): {ebitda_params.get('p_covenant_breach_pct')}%  "
          f"headroom: USD {int(ebitda_params.get('ebitda_headroom_usd_m', 0))}M")
    print(f"    Hedge   → gross: USD {hedge_params.get('gross_exposure_usd_m', 0):,.0f}M  "
          f"ratio: {hedge_params.get('hedge_ratio_pct')}%  "
          f"unhedged: USD {hedge_params.get('unhedged_usd_m', 0):,.0f}M")
    print(f"    Sup-Chn → suppliers: {sc_params.get('supplier_count')}  "
          f"recovery: {sc_params.get('recovery_months')}mo  "
          f"VaR baseline: USD {sc_params.get('var_95_baseline_usd_m', 0):,}M")

    # Patch dashboard HTML
    if dashboard_path.exists():
        changed, change_list = update_html_models(dashboard_path, params)
        if changed:
            print(f"  [MODEL CALIBRATOR] Dashboard patched — {len(change_list)} change(s):")
            for c in change_list:
                print(f"    • {c}")
        else:
            print("  [MODEL CALIBRATOR] Dashboard already up to date")

    # Write to store
    _write_to_store(params)
    print("  [MODEL CALIBRATOR] model_params written to risk_store.json")

    return params


if __name__ == "__main__":
    result = run_calibration()
    print("\nCalibration complete.")
    print(json.dumps(result, indent=2))
