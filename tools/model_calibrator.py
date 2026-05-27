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

N_SIMS  = 5_000
SEED    = 42

# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv(name: str) -> list[dict]:
    p = DATA_DIR / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


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
    Derive EBITDA model parameters from financial_summary.csv and
    covenant_tracker.csv, then run the Monte Carlo simulation.

    Returns a dict of params + simulation outputs.
    """
    fin = raw.get("financial_summary", [])
    latest = fin[-1] if fin else {}

    # Core parameters — fall back to Lenovo Q2 FY26 defaults
    revenue_b    = float(latest.get("revenue_usd_b",    57.0))
    cost_ratio   = float(latest.get("cost_ratio_pct",   96.0))
    ebitda_b     = float(latest.get("ebitda_usd_b",      2.28))
    volatility   = 20.0   # % p.a. — calibrated to hardware sector; no direct CSV source
    drift        = 0.0
    demand_var   = 12.0   # demand variability %

    # Covenant floor from tracker
    cov_rows = raw.get("covenant_tracker", [])
    cov5 = next((r for r in cov_rows if r.get("covenant_id") == "COV005"), None)
    cov1 = next((r for r in cov_rows if r.get("covenant_id") == "COV001"), None)

    # Covenant floor on EBITDA (from COV005 threshold)
    cov_floor_b = float(cov5["threshold"]) if cov5 else 1.80

    # Net-Debt/EBITDA ceiling (COV001)
    net_debt_ebitda_ceil = float(cov1["threshold"]) if cov1 else 3.0

    # Implied net debt from financial summary
    fin_latest = fin[-1] if fin else {}
    net_debt_b = float(fin_latest.get("net_debt_usd_b", 6.384))
    # EBITDA headroom = binding constraint (minimum of COV001 and COV005)
    # COV005: EBITDA floor — headroom = ebitda - floor
    headroom_cov5_m = (ebitda_b - cov_floor_b) * 1000
    # COV001: Net Debt/EBITDA ceiling — EBITDA must stay above net_debt / ceiling
    ebitda_min_cov1 = net_debt_b / net_debt_ebitda_ceil if net_debt_ebitda_ceil else 0
    headroom_cov1_m = (ebitda_b - ebitda_min_cov1) * 1000
    # Binding = lower (more conservative) headroom
    ebitda_headroom_m = round(min(headroom_cov5_m, headroom_cov1_m), 0)

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

    for i in range(N_SIMS):
        # Price path (1-year horizon, 3 steps)
        p = base_price
        for _ in range(3):
            p = p * math.exp((dr - 0.5 * vl * vl) + vl * _randn())
        # Revenue with demand variability
        rev = rev_b * (p / 100) * (1 + (demand_var / 100) * _randn())
        rev = max(rev, 0)
        # EBITDA = revenue × (1 - cost_ratio) + noise
        ebitda = rev * (1 - cr) + 0.05 * rev * _randn()
        ebitda_sims.append(ebitda)

        # Breach checks
        breaches_cov5 = ebitda < cov_floor_b
        nd_ebitda = net_debt_b / ebitda if ebitda > 0 else 99
        breaches_cov1 = nd_ebitda > net_debt_ebitda_ceil
        cov5_breach.append(int(breaches_cov5))
        cov1_breach.append(int(breaches_cov5 or breaches_cov1))

        # +1pp cost sensitivity
        ebitda_c1 = rev * (1 - (cr + 0.01)) + 0.05 * rev * _randn()
        cost_up1pp_breach.append(int(ebitda_c1 < cov_floor_b or
                                     (net_debt_b / ebitda_c1 if ebitda_c1 > 0 else 99) > net_debt_ebitda_ceil))

        # Top customer loss (remove ~22% of ISG revenue ≈ 5% of total)
        ebitda_cl = (rev * 0.95) * (1 - cr) + 0.05 * (rev * 0.95) * _randn()
        top_cust_loss_breach.append(int(ebitda_cl < cov_floor_b or
                                        (net_debt_b / ebitda_cl if ebitda_cl > 0 else 99) > net_debt_ebitda_ceil))

    p_breach         = round(sum(cov1_breach) / N_SIMS * 100, 1)
    p_cost_up1pp     = round(sum(cost_up1pp_breach) / N_SIMS * 100, 1)
    p_top_cust_loss  = round(sum(top_cust_loss_breach) / N_SIMS * 100, 1)
    var_95           = round(_percentile(ebitda_sims, 5) * 1000, 0)   # USD M
    cvar_95          = round(sum(e for e in ebitda_sims
                                  if e <= _percentile(ebitda_sims, 5)) /
                              max(1, sum(1 for e in ebitda_sims
                                         if e <= _percentile(ebitda_sims, 5))) * 1000, 0)

    return {
        # Slider parameters
        "revenue_usd_b":   round(revenue_b, 1),
        "cost_ratio_pct":  round(cost_ratio, 1),
        "volatility_pct":  round(volatility, 1),
        "drift_pct":       round(drift, 1),
        "demand_var_pct":  round(demand_var, 1),
        # Covenant
        "covenant_floor_usd_b":     round(cov_floor_b, 2),
        "ebitda_headroom_usd_m":    ebitda_headroom_m,
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
    Re-run the hedge VaR simulation.
    """
    treasury = raw.get("treasury", [])
    if not treasury:
        return {}

    gross_total  = sum(float(r["gross_exposure_usd_m"]) for r in treasury)
    hedged_total = sum(float(r["hedged_amount_usd_m"])  for r in treasury)
    pnl_total    = sum(float(r["unrealised_pnl_usd_m"]) for r in treasury)

    hedge_ratio  = round(hedged_total / gross_total * 100, 1) if gross_total else 0
    unhedged     = round(gross_total - hedged_total, 0)
    gross_b      = round(gross_total / 1000, 2)          # USD B (for JS base variable)

    # Estimate hedge cost: all-in cost on hedged notional
    # Includes bid/ask spread, collateral cost, and time-value adjustment.
    # Industry benchmark for mixed forward/option book: ~1.5% p.a. on hedged notional.
    hedge_cost_m = round(hedged_total * 0.015, 0)   # 1.5% × hedged notional (1yr)

    # Weighted average spot and forward rates (revenue exposures only)
    rev_rows = [r for r in treasury if r.get("exposure_type") == "Revenue"]
    if rev_rows:
        avg_spot    = round(sum(float(r["spot_rate"])    for r in rev_rows) / len(rev_rows), 1)
        avg_forward = round(sum(float(r["forward_rate"]) for r in rev_rows) / len(rev_rows), 1)
    else:
        avg_spot, avg_forward = 95, 100

    # Volatility: use a sector-calibrated default (FX vol ~ 12-18% p.a.)
    volatility = 18.0

    # ── Monte Carlo ─────────────────────────────────────────────────────────
    random.seed(SEED)
    vl = volatility / 100
    hr = hedge_ratio / 100
    sp = avg_spot if avg_spot > 0 else 95
    fp = avg_forward if avg_forward > 0 else 100
    hc = hedge_cost_m

    uh_sims, hd_sims = [], []
    for _ in range(N_SIMS):
        total_uh = 0
        total_hd = 0
        p = sp
        for _ in range(3):
            p = p * math.exp(-0.5 * vl * vl + vl * _randn())
            av = gross_b * (p / 100) * (1 + 0.12 * _randn())
            hv = min(gross_b * (fp / 100) * hr, av)
            total_uh += av * 1000
            total_hd += (hv * 1000 + (av - min(gross_b * (fp / 100) * hr, av)) * 1000 - hc)
        uh_sims.append(total_uh / 3)
        hd_sims.append(total_hd / 3)

    # 80% revenue base threshold for P(revenue < 80% base)
    base_80 = gross_b * 0.80 * 1000

    var_uh   = round(_percentile(uh_sims, 5))
    var_hd   = round(_percentile(hd_sims, 5))
    p_uh_80  = round(sum(1 for v in uh_sims if v < base_80) / N_SIMS * 100, 1)
    p_hd_80  = round(sum(1 for v in hd_sims if v < base_80) / N_SIMS * 100, 1)
    var_impr = round(var_hd - var_uh)

    return {
        # Slider parameters
        "gross_exposure_usd_m":   round(gross_total, 0),
        "gross_exposure_usd_b":   gross_b,
        "hedged_amount_usd_m":    round(hedged_total, 0),
        "unhedged_usd_m":         unhedged,
        "hedge_ratio_pct":        hedge_ratio,
        "hedge_ratio_slider":     int(round(hedge_ratio / 5) * 5),  # nearest step-5
        "unrealised_pnl_usd_m":   round(pnl_total, 1),
        "hedge_cost_usd_m":       int(round(hedge_cost_m / 10) * 10),  # nearest 10
        "avg_spot":               avg_spot,
        "avg_forward":            avg_forward,
        "volatility_pct":         volatility,
        # Simulation outputs
        "var_95_unhedged_usd_m":  var_uh,
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
    sc = raw.get("supply_chain", [])
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
    fin_rows = raw.get("financial_summary", [])
    fin_latest_sc = fin_rows[-1] if fin_rows else {}
    rev_b           = float(fin_latest_sc.get("revenue_usd_b",     57.0))
    gross_margin_pct = float(fin_latest_sc.get("gross_margin_pct", 12.5))

    company_cogs_m  = rev_b * 1000 * (1 - gross_margin_pct / 100)
    single_spend_m  = sum(float(r["our_spend_usd_m"]) for r in single_src)

    # Cap at 30% of revenue (conservative upper bound — not all spend = unique revenue)
    rev_at_risk_b = round(
        min(single_spend_m / company_cogs_m * rev_b, rev_b * 0.30), 0
    ) if company_cogs_m else 12.0

    # Demand shock parameters — derive from market intelligence signal count
    # Column is "signal_type", geopolitical signals use value "geopolitical"
    mkt = raw.get("market_intel", [])
    geo_signals   = [r for r in mkt if r.get("signal_type", "").lower() == "geopolitical"
                     or r.get("category", "").lower() == "geopolitical"]
    high_sev_geo  = [r for r in geo_signals if r.get("severity", "").lower() == "high"]
    demand_shock_prob = min(10 + len(geo_signals) * 4 + len(high_sev_geo) * 3, 60)  # base 10% + signal count
    demand_shock_impact = 15.0   # % impact — calibrated to sector history
    impact_vol = 8.0

    # Emergency sourcing premium — from market conditions
    emerg_premium = 25   # % — standard industry benchmark

    # MTBF — use mean financial health score as a proxy
    # Lower health score → shorter MTBF
    mean_health = sum(float(r["financial_health_score"]) for r in sc) / len(sc)
    # Map health score (0-100) → MTBF (2-15 years): MTBF = 2 + 13*(health/100)
    mtbf_years = round(2 + 13 * (mean_health / 100), 0)
    mtbf_years = max(2, min(15, int(mtbf_years)))

    # ── Monte Carlo (Poisson failure model) ──────────────────────────────────
    random.seed(SEED)
    fl  = supplier_count
    mt  = mtbf_years
    rc  = recovery_months
    ec  = emerg_premium / 100
    rv  = rev_at_risk_b
    dp  = demand_shock_prob / 100
    di  = demand_shock_impact / 100
    dv  = impact_vol / 100

    def _run_sim(dual_source: bool = False, inventory_buffer: bool = False) -> list[float]:
        results = []
        _mt = mt * 1.8 if dual_source else mt   # dual-source extends effective MTBF
        _rv = rv * 0.85 if inventory_buffer else rv  # buffer reduces effective exposure
        for _ in range(N_SIMS):
            annual_loss = 0.0
            for s in range(fl):
                # Poisson failure probability
                fail_prob = 1 - math.exp(-1.0 / _mt)
                if random.random() < fail_prob:
                    # Recovery time impact
                    impact_months = rc * (1 + 0.3 * _randn())
                    impact_months = max(1, impact_months)
                    recovery_cost = (_rv / fl) * (impact_months / 12) * (1 + ec)
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
    urgent_dual_cost_m = per_supplier_costs[0]["annual_cost_usd_m"] if per_supplier_costs else 15.0
    urgent_supplier    = per_supplier_costs[0]["supplier"] if per_supplier_costs else "Quanta"

    # ── Inventory buffer cost — derived from additional weeks target ───────────
    #
    # Cost to hold additional safety stock above current level:
    #   additional_inventory_value = (target_weeks - current_weeks) × weekly_spend
    #   annual_carrying_cost = inventory_value × storage_rate (warehousing + insurance
    #                          + obsolescence, typically 2% of inventory value per year)
    #
    STORAGE_RATE = 0.02   # 2% p.a. of inventory value (cash cost, not WACC)
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
        inv_cost_m = 8.0   # fallback if target_inventory_weeks not set

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
        "rev_at_risk_usd_b":       int(rev_at_risk_b),
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
