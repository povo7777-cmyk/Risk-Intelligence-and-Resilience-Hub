"""
tools/dashboard_updater.py
Patches the Risk Intelligence and Resilience Hub index.html with
updated KRI values and approved executive recommendation text.
Includes idempotency check and sparkline update.
"""

import json, re, shutil
from datetime import datetime, timezone
from pathlib import Path

STORE_PATH = Path(__file__).parent.parent / "api" / "risk_store.json"
BACKUP_DIR = Path(__file__).parent.parent / "dashboard" / "backups"

STATUS_TO_TS = {"ok": "ok", "amber": "am", "breach": "br"}

KRI_NAME_MAP = {
    "single_source_concentration":   "Single-source concentration",
    "inventory_cover_weeks":         "Inventory cover (weeks)",
    "supplier_distress_flags":       "Supplier financial distress flags",
    "mttd_days":                     "Mean time to detect (MTTD)",
    "patch_compliance_pct":          "Patch compliance rate",
    "critical_vulns_open_gt30d":     "Critical vulnerabilities open >30 days",
    "it_rto_hours":                  "IT system RTO",
    "field_failure_rate_pct":        "Field failure rate",
    "recall_readiness_score_pct":    "Recall readiness score",
    "safety_incidents_ytd":          "Confirmed product safety incidents YTD",
    "tech_attrition_rate_pct":       "Tech role attrition rate",
    "critical_open_roles_gt60d":     "Critical open roles >60 days",
    "svp_succession_coverage_pct":   "SVP+ succession plan coverage",
}

KRI_FORMAT = {
    "single_source_concentration":   lambda v: f"{v}%",
    "inventory_cover_weeks":         lambda v: f"{v}wk",
    "supplier_distress_flags":       lambda v: str(int(v)),
    "mttd_days":                     lambda v: f"{int(v)} days",
    "patch_compliance_pct":          lambda v: f"{v}%",
    "critical_vulns_open_gt30d":     lambda v: str(int(v)),
    "it_rto_hours":                  lambda v: f"{v}h",
    "field_failure_rate_pct":        lambda v: f"{v}%",
    "recall_readiness_score_pct":    lambda v: f"{int(v)}%",
    "safety_incidents_ytd":          lambda v: str(int(v)),
    "tech_attrition_rate_pct":       lambda v: f"{v}%",
    "critical_open_roles_gt60d":     lambda v: str(int(v)),
    "svp_succession_coverage_pct":   lambda v: f"{int(v)}%",
}

EXEC_REC_IDS = {
    "bcm":          "ec-bcm",
    "ebitda":       "ec-mc",
    "fx":           "ec-hg",
    "supply_chain": "ec-op",
}


def backup_dashboard(html_path):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"index_{ts}.html"
    shutil.copy2(html_path, backup_path)
    return backup_path


def _update_kri(html, kri_name, new_value, new_status):
    js_name = KRI_NAME_MAP.get(kri_name)
    if not js_name:
        return html, False
    fmt = KRI_FORMAT.get(kri_name, lambda v: str(v))
    new_cur = fmt(new_value)
    new_ts = STATUS_TO_TS.get(new_status, "ok")
    pattern = re.compile(
        r"(\{n:'" + re.escape(js_name) + r"',cur:')(.*?)(',a:'[^']*',r:'[^']*',tr:\[[^\]]*\],ts:')(.*?)(')",
        re.DOTALL
    )
    match = pattern.search(html)
    if not match:
        return html, False
    if match.group(2) == new_cur and match.group(4) == new_ts:
        return html, False
    new_frag = match.group(1) + new_cur + match.group(3) + new_ts + match.group(5)
    return html[:match.start()] + new_frag + html[match.end():], True


def _update_sparkline(html, kri_name, new_value):
    js_name = KRI_NAME_MAP.get(kri_name)
    if not js_name:
        return html
    pattern = re.compile(
        r"(n:'" + re.escape(js_name) + r"'.*?tr:\[)([^\]]+)(\])",
        re.DOTALL
    )
    match = pattern.search(html)
    if not match:
        return html
    old_values = [float(x.strip()) for x in match.group(2).split(",")]
    new_values = old_values[1:] + [round(new_value, 3)]
    new_array = ",".join(str(v) for v in new_values)
    return html[:match.start(1)] + match.group(1) + new_array + match.group(3) + html[match.end(3):]


def run_dashboard_update(dashboard_path):
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Dashboard not found: {dashboard_path}")
    store = json.loads(STORE_PATH.read_text())
    html = dashboard_path.read_text()
    backup = backup_dashboard(dashboard_path)
    changes = []
    total_updates = 0

    for risk_id, risk in store.get("operational_risks", {}).items():
        kri_changes = 0
        for kri_name, kri_data in risk.get("kris", {}).items():
            updated_html, changed = _update_kri(
                html, kri_name, kri_data["value"], kri_data["status"]
            )
            if changed:
                html = updated_html
                html = _update_sparkline(html, kri_name, kri_data["value"])
                kri_changes += 1
                total_updates += 1
        if kri_changes > 0:
            changes.append(f"{risk_id}: {kri_changes} KRI(s) updated")

    if total_updates > 0:
        dashboard_path.write_text(html)

    return {
        "dashboard_path": str(dashboard_path),
        "backup_path": str(backup),
        "total_kri_updates": total_updates,
        "risks_updated": changes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def update_exec_recommendations(dashboard_path, approved):
    if not dashboard_path.exists():
        return 0
    html = dashboard_path.read_text()
    backup_dashboard(dashboard_path)
    changes = 0

    for section_key, new_text in approved.items():
        if not new_text:
            continue
        ec_id = EXEC_REC_IDS.get(section_key)
        if not ec_id:
            continue
        pattern = re.compile(
            r'(<div class="exec" id="' + re.escape(ec_id) + r'">'
            r'.*?<div class="exec-hd">.*?</div>\s*)'
            r'(<p>.*?</p>)',
            re.DOTALL
        )
        match = pattern.search(html)
        if match:
            html = html[:match.start(2)] + f"<p>{new_text}</p>" + html[match.end(2):]
            changes += 1

    if changes > 0:
        dashboard_path.write_text(html)
    return changes


def _format_board_summary(text: str) -> str:
    import re

    DOMAIN_CONFIG = [
        ('COMPOUND SCENARIOS', '#7f8c8d'),
        ('COMPOUND',           '#7f8c8d'),
        ('STRATEGIC',          'var(--red-md)'),
        ('OPERATIONAL',        'var(--amb-md)'),
        ('FINANCIAL',          'var(--grn-md)'),
        ('COMPLIANCE',         'var(--pur-md)'),
    ]

    segments = []
    for label, color in DOMAIN_CONFIG:
        for m in re.finditer(re.escape(label) + r'\s*:\s*', text, re.IGNORECASE):
            segments.append((m.start(), m.end(), label, color))

    if not segments:
        return f'<p style="font-size:12.5px;color:var(--txt);line-height:1.6;margin:0">{text}</p>'

    segments.sort(key=lambda x: x[0])
    # dedupe: if two labels start at overlapping positions keep the longer one
    deduped = []
    for seg in segments:
        if deduped and seg[0] < deduped[-1][1]:
            if len(seg[2]) > len(deduped[-1][2]):
                deduped[-1] = seg
        else:
            deduped.append(seg)
    segments = deduped

    overview = text[:segments[0][0]].strip()
    parts = []
    if overview:
        parts.append(
            f'<p style="font-size:12.5px;font-weight:600;color:var(--navy);'
            f'margin:0 0 0.75rem;line-height:1.5">{overview}</p>'
        )

    for i, (start, end, label, color) in enumerate(segments):
        body_end = segments[i + 1][0] if i + 1 < len(segments) else len(text)
        body = text[end:body_end].strip()
        mb = '0' if i == len(segments) - 1 else '0.55rem'
        parts.append(
            f'<div style="margin-bottom:{mb};padding:0.4rem 0.6rem 0.4rem 0.7rem;'
            f'border-left:3px solid {color};background:rgba(0,0,0,0.02)">'
            f'<div style="font-size:10px;font-weight:700;color:{color};'
            f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">'
            f'{label.title()}</div>'
            f'<div style="font-size:12.5px;color:var(--txt);line-height:1.6">{body}</div>'
            f'</div>'
        )

    return ''.join(parts)


def update_board_summary(dashboard_path, summary_text: str, run_id: str) -> bool:
    """Write Risk Committee Summary into the dashboard panel as colour-coded domain sections."""
    import re
    from datetime import datetime, timezone
    html = Path(dashboard_path).read_text()
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    formatted = _format_board_summary(summary_text)

    # Update run/timestamp meta (handles both <div> and <p> closing tags)
    html = re.sub(
        r'id="board-summary-meta"[^>]*>[^<]*</(?:div|p)>',
        f'id="board-summary-meta" style="font-size:11px;color:var(--txt-m);margin-top:2px">'
        f'Run {run_id[:8]} · {timestamp}</div>',
        html
    )

    # Update content block — sentinel pattern for <div> structure
    new_html = re.sub(
        r'id="board-summary-text"[^>]*>.*?</div><!-- /bst -->',
        f'id="board-summary-text">{formatted}</div><!-- /bst -->',
        html, flags=re.DOTALL
    )
    if new_html == html:
        # Fallback: legacy <p> structure
        new_html = re.sub(
            r'id="board-summary-text"[^>]*>.*?</p>',
            f'id="board-summary-text">{formatted}</p>',
            html, flags=re.DOTALL
        )
    html = new_html

    Path(dashboard_path).write_text(html)
    return True

def update_signals_panel(dashboard_path, regulatory: dict, emerging: dict, run_id: str) -> bool:
    """Write regulatory and emerging signals into the dashboard signals panel."""
    import re
    from datetime import datetime, timezone
    html = Path(dashboard_path).read_text()
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    items = []

    # Regulatory deadline alerts
    for d in regulatory.get('deadline_alerts', []):
        urgency = d.get('urgency', 'watch')
        color = '#c0392b' if urgency == 'immediate' else '#e67e22' if urgency == '90_days' else '#7f8c8d'
        items.append(
            f'<div style="margin-bottom:0.7rem;padding-bottom:0.7rem;border-bottom:1px solid var(--bdr)">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:2px">'
            f'<span style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase">{urgency.replace("_"," ")}</span>'
            f'<span style="font-size:11px;font-weight:600">{d.get("regulation","")} — {d.get("jurisdiction","")}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:var(--txt)">{d.get("description","")}</div>'
            f'<div style="font-size:11px;color:var(--txt-m)">Deadline: {d.get("deadline","")} · Affects: {d.get("affects_risk","")}</div>'
            f'</div>'
        )

    # New regulatory signals
    for s in regulatory.get('new_signals', [])[:2]:
        items.append(
            f'<div style="margin-bottom:0.7rem;padding-bottom:0.7rem;border-bottom:1px solid var(--bdr)">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:2px">'
            f'<span style="font-size:10px;font-weight:700;color:#2980b9;text-transform:uppercase">NEW SIGNAL</span>'
            f'<span style="font-size:11px;font-weight:600">{s.get("title","")}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:var(--txt)">{s.get("summary","")[:150]}</div>'
            f'<div style="font-size:11px;color:var(--txt-m)">{s.get("regulation","")} · {s.get("jurisdiction","")} · Confidence: {s.get("confidence","")}</div>'
            f'</div>'
        )

    # Emerging risk candidates
    for c in emerging.get('risk_candidates', [])[:2]:
        l = c.get('initial_L', 0)
        i = c.get('initial_I', 0)
        action = c.get('recommended_action', 'watch_list')
        color = '#c0392b' if action == 'immediate_board_attention' else '#e67e22' if action == 'assess_for_register' else '#7f8c8d'
        items.append(
            f'<div style="margin-bottom:0.7rem;padding-bottom:0.7rem;border-bottom:1px solid var(--bdr)">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:2px">'
            f'<span style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase">{action.replace("_"," ")}</span>'
            f'<span style="font-size:11px;font-weight:600">Emerging: {c.get("proposed_id","")} ({c.get("proposed_domain","")})</span>'
            f'</div>'
            f'<div style="font-size:12px;color:var(--txt)">{c.get("signal","")[:150]}</div>'
            f'<div style="font-size:11px;color:var(--txt-m)">L={l} I={i} · Horizon: {c.get("horizon","")} · {c.get("rationale","")[:80]}</div>'
            f'</div>'
        )

    if not items:
        items = ['<div style="color:var(--txt-m);font-style:italic">No new signals this run.</div>']

    signals_html = ''.join(items)

    # Update meta
    html = re.sub(
        r'id="signals-meta"[^>]*>[^<]*</div>',
        f'id="signals-meta" style="font-size:11px;color:var(--txt-m);margin-top:2px">Run {run_id[:8]} · {timestamp}</div>',
        html
    )
    # Update signals list
    html = re.sub(
        r'id="signals-list"[^>]*>.*?</div>(?=\s*</div>\s*</div>\s*</div>)',
        f'id="signals-list" style="font-size:12.5px;color:var(--txt)">{signals_html}',
        html, flags=re.DOTALL
    )
    Path(dashboard_path).write_text(html)
    return True
