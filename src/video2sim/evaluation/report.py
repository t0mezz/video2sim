"""JSON and HTML report of an evaluation (P0.6). The JSON is the source of truth."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from video2sim.evaluation.metrics import SceneResult, aggregate, scene_to_dict


def build_report(
    scenes: Sequence[SceneResult],
    targets: dict[str, Any],
    header: dict[str, Any],
    missing_ground_truth: Sequence[str],
) -> dict[str, Any]:
    numeric = {k: v for k, v in targets.get("targets", {}).items() if isinstance(v, int | float)}
    return {
        "header": {
            **header,
            "targets_status": targets.get("status"),
            "scenes_without_ground_truth": list(missing_ground_truth),
        },
        "summary": aggregate(scenes, numeric),
        "scenes": [scene_to_dict(s) for s in scenes],
    }


def write_json(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


_CSS = """
body{background:#14161a;color:#d8dbe0;font:14px/1.4 system-ui,sans-serif;margin:24px}
h1,h2{font-weight:600}table{border-collapse:collapse;margin:8px 0 24px}
th,td{border:1px solid #2d3139;padding:4px 10px;text-align:right}
th{background:#1d2026}td:first-child,th:first-child{text-align:left}
.pass{color:#6fcf97}.fail{color:#eb5757}.mute{color:#8a8f98}
.warn{background:#3a2f12;color:#f2c94c;padding:8px 12px;border-radius:4px}
"""


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return html.escape(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _metric_rows(metrics: Sequence[dict[str, Any]], with_target: bool) -> list[list[str]]:
    rows = []
    for m in metrics:
        status = m["status"]
        if m["passed"] is True:
            verdict = '<span class="pass">pass</span>'
        elif m["passed"] is False:
            verdict = '<span class="fail">fail</span>'
        else:
            verdict = f'<span class="mute">{html.escape(status.replace("_", " "))}</span>'
        row = [html.escape(m["name"]), _fmt(m["value"]), _fmt(m["n"])]
        if with_target:
            row += [_fmt(m["target"]), verdict]
        rows.append(row)
    return rows


def render_html(report: dict[str, Any]) -> str:
    head, summary = report["header"], report["summary"]
    parts = ["<h1>video2sim evaluation</h1>"]
    meta = [
        [html.escape(str(k)), _fmt(v)]
        for k, v in head.items()
        if k != "scenes_without_ground_truth"
    ]
    parts.append(_table(["field", "value"], meta))
    missing = head["scenes_without_ground_truth"]
    if missing:
        parts.append(
            f'<p class="warn">{len(missing)} scene folder(s) had no usable ground truth and '
            f"were not scored: {html.escape(', '.join(missing))}</p>"
        )
    if head.get("targets_status") == "provisional":
        parts.append('<p class="warn">Targets are provisional (configs/targets.yaml).</p>')
    parts.append(
        f"<p>{summary['n_scenes']} scenes, {summary['n_evaluable_objects']} scored objects, "
        f"{summary['n_excluded_objects']} excluded (incomplete ground truth), "
        f"{summary['n_false_positives']} false positives.</p>"
    )
    parts.append("<h2>Metrics against targets</h2>")
    parts.append(
        _table(["metric", "value", "N", "target", "result"], _metric_rows(summary["pooled"], True))
    )
    seg = report.get("segmentation")
    if seg:
        parts.append("<h2>Phase 2 exit criteria</h2>")
        parts.append(
            _table(["metric", "value", "N", "target", "result"], _metric_rows(seg["rows"], True))
        )
        if seg["flag_scenes_missing"]:
            parts.append(
                '<p class="warn">Flag scenes not scored: '
                f"{html.escape(', '.join(seg['flag_scenes_missing']))}</p>"
            )
        flag_rows = [
            [html.escape(f"{s}/{o}"), html.escape(", ".join(f) or "none")]
            for s, objs in seg["flags_by_object"].items()
            for o, f in objs.items()
        ]
        if flag_rows:
            parts.append(_table(["object", "flags"], flag_rows))
    for tier, metrics in summary["per_tier"].items():
        parts.append(f"<h2>Tier: {html.escape(tier)}</h2>")
        parts.append(_table(["metric", "value", "N"], _metric_rows(metrics, False)))

    parts.append("<h2>Scenes</h2>")
    scene_rows = [
        [
            html.escape(s["scene_id"]),
            html.escape(s["tier"]),
            html.escape(s["split"]),
            _fmt(s["n_evaluable"]),
            _fmt(s["n_matched"]),
            _fmt(s["n_false_positive"]),
            _fmt(s["n_excluded"]),
            _fmt(s["runtime_s"]),
        ]
        for s in report["scenes"]
    ]
    parts.append(
        _table(
            ["scene", "tier", "split", "objects", "matched", "false pos.", "excluded", "runtime s"],
            scene_rows,
        )
    )

    parts.append("<h2>Objects</h2>")
    object_rows = [
        [
            html.escape(f"{s['scene_id']}/{o['object_id']}"),
            html.escape(o["label"]),
            "yes" if o["matched"] else '<span class="fail">missed</span>',
            _fmt(o["size_error"]),
            _fmt(o["position_error_m"]),
            _fmt(o["orientation_error_deg"]),
            _fmt(o["mass_error"]),
        ]
        for s in report["scenes"]
        for o in s["objects"]
    ]
    parts.append(
        _table(
            ["object", "label", "matched", "size err", "pos err m", "yaw err deg", "mass err"],
            object_rows,
        )
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>video2sim evaluation</title><style>{_CSS}</style></head>"
        f"<body>{''.join(parts)}</body></html>\n"
    )


def write_html(report: dict[str, Any], path: Path) -> None:
    path.write_text(render_html(report), encoding="utf-8")
