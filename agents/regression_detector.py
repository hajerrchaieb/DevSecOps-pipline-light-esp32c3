"""
agents/regression_detector.py  —  Regression Detector Automatique

Compares current pipeline-summary.json with previous runs stored in
reports/run-history.json. Flags regressions (score drop > 2 points)
and improvements. Updates history and writes regression-report.json.

Usage:
  python3 agents/regression_detector.py
  python3 agents/regression_detector.py --threshold 2

Called automatically from run_demo.py after the orchestrator.
Also runs in CI after the AI agents stage.
"""

import json
import os
from datetime import datetime
from pathlib import Path

REPORTS   = Path("reports")
THRESHOLD = float(os.getenv("REGRESSION_THRESHOLD", "2.0"))

TRACKED_METRICS = [
    ("code_quality",  ["stage_results", "code_quality",    "score"]),
    ("security",      ["stage_results", "security",        "score"]),
    ("robustness",    ["stage_results", "fault_injection", "robustness_score"]),
    ("dynamic_score", ["stage_results", "dynamic_score"]),
    ("flash_pct",     ["stage_results", "memory",          "flash_pct"]),
    ("tests_count",   ["stage_results", "tests_generated"]),
]


# ── Helpers ────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_nested(data: dict, keys: list):
    """Safely traverse nested dict."""
    val = data
    for k in keys:
        if not isinstance(val, dict):
            return None
        val = val.get(k)
    return val


def _to_float(v) -> float | None:
    try:
        return float(str(v).replace("N/A", "").strip())
    except Exception:
        return None


def _extract_metrics(summary: dict) -> dict:
    """Extract all tracked metrics from pipeline-summary into flat dict."""
    result = {}
    for name, keys in TRACKED_METRICS:
        raw = _get_nested(summary, keys)
        num = _to_float(raw)
        result[name] = num if num is not None else raw
    # Add metadata
    result["target"]     = summary.get("target", "esp32c3")
    result["version"]    = summary.get("version", "v1.0.0")
    result["passed"]     = summary.get("pipeline_passed", False)
    result["run_time"]   = datetime.now().isoformat()
    return result


def _build_run_snapshot(summary: dict, metrics: dict) -> dict:
    """Build a compact run entry for the history file."""
    return {
        "run_time":    metrics["run_time"],
        "target":      metrics["target"],
        "version":     metrics["version"],
        "passed":      metrics["passed"],
        "code_quality": metrics.get("code_quality"),
        "security":    metrics.get("security"),
        "robustness":  metrics.get("robustness"),
        "dynamic":     metrics.get("dynamic_score"),
        "flash_pct":   metrics.get("flash_pct"),
        "tests_count": metrics.get("tests_count"),
    }


# ── Regression detection ───────────────────────────────────────────

def detect_regressions(
    current: dict,
    previous: dict,
    threshold: float = THRESHOLD,
) -> tuple[list, list]:
    """
    Compare current vs previous run metrics.
    Returns (regressions, improvements).

    A regression = score decreased by more than `threshold`.
    For flash_pct: increase is bad (more memory used).
    """
    regressions  = []
    improvements = []

    # Metrics where LOWER is BETTER (flash usage)
    lower_is_better = {"flash_pct"}

    for metric_name, _ in TRACKED_METRICS:
        cur_val  = _to_float(current.get(metric_name))
        prev_val = _to_float(previous.get(metric_name))

        if cur_val is None or prev_val is None:
            continue

        delta = cur_val - prev_val

        if metric_name in lower_is_better:
            # Higher flash usage = regression
            if delta > threshold:
                regressions.append({
                    "metric":   metric_name,
                    "previous": prev_val,
                    "current":  cur_val,
                    "delta":    round(delta, 2),
                    "message":  f"Flash usage increased by {delta:.1f}% (threshold: {threshold}%)",
                    "severity": "high" if delta > threshold * 2 else "medium",
                })
            elif delta < -threshold:
                improvements.append({
                    "metric":   metric_name,
                    "previous": prev_val,
                    "current":  cur_val,
                    "delta":    round(delta, 2),
                })
        else:
            # Higher score = better
            if delta < -threshold:
                regressions.append({
                    "metric":   metric_name,
                    "previous": prev_val,
                    "current":  cur_val,
                    "delta":    round(delta, 2),
                    "message":  f"{metric_name} dropped by {abs(delta):.1f} points (threshold: {threshold})",
                    "severity": "critical" if abs(delta) > threshold * 2 else "high",
                })
            elif delta > threshold:
                improvements.append({
                    "metric":   metric_name,
                    "previous": prev_val,
                    "current":  cur_val,
                    "delta":    round(delta, 2),
                })

    return regressions, improvements


# ── Main ───────────────────────────────────────────────────────────

def run_regression_detector(threshold: float = THRESHOLD) -> dict:
    print(f"\n[Regression Detector] Threshold: ±{threshold} points")

    # Load current run
    summary_path = REPORTS / "pipeline-summary.json"
    if not summary_path.exists():
        print("[Regression Detector] pipeline-summary.json not found — skipping")
        return {"status": "skipped", "reason": "no pipeline-summary.json"}

    current_summary = _load(summary_path)
    current_metrics = _extract_metrics(current_summary)

    # Load history
    history_path = REPORTS / "run-history.json"
    history      = _load(history_path)
    runs         = history.get("runs", [])

    # Get previous run metrics
    previous_metrics = runs[-1] if runs else {}
    has_previous     = bool(previous_metrics)

    regressions, improvements = [], []

    if has_previous:
        regressions, improvements = detect_regressions(
            current_metrics, previous_metrics, threshold
        )
        print(f"[Regression Detector] vs run at {previous_metrics.get('run_time','?')}")
        print(f"  Regressions:  {len(regressions)}")
        print(f"  Improvements: {len(improvements)}")
    else:
        print("[Regression Detector] No previous run found — establishing baseline")

    # Print details
    for reg in regressions:
        print(f"  [REGRESSION] {reg['metric']}: {reg['previous']} → {reg['current']} ({reg['delta']:+.1f})")
    for imp in improvements:
        print(f"  [IMPROVED]   {imp['metric']}: {imp['previous']} → {imp['current']} ({imp['delta']:+.1f})")

    # Append current run to history
    snapshot = _build_run_snapshot(current_summary, current_metrics)
    runs.append(snapshot)

    # Keep max 50 runs in history
    if len(runs) > 50:
        runs = runs[-50:]

    history_data = {
        "last_updated": datetime.now().isoformat(),
        "run_count":    len(runs),
        "runs":         runs,
    }
    history_path.write_text(json.dumps(history_data, indent=2), encoding="utf-8")
    print(f"[Regression Detector] History updated: {len(runs)} run(s) stored")

    # Build regression report
    status = "clean"
    if any(r["severity"] == "critical" for r in regressions):
        status = "critical_regression"
    elif regressions:
        status = "regression_detected"
    elif improvements:
        status = "improved"
    elif has_previous:
        status = "stable"
    else:
        status = "baseline"

    report = {
        "timestamp":      datetime.now().isoformat(),
        "target":         current_metrics.get("target", "esp32c3"),
        "version":        current_metrics.get("version", "v1.0.0"),
        "threshold":      threshold,
        "has_previous":   has_previous,
        "status":         status,
        "regressions":    regressions,
        "improvements":   improvements,
        "current":        {k: current_metrics.get(k) for k, _ in TRACKED_METRICS},
        "previous":       {k: previous_metrics.get(k) for k, _ in TRACKED_METRICS} if has_previous else {},
        "run_history_count": len(runs),
        "summary": (
            f"{'No previous run — baseline established' if not has_previous else ''}"
            f"{f'{len(regressions)} regression(s) detected' if regressions else ''}"
            f"{f', {len(improvements)} improvement(s)' if improvements else ''}"
            f"{'. All stable.' if not regressions and not improvements and has_previous else ''}"
        ).strip(", "),
    }

    out = REPORTS / "regression-report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Regression Detector] Report: {out}")
    print(f"[Regression Detector] Status: {status.upper()}")

    # Alert if critical
    if status == "critical_regression":
        print(f"\n  ⚠ CRITICAL REGRESSION DETECTED:")
        for r in regressions:
            if r["severity"] == "critical":
                print(f"    {r['metric']}: {r['previous']} → {r['current']}")
        print(f"  Run: python3 generate_dashboard.py --open  to visualise\n")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESP32 Regression Detector")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Score drop threshold (default: {THRESHOLD})")
    args = parser.parse_args()
    report = run_regression_detector(threshold=args.threshold)
    print(json.dumps({k: v for k, v in report.items() if k not in ("regressions","improvements","current","previous")}, indent=2))
