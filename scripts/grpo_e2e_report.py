"""Generate a summary report from GRPO E2E test artifacts.

Reads the artifact directory produced by test_grpo_e2e.py and prints
a formatted summary to stdout. Optionally generates an HTML report.

Usage:
    uv run python scripts/grpo_e2e_report.py <artifact_dir>
    uv run python scripts/grpo_e2e_report.py <artifact_dir> --html

Examples:
    uv run python scripts/grpo_e2e_report.py test_artifacts/grpo_e2e/20260302_143000/
    uv run python scripts/grpo_e2e_report.py test_artifacts/grpo_e2e/20260302_143000/ --html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict | list | None:
    """Load a JSON file, returning None if it does not exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _find_latest_artifact_dir(base: Path) -> Path | None:
    """Find the most recent timestamped subdirectory in the base path."""
    if not base.is_dir():
        return None
    subdirs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def _section(title: str, char: str = "=") -> str:
    """Format a section header."""
    line = char * 60
    return f"\n{line}\n{title}\n{line}"


def report(artifact_dir: str, html: bool = False) -> None:
    """Generate and print a summary report from GRPO E2E test artifacts.

    Args:
        artifact_dir: Path to the artifact directory. Can also be
            'latest' to use the most recent run.
        html: If True, also write an HTML report to the artifact directory.
    """
    base = Path(artifact_dir)

    # Handle 'latest' shortcut
    if str(artifact_dir).lower() == "latest":
        base = _find_latest_artifact_dir(Path("test_artifacts") / "grpo_e2e")
        if base is None:
            print("ERROR: No artifact directories found in test_artifacts/grpo_e2e/")
            sys.exit(1)
        print(f"Using latest artifact directory: {base}")

    if not base.is_dir():
        print(f"ERROR: {base} is not a directory")
        sys.exit(1)

    lines: list[str] = []
    lines.append(_section("GRPO E2E TEST REPORT"))
    lines.append(f"Artifact directory: {base.resolve()}")

    # ---------------------------------------------------------------------------
    # Test report
    # ---------------------------------------------------------------------------
    test_report = _load_json(base / "test_report.json")
    if test_report:
        lines.append(_section("Test Summary", "-"))
        lines.append(f"  Test:             {test_report.get('test_name', 'unknown')}")
        lines.append(f"  Status:           {test_report.get('status', 'unknown')}")
        lines.append(f"  Duration:         {test_report.get('duration_s', 0):.2f}s")
        lines.append(f"  Training steps:   {test_report.get('num_training_steps', 0)}")
        lines.append(f"  Total rollouts:   {test_report.get('total_rollouts', 0)}")
        lines.append(
            f"  Weights changed:  {test_report.get('model_weights_changed', 'N/A')}"
        )
        lines.append(
            f"  Delta L2 norm:    {test_report.get('total_delta_norm', 0):.6f}"
        )
        lines.append(
            f"  Checkpoint OK:    {test_report.get('checkpoint_loadable', 'N/A')}"
        )
        lines.append(f"  Final loss:       {test_report.get('final_loss', 0):.4f}")
        lines.append(
            f"  Final reward:     {test_report.get('final_reward_mean', 0):.2f}"
        )

    # ---------------------------------------------------------------------------
    # Training log
    # ---------------------------------------------------------------------------
    training_log = _load_json(base / "training_log.json")
    if training_log and isinstance(training_log, list):
        lines.append(_section("Training Metrics", "-"))
        lines.append(
            f"  {'Step':>4}  {'Loss':>8}  {'Reward':>8}  {'GradNorm':>8}  {'Valid':>5}  {'Time':>6}"
        )
        for m in training_log:
            lines.append(
                f"  {m.get('step', 0):4d}  "
                f"{m.get('loss', 0):8.4f}  "
                f"{m.get('reward_mean', 0):8.2f}  "
                f"{m.get('grad_norm', 0):8.4f}  "
                f"{m.get('valid_terms', 0):5d}  "
                f"{m.get('step_time', 0):6.2f}s"
            )

    # ---------------------------------------------------------------------------
    # Model diff
    # ---------------------------------------------------------------------------
    model_diff = _load_json(base / "model_diff.json")
    if model_diff:
        lines.append(_section("Model Weight Changes", "-"))
        any_changed = model_diff.get("_any_changed", False)
        total_delta = model_diff.get("_total_delta_norm", 0)
        lines.append(f"  Any weights changed:  {any_changed}")
        lines.append(f"  Total delta L2 norm:  {total_delta:.6f}")
        lines.append("")
        for name, info in model_diff.items():
            if name.startswith("_"):
                continue
            lines.append(
                f"  {name}:"
                f"  delta={info.get('delta_norm', 0):.6f}"
                f"  norm: {info.get('initial_norm', 0):.4f} -> {info.get('final_norm', 0):.4f}"
                f"  changed={info.get('changed', False)}"
            )

    # ---------------------------------------------------------------------------
    # Weight diff detailed
    # ---------------------------------------------------------------------------
    weight_diff = _load_json(base / "weight_diff" / "detailed_diff.json")
    if weight_diff:
        lines.append(_section("Detailed Weight Diff", "-"))
        summary = weight_diff.get("summary", {})
        lines.append(f"  Total params:          {summary.get('total_params', 0)}")
        lines.append(f"  Trainable params:      {summary.get('trainable_params', 0)}")
        lines.append(
            f"  Changed param groups:  "
            f"{summary.get('changed_param_groups', 0)}/"
            f"{summary.get('total_param_groups', 0)}"
        )
        params = weight_diff.get("parameters", {})
        for name, info in params.items():
            lines.append(
                f"  {name}: "
                f"delta_l2={info.get('delta_l2', 0):.6f} "
                f"grad_norm={info.get('grad_norm', 0):.6f} "
                f"delta_max={info.get('delta_max', 0):.6f} "
                f"shape={info.get('shape', [])}"
            )

    # ---------------------------------------------------------------------------
    # Convergence
    # ---------------------------------------------------------------------------
    convergence = _load_json(base / "convergence" / "convergence_data.json")
    if convergence:
        lines.append(_section("GRPO Loss Convergence", "-"))
        lines.append(f"  Steps:          {convergence.get('num_steps', 0)}")
        lines.append(f"  Rewards:        {convergence.get('rewards', [])}")

        initial = convergence.get("initial_policy", [])
        final = convergence.get("final_policy", [])
        if initial:
            lines.append(
                f"  Initial policy: [{', '.join(f'{p:.4f}' for p in initial)}]"
            )
        if final:
            lines.append(f"  Final policy:   [{', '.join(f'{p:.4f}' for p in final)}]")

        loss_hist = convergence.get("loss_history", [])
        if loss_hist:
            lines.append(f"  Initial loss:   {loss_hist[0]:.4f}")
            lines.append(f"  Final loss:     {loss_hist[-1]:.4f}")

        # Check if good actions dominate
        rewards = convergence.get("rewards", [])
        if final and rewards:
            good_prob = sum(p for p, r in zip(final, rewards) if r > 0.5)
            bad_prob = sum(p for p, r in zip(final, rewards) if r <= 0.5)
            lines.append(f"  Good action prob: {good_prob:.4f}")
            lines.append(f"  Bad action prob:  {bad_prob:.4f}")
            lines.append(
                f"  Converged:        {'YES' if good_prob > bad_prob else 'NO'}"
            )

    # ---------------------------------------------------------------------------
    # GRPO loss properties
    # ---------------------------------------------------------------------------
    # Try both old and new filename for backwards compatibility
    loss_props = _load_json(base / "policy_gradient_loss_properties.json")
    if not loss_props:
        loss_props = _load_json(base / "grpo_loss_properties.json")
    if loss_props:
        lines.append(_section("GRPO Loss Mathematical Properties", "-"))
        tests = loss_props.get("tests", [])
        for t in tests:
            status = "PASS" if t.get("passed") else "FAIL"
            lines.append(f"  [{status}] {t.get('name', 'unknown')}")
        lines.append(
            f"\n  Overall: {'ALL PASSED' if loss_props.get('all_passed') else 'SOME FAILED'}"
        )

    # ---------------------------------------------------------------------------
    # Rollout collection
    # ---------------------------------------------------------------------------
    collection_summary = _load_json(base / "rollout_collection" / "summary.json")
    if collection_summary:
        lines.append(_section("Rollout Collection", "-"))
        lines.append(f"  Rollouts:         {collection_summary.get('num_rollouts', 0)}")
        lines.append(
            f"  Steps/rollout:    {collection_summary.get('steps_per_rollout', 0)}"
        )
        lines.append(f"  Rewards:          {collection_summary.get('rewards', [])}")
        lines.append(
            f"  Reward mean:      {collection_summary.get('reward_mean', 0):.2f}"
        )
        advs = collection_summary.get("advantages", [])
        if advs:
            lines.append(f"  Advantages:       [{', '.join(f'{a:.3f}' for a in advs)}]")

    # ---------------------------------------------------------------------------
    # Rollout traces
    # ---------------------------------------------------------------------------
    trace_dir = base / "rollout_traces"
    if trace_dir.is_dir():
        trace_files = sorted(trace_dir.glob("*.json"))
        if trace_files:
            lines.append(_section("Rollout Traces", "-"))
            for tf in trace_files[:10]:  # Show at most 10
                trace = _load_json(tf)
                if trace:
                    lines.append(
                        f"  {tf.name}: "
                        f"reward={trace.get('reward', 0):.1f} "
                        f"steps={trace.get('num_steps', 0)}"
                    )
                    for a in trace.get("actions", []):
                        lines.append(
                            f"    Step {a.get('step_idx', 0)}: {a.get('action_text', '')}"
                        )
            if len(trace_files) > 10:
                lines.append(f"  ... and {len(trace_files) - 10} more trace files")

    # ---------------------------------------------------------------------------
    # File listing
    # ---------------------------------------------------------------------------
    lines.append(_section("Artifact Files", "-"))
    all_files = sorted(base.rglob("*"))
    for f in all_files:
        if f.is_file():
            rel = f.relative_to(base)
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"
            lines.append(f"  {rel} ({size_str})")

    lines.append("")
    full_report = "\n".join(lines)
    print(full_report)

    # Optionally generate HTML
    if html:
        html_path = base / "report.html"
        _generate_html(base, full_report, html_path)
        print(f"\nHTML report written to: {html_path}")


def _generate_html(
    artifact_dir: Path,
    text_report: str,
    output_path: Path,
) -> None:
    """Generate a simple HTML report with embedded images."""
    # Collect screenshot paths
    screenshots: list[Path] = sorted(artifact_dir.rglob("*.png"))

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>GRPO E2E Test Report</title>",
        "<style>",
        "body { font-family: monospace; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "pre { background: #f5f5f5; padding: 16px; overflow-x: auto; border-radius: 4px; }",
        ".screenshots { display: flex; flex-wrap: wrap; gap: 8px; }",
        ".screenshots img { max-width: 320px; border: 1px solid #ccc; border-radius: 4px; }",
        ".screenshot-label { font-size: 12px; color: #666; text-align: center; }",
        "h2 { border-bottom: 2px solid #333; padding-bottom: 4px; }",
        "</style>",
        "</head><body>",
        "<h1>GRPO E2E Test Report</h1>",
        f"<p>Artifact directory: <code>{artifact_dir.resolve()}</code></p>",
        "<h2>Text Report</h2>",
        f"<pre>{text_report}</pre>",
    ]

    if screenshots:
        html_parts.append("<h2>Screenshots</h2>")
        html_parts.append("<div class='screenshots'>")
        for s in screenshots[:50]:  # Limit to 50 images
            rel = s.relative_to(artifact_dir)
            html_parts.append(
                f"<div>"
                f"<img src='{rel}' alt='{rel}'>"
                f"<div class='screenshot-label'>{rel}</div>"
                f"</div>"
            )
        html_parts.append("</div>")
        if len(screenshots) > 50:
            html_parts.append(
                f"<p>... and {len(screenshots) - 50} more screenshots</p>"
            )

    # Convergence chart (simple text-based, no external deps)
    convergence = _load_json(artifact_dir / "convergence" / "convergence_data.json")
    if convergence:
        loss_hist = convergence.get("loss_history", [])
        policy_hist = convergence.get("policy_history", [])
        if loss_hist:
            html_parts.append("<h2>Loss Convergence</h2>")
            html_parts.append("<pre>")
            # Simple ASCII chart
            if loss_hist:
                min_loss = min(loss_hist)
                max_loss = max(loss_hist)
                rng = max_loss - min_loss if max_loss != min_loss else 1.0
                chart_width = 50
                for i, loss in enumerate(loss_hist):
                    if i % 5 == 0:
                        bar_len = int((loss - min_loss) / rng * chart_width)
                        bar = "#" * bar_len
                        html_parts.append(
                            f"Step {i:3d} | {bar:<{chart_width}} | {loss:.4f}"
                        )
            html_parts.append("</pre>")

        if policy_hist:
            html_parts.append("<h2>Policy Evolution</h2>")
            html_parts.append("<pre>")
            rewards = convergence.get("rewards", [])
            for i in range(0, len(policy_hist), 5):
                probs = policy_hist[i]
                labels = [
                    f"{'G' if r > 0.5 else 'B'}:{p:.3f}" for p, r in zip(probs, rewards)
                ]
                html_parts.append(f"Step {i:3d}: {' | '.join(labels)}")
            html_parts.append("</pre>")

    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts))


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate summary report from GRPO E2E test artifacts."
    )
    parser.add_argument(
        "artifact_dir",
        help="Path to artifact directory, or 'latest' for most recent.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also generate an HTML report.",
    )
    args = parser.parse_args()
    report(args.artifact_dir, html=args.html)


if __name__ == "__main__":
    main()
