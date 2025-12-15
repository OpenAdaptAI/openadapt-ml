"""Compare human actions vs model predictions on a capture.

Generates an enhanced viewer showing both human and predicted actions side-by-side.

Usage:
    uv run python -m openadapt_ml.scripts.compare \
        --capture /path/to/capture \
        --checkpoint checkpoints/qwen3vl2b_capture_lora \
        --output comparison.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openadapt_ml.ingest.capture import capture_to_episode
from openadapt_ml.schemas.sessions import Episode, Step
from openadapt_ml.datasets.next_action import SYSTEM_PROMPT, format_action
from openadapt_ml.training.trainer import _get_shared_header_css, _generate_shared_header_html


def load_model(checkpoint_path: str | None, config_path: str | None = None):
    """Load a trained model for inference.

    Returns None if no checkpoint specified (will skip predictions).
    """
    if not checkpoint_path:
        return None

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"Warning: Checkpoint not found at {checkpoint}")
        return None

    try:
        from openadapt_ml.models.qwen_vl import QwenVLAdapter

        # Load base model with LoRA weights
        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        adapter = QwenVLAdapter.from_pretrained(
            model_name=model_name,
            lora_config={"weights_path": str(checkpoint)},
            load_in_4bit=True,  # Use 4-bit for inference too
        )
        print(f"Loaded model from {checkpoint}")
        return adapter
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_action(
    model,
    observation_image: str,
    goal: str,
    step_index: int = 0,
    total_steps: int = 1,
    action_history: list[str] | None = None,
) -> dict[str, Any] | None:
    """Run inference to predict an action given an observation.

    Returns predicted action dict or None if no model.
    """
    if model is None:
        return None

    try:
        import re

        # Build history section matching training format
        if action_history:
            history_text = "ACTIONS COMPLETED SO FAR:\n"
            for i, action_text in enumerate(action_history, 1):
                history_text += f"  {i}. {action_text}\n"
            history_text += f"\nThis is step {step_index + 1} of {total_steps}. "
        else:
            history_text = f"This is step 1 of {total_steps} (no actions completed yet). "

        # Match training prompt format exactly
        user_content = (
            f"Goal: {goal}\n\n"
            f"{history_text}"
            "Look at the screenshot and determine the NEXT action.\n\n"
            "Thought: [what element to interact with and why]\n"
            "Action: [CLICK(x=..., y=...) or TYPE(text=\"...\") or WAIT() or DONE()]"
        )

        # Build sample in the format expected by the adapter
        sample = {
            "images": [observation_image],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        # Run inference using generate method
        result = model.generate(sample, max_new_tokens=128)

        # Parse result - look for CLICK(x=..., y=...) or similar patterns
        action = {"type": "predicted", "raw_output": result}

        # Try to extract coordinates from output
        # Match patterns like: CLICK(x=0.42, y=0.31) or click at (0.42, 0.31)
        click_match = re.search(r'CLICK\s*\(\s*x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)\s*\)', result, re.IGNORECASE)
        if not click_match:
            click_match = re.search(r'click.*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', result, re.IGNORECASE)
        if not click_match:
            # Try to find any two decimal numbers
            nums = re.findall(r'(0\.\d+)', result)
            if len(nums) >= 2:
                click_match = type('Match', (), {'group': lambda s, i: nums[i-1]})()

        if click_match:
            action["x"] = float(click_match.group(1))
            action["y"] = float(click_match.group(2))
            action["type"] = "click"

        return action
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"type": "error", "error": str(e)}


def generate_comparison_data(
    episode: Episode,
    model=None,
) -> list[dict[str, Any]]:
    """Generate comparison data for each step.

    Returns list of dicts with human action, predicted action, and metadata.
    """
    comparison_data = []
    action_history: list[str] = []
    total_steps = len(episode.steps)

    for i, step in enumerate(episode.steps):
        step_data = {
            "index": i,
            "time": step.t,
            "image_path": step.observation.image_path,
            "human_action": {
                "type": step.action.type,
                "x": step.action.x,
                "y": step.action.y,
                "text": step.action.text,
            },
            "predicted_action": None,
            "match": None,
        }

        # Get prediction if model available
        if model and step.observation.image_path:
            predicted = predict_action(
                model,
                step.observation.image_path,
                episode.goal,
                step_index=i,
                total_steps=total_steps,
                action_history=action_history.copy(),
            )
            step_data["predicted_action"] = predicted

            # Check if prediction matches human action
            if predicted and predicted.get("type") == step.action.type:
                step_data["match"] = True
            else:
                step_data["match"] = False

        # Add this step's action to history for next iteration
        action_history.append(format_action(step.action, use_som=False))
        comparison_data.append(step_data)

    return comparison_data


def generate_comparison_html(
    capture_path: Path,
    episode: Episode,
    comparison_data: list[dict],
    output_path: Path,
) -> None:
    """Generate an HTML viewer with comparison data."""

    # Use openadapt-capture's viewer as base, then enhance
    try:
        from openadapt_capture.visualize.html import create_html

        # Generate base viewer
        base_html = create_html(capture_path, output=None)

        # Inject comparison data and UI
        comparison_json = json.dumps(comparison_data)

        # Add comparison panel above screenshot in main content
        comparison_panel = '''
        <div class="comparison-panel" id="comparison-panel">
            <div class="comparison-header">
                <h2>Action Comparison</h2>
                <div class="metrics-summary"></div>
                <div class="overlay-toggles"></div>
            </div>
            <div class="comparison-content">
                <div class="action-box human">
                    <div class="action-label">Human Action</div>
                    <div class="action-details" id="human-action"></div>
                </div>
                <div class="action-box predicted">
                    <div class="action-label">Model Prediction</div>
                    <div class="action-details" id="predicted-action"></div>
                </div>
                <div class="match-indicator" id="match-indicator"></div>
            </div>
        </div>
        '''

        comparison_styles = '''
        <style>
        /* Navigation bar */
        .nav-bar {
            display: flex;
            gap: 8px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .nav-link {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.8rem;
            text-decoration: none;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            transition: all 0.2s;
        }
        .nav-link:hover {
            border-color: var(--accent);
            color: var(--text-primary);
        }
        .nav-link.active {
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
            font-weight: 600;
        }
        .nav-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-right: 8px;
            align-self: center;
        }
        .comparison-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 16px;
            width: 100%;
        }
        .comparison-header {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 18px;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
        }
        .comparison-panel h2 {
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0;
            white-space: nowrap;
        }
        .comparison-content {
            padding: 14px 18px;
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 16px;
            align-items: start;
        }
        .action-box {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 0;
        }
        .action-box.human {
            background: rgba(0, 212, 170, 0.1);
            border: 1px solid rgba(0, 212, 170, 0.3);
        }
        .action-box.predicted {
            background: rgba(167, 139, 250, 0.1);
            border: 1px solid rgba(167, 139, 250, 0.3);
        }
        .action-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .action-details {
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.85rem;
        }
        .match-indicator {
            text-align: center;
            padding: 8px;
            border-radius: 6px;
            font-weight: 600;
        }
        .match-indicator.match {
            background: rgba(52, 211, 153, 0.2);
            color: #34d399;
        }
        .match-indicator.mismatch {
            background: rgba(255, 95, 95, 0.2);
            color: #ff5f5f;
        }
        .match-indicator.pending {
            background: var(--bg-tertiary);
            color: var(--text-muted);
        }
        /* Visual overlays for clicks on screenshot */
        .click-overlay {
            position: absolute;
            pointer-events: none;
            z-index: 100;
        }
        .click-marker {
            position: absolute;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            animation: pulse-marker 1.5s ease-in-out infinite;
        }
        .click-marker.human {
            background: rgba(0, 212, 170, 0.3);
            border: 3px solid #00d4aa;
            color: #00d4aa;
        }
        .click-marker.predicted {
            background: rgba(167, 139, 250, 0.3);
            border: 3px solid #a78bfa;
            color: #a78bfa;
        }
        .click-marker.human::after {
            content: 'H';
        }
        .click-marker.predicted::after {
            content: 'AI';
            font-size: 10px;
        }
        @keyframes pulse-marker {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
            50% { transform: translate(-50%, -50%) scale(1.1); opacity: 0.8; }
        }
        /* Distance line between human and predicted */
        .distance-line {
            position: absolute;
            height: 2px;
            background: linear-gradient(90deg, #00d4aa, #a78bfa);
            transform-origin: left center;
            pointer-events: none;
            z-index: 99;
        }
        /* Metrics summary - inline in header */
        .metrics-summary {
            display: flex;
            gap: 16px;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }
        .metric-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .metric-value {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent);
        }
        .metric-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        /* Toggle buttons - inline in header */
        .overlay-toggles {
            display: flex;
            gap: 6px;
            margin-left: auto;
        }
        .toggle-btn {
            padding: 6px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.2s;
            white-space: nowrap;
        }
        .toggle-btn.active {
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }
        .toggle-btn:hover {
            border-color: var(--accent);
        }
        </style>
        '''

        comparison_script = f'''
        <script>
        // Consolidated viewer script - all variables and functions in one scope
        // Export to window for cross-script access (for checkpoint dropdown script)
        window.comparisonData = {comparison_json};
        const comparisonData = window.comparisonData;  // Local alias
        window.currentIndex = 0;  // Explicit currentIndex declaration
        let currentIndex = window.currentIndex;  // Local alias
        let showHumanOverlay = true;
        let showPredictedOverlay = true;

        // Compute aggregate metrics
        window.computeMetrics = function() {{
            let matches = 0;
            let total = 0;
            let totalDistance = 0;
            let distanceCount = 0;

            comparisonData.forEach(d => {{
                if (d.match !== null) {{
                    total++;
                    if (d.match) matches++;
                }}
                // Compute spatial distance if both have coordinates
                if (d.human_action.x !== null && d.predicted_action && d.predicted_action.x !== undefined) {{
                    const dx = d.human_action.x - d.predicted_action.x;
                    const dy = d.human_action.y - d.predicted_action.y;
                    totalDistance += Math.sqrt(dx*dx + dy*dy);
                    distanceCount++;
                }}
            }});

            return {{
                accuracy: total > 0 ? (matches / total * 100).toFixed(1) : 'N/A',
                avgDistance: distanceCount > 0 ? (totalDistance / distanceCount * 100).toFixed(1) : 'N/A',
                total: comparisonData.length
            }};
        }};
        const computeMetrics = window.computeMetrics;  // Local alias

        window.updateClickOverlays = function(index) {{
            // Remove existing overlays
            document.querySelectorAll('.click-marker, .distance-line').forEach(el => el.remove());

            const data = comparisonData[index];
            if (!data) return;

            const imgContainer = document.querySelector('.display-container');
            if (!imgContainer) return;

            // Make container relative for absolute positioning
            imgContainer.style.position = 'relative';

            // Human click marker
            if (showHumanOverlay && data.human_action.x !== null) {{
                const humanMarker = document.createElement('div');
                humanMarker.className = 'click-marker human';
                humanMarker.style.left = (data.human_action.x * 100) + '%';
                humanMarker.style.top = (data.human_action.y * 100) + '%';
                imgContainer.appendChild(humanMarker);
            }}

            // Predicted click marker
            if (showPredictedOverlay && data.predicted_action && data.predicted_action.x !== undefined) {{
                const predMarker = document.createElement('div');
                predMarker.className = 'click-marker predicted';
                predMarker.style.left = (data.predicted_action.x * 100) + '%';
                predMarker.style.top = (data.predicted_action.y * 100) + '%';
                imgContainer.appendChild(predMarker);

                // Draw line between human and predicted if both visible
                if (showHumanOverlay && data.human_action.x !== null) {{
                    const line = document.createElement('div');
                    line.className = 'distance-line';
                    const x1 = data.human_action.x * imgContainer.offsetWidth;
                    const y1 = data.human_action.y * imgContainer.offsetHeight;
                    const x2 = data.predicted_action.x * imgContainer.offsetWidth;
                    const y2 = data.predicted_action.y * imgContainer.offsetHeight;
                    const length = Math.sqrt((x2-x1)**2 + (y2-y1)**2);
                    const angle = Math.atan2(y2-y1, x2-x1) * 180 / Math.PI;
                    line.style.left = x1 + 'px';
                    line.style.top = y1 + 'px';
                    line.style.width = length + 'px';
                    line.style.transform = `rotate(${{angle}}deg)`;
                    imgContainer.appendChild(line);
                }}
            }}
        }};
        const updateClickOverlays = window.updateClickOverlays;  // Local alias

        window.updateComparison = function(index) {{
            const data = comparisonData[index];
            if (!data) return;

            const humanEl = document.getElementById('human-action');
            const predictedEl = document.getElementById('predicted-action');
            const matchEl = document.getElementById('match-indicator');

            // Human action
            humanEl.innerHTML = `
                <div>Type: ${{data.human_action.type}}</div>
                ${{data.human_action.x !== null ? `<div>Position: (${{(data.human_action.x * 100).toFixed(1)}}%, ${{(data.human_action.y * 100).toFixed(1)}}%)</div>` : ''}}
                ${{data.human_action.text ? `<div>Text: ${{data.human_action.text}}</div>` : ''}}
            `;

            // Predicted action
            if (data.predicted_action) {{
                const pred = data.predicted_action;
                if (pred.x !== undefined) {{
                    predictedEl.innerHTML = `
                        <div>Type: ${{pred.type || 'click'}}</div>
                        <div>Position: (${{(pred.x * 100).toFixed(1)}}%, ${{(pred.y * 100).toFixed(1)}}%)</div>
                    `;
                }} else {{
                    predictedEl.innerHTML = `<div>${{pred.raw_output || JSON.stringify(pred)}}</div>`;
                }}
            }} else {{
                predictedEl.innerHTML = '<em style="color: var(--text-muted);">No model loaded</em>';
            }}

            // Match indicator
            if (data.match === true) {{
                matchEl.className = 'match-indicator match';
                matchEl.textContent = '✓ Match';
            }} else if (data.match === false) {{
                matchEl.className = 'match-indicator mismatch';
                matchEl.textContent = '✗ Mismatch';
            }} else {{
                matchEl.className = 'match-indicator pending';
                matchEl.textContent = '— No prediction';
            }}

            // Update visual overlays
            updateClickOverlays(index);

            // Sync currentIndex to window
            window.currentIndex = index;
        }};
        const updateComparison = window.updateComparison;  // Local alias

        window.setupOverlayToggles = function() {{
            const togglesContainer = document.querySelector('.overlay-toggles');
            if (!togglesContainer) return;

            togglesContainer.innerHTML = `
                <button class="toggle-btn active" id="toggle-human">Human (H)</button>
                <button class="toggle-btn active" id="toggle-predicted">AI (P)</button>
            `;

            document.getElementById('toggle-human').addEventListener('click', function() {{
                showHumanOverlay = !showHumanOverlay;
                this.classList.toggle('active', showHumanOverlay);
                updateClickOverlays(currentIndex);
            }});

            document.getElementById('toggle-predicted').addEventListener('click', function() {{
                showPredictedOverlay = !showPredictedOverlay;
                this.classList.toggle('active', showPredictedOverlay);
                updateClickOverlays(currentIndex);
            }});

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'h' || e.key === 'H') {{
                    document.getElementById('toggle-human').click();
                }} else if (e.key === 'p' || e.key === 'P') {{
                    document.getElementById('toggle-predicted').click();
                }}
            }});
        }};
        const setupOverlayToggles = window.setupOverlayToggles;  // Local alias

        window.setupMetricsSummary = function() {{
            const metricsEl = document.querySelector('.metrics-summary');
            if (!metricsEl) return;

            const metrics = computeMetrics();
            metricsEl.innerHTML = `
                <div class="metric-item">
                    <span class="metric-label">Accuracy:</span>
                    <span class="metric-value">${{metrics.accuracy}}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Avg Dist:</span>
                    <span class="metric-value">${{metrics.avgDistance}}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Steps:</span>
                    <span class="metric-value">${{metrics.total}}</span>
                </div>
            `;
        }};
        const setupMetricsSummary = window.setupMetricsSummary;  // Local alias

        // Hook into existing updateDisplay
        const originalUpdateDisplay = typeof updateDisplay !== 'undefined' ? updateDisplay : function() {{}};
        window.updateDisplay = updateDisplay = function(skipAudioSync) {{
            originalUpdateDisplay(skipAudioSync);
            // Sync currentIndex from base viewer if it exists
            if (typeof currentIndex !== 'undefined') {{
                window.currentIndex = currentIndex;
            }}
            updateComparison(window.currentIndex);
        }};

        // Discover other dashboards in the same directory
        async function discoverDashboards() {{
            const currentFile = window.location.pathname.split('/').pop() || 'comparison.html';

            // Create nav bar at top of container
            const container = document.querySelector('.container') || document.body.firstElementChild;
            if (!container) return;

            const navBar = document.createElement('nav');
            navBar.className = 'nav-bar';
            navBar.innerHTML = '';
            container.insertBefore(navBar, container.firstChild);

            // Known dashboard patterns to look for
            const patterns = [
                'dashboard.html',
                'comparison.html',
                'comparison_preview.html',
                'comparison_epoch0.html', 'comparison_epoch1.html', 'comparison_epoch2.html',
                'comparison_epoch3.html', 'comparison_epoch4.html', 'comparison_epoch5.html',
                'viewer.html'
            ];

            // For file:// protocol, only show essential links (fetch doesn't work)
            const isFileProtocol = window.location.protocol === 'file:';

            // Minimal links for file:// protocol - just the main ones
            const fileProtocolLinks = ['dashboard.html', currentFile];

            for (const file of patterns) {{
                try {{
                    let exists = false;
                    if (isFileProtocol) {{
                        // For file://, only show essential links
                        exists = fileProtocolLinks.includes(file);
                    }} else {{
                        const response = await fetch(file, {{ method: 'HEAD' }});
                        exists = response.ok;
                    }}

                    if (exists) {{
                        const link = document.createElement('a');
                        link.href = file;
                        link.className = 'nav-link' + (file === currentFile ? ' active' : '');
                        // Pretty name - make comparison labels clear
                        if (file === 'dashboard.html') {{
                            link.textContent = 'Training';
                        }} else if (file.startsWith('comparison_epoch')) {{
                            const epoch = file.match(/epoch(\\d+)/)?.[1];
                            link.textContent = `Comparison (E${{epoch}})`;
                        }} else if (file === 'comparison.html') {{
                            link.textContent = 'Comparison';
                        }} else if (file === 'comparison_preview.html') {{
                            link.textContent = 'Preview';
                        }} else if (file === 'viewer.html') {{
                            link.textContent = 'Viewer';
                        }} else {{
                            link.textContent = file.replace('.html', '');
                        }}
                        navBar.appendChild(link);
                    }}
                }} catch (e) {{
                    // File doesn't exist, skip
                }}
            }}
        }}

        // Initial setup
        setTimeout(() => {{
            setupOverlayToggles();
            setupMetricsSummary();
            updateComparison(window.currentIndex);
            // Note: Nav is now injected via shared header HTML, no need for discoverDashboards()
        }}, 100);
        </script>
        '''

        # Insert into HTML
        # Add shared header CSS and comparison styles before </head>
        shared_header_css = f'<style>{_get_shared_header_css()}</style>'
        html = base_html.replace('</head>', shared_header_css + comparison_styles + '</head>')

        # Add shared header HTML after container div
        shared_header_html = _generate_shared_header_html("viewer")
        html = html.replace(
            '<div class="container">',
            '<div class="container">\n' + shared_header_html
        )

        # Add comparison panel as full-width row BEFORE the main-content/sidebar flex row
        # Insert right BEFORE <div class="main-content"> as a sibling
        html = html.replace(
            '<div class="main-content">',
            comparison_panel + '\n        <div class="main-content">'
        )

        # Add script before </body>
        html = html.replace('</body>', comparison_script + '</body>')

        # Write output
        output_path.write_text(html, encoding='utf-8')
        print(f"Generated comparison viewer: {output_path}")

    except ImportError:
        print("Error: openadapt-capture is required for visualization")
        print("Install with: pip install openadapt-capture")


def main():
    parser = argparse.ArgumentParser(
        description="Compare human actions vs model predictions on a capture."
    )
    parser.add_argument(
        "--capture", "-c",
        required=True,
        help="Path to openadapt-capture recording directory",
    )
    parser.add_argument(
        "--checkpoint", "-m",
        help="Path to trained model checkpoint (optional)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output HTML path (default: capture_dir/comparison.html)",
    )
    parser.add_argument(
        "--goal", "-g",
        help="Task goal/description (auto-detected from capture if not provided)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open viewer in browser after generation",
    )
    args = parser.parse_args()

    capture_path = Path(args.capture)
    if not capture_path.exists():
        print(f"Error: Capture not found at {capture_path}")
        return 1

    # Convert capture to episode
    print(f"Loading capture from: {capture_path}")
    episode = capture_to_episode(capture_path, goal=args.goal)
    print(f"Loaded {len(episode.steps)} steps")

    # Load model if checkpoint provided
    model = load_model(args.checkpoint)

    # Generate comparison data
    print("Generating comparison data...")
    comparison_data = generate_comparison_data(episode, model)

    # Compute stats
    if model:
        matches = sum(1 for d in comparison_data if d.get("match") is True)
        total = sum(1 for d in comparison_data if d.get("match") is not None)
        if total > 0:
            print(f"Match rate: {matches}/{total} ({100*matches/total:.1f}%)")

    # Generate HTML
    output_path = Path(args.output) if args.output else capture_path / "comparison.html"
    generate_comparison_html(capture_path, episode, comparison_data, output_path)

    # Open in browser
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{output_path.absolute()}")

    return 0


def generate_unified_viewer(
    capture_path: Path,
    episode: Episode,
    predictions_by_checkpoint: dict[str, list[dict]],
    output_path: Path,
    capture_id: str | None = None,
    available_captures: list[dict] | None = None,
) -> None:
    """Generate a unified viewer with dropdowns for capture and checkpoint selection.

    Args:
        capture_path: Path to the capture directory
        episode: The episode data
        predictions_by_checkpoint: Dict mapping checkpoint names to prediction lists
            e.g. {"Epoch 1": [...], "Epoch 3": [...], "None": [...]}
        output_path: Where to write the HTML
        capture_id: ID of the current capture (for display)
        available_captures: List of available captures for the dropdown
            e.g. [{"id": "31807990", "name": "Turn off nightshift", "steps": 21}]
    """
    try:
        from openadapt_capture.visualize.html import create_html

        # Generate base viewer
        base_html = create_html(capture_path, output=None)

        # Prepare capture info
        if capture_id is None:
            capture_id = capture_path.name if capture_path else "unknown"

        if available_captures is None:
            available_captures = [{
                "id": capture_id,
                "name": episode.goal or "Untitled",
                "steps": len(episode.steps),
            }]

        # Prepare base capture data (human actions only, no predictions)
        base_data = []
        for i, step in enumerate(episode.steps):
            base_data.append({
                "index": i,
                "time": step.t,
                "image_path": step.observation.image_path,
                "human_action": {
                    "type": step.action.type,
                    "x": step.action.x,
                    "y": step.action.y,
                    "text": step.action.text,
                },
            })

        # JSON encode all data
        base_data_json = json.dumps(base_data)
        predictions_json = json.dumps(predictions_by_checkpoint)
        captures_json = json.dumps(available_captures)
        current_capture_json = json.dumps(capture_id)

        # Unified viewer styles and controls
        unified_styles = '''
        <style>
        /* Navigation bar */
        .nav-bar {
            display: flex;
            gap: 8px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }
        .nav-link {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.8rem;
            text-decoration: none;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            transition: all 0.2s;
        }
        .nav-link:hover {
            border-color: var(--accent);
            color: var(--text-primary);
        }
        .nav-link.active {
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
            font-weight: 600;
        }
        .nav-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-right: 8px;
            align-self: center;
        }

        /* Dropdown selectors */
        .viewer-controls {
            display: flex;
            gap: 16px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .control-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .control-select {
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            cursor: pointer;
            min-width: 180px;
        }
        .control-select:hover {
            border-color: var(--accent);
        }
        .control-select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2);
        }
        .control-hint {
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        /* Comparison panel */
        .comparison-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 16px;
            width: 100%;
        }
        .comparison-header {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 18px;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
        }
        .comparison-panel h2 {
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0;
            white-space: nowrap;
        }
        .comparison-content {
            padding: 14px 18px;
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 16px;
            align-items: start;
        }
        .action-box {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 0;
        }
        .action-box.human {
            background: rgba(0, 212, 170, 0.1);
            border: 1px solid rgba(0, 212, 170, 0.3);
        }
        .action-box.predicted {
            background: rgba(167, 139, 250, 0.1);
            border: 1px solid rgba(167, 139, 250, 0.3);
        }
        .action-box.predicted.disabled {
            opacity: 0.5;
        }
        .action-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .action-details {
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.85rem;
        }
        .match-indicator {
            text-align: center;
            padding: 8px;
            border-radius: 6px;
            font-weight: 600;
            min-width: 80px;
        }
        .match-indicator.match {
            background: rgba(52, 211, 153, 0.2);
            color: #34d399;
        }
        .match-indicator.mismatch {
            background: rgba(255, 95, 95, 0.2);
            color: #ff5f5f;
        }
        .match-indicator.pending {
            background: var(--bg-tertiary);
            color: var(--text-muted);
        }

        /* Visual overlays */
        .click-marker {
            position: absolute;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            pointer-events: none;
            z-index: 100;
            animation: pulse-marker 1.5s ease-in-out infinite;
        }
        .click-marker.human {
            background: rgba(0, 212, 170, 0.3);
            border: 3px solid #00d4aa;
            color: #00d4aa;
        }
        .click-marker.predicted {
            background: rgba(167, 139, 250, 0.3);
            border: 3px solid #a78bfa;
            color: #a78bfa;
        }
        .click-marker.human::after { content: 'H'; }
        .click-marker.predicted::after { content: 'AI'; font-size: 10px; }
        @keyframes pulse-marker {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
            50% { transform: translate(-50%, -50%) scale(1.1); opacity: 0.8; }
        }
        .distance-line {
            position: absolute;
            height: 2px;
            background: linear-gradient(90deg, #00d4aa, #a78bfa);
            transform-origin: left center;
            pointer-events: none;
            z-index: 99;
        }

        /* Metrics summary */
        .metrics-summary {
            display: flex;
            gap: 16px;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }
        .metric-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .metric-value {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent);
        }
        .metric-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        /* Toggle buttons */
        .overlay-toggles {
            display: flex;
            gap: 6px;
            margin-left: auto;
        }
        .toggle-btn {
            padding: 6px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.2s;
            white-space: nowrap;
        }
        .toggle-btn.active {
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }
        .toggle-btn:hover {
            border-color: var(--accent);
        }
        </style>
        '''

        # Comparison panel HTML
        comparison_panel = '''
        <div class="viewer-controls" id="viewer-controls">
            <div class="control-group">
                <span class="control-label">Training Example:</span>
                <select class="control-select" id="capture-select"></select>
                <span class="control-hint" id="capture-hint"></span>
            </div>
            <div class="control-group">
                <span class="control-label">Checkpoint:</span>
                <select class="control-select" id="checkpoint-select"></select>
            </div>
        </div>
        <div class="comparison-panel" id="comparison-panel">
            <div class="comparison-header">
                <h2>Action Comparison</h2>
                <div class="metrics-summary" id="metrics-summary"></div>
                <div class="overlay-toggles" id="overlay-toggles"></div>
            </div>
            <div class="comparison-content">
                <div class="action-box human">
                    <div class="action-label">Human Action</div>
                    <div class="action-details" id="human-action"></div>
                </div>
                <div class="action-box predicted" id="predicted-box">
                    <div class="action-label">Model Prediction</div>
                    <div class="action-details" id="predicted-action"></div>
                </div>
                <div class="match-indicator" id="match-indicator"></div>
            </div>
        </div>
        '''

        # Unified viewer script
        unified_script = f'''
        <script>
        // Consolidated unified viewer script - all variables in one scope
        // Data
        const baseData = {base_data_json};
        const predictionsByCheckpoint = {predictions_json};
        const availableCaptures = {captures_json};
        const currentCaptureId = {current_capture_json};

        // State
        let currentIndex = 0;  // Explicit currentIndex declaration
        let currentCheckpoint = 'None';
        let showHumanOverlay = true;
        let showPredictedOverlay = true;

        // Get merged data for current checkpoint
        function getMergedData() {{
            const predictions = predictionsByCheckpoint[currentCheckpoint] || [];
            return baseData.map((base, i) => {{
                const pred = predictions[i] || {{}};
                return {{
                    ...base,
                    predicted_action: pred.predicted_action || null,
                    match: pred.match !== undefined ? pred.match : null,
                }};
            }});
        }}

        // Initialize dropdowns
        function initDropdowns() {{
            const captureSelect = document.getElementById('capture-select');
            const checkpointSelect = document.getElementById('checkpoint-select');
            const captureHint = document.getElementById('capture-hint');

            // Populate capture dropdown
            captureSelect.innerHTML = '';
            availableCaptures.forEach(cap => {{
                const opt = document.createElement('option');
                opt.value = cap.id;
                opt.textContent = `${{cap.name}} (${{cap.steps}} steps)`;
                opt.selected = cap.id === currentCaptureId;
                captureSelect.appendChild(opt);
            }});

            // Show hint about available captures
            captureHint.textContent = `(${{availableCaptures.length}} available)`;

            // Populate checkpoint dropdown
            checkpointSelect.innerHTML = '';
            const checkpointNames = Object.keys(predictionsByCheckpoint);
            // Sort: "None" first, then by epoch number
            checkpointNames.sort((a, b) => {{
                if (a === 'None') return -1;
                if (b === 'None') return 1;
                const aNum = parseInt(a.match(/\\d+/)?.[0] || '999');
                const bNum = parseInt(b.match(/\\d+/)?.[0] || '999');
                return aNum - bNum;
            }});

            checkpointNames.forEach(name => {{
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name === 'None' ? 'None (Capture Only)' : name;
                checkpointSelect.appendChild(opt);
            }});

            // Set default to latest non-None checkpoint if available
            const latestCheckpoint = checkpointNames.filter(n => n !== 'None').pop();
            if (latestCheckpoint) {{
                checkpointSelect.value = latestCheckpoint;
                currentCheckpoint = latestCheckpoint;
            }}

            // Event handlers
            captureSelect.addEventListener('change', (e) => {{
                // In future: load different capture
                // For now, just show that we'd switch
                console.log('Would switch to capture:', e.target.value);
            }});

            checkpointSelect.addEventListener('change', (e) => {{
                currentCheckpoint = e.target.value;
                updateMetrics();
                updateComparison(typeof currentIndex !== 'undefined' ? currentIndex : 0);
            }});
        }}

        // Compute metrics for current checkpoint
        function computeMetrics() {{
            const data = getMergedData();
            let matches = 0;
            let total = 0;
            let totalDistance = 0;
            let distanceCount = 0;

            data.forEach(d => {{
                if (d.match !== null) {{
                    total++;
                    if (d.match) matches++;
                }}
                if (d.human_action.x !== null && d.predicted_action && d.predicted_action.x !== undefined) {{
                    const dx = d.human_action.x - d.predicted_action.x;
                    const dy = d.human_action.y - d.predicted_action.y;
                    totalDistance += Math.sqrt(dx*dx + dy*dy);
                    distanceCount++;
                }}
            }});

            return {{
                accuracy: total > 0 ? (matches / total * 100).toFixed(1) : 'N/A',
                avgDistance: distanceCount > 0 ? (totalDistance / distanceCount * 100).toFixed(1) : 'N/A',
                total: data.length,
                hasPredictions: total > 0,
            }};
        }}

        // Update metrics display
        function updateMetrics() {{
            const metricsEl = document.getElementById('metrics-summary');
            const metrics = computeMetrics();

            if (!metrics.hasPredictions) {{
                metricsEl.innerHTML = `
                    <div class="metric-item">
                        <span class="metric-label">Steps:</span>
                        <span class="metric-value">${{metrics.total}}</span>
                    </div>
                    <div class="metric-item">
                        <span style="color: var(--text-muted); font-size: 0.75rem;">No predictions - select a checkpoint</span>
                    </div>
                `;
            }} else {{
                metricsEl.innerHTML = `
                    <div class="metric-item">
                        <span class="metric-label">Accuracy:</span>
                        <span class="metric-value">${{metrics.accuracy}}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Avg Dist:</span>
                        <span class="metric-value">${{metrics.avgDistance}}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Steps:</span>
                        <span class="metric-value">${{metrics.total}}</span>
                    </div>
                `;
            }}
        }}

        // Update click overlays on screenshot
        function updateClickOverlays(index) {{
            document.querySelectorAll('.click-marker, .distance-line').forEach(el => el.remove());

            const data = getMergedData()[index];
            if (!data) return;

            const imgContainer = document.querySelector('.display-container');
            if (!imgContainer) return;
            imgContainer.style.position = 'relative';

            // Human click marker
            if (showHumanOverlay && data.human_action.x !== null) {{
                const humanMarker = document.createElement('div');
                humanMarker.className = 'click-marker human';
                humanMarker.style.left = (data.human_action.x * 100) + '%';
                humanMarker.style.top = (data.human_action.y * 100) + '%';
                imgContainer.appendChild(humanMarker);
            }}

            // Predicted click marker
            if (showPredictedOverlay && data.predicted_action && data.predicted_action.x !== undefined) {{
                const predMarker = document.createElement('div');
                predMarker.className = 'click-marker predicted';
                predMarker.style.left = (data.predicted_action.x * 100) + '%';
                predMarker.style.top = (data.predicted_action.y * 100) + '%';
                imgContainer.appendChild(predMarker);

                // Draw line between human and predicted
                if (showHumanOverlay && data.human_action.x !== null) {{
                    const line = document.createElement('div');
                    line.className = 'distance-line';
                    const x1 = data.human_action.x * imgContainer.offsetWidth;
                    const y1 = data.human_action.y * imgContainer.offsetHeight;
                    const x2 = data.predicted_action.x * imgContainer.offsetWidth;
                    const y2 = data.predicted_action.y * imgContainer.offsetHeight;
                    const length = Math.sqrt((x2-x1)**2 + (y2-y1)**2);
                    const angle = Math.atan2(y2-y1, x2-x1) * 180 / Math.PI;
                    line.style.left = x1 + 'px';
                    line.style.top = y1 + 'px';
                    line.style.width = length + 'px';
                    line.style.transform = `rotate(${{angle}}deg)`;
                    imgContainer.appendChild(line);
                }}
            }}
        }}

        // Update comparison display
        function updateComparison(index) {{
            const data = getMergedData()[index];
            if (!data) return;

            const humanEl = document.getElementById('human-action');
            const predictedEl = document.getElementById('predicted-action');
            const predictedBox = document.getElementById('predicted-box');
            const matchEl = document.getElementById('match-indicator');

            // Human action
            humanEl.innerHTML = `
                <div>Type: ${{data.human_action.type}}</div>
                ${{data.human_action.x !== null ? `<div>Position: (${{(data.human_action.x * 100).toFixed(1)}}%, ${{(data.human_action.y * 100).toFixed(1)}}%)</div>` : ''}}
                ${{data.human_action.text ? `<div>Text: ${{data.human_action.text}}</div>` : ''}}
            `;

            // Predicted action
            const hasPredictions = currentCheckpoint !== 'None';
            predictedBox.classList.toggle('disabled', !hasPredictions);

            if (!hasPredictions) {{
                predictedEl.innerHTML = '<em style="color: var(--text-muted);">Select a checkpoint to see predictions</em>';
            }} else if (data.predicted_action) {{
                const pred = data.predicted_action;
                if (pred.x !== undefined) {{
                    predictedEl.innerHTML = `
                        <div>Type: ${{pred.type || 'click'}}</div>
                        <div>Position: (${{(pred.x * 100).toFixed(1)}}%, ${{(pred.y * 100).toFixed(1)}}%)</div>
                    `;
                }} else {{
                    predictedEl.innerHTML = `<div>${{pred.raw_output || JSON.stringify(pred)}}</div>`;
                }}
            }} else {{
                predictedEl.innerHTML = '<em style="color: var(--text-muted);">No prediction available</em>';
            }}

            // Match indicator
            if (!hasPredictions) {{
                matchEl.className = 'match-indicator pending';
                matchEl.textContent = '—';
            }} else if (data.match === true) {{
                matchEl.className = 'match-indicator match';
                matchEl.textContent = '✓ Match';
            }} else if (data.match === false) {{
                matchEl.className = 'match-indicator mismatch';
                matchEl.textContent = '✗ Mismatch';
            }} else {{
                matchEl.className = 'match-indicator pending';
                matchEl.textContent = '— No prediction';
            }}

            updateClickOverlays(index);
        }}

        // Setup overlay toggle buttons
        function setupOverlayToggles() {{
            const togglesContainer = document.getElementById('overlay-toggles');
            togglesContainer.innerHTML = `
                <button class="toggle-btn active" id="toggle-human">Human (H)</button>
                <button class="toggle-btn active" id="toggle-predicted">AI (P)</button>
            `;

            document.getElementById('toggle-human').addEventListener('click', function() {{
                showHumanOverlay = !showHumanOverlay;
                this.classList.toggle('active', showHumanOverlay);
                updateClickOverlays(typeof currentIndex !== 'undefined' ? currentIndex : 0);
            }});

            document.getElementById('toggle-predicted').addEventListener('click', function() {{
                showPredictedOverlay = !showPredictedOverlay;
                this.classList.toggle('active', showPredictedOverlay);
                updateClickOverlays(typeof currentIndex !== 'undefined' ? currentIndex : 0);
            }});

            document.addEventListener('keydown', (e) => {{
                if (e.key === 'h' || e.key === 'H') document.getElementById('toggle-human').click();
                if (e.key === 'p' || e.key === 'P') document.getElementById('toggle-predicted').click();
            }});
        }}

        // Create navigation bar
        function createNavBar() {{
            const container = document.querySelector('.container') || document.body.firstElementChild;
            if (!container) return;

            const navBar = document.createElement('nav');
            navBar.className = 'nav-bar';
            navBar.id = 'nav-bar';
            navBar.innerHTML = `
                <a href="dashboard.html" class="nav-link">Training</a>
                <a href="viewer.html" class="nav-link active">Viewer</a>
            `;
            container.insertBefore(navBar, container.firstChild);
        }}

        // Hook into existing updateDisplay
        const originalUpdateDisplay = typeof updateDisplay !== 'undefined' ? updateDisplay : function() {{}};
        updateDisplay = function(skipAudioSync) {{
            originalUpdateDisplay(skipAudioSync);
            // Sync currentIndex from base viewer if it exists
            if (typeof currentIndex !== 'undefined') {{
                currentIndex = currentIndex;
            }}
            updateComparison(currentIndex);
        }};

        // Initialize
        setTimeout(() => {{
            createNavBar();
            initDropdowns();
            setupOverlayToggles();
            updateMetrics();
            updateComparison(currentIndex);
        }}, 100);
        </script>
        '''

        # Inject into HTML
        html = base_html.replace('</head>', unified_styles + '</head>')
        html = html.replace(
            '<div class="main-content">',
            comparison_panel + '\n        <div class="main-content">'
        )
        html = html.replace('</body>', unified_script + '</body>')

        # Write output
        output_path.write_text(html, encoding='utf-8')
        print(f"Generated unified viewer: {output_path}")

    except ImportError:
        print("Error: openadapt-capture is required for visualization")
        print("Install with: pip install openadapt-capture")


if __name__ == "__main__":
    exit(main())
