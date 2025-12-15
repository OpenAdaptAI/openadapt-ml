"""Unified viewer HTML generation.

This module generates the Viewer HTML with step-by-step playback,
transcript/audio sync, and model prediction comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

from openadapt_ml.training.shared_ui import (
    get_shared_header_css as _get_shared_header_css,
    generate_shared_header_html as _generate_shared_header_html,
)


def _copy_transcript_and_audio(capture_path: Path | None, output_dir: Path) -> None:
    """Copy transcript.json and convert audio to mp3 for viewer playback.

    Args:
        capture_path: Path to the capture directory (may be None)
        output_dir: Output directory for the viewer
    """
    import shutil
    import subprocess

    if capture_path is None or not capture_path.exists():
        return

    # Copy transcript.json if it exists
    transcript_src = capture_path / "transcript.json"
    transcript_dst = output_dir / "transcript.json"
    if transcript_src.exists() and not transcript_dst.exists():
        shutil.copy2(transcript_src, transcript_dst)
        print(f"  Copied transcript.json from capture")

    # Convert audio to mp3 if it exists (ffmpeg required)
    audio_dst = output_dir / "audio.mp3"
    if not audio_dst.exists():
        # Try common audio formats
        for audio_ext in [".flac", ".wav", ".m4a", ".aac", ".ogg"]:
            audio_src = capture_path / f"audio{audio_ext}"
            if audio_src.exists():
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-i", str(audio_src), "-y", "-q:a", "2", str(audio_dst)],
                        capture_output=True,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        print(f"  Converted {audio_src.name} to audio.mp3")
                    else:
                        print(f"  Warning: ffmpeg conversion failed for {audio_src.name}")
                except FileNotFoundError:
                    print("  Warning: ffmpeg not found, cannot convert audio")
                except subprocess.TimeoutExpired:
                    print("  Warning: ffmpeg timed out")
                break


def generate_unified_viewer_from_output_dir(output_dir: Path) -> Path | None:
    """Generate the unified viewer.html from existing data in output_dir.

    Collects predictions from any comparison_epoch*.html or comparison_*.html files
    and consolidates them into a single viewer with checkpoint dropdown.

    If the original capture is not available locally, extracts all data from
    existing comparison HTML files.
    """
    import re

    output_dir = Path(output_dir)

    # Try to load training log to get capture path and goal
    training_log_path = output_dir / "training_log.json"
    capture_path = None
    goal = "Complete the recorded workflow"
    capture_id = "unknown"

    evaluations: list[dict] = []

    if training_log_path.exists():
        with open(training_log_path) as f:
            log_data = json.load(f)

        capture_path_str = log_data.get("capture_path", "")
        if capture_path_str:
            capture_path = Path(capture_path_str)
            capture_id = capture_path.name
            if not capture_path.exists():
                print(f"Capture path not found locally: {capture_path}")
                capture_path = None  # Will extract from HTML files

        # Load evaluations
        evaluations = log_data.get("evaluations", [])
        if evaluations:
            print(f"  Loaded {len(evaluations)} evaluations")

    # Collect predictions and base data from JSON files or HTML files
    predictions_by_checkpoint: dict[str, list[dict]] = {"None": []}
    base_data: list[dict] | None = None

    # First, try to load from JSON files (preferred)
    for json_file in sorted(output_dir.glob("predictions_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Determine checkpoint name from filename
            name_match = re.search(r'predictions_(.+)\.json', json_file.name)
            if name_match:
                raw_name = name_match.group(1)
                if raw_name.startswith('epoch'):
                    checkpoint_name = f"Epoch {raw_name[5:]}"
                elif raw_name == 'preview':
                    checkpoint_name = "Preview"
                else:
                    checkpoint_name = raw_name.title()
            else:
                checkpoint_name = json_file.stem

            # Extract base data from first file
            if base_data is None and 'base_data' in data:
                base_data = data['base_data']

            # Store predictions
            if 'predictions' in data:
                predictions_by_checkpoint[checkpoint_name] = data['predictions']
                print(f"  Loaded predictions from {json_file.name}")
        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")

    # Fallback: look for comparison_epoch*.html files and extract their data
    for comp_file in sorted(output_dir.glob("comparison_epoch*.html")):
        match = re.search(r'epoch(\d+)', comp_file.name)
        if not match:
            continue

        epoch_num = match.group(1)
        checkpoint_name = f"Epoch {epoch_num}"

        # Extract comparisonData from the HTML
        try:
            html_content = comp_file.read_text()
            # Look for const comparisonData = [...];
            data_match = re.search(
                r'const\s+comparisonData\s*=\s*(\[.*?\]);',
                html_content,
                re.DOTALL
            )
            if data_match:
                comparison_data = json.loads(data_match.group(1))

                # Extract base data from the first file we find
                if base_data is None:
                    base_data = []
                    for item in comparison_data:
                        base_data.append({
                            "index": item.get("index", 0),
                            "time": item.get("time", 0),
                            "image_path": item.get("image_path", ""),
                            "human_action": item.get("human_action", {}),
                        })

                # Extract predictions
                predictions = []
                for item in comparison_data:
                    predictions.append({
                        "predicted_action": item.get("predicted_action"),
                        "match": item.get("match"),
                    })
                predictions_by_checkpoint[checkpoint_name] = predictions
                print(f"  Loaded predictions from {comp_file.name}")
        except Exception as e:
            print(f"  Warning: Could not extract data from {comp_file.name}: {e}")

    # Also check comparison_preview.html
    preview_file = output_dir / "comparison_preview.html"
    if preview_file.exists():
        try:
            html_content = preview_file.read_text()
            data_match = re.search(
                r'const\s+comparisonData\s*=\s*(\[.*?\]);',
                html_content,
                re.DOTALL
            )
            if data_match:
                comparison_data = json.loads(data_match.group(1))

                # Extract base data if we haven't yet
                if base_data is None:
                    base_data = []
                    for item in comparison_data:
                        base_data.append({
                            "index": item.get("index", 0),
                            "time": item.get("time", 0),
                            "image_path": item.get("image_path", ""),
                            "human_action": item.get("human_action", {}),
                        })

                predictions = []
                for item in comparison_data:
                    predictions.append({
                        "predicted_action": item.get("predicted_action"),
                        "match": item.get("match"),
                    })
                # Only add if it has actual predictions
                has_predictions = any(p.get("predicted_action") for p in predictions)
                if has_predictions and "Preview" not in predictions_by_checkpoint:
                    predictions_by_checkpoint["Preview"] = predictions
                    print(f"  Loaded predictions from comparison_preview.html")
        except Exception as e:
            print(f"  Warning: Could not extract data from comparison_preview.html: {e}")

    # If we still don't have base data, we can't generate the viewer
    if base_data is None:
        print("No comparison data found, cannot generate unified viewer")
        return None

    # Copy transcript and audio files from capture if available
    _copy_transcript_and_audio(capture_path, output_dir)

    # Generate the unified viewer using standalone HTML template
    # (Consolidated approach - always use standalone for reliability)
    viewer_path = output_dir / "viewer.html"

    _generate_unified_viewer_from_extracted_data(
        base_data=base_data,
        predictions_by_checkpoint=predictions_by_checkpoint,
        output_path=viewer_path,
        capture_id=capture_id,
        goal=goal,
        evaluations=evaluations,
    )

    return viewer_path


def _generate_unified_viewer_from_extracted_data(
    base_data: list[dict],
    predictions_by_checkpoint: dict[str, list[dict]],
    output_path: Path,
    capture_id: str = "unknown",
    goal: str = "Untitled",
    evaluations: list[dict] | None = None,
) -> None:
    """Generate unified viewer from extracted comparison data.

    This is used when the original capture isn't available locally
    but we have comparison HTML files to extract from.
    """
    # Get shared header components for consistent nav
    shared_header_css = _get_shared_header_css()
    shared_header_html = _generate_shared_header_html("viewer")

    # Build base HTML from extracted data (standalone, no openadapt-capture dependency)
    base_data_json = json.dumps(base_data)
    predictions_json = json.dumps(predictions_by_checkpoint)
    evaluations_json = json.dumps(evaluations or [])
    captures_json = json.dumps([{
        "id": capture_id,
        "name": goal,
        "steps": len(base_data),
    }])
    current_capture_json = json.dumps(capture_id)

    # Find first image to get dimensions (for display)
    first_image_path = base_data[0].get("image_path", "") if base_data else ""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Viewer - {capture_id}</title>
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --border-color: rgba(255, 255, 255, 0.06);
            --text-primary: #f0f0f0;
            --text-secondary: #888;
            --text-muted: #555;
            --accent: #00d4aa;
            --accent-dim: rgba(0, 212, 170, 0.15);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }}
        {shared_header_css}
        .nav-bar {{
            display: flex;
            gap: 8px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .nav-link {{
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.8rem;
            text-decoration: none;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            transition: all 0.2s;
        }}
        .nav-link:hover {{ border-color: var(--accent); color: var(--text-primary); }}
        .nav-link.active {{
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
            font-weight: 600;
        }}
        .nav-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-right: 8px;
        }}
        .viewer-controls {{
            display: flex;
            gap: 16px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .control-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .control-select {{
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            cursor: pointer;
            min-width: 200px;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M3 4.5L6 7.5L9 4.5'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 12px center;
            padding-right: 32px;
            transition: all 0.2s;
        }}
        .control-select:hover {{ border-color: var(--accent); background-color: var(--bg-secondary); }}
        .control-select:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-dim); }}
        .control-hint {{ font-size: 0.7rem; color: var(--text-muted); }}
        .comparison-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 16px;
        }}
        .comparison-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 18px;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
        }}
        .comparison-panel h2 {{ font-size: 0.9rem; font-weight: 600; margin: 0; }}
        .comparison-content {{
            padding: 14px 18px;
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 16px;
            align-items: start;
        }}
        .action-box {{ padding: 12px; border-radius: 8px; }}
        .action-box.human {{
            background: rgba(0, 212, 170, 0.1);
            border: 1px solid rgba(0, 212, 170, 0.3);
        }}
        .action-box.predicted {{
            background: rgba(167, 139, 250, 0.1);
            border: 1px solid rgba(167, 139, 250, 0.3);
        }}
        .action-box.predicted.disabled {{ opacity: 0.5; }}
        .action-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .action-details {{ font-family: "SF Mono", Monaco, monospace; font-size: 0.85rem; }}
        .match-indicator {{
            text-align: center;
            padding: 8px;
            border-radius: 6px;
            font-weight: 600;
            min-width: 80px;
        }}
        .match-indicator.match {{ background: rgba(52, 211, 153, 0.2); color: #34d399; }}
        .match-indicator.mismatch {{ background: rgba(255, 95, 95, 0.2); color: #ff5f5f; }}
        .match-indicator.pending {{ background: var(--bg-tertiary); color: var(--text-muted); }}
        .metrics-summary {{
            display: flex;
            gap: 16px;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }}
        .metric-item {{ display: flex; align-items: center; gap: 6px; }}
        .metric-value {{ font-size: 0.9rem; font-weight: 600; color: var(--accent); }}
        .metric-label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }}
        .overlay-toggles {{ display: flex; gap: 6px; margin-left: auto; }}
        .toggle-btn {{
            padding: 6px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.75rem;
        }}
        .toggle-btn.active {{ background: var(--accent); color: var(--bg-primary); border-color: var(--accent); }}
        .main-content {{ display: grid; grid-template-columns: 1fr 340px; gap: 24px; }}
        .viewer-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }}
        .frame-container {{
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 420px;
        }}
        .frame-container img {{ max-width: 100%; max-height: 70vh; object-fit: contain; }}
        .click-marker {{
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
        }}
        .click-marker.human {{
            background: rgba(0, 212, 170, 0.3);
            border: 3px solid #00d4aa;
            color: #00d4aa;
        }}
        .click-marker.predicted {{
            background: rgba(167, 139, 250, 0.3);
            border: 3px solid #a78bfa;
            color: #a78bfa;
        }}
        .click-marker.human::after {{ content: 'H'; }}
        .click-marker.predicted::after {{ content: 'AI'; font-size: 10px; }}
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .step-list {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            max-height: 500px;
            overflow-y: auto;
        }}
        .step-item {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            transition: background 0.2s;
        }}
        .step-item:hover {{ background: var(--bg-tertiary); }}
        .step-item.active {{ background: var(--accent-dim); border-left: 3px solid var(--accent); }}
        .step-index {{ font-weight: 600; color: var(--accent); }}
        .step-action {{ font-size: 0.85rem; color: var(--text-secondary); }}
        .eval-badges {{
            display: flex;
            gap: 4px;
            margin-top: 4px;
        }}
        .eval-badge {{
            display: inline-flex;
            align-items: center;
            gap: 2px;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 600;
        }}
        .eval-badge.correct {{
            background: rgba(52, 211, 153, 0.2);
            color: #34d399;
        }}
        .eval-badge.incorrect {{
            background: rgba(255, 95, 95, 0.2);
            color: #ff5f5f;
        }}
        .eval-badge .epoch {{
            opacity: 0.7;
        }}
        .playback-controls {{
            display: flex;
            gap: 8px;
            padding: 12px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .playback-btn {{
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            min-width: 40px;
            text-align: center;
        }}
        .playback-btn:hover {{ border-color: var(--accent); }}
        .playback-btn.active {{ background: var(--accent); color: var(--bg-primary); border-color: var(--accent); }}
        .playback-btn.primary {{ flex: 1; min-width: 60px; }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-left: auto;
        }}
        .speed-control label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .speed-control select {{
            padding: 4px 8px;
            border-radius: 4px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            font-size: 0.8rem;
            cursor: pointer;
        }}
        .progress-bar {{
            width: 100%;
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
            cursor: pointer;
        }}
        .progress-bar .progress {{
            height: 100%;
            background: var(--accent);
            transition: width 0.1s ease;
        }}
        .timeline {{
            width: 100%;
            height: 32px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-top: 12px;
            position: relative;
            cursor: pointer;
            overflow: hidden;
        }}
        .timeline-segments {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 16px;
            display: flex;
        }}
        .timeline-segment {{
            height: 100%;
            background: rgba(0, 212, 170, 0.2);
            border-right: 1px solid var(--bg-secondary);
            transition: background 0.15s;
            position: relative;
        }}
        .timeline-segment:hover {{
            background: rgba(0, 212, 170, 0.4);
        }}
        .timeline-segment.active {{
            background: rgba(0, 212, 170, 0.5);
        }}
        .timeline-segment-tooltip {{
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s;
            z-index: 100;
        }}
        .timeline-segment:hover .timeline-segment-tooltip {{
            opacity: 1;
        }}
        .timeline-markers {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 16px;
        }}
        .timeline-marker {{
            position: absolute;
            bottom: 2px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            transform: translateX(-50%);
            border: 2px solid var(--bg-primary);
            cursor: pointer;
            transition: transform 0.1s;
        }}
        .timeline-marker:hover {{
            transform: translateX(-50%) scale(1.3);
        }}
        .timeline-marker.click {{ background: #ff5f5f; }}
        .timeline-marker.double_click {{ background: #ff5f5f; }}
        .timeline-marker.type {{ background: #34d399; }}
        .timeline-marker.scroll {{ background: #a78bfa; }}
        .timeline-marker.drag {{ background: #00d4aa; }}
        .timeline-marker.done {{ background: #888; }}
        .timeline-marker.active {{
            box-shadow: 0 0 8px currentColor;
        }}
        .timeline-playhead {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--accent);
            transform: translateX(-50%);
            pointer-events: none;
            z-index: 10;
        }}
        .timeline-playhead::after {{
            content: '';
            position: absolute;
            top: -4px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid var(--accent);
        }}
        .details-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-top: 16px;
        }}
        .details-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        .details-content {{
            padding: 12px 16px;
            font-size: 0.82rem;
            max-height: 400px;
            overflow-y: auto;
        }}
        .detail-row {{
            display: flex;
            margin-bottom: 6px;
        }}
        .detail-key {{
            color: var(--text-muted);
            min-width: 70px;
            font-size: 0.75rem;
            text-transform: uppercase;
        }}
        .detail-value {{
            font-family: "SF Mono", Monaco, monospace;
            color: var(--text-secondary);
        }}
        .copy-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.7rem;
            text-transform: uppercase;
        }}
        .copy-btn:hover {{ background: var(--bg-secondary); color: var(--text-primary); }}
        .copy-btn.copied {{ background: var(--accent-dim); color: var(--accent); border-color: var(--accent); }}
        .cost-panel {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05));
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            display: none;
        }}
        .cost-panel.visible {{ display: flex; }}
        .cost-panel .cost-items {{
            display: flex;
            gap: 24px;
            align-items: center;
            flex: 1;
        }}
        .cost-panel .cost-item {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        .cost-panel .cost-label {{
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .cost-panel .cost-value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #ef4444;
            font-family: "SF Mono", Monaco, monospace;
        }}
        .cost-panel .cost-info {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-left: auto;
        }}
        .transcript-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }}
        .transcript-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-color);
        }}
        .transcript-panel h2 {{
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0;
        }}
        .transcript-follow-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.7rem;
            transition: all 0.2s;
        }}
        .transcript-follow-btn:hover {{
            border-color: var(--accent);
            color: var(--text-secondary);
        }}
        .transcript-follow-btn.active {{
            background: var(--accent-dim);
            border-color: var(--accent);
            color: var(--accent);
        }}
        .transcript-content {{
            padding: 14px 18px;
            font-size: 0.85rem;
            line-height: 1.9;
            color: var(--text-secondary);
            max-height: 150px;
            overflow-y: auto;
        }}
        .transcript-segment {{
            display: inline;
            cursor: pointer;
            padding: 2px 6px;
            border-radius: 4px;
            transition: all 0.15s ease;
        }}
        .transcript-segment:hover {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        .transcript-segment.active {{
            background: var(--accent-dim);
            color: var(--accent);
        }}
        .transcript-time {{
            color: var(--text-muted);
            font-size: 0.7rem;
            font-family: "SF Mono", Monaco, monospace;
            margin-right: 4px;
        }}
        .transcript-empty {{
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 16px;
        }}
        .step-list-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        .step-list-header h3 {{
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0;
        }}
        .copy-all-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.7rem;
            text-transform: uppercase;
        }}
        .copy-all-btn:hover {{ background: var(--bg-secondary); color: var(--text-primary); }}
        .copy-all-btn.copied {{ background: var(--accent-dim); color: var(--accent); border-color: var(--accent); }}

        /* Gallery Panel (Compact - in sidebar) */
        .gallery-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-top: 16px;
        }}
        .gallery-panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
        }}
        .gallery-panel-controls {{
            display: flex;
            gap: 8px;
        }}
        .gallery-maximize-btn, .gallery-collapse-btn, .gallery-close-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
        }}
        .gallery-maximize-btn:hover, .gallery-collapse-btn:hover, .gallery-close-btn:hover {{
            background: var(--bg-secondary);
            color: var(--text-primary);
        }}
        .gallery-panel-content {{
            padding: 12px;
            max-height: 300px;
            overflow-y: auto;
        }}
        .gallery-panel.collapsed .gallery-panel-content {{
            display: none;
        }}
        .gallery-panel.collapsed .gallery-collapse-btn {{
            transform: rotate(-90deg);
        }}
        .gallery-filters-compact {{
            display: flex;
            gap: 8px;
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }}
        .gallery-filter-select {{
            padding: 4px 8px;
            border-radius: 4px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            font-size: 0.75rem;
            cursor: pointer;
        }}
        .gallery-grid-compact {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
        }}
        .gallery-card {{
            background: var(--bg-tertiary);
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            cursor: pointer;
            transition: border-color 0.2s;
        }}
        .gallery-card:hover {{
            border-color: var(--accent);
        }}
        .gallery-card.hidden {{
            display: none;
        }}
        .gallery-card .image-wrapper {{
            position: relative;
            background: #000;
            min-height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .gallery-card img {{
            width: 100%;
            height: auto;
            max-height: 120px;
            object-fit: contain;
        }}
        .gallery-card .overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }}
        .gallery-card .marker {{
            position: absolute;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            border: 2px solid white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 7px;
            font-weight: 700;
            z-index: 10;
        }}
        .gallery-card .marker.human {{
            background: rgba(0, 212, 170, 0.5);
            border-color: #00d4aa;
            color: #00d4aa;
        }}
        .gallery-card .marker.predicted {{
            background: rgba(167, 139, 250, 0.5);
            border-color: #a78bfa;
            color: #a78bfa;
        }}
        .gallery-card .card-content {{
            padding: 8px;
        }}
        .gallery-card .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.7rem;
        }}
        .gallery-card .step-num {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        .gallery-card .status {{
            font-size: 0.65rem;
            font-weight: 600;
        }}
        .gallery-card .status.correct {{
            color: #34d399;
        }}
        .gallery-card .status.incorrect {{
            color: #ff5f5f;
        }}
        .gallery-empty {{
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 0.8rem;
        }}

        /* Gallery Maximized Overlay */
        .gallery-maximized-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            padding: 20px;
            overflow-y: auto;
        }}
        .gallery-maximized-overlay.active {{
            display: block;
        }}
        .gallery-maximized-content {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .gallery-maximized-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        .gallery-maximized-controls {{
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .gallery-grid-maximized {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .gallery-grid-maximized .gallery-card .image-wrapper {{
            min-height: 150px;
        }}
        .gallery-grid-maximized .gallery-card img {{
            max-height: 250px;
        }}
        .gallery-grid-maximized .gallery-card .marker {{
            width: 22px;
            height: 22px;
            font-size: 9px;
        }}
        .gallery-grid-maximized .gallery-card .card-content {{
            padding: 12px;
        }}
        .gallery-grid-maximized .gallery-card .card-header {{
            font-size: 0.85rem;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }}
        .gallery-grid-maximized .gallery-card .card-details {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        .gallery-grid-maximized .gallery-card .coord-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }}
        .gallery-grid-maximized .gallery-card .coord-human {{
            color: #34d399;
        }}
        .gallery-grid-maximized .gallery-card .coord-pred {{
            color: #a78bfa;
        }}
    </style>
</head>
<body>
    {shared_header_html}

    <div class="container">
        <div class="viewer-controls">
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

        <div class="cost-panel" id="cost-panel">
            <div class="cost-items">
                <div class="cost-item">
                    <div class="cost-label">Running Cost</div>
                    <div class="cost-value" id="cost-running">$0.00</div>
                </div>
                <div class="cost-item">
                    <div class="cost-label">Total Cost</div>
                    <div class="cost-value" id="cost-total">$0.00</div>
                </div>
                <div class="cost-info" id="cost-info"></div>
            </div>
        </div>

        <div class="comparison-panel">
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

        <div class="main-content" id="main-content">
            <div class="viewer-section">
                <div class="frame-container" id="frame-container">
                    <img id="frame-image" src="" alt="Screenshot">
                    <div id="image-placeholder" style="display:none;flex-direction:column;align-items:center;justify-content:center;min-height:300px;width:100%;"></div>
                </div>
                <div class="gallery-panel" id="gallery-panel">
                    <div class="gallery-panel-header">
                        <span style="font-size:0.9rem;font-weight:600;">Evaluation Gallery</span>
                        <div class="gallery-panel-controls">
                            <button class="gallery-maximize-btn" id="gallery-maximize-btn" title="Maximize gallery">⤢</button>
                            <button class="gallery-collapse-btn" id="gallery-collapse-btn" title="Collapse">▼</button>
                        </div>
                    </div>
                    <div class="gallery-panel-content" id="gallery-panel-content">
                        <div class="gallery-filters-compact">
                            <select class="gallery-filter-select" id="gallery-epoch-filter">
                                <option value="all">All Epochs</option>
                            </select>
                            <select class="gallery-filter-select" id="gallery-status-filter">
                                <option value="all">All</option>
                                <option value="correct">Correct</option>
                                <option value="incorrect">Incorrect</option>
                            </select>
                            <span class="control-hint" id="gallery-count">0 samples</span>
                        </div>
                        <div class="gallery-grid-compact" id="gallery-grid"></div>
                        <div class="gallery-empty" id="gallery-empty" style="display:none;">
                            No evaluations available.
                        </div>
                    </div>
                </div>
            </div>
            <div class="sidebar">
                <div class="playback-controls">
                    <button class="playback-btn" id="rewind-btn" title="Rewind (Home)">⏮</button>
                    <button class="playback-btn" id="prev-btn" title="Previous (←)">◀</button>
                    <button class="playback-btn primary" id="play-btn" title="Play/Pause (Space)">▶ Play</button>
                    <button class="playback-btn" id="next-btn" title="Next (→)">▶</button>
                    <button class="playback-btn" id="end-btn" title="End (End)">⏭</button>
                    <div class="speed-control">
                        <label>Speed</label>
                        <select id="speed-select">
                            <option value="2000">0.5x</option>
                            <option value="1000" selected>1x</option>
                            <option value="500">2x</option>
                            <option value="250">4x</option>
                        </select>
                    </div>
                    <div class="progress-bar" id="progress-bar">
                        <div class="progress" id="progress"></div>
                    </div>
                    <div class="timeline" id="timeline">
                        <div class="timeline-segments" id="timeline-segments"></div>
                        <div class="timeline-markers" id="timeline-markers"></div>
                        <div class="timeline-playhead" id="timeline-playhead" style="left: 0%"></div>
                    </div>
                </div>
                <div class="step-list" id="step-list">
                    <div class="step-list-header">
                        <h3>Steps</h3>
                        <button class="copy-all-btn" id="copy-all-btn">Copy All</button>
                    </div>
                    <div id="step-list-items"></div>
                </div>
                <div class="transcript-panel" id="transcript-panel">
                    <div class="transcript-header">
                        <h2>Transcript</h2>
                        <button class="transcript-follow-btn active" id="transcript-follow-btn" title="Auto-scroll to active segment">Follow</button>
                    </div>
                    <div class="transcript-content" id="transcript-content"></div>
                </div>
                <audio id="audio" style="display:none;"></audio>
                <div class="details-panel" id="details-panel">
                    <div class="details-header">
                        <span style="font-size:0.9rem;font-weight:600;">Step Details</span>
                        <button class="copy-btn" id="copy-btn">Copy</button>
                    </div>
                    <div class="details-content" id="details-content"></div>
                </div>
            </div>
        </div>

        <div class="gallery-maximized-overlay" id="gallery-maximized-overlay">
            <div class="gallery-maximized-content">
                <div class="gallery-maximized-header">
                    <h2 style="margin:0;font-size:1.1rem;">Evaluation Gallery</h2>
                    <div class="gallery-maximized-controls">
                        <select class="gallery-filter-select" id="gallery-epoch-filter-max">
                            <option value="all">All Epochs</option>
                        </select>
                        <select class="gallery-filter-select" id="gallery-status-filter-max">
                            <option value="all">All</option>
                            <option value="correct">Correct</option>
                            <option value="incorrect">Incorrect</option>
                        </select>
                        <span class="control-hint" id="gallery-count-max">0 samples</span>
                        <button class="gallery-close-btn" id="gallery-close-btn" title="Close">✕</button>
                    </div>
                </div>
                <div class="gallery-grid-maximized" id="gallery-grid-max"></div>
            </div>
        </div>
    </div>

    <script>
    const baseData = {base_data_json};
    const predictionsByCheckpoint = {predictions_json};
    const evaluations = {evaluations_json};
    const availableCaptures = {captures_json};
    const currentCaptureId = {current_capture_json};

    // Build evaluation map: step_idx -> [{{epoch, correct, distance}}]
    const evalByStep = {{}};
    evaluations.forEach(ev => {{
        const idx = ev.sample_idx;
        if (!evalByStep[idx]) evalByStep[idx] = [];
        evalByStep[idx].push({{
            epoch: ev.epoch,
            correct: ev.correct,
            distance: ev.distance
        }});
    }});

    let currentIndex = 0;
    let currentCheckpoint = 'None';
    let showHumanOverlay = true;
    let showPredictedOverlay = true;
    let isPlaying = false;
    let playInterval = null;
    let playSpeed = 1000;  // ms per step

    // Cloud cost tracking
    const COST_RATES = {{
        'gpu_1x_a10': 0.75,      // Lambda Labs A10
        'gpu_8x_a100': 1.29,     // Lambda Labs A100 (per GPU)
        'a10': 0.75,             // Generic A10
        'a100': 1.29,            // Generic A100
    }};

    function getHourlyRate(instanceType) {{
        if (!instanceType) return 0;
        // Try exact match first
        const lowerType = instanceType.toLowerCase();
        if (COST_RATES[lowerType]) {{
            return COST_RATES[lowerType];
        }}
        // Try partial match
        if (lowerType.includes('a100')) return COST_RATES['a100'];
        if (lowerType.includes('a10')) return COST_RATES['a10'];
        // Default to A10 rate
        return COST_RATES['a10'];
    }}

    async function loadAndDisplayCosts() {{
        try {{
            const response = await fetch('training_log.json?t=' + Date.now());
            if (!response.ok) return;

            const data = await response.json();
            const instanceType = data.instance_type || '';

            // Only show costs for actual cloud training (not stub/local)
            if (!instanceType || instanceType === '' || instanceType === 'stub') {{
                document.getElementById('cost-panel').style.display = 'none';
                return;
            }}

            const hourlyRate = getHourlyRate(instanceType);
            const elapsedTime = data.elapsed_time || 0;
            const elapsedHours = elapsedTime / 3600;
            const totalCost = elapsedHours * hourlyRate;

            // Update display
            document.getElementById('cost-running').textContent = `$${{totalCost.toFixed(2)}}`;
            document.getElementById('cost-total').textContent = `$${{totalCost.toFixed(2)}}`;
            document.getElementById('cost-info').textContent = `${{instanceType}} @ $${{hourlyRate.toFixed(2)}}/hr`;
            document.getElementById('cost-panel').classList.add('visible');
        }} catch (e) {{
            // Silently fail if training_log.json not available
            console.log('Could not load training costs:', e);
        }}
    }}

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

    function parseModelOutput(rawOutput) {{
        // Parse model output for structured action commands
        let action = null;
        let thinking = '';

        // Try to extract SoM actions: CLICK([N]), TYPE([N], "text"), TYPE("text")
        const clickSomMatch = rawOutput.match(/CLICK\\s*\\(\\s*\\[\\s*(\\d+)\\s*\\]\\s*\\)/);
        const typeSomMatch = rawOutput.match(/TYPE\\s*\\(\\s*\\[\\s*(\\d+)\\s*\\]\\s*,\\s*["']([^"']*)["']\\s*\\)/);
        const typeSimpleMatch = rawOutput.match(/TYPE\\s*\\(\\s*["']([^"']*)["']\\s*\\)/);

        // Try coordinate-based: CLICK(x=0.5, y=0.5)
        const clickCoordMatch = rawOutput.match(/CLICK\\s*\\(\\s*x\\s*=\\s*([\\d.]+)\\s*,\\s*y\\s*=\\s*([\\d.]+)\\s*\\)/);

        // Try to extract thinking/reasoning
        const thinkMatch = rawOutput.match(/(?:Thought|Thinking|Reasoning|Analysis):\\s*([\\s\\S]*?)(?:Action:|$)/i);
        const actionMatch = rawOutput.match(/Action:\\s*([^\\n]+)/i);

        if (thinkMatch) thinking = thinkMatch[1].trim().substring(0, 150);

        if (clickSomMatch) {{
            action = {{ type: 'click', element: `[${{clickSomMatch[1]}}]` }};
        }} else if (typeSomMatch) {{
            action = {{ type: 'type', element: `[${{typeSomMatch[1]}}]`, text: typeSomMatch[2] }};
        }} else if (typeSimpleMatch) {{
            action = {{ type: 'type', text: typeSimpleMatch[1] }};
        }} else if (clickCoordMatch) {{
            action = {{ type: 'click', x: parseFloat(clickCoordMatch[1]), y: parseFloat(clickCoordMatch[2]) }};
        }} else if (actionMatch) {{
            // Extract the action line for cleaner display
            action = {{ type: 'raw', text: actionMatch[1].trim() }};
        }}

        // Generate HTML
        let html = '';
        if (action) {{
            if (action.type === 'click' && action.element) {{
                html = `<div style="font-weight:600;color:var(--accent);">CLICK(${{action.element}})</div>`;
            }} else if (action.type === 'click' && action.x !== undefined) {{
                html = `<div style="font-weight:600;color:var(--accent);">CLICK(x=${{action.x.toFixed(2)}}, y=${{action.y.toFixed(2)}})</div>`;
            }} else if (action.type === 'type') {{
                const elem = action.element ? `${{action.element}}, ` : '';
                html = `<div style="font-weight:600;color:var(--accent);">TYPE(${{elem}}"${{action.text}}")</div>`;
            }} else if (action.type === 'raw') {{
                html = `<div style="color:var(--accent);">${{action.text}}</div>`;
            }}
            if (thinking) {{
                html += `<div style="font-size:0.8rem;color:var(--text-muted);margin-top:4px;max-height:60px;overflow:hidden;">${{thinking}}...</div>`;
            }}
        }} else {{
            // No parseable action - show truncated raw output
            const truncated = rawOutput.substring(0, 200).replace(/\\n/g, ' ');
            html = `<div style="font-size:0.85rem;color:var(--text-muted);max-height:80px;overflow:hidden;">${{truncated}}${{rawOutput.length > 200 ? '...' : ''}}</div>`;
        }}

        return {{ action, thinking, html }};
    }}

    function initDropdowns() {{
        const captureSelect = document.getElementById('capture-select');
        const checkpointSelect = document.getElementById('checkpoint-select');
        const captureHint = document.getElementById('capture-hint');

        captureSelect.innerHTML = '';
        availableCaptures.forEach(cap => {{
            const opt = document.createElement('option');
            opt.value = cap.id;
            opt.textContent = `${{cap.name}} (${{cap.steps}} steps)`;
            opt.selected = cap.id === currentCaptureId;
            captureSelect.appendChild(opt);
        }});
        captureHint.textContent = `(${{availableCaptures.length}} available)`;

        checkpointSelect.innerHTML = '';
        const checkpointNames = Object.keys(predictionsByCheckpoint);
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
        const latestCheckpoint = checkpointNames.filter(n => n !== 'None').pop();
        if (latestCheckpoint) {{
            checkpointSelect.value = latestCheckpoint;
            currentCheckpoint = latestCheckpoint;
        }}
        checkpointSelect.addEventListener('change', (e) => {{
            currentCheckpoint = e.target.value;
            updateMetrics();
            updateDisplay();
        }});
    }}

    function computeMetrics() {{
        const data = getMergedData();
        let matches = 0, total = 0;
        data.forEach(d => {{
            if (d.match !== null) {{ total++; if (d.match) matches++; }}
        }});
        return {{
            accuracy: total > 0 ? (matches / total * 100).toFixed(1) : 'N/A',
            total: data.length,
            hasPredictions: total > 0,
        }};
    }}

    function updateMetrics() {{
        const metricsEl = document.getElementById('metrics-summary');
        const metrics = computeMetrics();
        if (!metrics.hasPredictions) {{
            metricsEl.innerHTML = `<div class="metric-item"><span class="metric-label">Steps:</span><span class="metric-value">${{metrics.total}}</span></div>`;
        }} else {{
            metricsEl.innerHTML = `<div class="metric-item"><span class="metric-label">Accuracy:</span><span class="metric-value">${{metrics.accuracy}}%</span></div><div class="metric-item"><span class="metric-label">Steps:</span><span class="metric-value">${{metrics.total}}</span></div>`;
        }}
    }}

    function updateDisplay() {{
        const data = getMergedData()[currentIndex];
        if (!data) return;

        // Update image - handle both local and remote paths
        const imgEl = document.getElementById('frame-image');
        const placeholderEl = document.getElementById('image-placeholder');

        // Check if image path is remote (Lambda Labs path)
        const imagePath = data.image_path || '';
        const isRemote = imagePath.startsWith('/home/ubuntu/') || imagePath.startsWith('/root/');

        // Try local screenshots folder first
        const localPath = isRemote ? 'screenshots/' + imagePath.split('/').pop() : imagePath;

        imgEl.src = localPath;
        imgEl.style.display = 'block';
        if (placeholderEl) placeholderEl.style.display = 'none';

        imgEl.onerror = () => {{
            imgEl.style.display = 'none';
            if (placeholderEl) {{
                placeholderEl.style.display = 'flex';
                placeholderEl.innerHTML = `
                    <div style="text-align:center;padding:40px;color:var(--text-muted);">
                        <div style="font-size:2rem;margin-bottom:12px;">📷</div>
                        <div style="margin-bottom:8px;color:var(--text-secondary);">Screenshots not downloaded</div>
                        <div style="font-size:0.8rem;margin-bottom:12px;">
                            Run: <code style="background:var(--bg-tertiary);padding:4px 8px;border-radius:4px;">uv run python -m openadapt_ml.cloud.lambda_labs rsync remote:/home/ubuntu/capture/screenshots/ training_output/screenshots/</code>
                        </div>
                        <div style="font-size:0.75rem;color:var(--text-muted);">Step ${{currentIndex + 1}} of ${{baseData.length}}</div>
                    </div>
                `;
            }}
        }};

        // Update human action
        const humanEl = document.getElementById('human-action');
        humanEl.innerHTML = `<div>Type: ${{data.human_action.type || 'unknown'}}</div>${{data.human_action.x !== null && data.human_action.x !== undefined ? `<div>Position: (${{(data.human_action.x * 100).toFixed(1)}}%, ${{(data.human_action.y * 100).toFixed(1)}}%)</div>` : ''}}${{data.human_action.text ? `<div>Text: ${{data.human_action.text}}</div>` : ''}}`;

        // Update predicted action
        const predictedEl = document.getElementById('predicted-action');
        const predictedBox = document.getElementById('predicted-box');
        const hasPredictions = currentCheckpoint !== 'None';
        predictedBox.classList.toggle('disabled', !hasPredictions);
        if (!hasPredictions) {{
            predictedEl.innerHTML = '<em style="color:var(--text-muted);">Select a checkpoint</em>';
        }} else if (data.predicted_action) {{
            const pred = data.predicted_action;
            if (pred.x !== undefined) {{
                predictedEl.innerHTML = `<div>Type: ${{pred.type || 'click'}}</div><div>Position: (${{(pred.x * 100).toFixed(1)}}%, ${{(pred.y * 100).toFixed(1)}}%)</div>`;
            }} else {{
                // Parse raw_output for actions
                const rawOutput = pred.raw_output || JSON.stringify(pred);
                const parsed = parseModelOutput(rawOutput);
                predictedEl.innerHTML = parsed.html;
            }}
        }} else {{
            predictedEl.innerHTML = '<em style="color:var(--text-muted);">No prediction</em>';
        }}

        // Update match indicator
        const matchEl = document.getElementById('match-indicator');
        if (!hasPredictions) {{
            matchEl.className = 'match-indicator pending'; matchEl.textContent = '—';
        }} else if (data.match === true) {{
            matchEl.className = 'match-indicator match'; matchEl.textContent = '✓ Match';
        }} else if (data.match === false) {{
            matchEl.className = 'match-indicator mismatch'; matchEl.textContent = '✗ Mismatch';
        }} else {{
            matchEl.className = 'match-indicator pending'; matchEl.textContent = '—';
        }}

        // Update click overlays
        updateClickOverlays();

        // Update step list active state
        document.querySelectorAll('.step-item').forEach((el, i) => {{
            el.classList.toggle('active', i === currentIndex);
        }});

        // Update details panel
        updateDetailsPanel(data);

        // Update progress bar
        updateProgressBar();
    }}

    function updateDetailsPanel(data) {{
        const detailsEl = document.getElementById('details-content');
        const action = data.human_action;

        // Build human action section
        let html = `
            <div style="font-weight:600;font-size:0.8rem;color:var(--accent);margin-bottom:8px;text-transform:uppercase;">Human Action</div>
            <div class="detail-row"><span class="detail-key">Step</span><span class="detail-value">${{currentIndex + 1}} of ${{baseData.length}}</span></div>
            <div class="detail-row"><span class="detail-key">Time</span><span class="detail-value">${{data.time ? data.time.toFixed(2) + 's' : '—'}}</span></div>
            <div class="detail-row"><span class="detail-key">Type</span><span class="detail-value">${{action.type}}</span></div>
        `;
        if (action.x !== null && action.x !== undefined) {{
            html += `<div class="detail-row"><span class="detail-key">Position</span><span class="detail-value">(${{(action.x * 100).toFixed(2)}}%, ${{(action.y * 100).toFixed(2)}}%)</span></div>`;
        }}
        if (action.text) {{
            html += `<div class="detail-row"><span class="detail-key">Text</span><span class="detail-value">"${{action.text}}"</span></div>`;
        }}

        // Build prediction section if available
        if (data.predicted_action && currentCheckpoint !== 'None') {{
            const pred = data.predicted_action;
            html += `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border-color);">`;
            html += `<div style="font-weight:600;font-size:0.8rem;color:#a78bfa;margin-bottom:8px;text-transform:uppercase;display:flex;justify-content:space-between;">
                <span>Model Prediction</span>
                <span style="color:${{data.match === true ? '#34d399' : data.match === false ? '#ff5f5f' : 'var(--text-muted)'}};">${{data.match === true ? '✓ Match' : data.match === false ? '✗ Mismatch' : '—'}}</span>
            </div>`;

            // Show predicted position if available
            if (pred.x !== undefined && pred.y !== undefined) {{
                html += `<div class="detail-row"><span class="detail-key">Type</span><span class="detail-value">${{pred.type || 'click'}}</span></div>`;
                html += `<div class="detail-row"><span class="detail-key">Position</span><span class="detail-value">(${{(pred.x * 100).toFixed(2)}}%, ${{(pred.y * 100).toFixed(2)}}%)</span></div>`;
            }}

            // Show raw output (model reasoning)
            if (pred.raw_output) {{
                const rawOutput = pred.raw_output;
                html += `<div class="detail-row" style="flex-direction:column;margin-top:8px;">
                    <span class="detail-key" style="margin-bottom:4px;">Model Output</span>
                    <div class="detail-value" style="font-size:0.75rem;max-height:150px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;background:var(--bg-tertiary);padding:8px;border-radius:4px;">${{rawOutput.replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
                </div>`;
            }} else {{
                // Show whatever fields are present
                const predStr = JSON.stringify(pred, null, 2);
                html += `<div class="detail-row" style="flex-direction:column;margin-top:8px;">
                    <span class="detail-key" style="margin-bottom:4px;">Prediction Data</span>
                    <div class="detail-value" style="font-size:0.75rem;max-height:100px;overflow-y:auto;white-space:pre;background:var(--bg-tertiary);padding:8px;border-radius:4px;">${{predStr}}</div>
                </div>`;
            }}
            html += `</div>`;
        }}

        // Add evaluation history if available
        const stepEvals = evalByStep[currentIndex];
        if (stepEvals && stepEvals.length > 0) {{
            html += `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border-color);">`;
            html += `<div style="font-weight:600;font-size:0.8rem;color:var(--text-secondary);margin-bottom:8px;text-transform:uppercase;">Evaluation History</div>`;

            // Sort by epoch
            const sorted = [...stepEvals].sort((a, b) => a.epoch - b.epoch);
            sorted.forEach(ev => {{
                const icon = ev.correct ? '✓' : '✗';
                const color = ev.correct ? '#34d399' : '#ff5f5f';
                const dist = ev.distance ? ev.distance.toFixed(2) + 'px' : '—';
                html += `<div class="detail-row">
                    <span class="detail-key">Epoch ${{ev.epoch}}</span>
                    <span class="detail-value" style="color:${{color}};">${{icon}} ${{dist}}</span>
                </div>`;
            }});

            // Show improvement trend if multiple epochs
            if (sorted.length > 1) {{
                const first = sorted[0];
                const last = sorted[sorted.length - 1];
                if (first.distance && last.distance) {{
                    const improvement = first.distance - last.distance;
                    const pct = ((improvement / first.distance) * 100).toFixed(1);
                    const improved = improvement > 0;
                    html += `<div class="detail-row" style="margin-top:4px;padding-top:4px;border-top:1px dashed var(--border-color);">
                        <span class="detail-key">Trend</span>
                        <span class="detail-value" style="color:${{improved ? '#34d399' : '#ff5f5f'}};">
                            ${{improved ? '↓' : '↑'}} ${{Math.abs(improvement).toFixed(1)}}px (${{improved ? '-' : '+'}}${{Math.abs(pct)}}%)
                        </span>
                    </div>`;
                }}
            }}

            html += `</div>`;
        }}

        detailsEl.innerHTML = html;
    }}

    function setupCopyButton() {{
        document.getElementById('copy-btn').onclick = function() {{
            const data = getMergedData()[currentIndex];
            const text = JSON.stringify(data, null, 2);
            navigator.clipboard.writeText(text);
            this.textContent = 'Copied!';
            this.classList.add('copied');
            setTimeout(() => {{
                this.textContent = 'Copy';
                this.classList.remove('copied');
            }}, 1500);
        }};
    }}

    function setupCopyAllButton() {{
        const btn = document.getElementById('copy-all-btn');
        if (!btn) return;

        btn.onclick = function() {{
            const allData = getMergedData();
            const text = JSON.stringify(allData, null, 2);
            navigator.clipboard.writeText(text);
            this.textContent = 'Copied!';
            this.classList.add('copied');
            setTimeout(() => {{
                this.textContent = 'Copy All';
                this.classList.remove('copied');
            }}, 1500);
        }};
    }}

    function updateClickOverlays() {{
        document.querySelectorAll('.click-marker').forEach(el => el.remove());
        const data = getMergedData()[currentIndex];
        if (!data) return;
        const container = document.getElementById('frame-container');

        if (showHumanOverlay && data.human_action.x !== null && data.human_action.x !== undefined) {{
            const marker = document.createElement('div');
            marker.className = 'click-marker human';
            marker.style.left = (data.human_action.x * 100) + '%';
            marker.style.top = (data.human_action.y * 100) + '%';
            container.appendChild(marker);
        }}
        if (showPredictedOverlay && data.predicted_action && data.predicted_action.x !== undefined) {{
            const marker = document.createElement('div');
            marker.className = 'click-marker predicted';
            marker.style.left = (data.predicted_action.x * 100) + '%';
            marker.style.top = (data.predicted_action.y * 100) + '%';
            container.appendChild(marker);
        }}
    }}

    function buildStepList() {{
        const listEl = document.getElementById('step-list-items');
        if (!listEl) return;
        listEl.innerHTML = '';
        const typeColors = {{
            click: '#ff5f5f',
            double_click: '#ff5f5f',
            type: '#34d399',
            scroll: '#a78bfa',
            drag: '#00d4aa',
            done: '#888',
        }};
        baseData.forEach((step, i) => {{
            const item = document.createElement('div');
            item.className = 'step-item' + (i === currentIndex ? ' active' : '');
            const action = step.human_action;
            const time = step.time ? step.time.toFixed(1) + 's' : '';
            const typeColor = typeColors[action.type] || 'var(--text-secondary)';
            const actionDetail = action.type === 'type' && action.text
                ? `"${{action.text.length > 15 ? action.text.slice(0,15) + '...' : action.text}}"`
                : (action.x !== null && action.x !== undefined ? `(${{(action.x*100).toFixed(0)}}%, ${{(action.y*100).toFixed(0)}}%)` : '');

            // Build evaluation badges
            let evalBadgesHtml = '';
            const stepEvals = evalByStep[i];
            if (stepEvals && stepEvals.length > 0) {{
                const badges = stepEvals.map(ev => {{
                    const cls = ev.correct ? 'correct' : 'incorrect';
                    const icon = ev.correct ? '✓' : '✗';
                    const dist = ev.distance ? ev.distance.toFixed(1) + 'px' : '';
                    return `<span class="eval-badge ${{cls}}" title="Epoch ${{ev.epoch}}: ${{dist}}"><span class="epoch">E${{ev.epoch}}</span>${{icon}}</span>`;
                }}).join('');
                evalBadgesHtml = `<div class="eval-badges">${{badges}}</div>`;
            }}

            item.innerHTML = `
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-family:monospace;font-size:0.7rem;color:var(--text-muted);min-width:40px;">${{time}}</span>
                    <span style="font-weight:600;color:${{typeColor}};text-transform:uppercase;font-size:0.75rem;">${{action.type}}</span>
                </div>
                <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:2px;font-family:monospace;">${{actionDetail}}</div>
                ${{evalBadgesHtml}}
            `;
            item.onclick = () => {{ currentIndex = i; updateDisplay(); }};
            listEl.appendChild(item);
        }});
    }}

    function setupOverlayToggles() {{
        const container = document.getElementById('overlay-toggles');
        container.innerHTML = `<button class="toggle-btn active" id="toggle-human" title="Toggle human overlay (H)">Human</button><button class="toggle-btn active" id="toggle-predicted" title="Toggle AI overlay (A)">AI</button>`;
        document.getElementById('toggle-human').onclick = function() {{
            showHumanOverlay = !showHumanOverlay;
            this.classList.toggle('active', showHumanOverlay);
            // Also dim the human action box
            const humanBox = document.querySelector('.action-box.human');
            if (humanBox) humanBox.style.opacity = showHumanOverlay ? '1' : '0.4';
            updateClickOverlays();
        }};
        document.getElementById('toggle-predicted').onclick = function() {{
            showPredictedOverlay = !showPredictedOverlay;
            this.classList.toggle('active', showPredictedOverlay);
            // Also dim the predicted action box
            const predictedBox = document.getElementById('predicted-box');
            if (predictedBox) predictedBox.style.opacity = showPredictedOverlay ? '1' : '0.4';
            updateClickOverlays();
        }};
    }}

    function updateProgressBar() {{
        const progress = document.getElementById('progress');
        if (progress) {{
            const pct = (currentIndex / (baseData.length - 1)) * 100;
            progress.style.width = pct + '%';
        }}
    }}

    function stopPlayback() {{
        isPlaying = false;
        if (playInterval) {{
            clearInterval(playInterval);
            playInterval = null;
        }}
        const playBtn = document.getElementById('play-btn');
        if (playBtn) {{
            playBtn.textContent = '▶ Play';
            playBtn.classList.remove('active');
        }}
        // Pause audio if playing
        if (audioElement && !audioElement.paused) {{
            audioElement.pause();
        }}
    }}

    function startPlayback() {{
        isPlaying = true;
        const playBtn = document.getElementById('play-btn');
        if (playBtn) {{
            playBtn.textContent = '⏸ Pause';
            playBtn.classList.add('active');
        }}
        // Start audio if available
        if (audioElement && audioElement.src) {{
            audioElement.play().catch(e => console.log('Audio play failed:', e));
        }}
        playInterval = setInterval(() => {{
            if (currentIndex < baseData.length - 1) {{
                currentIndex++;
                updateDisplay();
            }} else {{
                stopPlayback();
            }}
        }}, playSpeed);
    }}

    function togglePlayback() {{
        if (isPlaying) {{
            stopPlayback();
        }} else {{
            startPlayback();
        }}
    }}

    function setupPlaybackControls() {{
        // Rewind
        document.getElementById('rewind-btn').onclick = () => {{
            stopPlayback();
            currentIndex = 0;
            updateDisplay();
        }};

        // Previous
        document.getElementById('prev-btn').onclick = () => {{
            stopPlayback();
            if (currentIndex > 0) {{ currentIndex--; updateDisplay(); }}
        }};

        // Play/Pause
        document.getElementById('play-btn').onclick = togglePlayback;

        // Next
        document.getElementById('next-btn').onclick = () => {{
            stopPlayback();
            if (currentIndex < baseData.length - 1) {{ currentIndex++; updateDisplay(); }}
        }};

        // End
        document.getElementById('end-btn').onclick = () => {{
            stopPlayback();
            currentIndex = baseData.length - 1;
            updateDisplay();
        }};

        // Speed control
        document.getElementById('speed-select').onchange = (e) => {{
            playSpeed = parseInt(e.target.value);
            // Map step interval to audio playback rate: 2000ms=0.5x, 1000ms=1x, 500ms=2x, 250ms=4x
            const playbackRate = 1000 / playSpeed;
            if (audioElement) {{
                audioElement.playbackRate = playbackRate;
            }}
            if (isPlaying) {{
                stopPlayback();
                startPlayback();
            }}
        }};

        // Progress bar click to seek
        document.getElementById('progress-bar').onclick = (e) => {{
            const rect = e.currentTarget.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            currentIndex = Math.round(pct * (baseData.length - 1));
            updateDisplay();
        }};

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            // Ignore if focused on an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            switch(e.key) {{
                case 'ArrowLeft':
                    document.getElementById('prev-btn').click();
                    break;
                case 'ArrowRight':
                    document.getElementById('next-btn').click();
                    break;
                case ' ':  // Space
                    e.preventDefault();
                    togglePlayback();
                    break;
                case 'Home':
                    document.getElementById('rewind-btn').click();
                    break;
                case 'End':
                    document.getElementById('end-btn').click();
                    break;
                case 'h':
                case 'H':
                    document.getElementById('toggle-human').click();
                    break;
                case 'a':
                case 'A':
                    document.getElementById('toggle-predicted').click();
                    break;
            }}
        }});
    }}

    // Transcript/audio sync variables
    let transcriptSegments = [];
    let audioElement = null;
    let lastActiveSegmentIndex = -1;
    let autoScrollTranscript = true;

    async function loadTranscript() {{
        // Try to load transcript.json
        try {{
            const response = await fetch('transcript.json?t=' + Date.now());
            if (response.ok) {{
                const data = await response.json();
                if (data.segments && data.segments.length > 0) {{
                    transcriptSegments = data.segments;
                    renderTranscript();
                    setupAudioSync();
                    return;
                }}
            }}
        }} catch (e) {{
            console.log('No transcript.json found');
        }}

        // Check if any base data has transcript info
        const hasTranscript = baseData.some(d => d.transcript_text || d.audio_start !== undefined);
        if (!hasTranscript) {{
            document.getElementById('transcript-content').innerHTML = '<div class="transcript-empty">No transcript available</div>';
            return;
        }}

        // Build segments from base data
        baseData.forEach((step, i) => {{
            if (step.transcript_text) {{
                transcriptSegments.push({{
                    start: step.audio_start || step.time || 0,
                    end: step.audio_end || (baseData[i + 1]?.time || step.time + 5),
                    text: step.transcript_text,
                    stepIndex: i
                }});
            }}
        }});

        if (transcriptSegments.length > 0) {{
            renderTranscript();
            setupAudioSync();
        }} else {{
            document.getElementById('transcript-content').innerHTML = '<div class="transcript-empty">No transcript available</div>';
        }}
    }}

    function renderTranscript() {{
        const container = document.getElementById('transcript-content');
        if (transcriptSegments.length === 0) {{
            container.innerHTML = '<div class="transcript-empty">No transcript available</div>';
            return;
        }}

        container.innerHTML = transcriptSegments.map((seg, i) => {{
            const timeStr = formatTime(seg.start);
            return `<span class="transcript-segment" data-index="${{i}}" data-start="${{seg.start}}" data-end="${{seg.end}}">` +
                   `<span class="transcript-time">${{timeStr}}</span>${{seg.text}} </span>`;
        }}).join('');

        // Add click handlers for seek
        container.querySelectorAll('.transcript-segment').forEach(el => {{
            el.onclick = () => {{
                const start = parseFloat(el.dataset.start);
                seekAudio(start);

                // Also jump to corresponding step if available
                const segIndex = parseInt(el.dataset.index);
                if (transcriptSegments[segIndex]?.stepIndex !== undefined) {{
                    currentIndex = transcriptSegments[segIndex].stepIndex;
                    updateDisplay();
                }}
            }};
        }});
    }}

    function formatTime(seconds) {{
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
    }}

    function seekAudio(time) {{
        if (!audioElement) {{
            audioElement = document.getElementById('audio');
        }}
        if (audioElement && audioElement.src) {{
            audioElement.currentTime = time;
            if (audioElement.paused) {{
                audioElement.play().catch(e => console.log('Audio play failed:', e));
            }}
        }}
    }}

    function setupAudioSync() {{
        audioElement = document.getElementById('audio');

        // Try to load audio file
        const audioSrc = 'audio.mp3';
        audioElement.src = audioSrc;
        audioElement.load();

        // Auto-highlight during playback
        audioElement.ontimeupdate = () => {{
            const currentTime = audioElement.currentTime;
            highlightCurrentSegment(currentTime);
        }};

        audioElement.onerror = () => {{
            console.log('Audio file not available');
        }};

        // Setup follow toggle button
        const followBtn = document.getElementById('transcript-follow-btn');
        if (followBtn) {{
            followBtn.onclick = () => {{
                autoScrollTranscript = !autoScrollTranscript;
                followBtn.classList.toggle('active', autoScrollTranscript);
            }};
        }}
    }}

    function highlightCurrentSegment(currentTime) {{
        const segments = document.querySelectorAll('.transcript-segment');
        let newActiveIndex = -1;

        segments.forEach((el, i) => {{
            const start = parseFloat(el.dataset.start);
            const end = parseFloat(el.dataset.end);
            const isActive = currentTime >= start && currentTime < end;
            el.classList.toggle('active', isActive);

            if (isActive) {{
                newActiveIndex = i;
            }}
        }});

        // Only scroll when active segment changes (not on every timeupdate)
        if (newActiveIndex !== lastActiveSegmentIndex && newActiveIndex !== -1) {{
            lastActiveSegmentIndex = newActiveIndex;
            if (autoScrollTranscript) {{
                segments[newActiveIndex].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
            }}
        }}
    }}

    // Timeline visualizer
    let totalDuration = 0;

    function renderTimeline() {{
        const timeline = document.getElementById('timeline');
        const segmentsContainer = document.getElementById('timeline-segments');
        const markersContainer = document.getElementById('timeline-markers');

        if (!timeline || !segmentsContainer || !markersContainer) return;

        // Calculate total duration from audio or last step
        totalDuration = audioElement?.duration || baseData[baseData.length - 1]?.time || 60;

        // Clear existing
        segmentsContainer.innerHTML = '';
        markersContainer.innerHTML = '';

        // Render transcript segments
        if (transcriptSegments.length > 0) {{
            transcriptSegments.forEach((seg, i) => {{
                const left = (seg.start / totalDuration) * 100;
                const width = Math.max(1, ((seg.end - seg.start) / totalDuration) * 100);
                const div = document.createElement('div');
                div.className = 'timeline-segment';
                div.style.width = width + '%';
                div.dataset.index = i;
                div.dataset.start = seg.start;
                div.dataset.end = seg.end;

                // Tooltip with truncated text
                const tooltip = document.createElement('div');
                tooltip.className = 'timeline-segment-tooltip';
                tooltip.textContent = seg.text.length > 40 ? seg.text.slice(0, 40) + '...' : seg.text;
                div.appendChild(tooltip);

                div.onclick = (e) => {{
                    e.stopPropagation();
                    seekAudio(seg.start);
                }};
                segmentsContainer.appendChild(div);
            }});
        }} else {{
            // No segments - fill with empty space
            segmentsContainer.innerHTML = '<div class="timeline-segment" style="width:100%;background:transparent;"></div>';
        }}

        // Render action markers
        baseData.forEach((step, i) => {{
            const left = (step.time / totalDuration) * 100;
            const marker = document.createElement('div');
            marker.className = `timeline-marker ${{step.human_action?.type || 'unknown'}}`;
            marker.style.left = left + '%';
            marker.dataset.index = i;
            marker.title = `Step ${{i + 1}}: ${{step.human_action?.type || 'unknown'}} @ ${{formatTime(step.time)}}`;
            marker.onclick = (e) => {{
                e.stopPropagation();
                currentIndex = i;
                updateDisplay();
                if (step.time && audioElement) {{
                    seekAudio(step.time);
                }}
            }};
            markersContainer.appendChild(marker);
        }});

        // Timeline click to seek
        timeline.onclick = (e) => {{
            const rect = timeline.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            const time = pct * totalDuration;
            seekAudio(time);

            // Find closest step
            let closest = 0;
            let minDist = Infinity;
            baseData.forEach((step, i) => {{
                const dist = Math.abs(step.time - time);
                if (dist < minDist) {{
                    minDist = dist;
                    closest = i;
                }}
            }});
            currentIndex = closest;
            updateDisplay();
        }};
    }}

    function updateTimelinePlayhead() {{
        const playhead = document.getElementById('timeline-playhead');
        if (!playhead || !totalDuration) return;

        const currentTime = audioElement?.currentTime || baseData[currentIndex]?.time || 0;
        const pct = (currentTime / totalDuration) * 100;
        playhead.style.left = pct + '%';

        // Update active segment
        document.querySelectorAll('.timeline-segment').forEach(el => {{
            const start = parseFloat(el.dataset.start) || 0;
            const end = parseFloat(el.dataset.end) || 0;
            el.classList.toggle('active', currentTime >= start && currentTime < end);
        }});

        // Update active marker
        document.querySelectorAll('.timeline-marker').forEach((el, i) => {{
            el.classList.toggle('active', i === currentIndex);
        }});
    }}

    // Hook into audio timeupdate for playhead
    const originalHighlight = highlightCurrentSegment;
    highlightCurrentSegment = function(currentTime) {{
        originalHighlight(currentTime);
        updateTimelinePlayhead();
    }};

    // Initialize
    initDropdowns();
    buildStepList();
    setupOverlayToggles();
    setupPlaybackControls();
    setupCopyButton();
    setupCopyAllButton();
    updateMetrics();
    updateDisplay();
    loadAndDisplayCosts();
    loadTranscript();  // Load transcript and setup audio sync

    // Render timeline after transcript loads (needs segment data)
    setTimeout(() => {{
        renderTimeline();
        updateTimelinePlayhead();
    }}, 500);

    // Gallery Panel Functions
    let currentGalleryEpoch = 'all';
    let currentGalleryStatus = 'all';

    function setupGalleryPanel() {{
        const panel = document.getElementById('gallery-panel');
        const collapseBtn = document.getElementById('gallery-collapse-btn');
        const maximizeBtn = document.getElementById('gallery-maximize-btn');
        const overlay = document.getElementById('gallery-maximized-overlay');
        const closeBtn = document.getElementById('gallery-close-btn');

        // Collapse/expand
        collapseBtn.onclick = function(e) {{
            e.stopPropagation();
            panel.classList.toggle('collapsed');
        }};

        // Maximize
        maximizeBtn.onclick = function(e) {{
            e.stopPropagation();
            overlay.classList.add('active');
            renderGalleryMaximized();
        }};

        // Close maximized
        closeBtn.onclick = function() {{
            overlay.classList.remove('active');
        }};

        // Close on escape
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape' && overlay.classList.contains('active')) {{
                overlay.classList.remove('active');
            }}
        }});

        // Close on overlay background click
        overlay.onclick = function(e) {{
            if (e.target === overlay) {{
                overlay.classList.remove('active');
            }}
        }};

        // Setup compact filters
        document.getElementById('gallery-epoch-filter').onchange = function() {{
            currentGalleryEpoch = this.value;
            filterGallery('compact');
        }};

        document.getElementById('gallery-status-filter').onchange = function() {{
            currentGalleryStatus = this.value;
            filterGallery('compact');
        }};

        // Setup maximized filters
        document.getElementById('gallery-epoch-filter-max').onchange = function() {{
            currentGalleryEpoch = this.value;
            document.getElementById('gallery-epoch-filter').value = this.value;
            filterGallery('maximized');
        }};

        document.getElementById('gallery-status-filter-max').onchange = function() {{
            currentGalleryStatus = this.value;
            document.getElementById('gallery-status-filter').value = this.value;
            filterGallery('maximized');
        }};

        // Initial render
        renderGalleryCompact();
    }}

    function buildGalleryCards(evaluations, compact = true) {{
        return evaluations.map((ev, i) => {{
            const statusClass = ev.correct ? 'correct' : 'incorrect';
            const statusText = ev.correct ? '✓' : '✗';
            const humanX = ((ev.human_action?.x || 0) * 100).toFixed(1);
            const humanY = ((ev.human_action?.y || 0) * 100).toFixed(1);
            const predX = ((ev.predicted_action?.x || 0) * 100).toFixed(1);
            const predY = ((ev.predicted_action?.y || 0) * 100).toFixed(1);
            const distance = ev.distance ? ev.distance.toFixed(3) : '—';

            const stepData = baseData[ev.sample_idx];
            const imagePath = stepData?.screenshot_path || ev.image_path || 'screenshots/sample.png';

            if (compact) {{
                return `
                    <div class="gallery-card" data-epoch="${{ev.epoch}}" data-correct="${{ev.correct}}" data-step="${{ev.sample_idx}}">
                        <div class="image-wrapper">
                            <img src="${{imagePath}}" alt="Step ${{ev.sample_idx + 1}}" onerror="this.src='screenshots/sample.png'">
                            <div class="overlay">
                                <div class="marker human" style="left: ${{humanX}}%; top: ${{humanY}}%;">H</div>
                                <div class="marker predicted" style="left: ${{predX}}%; top: ${{predY}}%;">AI</div>
                            </div>
                        </div>
                        <div class="card-content">
                            <div class="card-header">
                                <span class="step-num">S${{ev.sample_idx + 1}} E${{ev.epoch + 1}}</span>
                                <span class="status ${{statusClass}}">${{statusText}}</span>
                            </div>
                        </div>
                    </div>
                `;
            }} else {{
                return `
                    <div class="gallery-card" data-epoch="${{ev.epoch}}" data-correct="${{ev.correct}}" data-step="${{ev.sample_idx}}">
                        <div class="image-wrapper">
                            <img src="${{imagePath}}" alt="Step ${{ev.sample_idx + 1}}" onerror="this.src='screenshots/sample.png'">
                            <div class="overlay">
                                <div class="marker human" style="left: ${{humanX}}%; top: ${{humanY}}%;">H</div>
                                <div class="marker predicted" style="left: ${{predX}}%; top: ${{predY}}%;">AI</div>
                            </div>
                        </div>
                        <div class="card-content">
                            <div class="card-header">
                                <span class="step-num">Step ${{ev.sample_idx + 1}} | Epoch ${{ev.epoch + 1}}</span>
                                <span class="status ${{statusClass}}">${{ev.correct ? '✓ Correct' : '✗ Incorrect'}}</span>
                            </div>
                            <div class="card-details">
                                <div class="coord-row">
                                    <span class="coord-human">H: (${{humanX}}%, ${{humanY}}%)</span>
                                    <span class="coord-pred">AI: (${{predX}}%, ${{predY}}%)</span>
                                </div>
                                <div>Distance: ${{distance}}</div>
                            </div>
                        </div>
                    </div>
                `;
            }}
        }}).join('');
    }}

    function renderGalleryCompact() {{
        const grid = document.getElementById('gallery-grid');
        const emptyState = document.getElementById('gallery-empty');
        const epochFilter = document.getElementById('gallery-epoch-filter');
        const countEl = document.getElementById('gallery-count');

        if (evaluations.length === 0) {{
            grid.style.display = 'none';
            emptyState.style.display = 'block';
            countEl.textContent = '0 samples';
            return;
        }}

        grid.style.display = 'grid';
        emptyState.style.display = 'none';

        // Populate epoch filter options
        const epochs = [...new Set(evaluations.map(e => e.epoch))].sort((a, b) => a - b);
        epochFilter.innerHTML = '<option value="all">All Epochs</option>' +
            epochs.map(ep => `<option value="${{ep}}">Epoch ${{ep + 1}}</option>`).join('');

        grid.innerHTML = buildGalleryCards(evaluations, true);
        setupCardClickHandlers(grid, false);
        filterGallery('compact');
    }}

    function renderGalleryMaximized() {{
        const grid = document.getElementById('gallery-grid-max');
        const epochFilter = document.getElementById('gallery-epoch-filter-max');
        const countEl = document.getElementById('gallery-count-max');

        if (evaluations.length === 0) {{
            grid.innerHTML = '<div class="gallery-empty">No evaluations available.</div>';
            countEl.textContent = '0 samples';
            return;
        }}

        // Populate epoch filter options
        const epochs = [...new Set(evaluations.map(e => e.epoch))].sort((a, b) => a - b);
        epochFilter.innerHTML = '<option value="all">All Epochs</option>' +
            epochs.map(ep => `<option value="${{ep}}">Epoch ${{ep + 1}}</option>`).join('');

        // Sync filter values
        epochFilter.value = currentGalleryEpoch;
        document.getElementById('gallery-status-filter-max').value = currentGalleryStatus;

        grid.innerHTML = buildGalleryCards(evaluations, false);
        setupCardClickHandlers(grid, true);
        filterGallery('maximized');
    }}

    function setupCardClickHandlers(grid, closeOverlay) {{
        grid.querySelectorAll('.gallery-card').forEach(card => {{
            card.onclick = function() {{
                const stepIdx = parseInt(this.dataset.step);
                currentIndex = stepIdx;
                updateDisplay();
                if (closeOverlay) {{
                    document.getElementById('gallery-maximized-overlay').classList.remove('active');
                }}
            }};
        }});
    }}

    function filterGallery(mode) {{
        const isCompact = mode === 'compact';
        const grid = document.getElementById(isCompact ? 'gallery-grid' : 'gallery-grid-max');
        const countEl = document.getElementById(isCompact ? 'gallery-count' : 'gallery-count-max');
        const cards = grid.querySelectorAll('.gallery-card');
        let visible = 0;

        cards.forEach(card => {{
            const epoch = card.dataset.epoch;
            const correct = card.dataset.correct === 'true';

            const epochMatch = currentGalleryEpoch === 'all' || epoch === currentGalleryEpoch;
            const statusMatch = currentGalleryStatus === 'all' ||
                (currentGalleryStatus === 'correct' && correct) ||
                (currentGalleryStatus === 'incorrect' && !correct);

            if (epochMatch && statusMatch) {{
                card.classList.remove('hidden');
                visible++;
            }} else {{
                card.classList.add('hidden');
            }}
        }});

        countEl.textContent = `${{visible}} of ${{cards.length}} samples`;
    }}

    // Initialize gallery panel
    setupGalleryPanel();
    </script>
</body>
</html>'''

    output_path.write_text(html, encoding='utf-8')
    print(f"Generated unified viewer: {output_path}")


def _enhance_comparison_to_unified_viewer(
    base_html_file: Path,
    predictions_by_checkpoint: dict[str, list[dict]],
    output_path: Path,
    capture_id: str = "unknown",
    goal: str = "Untitled",
) -> None:
    """Enhance an existing comparison HTML file into a unified viewer.

    DEPRECATED: This function uses script injection which is fragile.
    Use _generate_unified_viewer_from_extracted_data() instead for a
    standalone viewer that doesn't depend on the comparison.html structure.

    Takes the nice openadapt-capture viewer and adds:
    - Simplified nav (Training + Viewer only)
    - Checkpoint dropdown to switch between predictions
    - Training example dropdown (stub for future)
    """
    import re

    html = base_html_file.read_text()

    # Extract base data from the existing comparisonData
    data_match = re.search(
        r'const\s+comparisonData\s*=\s*(\[.*?\]);',
        html,
        re.DOTALL
    )
    if not data_match:
        print(f"Could not find comparisonData in {base_html_file}")
        return

    base_comparison_data = json.loads(data_match.group(1))

    # Build base data (human actions only) and ensure predictions dict has base data
    base_data = []
    for item in base_comparison_data:
        base_data.append({
            "index": item.get("index", 0),
            "time": item.get("time", 0),
            "image_path": item.get("image_path", ""),
            "human_action": item.get("human_action", {}),
        })

    # JSON encode predictions
    predictions_json = json.dumps(predictions_by_checkpoint)
    captures_json = json.dumps([{
        "id": capture_id,
        "name": goal,
        "steps": len(base_data),
    }])

    # 1. Replace nav bar with unified header combining nav + controls
    # Use shared header CSS and HTML for consistency with training dashboard
    header_css = f'<style>{_get_shared_header_css()}</style>'

    # Build the controls HTML for the viewer (example + checkpoint dropdowns)
    controls_html = f'''
            <div class="control-group">
                <span class="control-label">Example</span>
                <select id="capture-select">
                    <option value="{capture_id}">{goal[:40]}{'...' if len(goal) > 40 else ''} ({len(base_data)})</option>
                </select>
            </div>
            <div class="control-group">
                <span class="control-label">Checkpoint</span>
                <select id="checkpoint-select"></select>
            </div>
    '''

    unified_header = header_css + _generate_shared_header_html(
        "viewer",
        controls_html=controls_html,
        meta_html=f"ID: {capture_id}"
    )

    # Remove any old viewer-controls div if it exists (from previous runs)
    html = re.sub(
        r'<div class="viewer-controls"[^>]*>.*?</div>\s*(?=<)',
        '',
        html,
        flags=re.DOTALL
    )

    # Try to replace existing nav with unified header
    nav_replaced = False
    if re.search(r'<nav class="nav-bar"', html):
        html = re.sub(
            r'<nav class="nav-bar"[^>]*>.*?</nav>\s*',
            unified_header,
            html,
            flags=re.DOTALL
        )
        nav_replaced = True

    # Remove the old <header> element - unified header already contains all info
    html = re.sub(
        r'<header[^>]*>.*?</header>\s*',
        '',
        html,
        flags=re.DOTALL
    )

    # If no nav was found/replaced, insert unified header after <body>
    if not nav_replaced:
        html = re.sub(
            r'(<body[^>]*>)',
            r'\1\n' + unified_header,
            html,
            count=1
        )

    # 3. Replace the comparisonData with multi-checkpoint system
    # We need to modify the JavaScript to use our checkpoint system

    checkpoint_script = f'''
    <script>
    // Unified viewer: multi-checkpoint support
    // Bridge local comparisonData to window scope for cross-script access
    if (typeof comparisonData !== 'undefined' && typeof window.comparisonData === 'undefined') {{
        window.comparisonData = comparisonData;
    }}

    // Parse model output for SoM actions
    window.parseModelOutput = function(rawOutput) {{
        if (!rawOutput) return {{ html: '<em style="color:var(--text-muted);">No prediction</em>' }};

        // Try to extract SoM actions: CLICK([N]), TYPE([N], "text"), TYPE("text")
        const clickSomMatch = rawOutput.match(/CLICK\\s*\\(\\s*\\[\\s*(\\d+)\\s*\\]\\s*\\)/);
        const typeSomMatch = rawOutput.match(/TYPE\\s*\\(\\s*\\[\\s*(\\d+)\\s*\\]\\s*,\\s*["']([^"']*)["']\\s*\\)/);
        const typeSimpleMatch = rawOutput.match(/TYPE\\s*\\(\\s*["']([^"']*)["']\\s*\\)/);
        const clickCoordMatch = rawOutput.match(/CLICK\\s*\\(\\s*x\\s*=\\s*([\\d.]+)\\s*,\\s*y\\s*=\\s*([\\d.]+)\\s*\\)/);

        let html = '';

        if (clickSomMatch) {{
            html = `<div style="font-weight:600;color:#00d4aa;">CLICK([${{clickSomMatch[1]}}])</div>`;
        }} else if (typeSomMatch) {{
            html = `<div style="font-weight:600;color:#00d4aa;">TYPE([${{typeSomMatch[1]}}], "${{typeSomMatch[2]}}")</div>`;
        }} else if (typeSimpleMatch) {{
            html = `<div style="font-weight:600;color:#00d4aa;">TYPE("${{typeSimpleMatch[1]}}")</div>`;
        }} else if (clickCoordMatch) {{
            html = `<div style="font-weight:600;color:#00d4aa;">CLICK(x=${{clickCoordMatch[1]}}, y=${{clickCoordMatch[2]}})</div>`;
        }} else {{
            // No structured action - show truncated output
            const truncated = rawOutput.replace(/\\n/g, ' ').substring(0, 150);
            html = `<div style="font-size:0.85rem;color:var(--text-muted);max-height:60px;overflow:hidden;">${{truncated}}${{rawOutput.length > 150 ? '...' : ''}}</div>`;
        }}

        return {{ html }};
    }};

    // Override prediction display in comparison viewer
    window.formatPrediction = function(pred) {{
        if (!pred) return '<em style="color:var(--text-muted);">No prediction</em>';
        if (pred.x !== undefined) {{
            return `<div>Type: ${{pred.type || 'click'}}</div><div>Position: (${{(pred.x * 100).toFixed(1)}}%, ${{(pred.y * 100).toFixed(1)}}%)</div>`;
        }}
        return window.parseModelOutput(pred.raw_output || JSON.stringify(pred)).html;
    }};

    // Use window. prefix for cross-script variable access
    window.predictionsByCheckpoint = {predictions_json};
    window.availableCaptures = {captures_json};
    window.currentCheckpoint = 'None';

    // Initialize checkpoint dropdown
    window.initCheckpointDropdown = function() {{
        const select = document.getElementById('checkpoint-select');
        if (!select) return;

        const checkpointNames = Object.keys(window.predictionsByCheckpoint);
        checkpointNames.sort((a, b) => {{
            if (a === 'None') return -1;
            if (b === 'None') return 1;
            const aNum = parseInt(a.match(/\\d+/)?.[0] || '999');
            const bNum = parseInt(b.match(/\\d+/)?.[0] || '999');
            return aNum - bNum;
        }});

        select.innerHTML = '';
        checkpointNames.forEach(name => {{
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name === 'None' ? 'None (Capture Only)' : name;
            select.appendChild(opt);
        }});

        // Default to latest epoch checkpoint (prefer actual trained checkpoints over Preview)
        const epochCheckpoints = checkpointNames.filter(n => n.toLowerCase().includes('epoch'));
        const latestCheckpoint = epochCheckpoints.length > 0
            ? epochCheckpoints.pop()
            : checkpointNames.filter(n => n !== 'None').pop();
        if (latestCheckpoint) {{
            select.value = latestCheckpoint;
            window.currentCheckpoint = latestCheckpoint;
            window.applyCheckpointPredictions(latestCheckpoint);
        }}

        select.addEventListener('change', (e) => {{
            window.currentCheckpoint = e.target.value;
            window.applyCheckpointPredictions(window.currentCheckpoint);
        }});
    }};

    // Apply predictions from selected checkpoint to comparisonData
    window.applyCheckpointPredictions = function(checkpointName) {{
        const predictions = window.predictionsByCheckpoint[checkpointName] || [];

        // Update comparisonData with new predictions (access from window)
        if (typeof window.comparisonData !== 'undefined') {{
            window.comparisonData.forEach((item, i) => {{
                const pred = predictions[i] || {{}};
                item.predicted_action = pred.predicted_action || null;
                item.match = pred.match !== undefined ? pred.match : null;
            }});
        }}

        // Refresh display if updateComparison exists (check both window and global scope)
        const idx = typeof window.currentIndex !== 'undefined' ? window.currentIndex :
                    (typeof currentIndex !== 'undefined' ? currentIndex : 0);
        if (typeof window.updateComparison === 'function') {{
            window.updateComparison(idx);
        }} else if (typeof updateComparison === 'function') {{
            updateComparison(idx);
        }}

        // Reformat prediction display after original updateComparison runs
        setTimeout(() => {{
            const predEl = document.getElementById('predicted-action') ||
                          document.querySelector('.action-box.predicted .action-details');
            if (predEl && window.comparisonData && window.comparisonData[idx]) {{
                const pred = window.comparisonData[idx].predicted_action;
                if (pred) {{
                    predEl.innerHTML = window.formatPrediction(pred);
                }}
            }}
        }}, 50);

        // Update metrics if setupMetricsSummary exists
        if (typeof window.setupMetricsSummary === 'function') {{
            window.setupMetricsSummary();
        }}
    }};

    // Initialize on load
    setTimeout(window.initCheckpointDropdown, 200);

    // Smart auto-scroll: scroll while playing, but stop if user scrolls up
    (function() {{
        let autoScrollEnabled = true;
        let lastScrollTop = 0;

        // Find the events list element
        const eventsList = document.querySelector('.events-list');
        if (!eventsList) return;

        // Detect user scroll - disable auto-scroll if scrolling up
        eventsList.addEventListener('scroll', function() {{
            const currentScrollTop = eventsList.scrollTop;

            // If user scrolled up, disable auto-scroll
            if (currentScrollTop < lastScrollTop - 10) {{
                autoScrollEnabled = false;
            }}

            // If user scrolled to bottom (within 50px), re-enable auto-scroll
            const isAtBottom = eventsList.scrollHeight - eventsList.scrollTop - eventsList.clientHeight < 50;
            if (isAtBottom) {{
                autoScrollEnabled = true;
            }}

            lastScrollTop = currentScrollTop;
        }});

        // Override scrollIntoView behavior for event items
        const originalScrollIntoView = Element.prototype.scrollIntoView;
        Element.prototype.scrollIntoView = function(options) {{
            // Only block scroll for event items when auto-scroll is disabled
            if (!autoScrollEnabled && this.classList && this.classList.contains('event-item')) {{
                return; // Skip scrollIntoView when user has scrolled up
            }}
            return originalScrollIntoView.call(this, options);
        }};

        // Add scroll lock indicator
        const indicator = document.createElement('div');
        indicator.id = 'scroll-lock-indicator';
        indicator.style.cssText = 'position:fixed;bottom:20px;right:20px;padding:8px 12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:4px;font-size:0.75rem;color:var(--text-muted);opacity:0;transition:opacity 0.3s;pointer-events:none;z-index:1000;';
        indicator.textContent = '⏸ Auto-scroll paused (scroll to bottom to resume)';
        document.body.appendChild(indicator);

        // Show/hide indicator based on scroll state
        setInterval(() => {{
            indicator.style.opacity = autoScrollEnabled ? '0' : '1';
        }}, 200);
    }})();
    </script>
    '''

    # Insert checkpoint script before </body>
    html = html.replace('</body>', checkpoint_script + '</body>')

    # 4. Disable the old discoverDashboards that creates wrong nav
    html = html.replace(
        'discoverDashboards();',
        '// discoverDashboards disabled - using unified viewer nav'
    )

    # Write output
    output_path.write_text(html, encoding='utf-8')
    print(f"Generated unified viewer from {base_html_file.name}: {output_path}")


def _add_static_nav_to_comparison(
    comparison_path: Path,
    output_dir: Path,
    nav_links: list[tuple[str, str]] | None = None,
) -> None:
    """Add or update static navigation in a comparison HTML file.

    Also moves the Action Comparison panel to main-content (above screenshot) if needed.

    Args:
        comparison_path: Path to the comparison HTML file
        output_dir: Directory containing all dashboard files
        nav_links: Pre-built list of (filename, label) tuples for consistency
    """
    import re

    html = comparison_path.read_text()

    # Move comparison panel to be a full-width sibling BEFORE main-content (not inside it)
    if '<div class="comparison-panel"' in html:
        # Check if panel is NOT already right before main-content
        if '<div class="comparison-panel"' in html and 'class="comparison-panel"' in html:
            # Check if it's in the wrong place (inside sidebar or main-content)
            in_sidebar = '<div class="sidebar">' in html and html.index('<div class="comparison-panel"') > html.index('<div class="sidebar">')
            in_main = '<div class="main-content">' in html and '<div class="main-content">\n' in html and '<div class="main-content">\n        <div class="comparison-panel"' in html

            if in_sidebar or in_main:
                # Extract comparison panel from wherever it is
                panel_match = re.search(
                    r'(\s*<div class="comparison-panel"[^>]*>.*?</div>\s*</div>\s*</div>)',
                    html,
                    re.DOTALL
                )
                if panel_match:
                    panel_html = panel_match.group(1)
                    # Remove from current location
                    html = html.replace(panel_html, '')
                    # Insert as sibling BEFORE main-content
                    html = html.replace(
                        '<div class="main-content">',
                        panel_html.strip() + '\n        <div class="main-content">'
                    )
                    print(f"  Moved Action Comparison above screenshot in {comparison_path.name}")

    # Build nav links if not provided
    if nav_links is None:
        nav_links = _build_nav_links()

    # Build nav HTML with active state for current file
    # NOTE: No "Dashboards:" label to match training dashboard nav
    current_file = comparison_path.name
    nav_html = '''
    <nav class="nav-bar" style="display:flex;gap:8px;padding:12px 16px;background:#12121a;border:1px solid rgba(255,255,255,0.06);border-radius:8px;margin-bottom:16px;flex-wrap:wrap;">
'''
    for filename, label in nav_links:
        is_active = filename == current_file
        active_style = "background:#00d4aa;color:#0a0a0f;border-color:#00d4aa;font-weight:600;" if is_active else ""
        nav_html += f'        <a href="{filename}" style="padding:8px 16px;border-radius:6px;font-size:0.8rem;text-decoration:none;color:#888;background:#1a1a24;border:1px solid rgba(255,255,255,0.06);{active_style}">{label}</a>\n'
    nav_html += '    </nav>\n'

    # ALWAYS replace existing nav or add new one (for consistency)
    if '<nav class="nav-bar"' in html:
        # Replace existing nav
        html = re.sub(
            r'<nav class="nav-bar"[^>]*>.*?</nav>\s*',
            nav_html,
            html,
            flags=re.DOTALL
        )
        print(f"  Updated navigation in {comparison_path.name}")
    elif '<div class="container">' in html:
        html = html.replace(
            '<div class="container">',
            '<div class="container">\n' + nav_html
        )
        print(f"  Added navigation to {comparison_path.name}")
    elif '<body>' in html:
        html = html.replace('<body>', '<body>\n' + nav_html)
        print(f"  Added navigation to {comparison_path.name}")

    comparison_path.write_text(html)


