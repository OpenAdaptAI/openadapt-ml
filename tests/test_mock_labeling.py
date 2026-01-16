#!/usr/bin/env python3
"""Test script to verify mock/real labeling in benchmark viewer.

This script:
1. Creates a temporary real benchmark result (modifies benchmark_name from mock to real)
2. Generates viewers for both mock and real runs
3. Opens both in browser for visual verification
"""

from pathlib import Path
import json
import shutil
import webbrowser

from openadapt_ml.training.benchmark_viewer import generate_benchmark_viewer

def main():
    benchmark_dir = Path("benchmark_results")

    # Use existing mock run
    mock_run = benchmark_dir / "waa-mock_eval_20251216_161049"

    # Create temporary real run by copying mock and modifying metadata
    real_run = benchmark_dir / "waa_eval_20251217_test_real"

    if real_run.exists():
        shutil.rmtree(real_run)

    shutil.copytree(mock_run, real_run)

    # Modify metadata to make it a "real" run
    metadata_path = real_run / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata["benchmark_name"] = "waa"  # Change from "waa-mock" to "waa"
    metadata["run_name"] = "waa_eval_20251217_test_real"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Also update summary.json
    summary_path = real_run / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    summary["benchmark_name"] = "waa"
    summary["run_name"] = "waa_eval_20251217_test_real"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate viewers
    print("Generating benchmark viewers...")
    mock_viewer = generate_benchmark_viewer(mock_run)
    real_viewer = generate_benchmark_viewer(real_run)

    print(f"\n✓ Generated mock viewer: {mock_viewer}")
    print(f"✓ Generated real viewer: {real_viewer}")

    print("\n" + "=" * 60)
    print("VERIFICATION CHECKLIST:")
    print("=" * 60)
    print("\nMock viewer should show:")
    print("  - Orange warning badge: '⚠️ MOCK DATA - Simulated results...'")
    print("  - Orange banner at top explaining mock data")
    print("  - No success rate displayed (or marked as not meaningful)")
    print(f"\n  → Open: file://{mock_viewer.absolute()}")

    print("\nReal viewer should show:")
    print("  - Green success badge: '✓ REAL - Actual Windows Agent Arena evaluation'")
    print("  - NO banner at top")
    print("  - Success rate displayed normally")
    print(f"\n  → Open: file://{real_viewer.absolute()}")

    print("\n" + "=" * 60)
    print("\n✓ Test complete!")
    print("\nTo view the generated HTML files, open them in your browser:")
    print(f"  - Mock: file://{mock_viewer.absolute()}")
    print(f"  - Real: file://{real_viewer.absolute()}")
    print(f"\nNote: The temporary real run directory can be deleted with:")
    print(f"  rm -rf {real_run}")

if __name__ == "__main__":
    main()
