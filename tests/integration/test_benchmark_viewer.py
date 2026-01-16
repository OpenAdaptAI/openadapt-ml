#!/usr/bin/env python
"""Test script for benchmark viewer generation.

This script tests the benchmark viewer by generating the HTML from existing
benchmark test data.

Usage:
    python test_benchmark_viewer.py
"""

from pathlib import Path
from openadapt_ml.training.benchmark_viewer import generate_benchmark_viewer

def main():
    # Test with the existing test_run_phase1 data
    benchmark_dir = Path("benchmark_results/test_run_phase1")

    if not benchmark_dir.exists():
        print(f"Error: Test data not found at {benchmark_dir}")
        print("Please run the data collection test first:")
        print("  python -m openadapt_ml.benchmarks.cli test-collection --tasks 5")
        return 1

    print("=" * 60)
    print("Testing Benchmark Viewer Generation")
    print("=" * 60)
    print(f"Benchmark directory: {benchmark_dir}")
    print()

    try:
        # Generate the benchmark viewer
        viewer_path = generate_benchmark_viewer(benchmark_dir)

        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Benchmark viewer generated at: {viewer_path}")
        print()
        print("To view the benchmark viewer:")
        print(f"  python -m openadapt_ml.cloud.local serve --benchmark {benchmark_dir} --open")
        print()
        print("Or open directly in your browser:")
        print(f"  open {viewer_path}")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR!")
        print("=" * 60)
        print(f"Failed to generate benchmark viewer: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
