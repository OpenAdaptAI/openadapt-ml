"""CLI for export utilities."""

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Main entry point for export CLI."""
    parser = argparse.ArgumentParser(
        description="Export Episode data to various formats",
        prog="python -m openadapt_ml.export",
    )
    subparsers = parser.add_subparsers(dest="command", help="Export format")

    # Parquet subcommand
    parquet_parser = subparsers.add_parser(
        "parquet",
        help="Export to Parquet format for analytics",
    )
    parquet_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory containing Episode JSON files",
    )
    parquet_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for .parquet file",
    )
    parquet_parser.add_argument(
        "--include-summary",
        action="store_true",
        help="Also generate episode-level summary table",
    )

    args = parser.parse_args()

    if args.command == "parquet":
        return export_parquet(args)
    else:
        parser.print_help()
        return 1


def export_parquet(args: argparse.Namespace) -> int:
    """Export Episodes to Parquet."""
    try:
        from openadapt_ml.export import to_parquet
        from openadapt_ml.ingest import load_episodes
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading episodes from: {input_path}")
    episodes = load_episodes(str(input_path))
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("Warning: No episodes found", file=sys.stderr)
        return 1

    total_steps = sum(len(ep.steps) for ep in episodes)
    print(f"Total steps: {total_steps}")

    print(f"Exporting to: {args.output}")
    to_parquet(
        episodes,
        args.output,
        include_summary=args.include_summary,
    )

    print("Done!")
    if args.include_summary:
        summary_path = args.output.replace(".parquet", "_summary.parquet")
        print(f"Summary written to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
