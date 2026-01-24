#!/usr/bin/env python3
"""
Sync package documentation from sub-repos to the main OpenAdapt docs.

This script fetches README.md files from OpenAdapt sub-packages and transforms
them into MkDocs-compatible documentation pages.

Usage:
    python scripts/sync_package_docs.py [--dry-run] [--package PACKAGE]

Arguments:
    --dry-run       Print what would be changed without making changes
    --package       Sync only a specific package (e.g., openadapt-ml)

Environment:
    GITHUB_TOKEN    GitHub token for API access (optional, increases rate limit)

This script should be run from the main OpenAdapt repository root.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError


# Package configuration: maps package name to doc output file
PACKAGES = {
    "openadapt-ml": {
        "repo": "OpenAdaptAI/openadapt-ml",
        "doc_file": "docs/packages/ml.md",
        "title": "openadapt-ml",
        "description": "Policy learning, training, and inference for GUI automation agents.",
    },
    "openadapt-evals": {
        "repo": "OpenAdaptAI/openadapt-evals",
        "doc_file": "docs/packages/evals.md",
        "title": "openadapt-evals",
        "description": "Benchmark evaluation framework for GUI automation agents.",
    },
    "openadapt-capture": {
        "repo": "OpenAdaptAI/openadapt-capture",
        "doc_file": "docs/packages/capture.md",
        "title": "openadapt-capture",
        "description": "Screen and input capture for demonstration recording.",
    },
    "openadapt-viewer": {
        "repo": "OpenAdaptAI/openadapt-viewer",
        "doc_file": "docs/packages/viewer.md",
        "title": "openadapt-viewer",
        "description": "Web-based viewer for demonstrations and training results.",
    },
    "openadapt-grounding": {
        "repo": "OpenAdaptAI/openadapt-grounding",
        "doc_file": "docs/packages/grounding.md",
        "title": "openadapt-grounding",
        "description": "Visual grounding for element detection and localization.",
    },
    "openadapt-retrieval": {
        "repo": "OpenAdaptAI/openadapt-retrieval",
        "doc_file": "docs/packages/retrieval.md",
        "title": "openadapt-retrieval",
        "description": "Demo retrieval for trajectory-conditioned prompting.",
    },
    "openadapt-privacy": {
        "repo": "OpenAdaptAI/openadapt-privacy",
        "doc_file": "docs/packages/privacy.md",
        "title": "openadapt-privacy",
        "description": "PII/PHI detection and scrubbing for privacy protection.",
    },
}


def fetch_readme(repo: str, token: Optional[str] = None) -> Optional[str]:
    """Fetch README.md from a GitHub repository."""
    url = f"https://api.github.com/repos/{repo}/readme"
    headers = {
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "OpenAdapt-Docs-Sync/1.0",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        request = Request(url, headers=headers)
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 404:
            print(f"  Warning: README not found for {repo}")
        else:
            print(f"  Error fetching README for {repo}: {e}")
        return None
    except Exception as e:
        print(f"  Error fetching README for {repo}: {e}")
        return None


def transform_readme(
    readme: str,
    package_name: str,
    config: dict,
) -> str:
    """Transform a package README into MkDocs-compatible documentation.

    Transformations:
    1. Replace the title with standardized format
    2. Add package metadata header
    3. Fix relative links to point to GitHub
    4. Remove badges (they don't render well in MkDocs)
    5. Add navigation hint at the end
    """
    repo = config["repo"]
    title = config["title"]
    description = config["description"]

    lines = readme.split("\n")
    output_lines = []

    # Skip existing title and badges
    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip title line
        if line.startswith("# "):
            i += 1
            continue
        # Skip badge lines (contain shields.io or badge images)
        if "shields.io" in line or "badge" in line.lower() or line.strip().startswith("[!["):
            i += 1
            continue
        # Skip empty lines after title/badges
        if not line.strip() and i < 5:
            i += 1
            continue
        break

    # Add standardized header
    output_lines.append(f"# {title}")
    output_lines.append("")
    output_lines.append(description)
    output_lines.append("")
    output_lines.append(f"**Repository**: [{repo}](https://github.com/{repo})")
    output_lines.append("")

    # Process remaining content
    while i < len(lines):
        line = lines[i]

        # Fix relative links to docs/ -> GitHub
        line = re.sub(
            r'\[([^\]]+)\]\(docs/([^)]+)\)',
            f'[\\1](https://github.com/{repo}/blob/main/docs/\\2)',
            line
        )

        # Fix relative links to other files -> GitHub
        line = re.sub(
            r'\[([^\]]+)\]\((?!http)([^)]+\.md)\)',
            f'[\\1](https://github.com/{repo}/blob/main/\\2)',
            line
        )

        output_lines.append(line)
        i += 1

    # Add footer
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    output_lines.append(f"*This page is auto-generated from the [{package_name} README](https://github.com/{repo}). "
                       f"To update, push changes to the source repository.*")

    return "\n".join(output_lines)


def sync_package(
    package_name: str,
    config: dict,
    token: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Sync documentation for a single package.

    Returns True if changes were made (or would be made in dry-run mode).
    """
    print(f"Syncing {package_name}...")

    # Fetch README
    readme = fetch_readme(config["repo"], token)
    if not readme:
        print(f"  Skipping {package_name}: Could not fetch README")
        return False

    # Transform to MkDocs format
    doc_content = transform_readme(readme, package_name, config)

    # Check if file needs updating
    doc_path = Path(config["doc_file"])
    if doc_path.exists():
        existing_content = doc_path.read_text()
        if existing_content == doc_content:
            print(f"  No changes for {package_name}")
            return False

    # Write or report
    if dry_run:
        print(f"  Would update {doc_path}")
        print(f"  Preview (first 500 chars):")
        print("  " + "-" * 40)
        for line in doc_content[:500].split("\n"):
            print(f"  {line}")
        print("  " + "-" * 40)
    else:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(doc_content)
        print(f"  Updated {doc_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync package documentation from sub-repos"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without making changes",
    )
    parser.add_argument(
        "--package",
        help="Sync only a specific package",
    )
    args = parser.parse_args()

    # Get GitHub token from environment
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Note: GITHUB_TOKEN not set, API rate limits may apply")

    # Determine which packages to sync
    if args.package:
        if args.package not in PACKAGES:
            print(f"Error: Unknown package '{args.package}'")
            print(f"Available packages: {', '.join(PACKAGES.keys())}")
            sys.exit(1)
        packages_to_sync = {args.package: PACKAGES[args.package]}
    else:
        packages_to_sync = PACKAGES

    # Sync each package
    changes_made = False
    for package_name, config in packages_to_sync.items():
        if sync_package(package_name, config, token, args.dry_run):
            changes_made = True

    # Summary
    if changes_made:
        if args.dry_run:
            print("\nDry run complete. Use without --dry-run to apply changes.")
        else:
            print("\nSync complete. Changes were made.")
    else:
        print("\nNo changes needed.")

    return 0 if not changes_made else 1  # Exit 1 if changes for CI detection


if __name__ == "__main__":
    sys.exit(main())
