from __future__ import annotations

import os
from pathlib import Path

from openadapt_ml.ingest.synthetic import generate_synthetic_sessions


def main() -> None:
    output_dir = Path("synthetic_debug")
    sessions = generate_synthetic_sessions(num_sessions=2, seed=42, output_dir=output_dir)

    print(f"Generated {len(sessions)} sessions into {output_dir.resolve()}")

    total_episodes = 0
    total_steps = 0
    missing_images: list[str] = []

    for session in sessions:
        total_episodes += len(session.episodes)
        for episode in session.episodes:
            total_steps += len(episode.steps)
            for step in episode.steps:
                path = step.observation.image_path
                if not path:
                    missing_images.append(f"[no path] in episode {episode.id}")
                    continue
                if not os.path.exists(path):
                    missing_images.append(path)

    print(f"Episodes: {total_episodes}, Steps: {total_steps}")

    if missing_images:
        print("Missing images:")
        for p in missing_images:
            print(" -", p)
        raise SystemExit(1)

    print("All observation image paths exist.")


if __name__ == "__main__":
    main()
