from __future__ import annotations

import os
from pathlib import Path

from openadapt_ml.ingest.synthetic import generate_synthetic_episodes


def main() -> None:
    output_dir = Path("synthetic") / "debug"
    episodes = generate_synthetic_episodes(num_episodes=2, seed=42, output_dir=output_dir)

    print(f"Generated {len(episodes)} episodes into {output_dir.resolve()}")

    total_steps = 0
    missing_images: list[str] = []

    for episode in episodes:
        total_steps += len(episode.steps)
        for step in episode.steps:
            path = step.observation.screenshot_path
            if not path:
                missing_images.append(f"[no path] in episode {episode.episode_id}")
                continue
            if not os.path.exists(path):
                missing_images.append(path)

    print(f"Episodes: {len(episodes)}, Steps: {total_steps}")

    if missing_images:
        print("Missing images:")
        for p in missing_images:
            print(" -", p)
        raise SystemExit(1)

    print("All observation image paths exist.")


if __name__ == "__main__":
    main()
