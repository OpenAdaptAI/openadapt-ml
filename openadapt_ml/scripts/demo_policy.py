from __future__ import annotations

import argparse

from openadapt_ml.datasets.next_action import build_next_action_sft_samples
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions
from openadapt_ml.models.dummy_adapter import DummyAdapter
from openadapt_ml.models.qwen_vl import QwenVLAdapter
from openadapt_ml.runtime.policy import AgentPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["dummy", "qwen3", "qwen2_5"], default="dummy")
    args = parser.parse_args()

    # Use synthetic data to build one SFT-style sample
    sessions = generate_synthetic_sessions(num_sessions=1, seed=99, output_dir="synthetic/demo")
    episodes = [ep for sess in sessions for ep in sess.episodes]
    samples = build_next_action_sft_samples(episodes)

    # Load first sample and overwrite assistant content so the dummy adapter
    # doesn't depend on any particular target.
    sample = samples[0]

    if args.backend == "dummy":
        adapter = DummyAdapter()
    elif args.backend == "qwen3":
        adapter = QwenVLAdapter.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            lora_config=None,
            load_in_4bit=False,
        )
    else:  # qwen2_5
        adapter = QwenVLAdapter.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            lora_config=None,
            load_in_4bit=False,
        )
    policy = AgentPolicy(adapter)

    action, thought = policy.predict_action_from_sample(sample)
    print("Raw sample messages:")
    for m in sample["messages"]:
        print(f"[{m['role']}] {m['content']}")

    print("\nPredicted action:", action)
    print("Thought:", thought)

if __name__ == "__main__":
    main()
