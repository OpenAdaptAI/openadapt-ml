#!/usr/bin/env python3
"""CLI entry point for standalone GRPO training.

Usage:
    python scripts/run_grpo.py \\
        --server-url http://localhost:5001 \\
        --task-ids abc-123 def-456 \\
        --model-name Qwen/Qwen2.5-VL-7B-Instruct \\
        --num-steps 100 \\
        --output-dir checkpoints/grpo_run1

    # Resume from existing SFT LoRA:
    python scripts/run_grpo.py \\
        --server-url http://localhost:5001 \\
        --task-ids abc-123 \\
        --lora-checkpoint checkpoints/sft/step_500 \\
        --num-steps 50
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run standalone GRPO training against a WAA VM",
    )
    parser.add_argument("--server-url", default="http://localhost:5001")
    parser.add_argument("--evaluate-url", default=None)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--max-steps-per-episode", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    args = parser.parse_args()

    from openadapt_ml.training.grpo.config import GRPOConfig
    from openadapt_ml.training.grpo.trainer import GRPOTrainer

    config = GRPOConfig(
        backend="standalone",
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        num_rollouts_per_step=args.num_rollouts,
        max_steps_per_episode=args.max_steps_per_episode,
        temperature=args.temperature,
        server_url=args.server_url,
        evaluate_url=args.evaluate_url,
        task_ids=args.task_ids,
        learning_rate=args.learning_rate,
        num_training_steps=args.num_steps,
        save_every_steps=args.save_every,
        output_dir=args.output_dir,
    )

    trainer = GRPOTrainer(config)
    checkpoint = trainer.train()
    print(f"Training complete. Final checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
