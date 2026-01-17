"""Main experiment runner for the Representation Shootout.

This module provides:
1. ExperimentRunner class for running the full experiment
2. CLI interface for experiment execution
3. Results reporting and recommendation generation

Usage:
    # Run full experiment
    python -m openadapt_ml.experiments.representation_shootout.runner run

    # Run specific condition
    python -m openadapt_ml.experiments.representation_shootout.runner run --condition marks

    # Evaluate under specific drift
    python -m openadapt_ml.experiments.representation_shootout.runner eval --drift resolution

    # Generate recommendation from existing results
    python -m openadapt_ml.experiments.representation_shootout.runner recommend --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from openadapt_ml.experiments.representation_shootout.conditions import (
    ConditionBase,
    Observation,
    UIElement,
    UIElementGraph,
    create_condition,
)
from openadapt_ml.experiments.representation_shootout.config import (
    ConditionName,
    DriftConfig,
    ExperimentConfig,
    MetricName,
)
from openadapt_ml.experiments.representation_shootout.evaluator import (
    DriftEvaluator,
    EvaluationResult,
    Recommendation,
    Sample,
    make_recommendation,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the Representation Shootout experiment.

    This class manages:
    1. Loading/generating evaluation data
    2. Running conditions under drift
    3. Computing metrics and generating recommendations
    4. Saving results
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.conditions: dict[ConditionName, ConditionBase] = {}
        self.evaluator: DriftEvaluator | None = None
        self.results: list[EvaluationResult] = []

        # Validate config
        issues = config.validate()
        if issues:
            raise ValueError(f"Invalid config: {issues}")

        # Initialize conditions
        for cond_config in config.conditions:
            self.conditions[cond_config.name] = create_condition(cond_config)

        # Initialize evaluator
        self.evaluator = DriftEvaluator(self.conditions, config.drift_tests)

    def load_samples(self, data_path: str | None = None) -> list[Sample]:
        """Load evaluation samples from data directory.

        This is a scaffolding implementation. Full implementation would:
        1. Load screenshots from data_path
        2. Load ground truth actions
        3. Load UI element annotations (for marks condition)

        Args:
            data_path: Path to data directory (uses config.dataset.eval_path if None).

        Returns:
            List of Sample objects.
        """
        data_path = data_path or self.config.dataset.eval_path

        if data_path and Path(data_path).exists():
            # Load from files
            samples = self._load_samples_from_path(data_path)
            if samples:
                return samples

        # Generate synthetic samples for scaffolding
        logger.info("Generating synthetic samples for scaffolding")
        return self._generate_synthetic_samples(num_samples=100)

    def _load_samples_from_path(self, data_path: str) -> list[Sample]:
        """Load samples from a data directory.

        Expected structure:
        data_path/
            samples.json  # List of sample metadata
            screenshots/
                sample_001.png
                sample_002.png
                ...

        Args:
            data_path: Path to data directory.

        Returns:
            List of Sample objects.
        """
        data_dir = Path(data_path)
        samples_file = data_dir / "samples.json"

        if not samples_file.exists():
            logger.warning(f"No samples.json found in {data_path}")
            return []

        with open(samples_file) as f:
            samples_data = json.load(f)

        samples = []
        for item in samples_data:
            # Build UI elements if present
            ui_elements = None
            if "ui_elements" in item:
                elements = [
                    UIElement(
                        element_id=el["id"],
                        role=el.get("role", "unknown"),
                        name=el.get("name"),
                        bbox=tuple(el["bbox"]),  # type: ignore
                    )
                    for el in item["ui_elements"]
                ]
                ui_elements = UIElementGraph(elements=elements)

            observation = Observation(
                screenshot_path=str(data_dir / "screenshots" / item["screenshot"]),
                screen_size=tuple(item.get("screen_size", (1920, 1080))),  # type: ignore
                ui_elements=ui_elements,
            )

            sample = Sample(
                sample_id=item["id"],
                observation=observation,
                goal=item["goal"],
                ground_truth=item["ground_truth"],
            )
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from {data_path}")
        return samples

    def _generate_synthetic_samples(self, num_samples: int = 100) -> list[Sample]:
        """Generate synthetic samples for scaffolding.

        These are placeholder samples for testing the framework.
        Real experiments should use actual UI data.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            List of synthetic Sample objects.
        """
        import random

        random.seed(self.config.seed)

        samples = []
        for i in range(num_samples):
            # Generate random UI elements
            num_elements = random.randint(5, 20)
            elements = []
            for j in range(num_elements):
                x1 = random.uniform(0, 0.8)
                y1 = random.uniform(0, 0.8)
                w = random.uniform(0.05, 0.2)
                h = random.uniform(0.03, 0.1)
                elements.append(
                    UIElement(
                        element_id=f"e{j + 1}",
                        role=random.choice(["button", "textfield", "link", "checkbox"]),
                        name=f"Element {j + 1}",
                        bbox=(x1, y1, x1 + w, y1 + h),
                    )
                )

            ui_elements = UIElementGraph(elements=elements)

            # Pick a random target element
            target_element = random.choice(elements)
            target_x, target_y = target_element.center

            observation = Observation(
                screenshot_path=None,  # No actual screenshot in scaffolding
                screen_size=(1920, 1080),
                ui_elements=ui_elements,
            )

            sample = Sample(
                sample_id=f"synthetic_{i:04d}",
                observation=observation,
                goal=f"Click the {target_element.role} named '{target_element.name}'",
                ground_truth={
                    "type": "click",
                    "x": target_x,
                    "y": target_y,
                    "element_id": target_element.element_id,
                    "target_bbox": target_element.bbox,
                },
            )
            samples.append(sample)

        logger.info(f"Generated {num_samples} synthetic samples")
        return samples

    def get_model_predictions(
        self,
        condition: ConditionBase,
        samples: list[Sample],
        drift_config: DriftConfig,
    ) -> list[str]:
        """Get model predictions for samples.

        This is a scaffolding implementation that returns mock predictions.
        Full implementation would:
        1. Prepare inputs using condition.prepare_input()
        2. Send to model (VLM API or local model)
        3. Return raw model outputs

        Args:
            condition: Condition to use for input preparation.
            samples: Samples to get predictions for.
            drift_config: Drift applied to samples.

        Returns:
            List of raw model output strings.
        """
        # Scaffolding: Generate plausible mock predictions
        import random

        random.seed(
            self.config.seed + hash(condition.name.value) + hash(drift_config.name)
        )

        predictions = []
        for sample in samples:
            gt = sample.ground_truth

            if condition.name == ConditionName.MARKS:
                # Generate element ID prediction
                # Simulate some errors based on drift
                error_rate = self._get_error_rate(drift_config)
                if random.random() < error_rate:
                    # Make an error - pick wrong element
                    if sample.observation.ui_elements:
                        wrong_el = random.choice(
                            sample.observation.ui_elements.elements
                        )
                        predictions.append(f"ACTION: CLICK([{wrong_el.element_id}])")
                    else:
                        predictions.append("ACTION: CLICK([e1])")
                else:
                    # Correct prediction
                    predictions.append(f"ACTION: CLICK([{gt.get('element_id', 'e1')}])")
            else:
                # Generate coordinate prediction
                # Add some noise based on drift
                noise_std = self._get_coordinate_noise(drift_config)
                pred_x = gt.get("x", 0.5) + random.gauss(0, noise_std)
                pred_y = gt.get("y", 0.5) + random.gauss(0, noise_std)
                # Clamp to valid range
                pred_x = max(0, min(1, pred_x))
                pred_y = max(0, min(1, pred_y))
                predictions.append(f"ACTION: CLICK({pred_x:.4f}, {pred_y:.4f})")

        return predictions

    def _get_error_rate(self, drift_config: DriftConfig) -> float:
        """Get expected error rate for marks condition under drift."""
        if drift_config.is_canonical:
            return 0.05  # 5% baseline error

        # Different drifts have different impacts
        drift_impact = {
            "resolution": 0.08,  # Small impact - elements still identifiable
            "translation": 0.05,  # Minimal impact - relative positions preserved
            "theme": 0.15,  # Moderate impact - visual appearance changes
            "scroll": 0.10,  # Some elements may be off-screen
        }

        drift_type = drift_config.drift_type.value
        base_rate = drift_impact.get(drift_type, 0.10)

        # Scale by drift severity
        if drift_config.drift_type.value == "resolution":
            scale = abs(drift_config.params.scale - 1.0)  # type: ignore
            return base_rate + scale * 0.2
        elif drift_config.drift_type.value == "scroll":
            scroll_amount = drift_config.params.offset_y  # type: ignore
            return base_rate + (scroll_amount / 1000) * 0.2

        return base_rate

    def _get_coordinate_noise(self, drift_config: DriftConfig) -> float:
        """Get expected coordinate noise for coords conditions under drift."""
        if drift_config.is_canonical:
            return 0.02  # 2% baseline noise (normalized coords)

        # Coordinates are more sensitive to drift than marks
        drift_impact = {
            "resolution": 0.08,  # Significant - coordinates may not scale correctly
            "translation": 0.06,  # Moderate - if using screen-absolute coords
            "theme": 0.03,  # Minimal - visual changes don't affect coordinates directly
            "scroll": 0.12,  # High - y-coordinates shift significantly
        }

        drift_type = drift_config.drift_type.value
        return drift_impact.get(drift_type, 0.05)

    def run_condition(
        self,
        condition_name: ConditionName,
        samples: list[Sample],
    ) -> list[EvaluationResult]:
        """Run a single condition across all drift tests.

        Args:
            condition_name: Name of condition to run.
            samples: Samples to evaluate.

        Returns:
            List of EvaluationResult for each drift test.
        """
        condition = self.conditions.get(condition_name)
        if not condition:
            raise ValueError(f"Condition {condition_name} not found")

        results = []
        for drift_config in self.config.drift_tests:
            logger.info(f"Evaluating {condition_name.value} under {drift_config.name}")

            # Get model predictions
            predictions = self.get_model_predictions(condition, samples, drift_config)

            # Evaluate
            result = self.evaluator.evaluate_condition_under_drift(  # type: ignore
                condition, samples, drift_config, predictions
            )
            results.append(result)

            # Log summary
            hit_rate = result.metrics.get(MetricName.CLICK_HIT_RATE.value, 0)
            logger.info(f"  Click-hit rate: {hit_rate:.1%}")

        return results

    def run(self) -> Recommendation:
        """Run the full experiment.

        Returns:
            Recommendation based on results.
        """
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Conditions: {[c.value for c in self.conditions.keys()]}")
        logger.info(f"Drift tests: {[d.name for d in self.config.drift_tests]}")

        # Load samples
        samples = self.load_samples()
        logger.info(f"Loaded {len(samples)} samples")

        # Run all conditions
        all_results = []
        for condition_name in self.conditions.keys():
            results = self.run_condition(condition_name, samples)
            all_results.extend(results)

        self.results = all_results

        # Generate recommendation
        recommendation = make_recommendation(
            all_results,
            tolerance=self.config.decision_tolerance,
        )

        # Save results
        self.save_results(recommendation)

        return recommendation

    def save_results(self, recommendation: Recommendation) -> None:
        """Save experiment results to output directory.

        Args:
            recommendation: Final recommendation.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"results_{timestamp}.json"

        # Serialize results
        results_data = {
            "experiment": self.config.name,
            "timestamp": timestamp,
            "config": {
                "conditions": [c.value for c in self.conditions.keys()],
                "drift_tests": [d.name for d in self.config.drift_tests],
                "decision_tolerance": self.config.decision_tolerance,
            },
            "results": [
                {
                    "condition": r.condition.value,
                    "drift": r.drift,
                    "num_samples": r.num_samples,
                    "metrics": r.metrics,
                }
                for r in self.results
            ],
            "recommendation": {
                "recommended": recommendation.recommended,
                "reason": recommendation.reason,
                "coords_cues_avg": recommendation.coords_cues_avg,
                "marks_avg": recommendation.marks_avg,
                "tolerance": recommendation.tolerance,
                "detailed_comparison": recommendation.detailed_comparison,
            },
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def print_summary(self, recommendation: Recommendation) -> None:
        """Print experiment summary to stdout.

        Args:
            recommendation: Final recommendation.
        """
        print("\n" + "=" * 70)
        print("REPRESENTATION SHOOTOUT - EXPERIMENT SUMMARY")
        print("=" * 70)

        print(f"\nExperiment: {self.config.name}")
        print(f"Conditions: {', '.join(c.value for c in self.conditions.keys())}")
        print(f"Drift tests: {len(self.config.drift_tests)}")
        print(f"Samples: {len(self.results[0].sample_results) if self.results else 0}")

        print("\n" + "-" * 70)
        print("RESULTS BY CONDITION AND DRIFT")
        print("-" * 70)

        # Group results by condition
        by_condition: dict[str, list[EvaluationResult]] = {}
        for r in self.results:
            key = r.condition.value
            if key not in by_condition:
                by_condition[key] = []
            by_condition[key].append(r)

        # Print table
        header = f"{'Condition':<15} {'Drift':<25} {'Hit Rate':<12} {'Distance':<12}"
        print(header)
        print("-" * len(header))

        for condition, results in by_condition.items():
            for r in results:
                hit_rate = r.metrics.get(MetricName.CLICK_HIT_RATE.value, 0)
                distance = r.metrics.get(MetricName.COORD_DISTANCE.value, 0)
                print(
                    f"{condition:<15} {r.drift:<25} {hit_rate:>10.1%} {distance:>10.4f}"
                )
            print()

        print("-" * 70)
        print("RECOMMENDATION")
        print("-" * 70)
        print(f"\nRecommended approach: {recommendation.recommended}")
        print(f"\nReason: {recommendation.reason}")
        print(f"\nCoords+Cues average: {recommendation.coords_cues_avg:.1%}")
        print(f"Marks average: {recommendation.marks_avg:.1%}")
        print(f"Tolerance: {recommendation.tolerance:.1%}")

        print("\n" + "=" * 70)


def run_experiment(
    config: ExperimentConfig | None = None,
    data_path: str | None = None,
    verbose: bool = True,
) -> Recommendation:
    """Convenience function to run the experiment.

    Args:
        config: Experiment configuration (uses default if None).
        data_path: Path to evaluation data.
        verbose: Whether to print progress.

    Returns:
        Final recommendation.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = config or ExperimentConfig.default()
    runner = ExperimentRunner(config)

    recommendation = runner.run()

    if verbose:
        runner.print_summary(recommendation)

    return recommendation


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Representation Shootout Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full experiment with default config
    python -m openadapt_ml.experiments.representation_shootout.runner run

    # Run with minimal config (quick test)
    python -m openadapt_ml.experiments.representation_shootout.runner run --minimal

    # Run specific condition only
    python -m openadapt_ml.experiments.representation_shootout.runner run --condition marks

    # Specify output directory
    python -m openadapt_ml.experiments.representation_shootout.runner run --output results/my_experiment

    # Generate recommendation from existing results
    python -m openadapt_ml.experiments.representation_shootout.runner recommend --results results/results_20260116.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the experiment")
    run_parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use minimal config for quick testing",
    )
    run_parser.add_argument(
        "--condition",
        choices=["raw_coords", "coords_cues", "marks"],
        help="Run only specific condition",
    )
    run_parser.add_argument(
        "--data",
        help="Path to evaluation data directory",
    )
    run_parser.add_argument(
        "--output",
        default="experiment_results/representation_shootout",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Recommend command (analyze existing results)
    rec_parser = subparsers.add_parser(
        "recommend", help="Generate recommendation from results"
    )
    rec_parser.add_argument(
        "--results",
        required=True,
        help="Path to results JSON file",
    )

    args = parser.parse_args()

    if args.command == "run":
        # Configure logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

        # Build config
        if args.minimal:
            config = ExperimentConfig.minimal()
        else:
            config = ExperimentConfig.default()

        # Override output dir
        config.output_dir = args.output
        config.seed = args.seed

        # Filter to specific condition if requested
        if args.condition:
            condition_name = ConditionName(args.condition)
            config.conditions = [
                c for c in config.conditions if c.name == condition_name
            ]
            if not config.conditions:
                print(f"Error: No matching condition found for {args.condition}")
                return 1

        try:
            runner = ExperimentRunner(config)
            if args.data:
                runner.load_samples(args.data)
            recommendation = runner.run()
            runner.print_summary(recommendation)
            return 0
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1

    elif args.command == "recommend":
        # Load existing results and generate recommendation
        try:
            with open(args.results) as f:
                data = json.load(f)

            # Reconstruct EvaluationResults
            results = []
            for r in data["results"]:
                results.append(
                    EvaluationResult(
                        condition=ConditionName(r["condition"]),
                        drift=r["drift"],
                        num_samples=r["num_samples"],
                        metrics=r["metrics"],
                    )
                )

            tolerance = data.get("config", {}).get("decision_tolerance", 0.05)
            recommendation = make_recommendation(results, tolerance=tolerance)

            print("\n" + "=" * 70)
            print("RECOMMENDATION FROM RESULTS")
            print("=" * 70)
            print(f"\nRecommended approach: {recommendation.recommended}")
            print(f"\nReason: {recommendation.reason}")
            return 0

        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
