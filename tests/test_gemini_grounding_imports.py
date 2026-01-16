"""Test that GeminiGrounder imports work correctly.

This is a minimal import test that doesn't require API keys or actual inference.
It verifies that the module structure is correct and all imports resolve.
"""

import pytest


def test_import_base_classes():
    """Test importing base grounding classes."""
    from openadapt_ml.grounding.base import (
        GroundingModule,
        OracleGrounder,
        RegionCandidate,
    )

    assert GroundingModule is not None
    assert OracleGrounder is not None
    assert RegionCandidate is not None


def test_import_gemini_grounder():
    """Test importing GeminiGrounder."""
    from openadapt_ml.grounding.detector import GeminiGrounder

    assert GeminiGrounder is not None


def test_import_utility_functions():
    """Test importing utility functions."""
    from openadapt_ml.grounding.detector import (
        extract_ui_elements,
        overlay_element_marks,
    )

    assert extract_ui_elements is not None
    assert overlay_element_marks is not None


def test_import_from_package():
    """Test importing from package __init__."""
    from openadapt_ml.grounding import (
        DetectorGrounder,
        GeminiGrounder,
        GroundingModule,
        OracleGrounder,
        RegionCandidate,
        extract_ui_elements,
        overlay_element_marks,
    )

    assert GeminiGrounder is not None
    assert DetectorGrounder is not None
    assert GroundingModule is not None
    assert OracleGrounder is not None
    assert RegionCandidate is not None
    assert extract_ui_elements is not None
    assert overlay_element_marks is not None


def test_region_candidate_creation():
    """Test creating a RegionCandidate object."""
    from openadapt_ml.grounding.base import RegionCandidate

    candidate = RegionCandidate(
        bbox=(0.1, 0.2, 0.3, 0.4),
        centroid=(0.2, 0.3),
        confidence=0.95,
        element_label="button",
        text_content="Submit",
    )

    assert candidate.bbox == (0.1, 0.2, 0.3, 0.4)
    assert candidate.centroid == (0.2, 0.3)
    assert candidate.confidence == 0.95
    assert candidate.element_label == "button"
    assert candidate.text_content == "Submit"


def test_region_candidate_validation():
    """Test RegionCandidate coordinate validation."""
    from openadapt_ml.grounding.base import RegionCandidate

    # Valid coordinates
    RegionCandidate(
        bbox=(0.0, 0.0, 1.0, 1.0),
        centroid=(0.5, 0.5),
        confidence=1.0,
    )

    # Invalid coordinates (out of range)
    with pytest.raises(ValueError):
        RegionCandidate(
            bbox=(0.0, 0.0, 1.5, 1.0),  # x2 > 1.0
            centroid=(0.5, 0.5),
            confidence=1.0,
        )

    # Invalid confidence
    with pytest.raises(ValueError):
        RegionCandidate(
            bbox=(0.0, 0.0, 1.0, 1.0),
            centroid=(0.5, 0.5),
            confidence=1.5,  # > 1.0
        )


def test_region_candidate_methods():
    """Test RegionCandidate utility methods."""
    from openadapt_ml.grounding.base import RegionCandidate

    candidate1 = RegionCandidate(
        bbox=(0.0, 0.0, 0.5, 0.5),
        centroid=(0.25, 0.25),
        confidence=0.9,
    )

    candidate2 = RegionCandidate(
        bbox=(0.25, 0.25, 0.75, 0.75),
        centroid=(0.5, 0.5),
        confidence=0.8,
    )

    # Test area calculation
    assert candidate1.area == 0.25  # 0.5 * 0.5

    # Test IoU calculation
    iou = candidate1.iou(candidate2)
    assert 0.0 <= iou <= 1.0

    # Test point containment
    assert candidate1.contains_point(0.25, 0.25)  # Inside
    assert not candidate1.contains_point(0.75, 0.75)  # Outside


def test_gemini_grounder_initialization():
    """Test GeminiGrounder can be instantiated (no API call)."""
    from openadapt_ml.grounding.detector import GeminiGrounder

    # Should initialize without error (lazy loading)
    grounder = GeminiGrounder()
    assert grounder is not None
    assert grounder.name == "GeminiGrounder"
    assert grounder.supports_batch is False

    # Test with custom model name
    grounder2 = GeminiGrounder(model="gemini-2.5-pro")
    assert grounder2._model_name == "gemini-2.5-pro"


def test_oracle_grounder():
    """Test OracleGrounder basic functionality."""
    from openadapt_ml.grounding.base import OracleGrounder

    oracle = OracleGrounder()

    # Set ground truth
    oracle.set_ground_truth(
        target_description="login button",
        bbox=(0.3, 0.5, 0.7, 0.6),
    )

    # Ground should return the set ground truth
    from PIL import Image

    dummy_img = Image.new("RGB", (100, 100))
    candidates = oracle.ground(dummy_img, "login button")

    assert len(candidates) == 1
    assert candidates[0].bbox == (0.3, 0.5, 0.7, 0.6)
    assert candidates[0].confidence == 1.0

    # Unknown target should return empty list
    candidates = oracle.ground(dummy_img, "unknown button")
    assert len(candidates) == 0


if __name__ == "__main__":
    # Run tests without pytest
    test_import_base_classes()
    test_import_gemini_grounder()
    test_import_utility_functions()
    test_import_from_package()
    test_region_candidate_creation()
    test_region_candidate_validation()
    test_region_candidate_methods()
    test_gemini_grounder_initialization()
    test_oracle_grounder()
    print("✅ All import tests passed!")
