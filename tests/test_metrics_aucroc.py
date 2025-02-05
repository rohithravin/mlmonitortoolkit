"""
Unit tests for the AUCROC class.

Author: Rohith Ravindranath
Date: Jan 30, 2025
"""

import pytest
import numpy as np
import pandas as pd
from mlmonkit.metrics.classification.aucroc import AUCROC

@pytest.fixture
def aucroc_batch():
    """Fixture to initialize an AUCROC instance in batch mode."""
    return AUCROC(use_streaming=False)

@pytest.fixture
def aucroc_streaming():
    """Fixture to initialize an AUCROC instance in streaming mode."""
    return AUCROC(use_streaming=True)

@pytest.mark.parametrize(
    "true_labels, pred_scores, expected_aucroc",
    [
        # Test NumPy arrays
        (np.array([1, 0, 1, 1]), np.array([0.6, 0.4, 0.7, 0.3]), 2/3),
        # Test Python lists
        ([1, 0, 1, 1], [0.6, 0.4, 0.7, 0.3], 2/3),
        # Test pandas Series
        (pd.Series([1, 0, 1, 1]), pd.Series([0.6, 0.4, 0.7, 0.3]), 2/3),
        # Test pandas DataFrame (single-column)
        (pd.DataFrame({"label": [1, 0, 1, 1]}), pd.DataFrame({"score": [0.6, 0.4, 0.7, 0.3]}), 2/3),
    ],
)
def test_batch_aucroc(aucroc_batch, true_labels, pred_scores, expected_aucroc):
    """Test batch AUCROC calculation for different input types."""
    assert aucroc_batch.do_score(true_labels, pred_scores) == pytest.approx(expected_aucroc)

@pytest.mark.parametrize(
    "true_labels, pred_scores, expected_intermediate, expected_final",
    [
        # Test NumPy arrays
        (np.array([1, 0, 1, 1]), np.array([0.6, 0.4, 0.7, 0.3]), 2/3, 0.6875),
        # Test Python lists
        ([1, 0, 1, 1], [0.6, 0.4, 0.7, 0.3], 2/3, 0.6875),
        # Test pandas Series
        (pd.Series([1, 0, 1, 1]), pd.Series([0.6, 0.4, 0.7, 0.3]), 2/3, 0.6875),
        # Test pandas DataFrame (single-column)
        (pd.DataFrame({"label": [1, 0, 1, 1]}), pd.DataFrame({"score": [0.6, 0.4, 0.7, 0.3]}),
            2/3, 0.6875),
    ],
)
def test_streaming_aucroc(aucroc_streaming, true_labels, pred_scores,
                          expected_intermediate, expected_final):
    """Test streaming AUCROC calculation for different input types."""
    assert aucroc_streaming.do_score(true_labels, pred_scores) == \
        pytest.approx(expected_intermediate)

    # Second batch
    new_true_labels = np.array([1, 0])
    new_pred_scores = np.array([0.85, 0.6])

    if isinstance(true_labels, pd.DataFrame):
        new_true_labels = pd.DataFrame({"label": new_true_labels})
        new_pred_scores = pd.DataFrame({"score": new_pred_scores})

    assert aucroc_streaming.do_score(new_true_labels, new_pred_scores) == \
        pytest.approx(expected_final)

def test_empty_inputs(aucroc_batch):
    """Test behavior when empty inputs are provided."""
    empty_inputs = [np.array([]), [], pd.Series([], dtype=int), pd.DataFrame({"label": []})]

    for empty in empty_inputs:
        assert aucroc_batch.do_score(empty, empty) == 0.0

def test_mismatched_lengths(aucroc_batch):
    """Test that ValueError is raised for mismatched input lengths."""
    mismatched_cases = [
        (np.array([1, 0, 1]), np.array([0.9, 0.2])),
        ([1, 0, 1], [0.9, 0.2]),
        (pd.Series([1, 0, 1]), pd.Series([0.9, 0.2])),
        (pd.DataFrame({"label": [1, 0, 1]}), pd.DataFrame({"score": [0.9, 0.2]})),
    ]

    for true_labels, pred_scores in mismatched_cases:
        with pytest.raises(ValueError, match="The length of true_labels and " + \
                           "pred_probs must be the same."):
            aucroc_batch.do_score(true_labels, pred_scores)

@pytest.mark.parametrize(
    "true_labels, pred_scores",
    [
        ("invalid", [0.9, 0.2, 0.8]),
        ([1, 0, 1], "invalid"),
        (None, [0.9, 0.2, 0.8]),
        ([1, 0, 1], None),
        ({1, 0, 1}, [0.9, 0.2, 0.8]),  # Set type (unsupported)
    ],
)
def test_invalid_input_types(aucroc_batch, true_labels, pred_scores):
    """Test that TypeError is raised for unsupported input types."""
    with pytest.raises(TypeError):
        aucroc_batch.do_score(true_labels, pred_scores)
