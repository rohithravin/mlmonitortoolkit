"""
Unit tests for the Recall class.

Author: Rohith Ravindranath
Date: Jan 30, 2025
"""

import pytest
import numpy as np
import pandas as pd
from mlmonkit.metrics.classification.recall import Recall

@pytest.fixture
def recall_batch():
    """Fixture to initialize a Recall instance in batch mode."""
    return Recall(use_streaming=False)

@pytest.fixture
def recall_streaming():
    """Fixture to initialize a Recall instance in streaming mode."""
    return Recall(use_streaming=True)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_recall",
    [
        # Test NumPy arrays (Recall = 2/3)
        (np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1]), 2 / 3),

        # Test Python lists (Recall = 3/4)
        ([1,1,1,0,1,0], [1,1,0,0,1,0], 3 / 4),  # More true positives

        # Test pandas Series (Recall = 1/2)
        (pd.Series([1, 0, 1, 1]), pd.Series([0, 0, 1, 1]), 2 / 3),  # One true positive missed

        # Test pandas DataFrame (single-column) (Recall = 1/3)
        (pd.DataFrame({"label": [1, 0, 1, 1]}), 
         pd.DataFrame({"label": [0, 0, 1, 0]}), 1 / 3),  # Only one true positive
    ]
)
def test_batch_recall(recall_batch, true_labels, pred_labels, expected_recall):
    """Test batch recall calculation for different input types."""
    assert recall_batch.do_score(true_labels, pred_labels) == pytest.approx(expected_recall)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_intermediate, expected_final",
    [
        # Test NumPy arrays
        (np.array([1, 0, 1]), np.array([1, 0, 0]), 1 / 2, 2 / 3),
        # Test Python lists
        ([1, 0, 1], [1, 0, 0], 1 / 2, 2 / 3),
        # Test pandas Series
        (pd.Series([1, 0, 1]), pd.Series([1, 0, 0]), 1 / 2, 2 / 3),
        # Test pandas DataFrame (single-column)
        (pd.DataFrame({"label": [1, 0, 1]}), pd.DataFrame({"label": [1, 0, 0]}), 1 / 2, 2 / 3),
    ],
)
def test_streaming_recall(recall_streaming, true_labels, pred_labels,
                          expected_intermediate, expected_final):
    """Test streaming recall calculation for different input types."""
    assert recall_streaming.do_score(true_labels, pred_labels) == \
        pytest.approx(expected_intermediate)

    # Second batch
    new_true_labels = np.array([1, 0])
    new_pred_labels = np.array([1, 1])
    assert recall_streaming.do_score(new_true_labels, new_pred_labels) == \
        pytest.approx(expected_final)

def test_empty_inputs(recall_batch):
    """Test behavior when empty inputs are provided."""
    empty_inputs = [np.array([]), [], pd.Series([], dtype=int), pd.DataFrame({"label": []})]

    for empty in empty_inputs:
        assert recall_batch.do_score(empty, empty) == 0.0

def test_mismatched_lengths(recall_batch):
    """Test that ValueError is raised for mismatched input lengths."""
    mismatched_cases = [
        (np.array([1, 0, 1]), np.array([1, 0])),
        ([1, 0, 1], [1, 0]),
        (pd.Series([1, 0, 1]), pd.Series([1, 0])),
        (pd.DataFrame({"label": [1, 0, 1]}), pd.DataFrame({"label": [1, 0]})),
    ]

    for true_labels, pred_labels in mismatched_cases:
        with pytest.raises(ValueError, match="The length of true_labels and " + \
                           "pred_labels must be the same."):
            recall_batch.do_score(true_labels, pred_labels)

@pytest.mark.parametrize(
    "true_labels, pred_labels",
    [
        ("invalid", [1, 0, 1]),
        ([1, 0, 1], "invalid"),
        (None, [1, 0, 1]),
        ([1, 0, 1], None),
        ({1, 0, 1}, [1, 0, 1]),  # Set type (unsupported)
    ],
)
def test_invalid_input_types(recall_batch, true_labels, pred_labels):
    """Test that TypeError is raised for unsupported input types."""
    with pytest.raises(TypeError):
        recall_batch.do_score(true_labels, pred_labels)
