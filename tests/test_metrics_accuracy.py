"""
Unit tests for the Accuracy class.

Author: Rohith Ravindranath
Date: Jan 30, 2025
"""

import pytest
import numpy as np
import pandas as pd
from mlmonkit.metrics.classification.accuracy import Accuracy

@pytest.fixture
def accuracy_batch():
    """Fixture to initialize an Accuracy instance in batch mode."""
    return Accuracy(use_streaming=False)

@pytest.fixture
def accuracy_streaming():
    """Fixture to initialize an Accuracy instance in streaming mode."""
    return Accuracy(use_streaming=True)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_accuracy",
    [
        # Test NumPy arrays
        (np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1]), 3 / 4),
        # Test Python lists
        ([1, 0, 1, 1], [1, 0, 0, 1], 3 / 4),
        # Test pandas Series
        (pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 1]), 3 / 4),
        # Test pandas DataFrame (single-column)
        (pd.DataFrame({"label": [1, 0, 1, 1]}), pd.DataFrame({"label": [1, 0, 0, 1]}), 3 / 4),
    ],
)
def test_batch_accuracy(accuracy_batch, true_labels, pred_labels, expected_accuracy):
    """Test batch accuracy calculation for different input types."""
    assert accuracy_batch.update(true_labels, pred_labels) == pytest.approx(expected_accuracy)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_intermediate, expected_final",
    [
        # Test NumPy arrays
        (np.array([1, 0, 1]), np.array([1, 0, 0]), 2 / 3, 3 / 5),
        # Test Python lists
        ([1, 0, 1], [1, 0, 0], 2 / 3, 3 / 5),
        # Test pandas Series
        (pd.Series([1, 0, 1]), pd.Series([1, 0, 0]), 2 / 3, 3 / 5),
        # Test pandas DataFrame (single-column)
        (pd.DataFrame({"label": [1, 0, 1]}), pd.DataFrame({"label": [1, 0, 0]}), 2 / 3, 3 / 5),
    ],
)
def test_streaming_accuracy(accuracy_streaming, true_labels, pred_labels,
                            expected_intermediate, expected_final):
    """Test streaming accuracy calculation for different input types."""
    assert accuracy_streaming.update(true_labels, pred_labels) == \
        pytest.approx(expected_intermediate)

    # Second batch
    new_true_labels = np.array([1, 0])
    new_pred_labels = np.array([1, 1])
    assert accuracy_streaming.update(new_true_labels, new_pred_labels) == \
        pytest.approx(expected_final)

def test_empty_inputs(accuracy_batch):
    """Test behavior when empty inputs are provided."""
    empty_inputs = [np.array([]), [], pd.Series([], dtype=int), pd.DataFrame({"label": []})]

    for empty in empty_inputs:
        assert accuracy_batch.update(empty, empty) == 0.0

def test_mismatched_lengths(accuracy_batch):
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
            accuracy_batch.update(true_labels, pred_labels)

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
def test_invalid_input_types(accuracy_batch, true_labels, pred_labels):
    """Test that TypeError is raised for unsupported input types."""
    with pytest.raises(TypeError):
        accuracy_batch.update(true_labels, pred_labels)
