"""
Unit tests for the Precision class.

Author: Rohith Ravindranath
Date: Feb 4, 2025
"""

import pytest
import numpy as np
import pandas as pd
from mlmonkit.metrics.classification.precision import Precision

@pytest.fixture
def precision_batch():
    """Fixture to initialize a Precision instance in batch mode."""
    return Precision(use_streaming=False)

@pytest.fixture
def precision_streaming():
    """Fixture to initialize a Precision instance in streaming mode."""
    return Precision(use_streaming=True)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_precision",
    [
        # Test NumPy arrays (Precision = 2/3)
        (np.array([1, 0, 1, 1]), np.array([1, 1, 0, 1]), 2 / 3),

        # Test Python lists (Precision = 3/4)
        ([1, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1], 3 / 4),  

        # Test pandas Series (Precision = 2/3)
        (pd.Series([1, 0, 1, 1]), pd.Series([0, 1, 1, 1]), 2 / 3), 

        # Test pandas DataFrame (single-column) (Precision = 1/2)
        (pd.DataFrame({"label": [1, 0, 1, 1]}),
         pd.DataFrame({"label": [0, 1, 1, 0]}), 1 / 2),
    ]
)
def test_batch_precision(precision_batch, true_labels, pred_labels, expected_precision):
    """Test batch precision calculation for different input types."""
    assert precision_batch.do_score(true_labels, pred_labels) == pytest.approx(expected_precision)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_intermediate, expected_final",
    [
        # Test NumPy arrays
        (np.array([1, 0, 1]), np.array([1, 1, 0]), 1 / 2, 1 / 2),
        # Test Python lists
        ([1, 0, 1], [1, 1, 0], 1 / 2, 1 / 2),
        # Test pandas Series
        (pd.Series([1, 0, 1]), pd.Series([1, 1, 0]), 1 / 2, 1 / 2),
        # Test pandas DataFrame (single-column)
        (pd.DataFrame({"label": [1, 0, 1]}), pd.DataFrame({"label": [1, 1, 0]}), 1 / 2, 1 / 2),
    ],
)
def test_streaming_precision(precision_streaming, true_labels, pred_labels,
                             expected_intermediate, expected_final):
    """Test streaming precision calculation for different input types."""
    assert precision_streaming.do_score(true_labels, pred_labels) == \
        pytest.approx(expected_intermediate)

    # Second batch
    new_true_labels = np.array([1, 0])
    new_pred_labels = np.array([1, 1])
    assert precision_streaming.do_score(new_true_labels, new_pred_labels) == \
        pytest.approx(expected_final)

def test_empty_inputs(precision_batch):
    """Test behavior when empty inputs are provided."""
    empty_inputs = [np.array([]), [], pd.Series([], dtype=int), pd.DataFrame({"label": []})]

    for empty in empty_inputs:
        assert precision_batch.do_score(empty, empty) == 0.0

def test_mismatched_lengths(precision_batch):
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
            precision_batch.do_score(true_labels, pred_labels)

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
def test_invalid_input_types(precision_batch, true_labels, pred_labels):
    """Test that TypeError is raised for unsupported input types."""
    with pytest.raises(TypeError):
        precision_batch.do_score(true_labels, pred_labels)
