"""
Unit tests for the F1 class.

Author: Rohith Ravindranath
Date: Jan 30, 2025
"""

import pytest
import numpy as np
import pandas as pd
from mlmonkit.metrics.classification.f1 import F1

@pytest.fixture
def f1_batch():
    """Fixture to initialize an F1 instance in batch mode."""
    return F1(use_streaming=False)

@pytest.fixture
def f1_streaming():
    """Fixture to initialize an F1 instance in streaming mode."""
    return F1(use_streaming=True)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_f1",
    [
        # Test NumPy arrays
        (np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1]), 0.8),
        # Test Python lists
        ([1, 0, 1, 1], [1, 0, 0, 1], 0.8),
        # Test pandas Series
        (pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 1]), 0.8),
        # Test pandas DataFrame (single-column)
        (pd.DataFrame({"label": [1, 0, 1, 1]}), pd.DataFrame({"label": [1, 0, 0, 1]}), 0.8),
    ],
)
def test_batch_f1(f1_batch, true_labels, pred_labels, expected_f1):
    """Test batch F1 score calculation for different input types."""
    assert f1_batch.do_score(true_labels, pred_labels) == pytest.approx(expected_f1)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_intermediate, expected_final",
    [
        # Test NumPy arrays
        (np.array([1, 0, 1]), np.array([1, 0, 0]), 2/3, 2/3),
        # Test Python lists
        ([1, 0, 1], [1, 0, 0], 2/3, 2/3),
        # Test pandas Series
        (pd.Series([1, 0, 1]), pd.Series([1, 0, 0]), 2/3, 2/3),
        # Test pandas DataFrame (single-column)
        (pd.DataFrame({"label": [1, 0, 1]}), pd.DataFrame({"label": [1, 0, 0]}), 2/3, 2/3),
    ],
)
def test_streaming_f1(f1_streaming, true_labels, pred_labels, expected_intermediate, expected_final):
    """Test streaming F1 score calculation for different input types."""
    assert f1_streaming.do_score(true_labels, pred_labels) == pytest.approx(expected_intermediate)

    # Second batch
    new_true_labels = np.array([1, 0])
    new_pred_labels = np.array([1, 1])
    assert f1_streaming.do_score(new_true_labels, new_pred_labels) == pytest.approx(expected_final)

def test_empty_inputs(f1_batch):
    """Test behavior when empty inputs are provided."""
    empty_inputs = [np.array([]), [], pd.Series([], dtype=int), pd.DataFrame({"label": []})]
    
    for empty in empty_inputs:
        assert f1_batch.do_score(empty, empty) == 0.0

def test_mismatched_lengths(f1_batch):
    """Test that ValueError is raised for mismatched input lengths."""
    mismatched_cases = [
        (np.array([1, 0, 1]), np.array([1, 0])),
        ([1, 0, 1], [1, 0]),
        (pd.Series([1, 0, 1]), pd.Series([1, 0])),
        (pd.DataFrame({"label": [1, 0, 1]}), pd.DataFrame({"label": [1, 0]})),
    ]
    
    for true_labels, pred_labels in mismatched_cases:
        with pytest.raises(ValueError, match="The length of true_labels and " +
                           "pred_labels must be the same."):
            f1_batch.do_score(true_labels, pred_labels)

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
def test_invalid_input_types(f1_batch, true_labels, pred_labels):
    """Test that TypeError is raised for unsupported input types."""
    with pytest.raises(TypeError):
        f1_batch.do_score(true_labels, pred_labels)
