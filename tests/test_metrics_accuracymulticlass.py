"""
Unit tests for the AccuracyMultiClass class.

Author: Rohith Ravindranath
Date: Feb 2, 2025
"""

import pytest
import warnings
import numpy as np
import pandas as pd
from mlmonkit.metrics.classification.accuracymulticlass import AccuracyMultiClass

@pytest.fixture
def accuracy_multiclass_batch():
    """Fixture to initialize an AccuracyMultiClass instance in batch mode."""
    return AccuracyMultiClass(use_streaming=False)

@pytest.fixture
def accuracy_multiclass_streaming():
    """Fixture to initialize an AccuracyMultiClass instance in streaming mode."""
    return AccuracyMultiClass(use_streaming=True)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_accuracy",
    [
        # Test NumPy arrays
        (np.array([0, 1, 2]),
         np.array([0, 1, 2]), 1.0),
        # Test Python lists
        ([0, 1, 2], [0, 1, 2], 1.0),
        # Test pandas Series
        (pd.Series([0, 1, 2]),
         pd.Series([0, 1, 2]), 1.0),
        # Test pandas DataFrame (single column for multi-class labels)
        (pd.DataFrame({"label": [0, 1, 2]}),
         pd.DataFrame({"label": [0, 1, 2]}), 1.0),
        # Test incorrect prediction (for partial match)
        (np.array([0, 1, 2]),
         np.array([0, 1, 1]), 2 / 3),
    ],
)
def test_batch_accuracy_multiclass(accuracy_multiclass_batch, true_labels,
                                   pred_labels, expected_accuracy):
    """Test batch accuracy calculation for multi-class classification with different input types."""
    assert accuracy_multiclass_batch.do_score(true_labels, pred_labels) == \
        pytest.approx(expected_accuracy)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_intermediate, expected_final",
    [
        # Test NumPy arrays
        (np.array([0, 1, 2]),
         np.array([0, 1, 2]), 1.0, 4/5),
        # Test Python lists
        ([0, 1, 2], [0, 1, 2], 1.0, 4/5),
        # Test pandas Series
        (pd.Series([0, 1, 2]),
         pd.Series([0, 1, 2]), 1.0, 4/5),
        # Test pandas DataFrame (single column for multi-class labels)
        (pd.DataFrame({"label": [0, 1, 2]}),
         pd.DataFrame({"label": [0, 1, 2]}), 1.0, 4/5),
    ],
)
def test_streaming_accuracy_multiclass(accuracy_multiclass_streaming, true_labels, pred_labels,
                                       expected_intermediate, expected_final):
    """Test streaming accuracy calculation for multi-class 
            classification with different input types."""
    assert accuracy_multiclass_streaming.do_score(true_labels, pred_labels) == \
        pytest.approx(expected_intermediate)

    # Second batch with a mistake to reduce accuracy
    new_true_labels = np.array([1, 0])
    new_pred_labels = np.array([1, 2])  # Introducing an error here (second label is incorrect)
    assert accuracy_multiclass_streaming.do_score(new_true_labels, new_pred_labels) == \
        pytest.approx(expected_final)


def test_empty_inputs(accuracy_multiclass_batch):
    """Test behavior when empty inputs are provided for multi-class."""
    empty_inputs = [np.array([]), [], pd.DataFrame({"label": []})]

    for empty in empty_inputs:
        with pytest.warns(Warning,
                          match="Either true_labels or pred_labels is empty. Returning 0.0."):
            assert accuracy_multiclass_batch.do_score(empty, empty) == 0.0

def test_mismatched_lengths(accuracy_multiclass_batch):
    """Test that ValueError is raised for mismatched input lengths in multi-class classification."""
    mismatched_cases = [
        (np.array([0, 1]), np.array([0])),
        ([0, 1], [0]),
        (pd.DataFrame({"label": [0, 1]}), pd.DataFrame({"label": [0]})),
    ]

    for true_labels, pred_labels in mismatched_cases:
        with pytest.raises(ValueError, match="The shape of true_labels and " + \
                           "pred_labels must be the same."):
            accuracy_multiclass_batch.do_score(true_labels, pred_labels)

@pytest.mark.parametrize(
    "true_labels, pred_labels",
    [
        ("invalid", [0, 1, 2]),
        ([0, 1, 2], "invalid"),
        (None, [0, 1, 2]),
        ([0, 1, 2], None),
        ({0, 1, 2}, [0, 1, 2]),  # Set type (unsupported)
    ],
)
def test_invalid_input_types(accuracy_multiclass_batch, true_labels, pred_labels):
    """Test that TypeError is raised for unsupported input types in multi-class classification."""
    with pytest.raises(TypeError):
        accuracy_multiclass_batch.do_score(true_labels, pred_labels)
