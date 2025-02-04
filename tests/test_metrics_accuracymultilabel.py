"""
Unit tests for the AccuracyMultiLabel class.

Author: Rohith Ravindranath
Date: Jan 30, 2025
"""

import pytest
import warnings
import numpy as np
import pandas as pd
from mlmonkit.metrics.classification.accuracymultilabel import AccuracyMultiLabel

@pytest.fixture
def accuracy_multilabel_batch():
    """Fixture to initialize an AccuracyMultiLabel instance in batch mode."""
    return AccuracyMultiLabel(use_streaming=False)

@pytest.fixture
def accuracy_multilabel_streaming():
    """Fixture to initialize an AccuracyMultiLabel instance in streaming mode."""
    return AccuracyMultiLabel(use_streaming=True)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_accuracy",
    [
        # Test NumPy arrays
        (np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]]),
         np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]]), 2 / 3),
        # Test Python lists
        ([[1, 0, 1], [1, 1, 0], [0, 1, 0]], [[1, 0, 1], [1, 1, 0], [0, 0, 0]], 2 / 3),
        # Test pandas Series
        (pd.DataFrame([[1, 0, 1], [1, 1, 0], [0, 1, 0]]),
         pd.DataFrame([[1, 0, 1], [1, 1, 0], [0, 0, 0]]), 2 / 3),
        # Test pandas DataFrame (multi-column)
        (pd.DataFrame({"label1": [1, 0, 1], "label2": [1, 1, 0], "label3": [0, 1, 0]}),
         pd.DataFrame({"label1": [1, 0, 1], "label2": [1, 1, 0], "label3": [0, 0, 0]}), 2 / 3),
    ],
)
def test_batch_accuracy_multilabel(accuracy_multilabel_batch, true_labels,
                                   pred_labels, expected_accuracy):
    """Test batch accuracy calculation for multi-label classification with different input types."""
    assert accuracy_multilabel_batch.do_score(true_labels, pred_labels) == \
        pytest.approx(expected_accuracy)

@pytest.mark.parametrize(
    "true_labels, pred_labels, expected_intermediate, expected_final",
    [
        # Test NumPy arrays
        (np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]]),
         np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]]), 2 / 3, 3 / 5),
        # Test Python lists
        ([[1, 0, 1], [1, 1, 0], [0, 1, 0]], [[1, 0, 1], [1, 1, 0], [0, 0, 0]],  2 / 3, 3 / 5),
        # Test pandas Series
        (pd.DataFrame([[1, 0, 1], [1, 1, 0], [0, 1, 0]]),
         pd.DataFrame([[1, 0, 1], [1, 1, 0], [0, 0, 0]]),  2 / 3, 3 / 5),
        # Test pandas DataFrame (multi-column)
        (pd.DataFrame({"label1": [1, 0, 1], "label2": [1, 1, 0], "label3": [0, 1, 0]}),
         pd.DataFrame({"label1": [1, 0, 1], "label2": [1, 1, 0], "label3": [0, 0, 0]}),
          2 / 3, 3 / 5),
    ],
)
def test_streaming_accuracy_multilabel(accuracy_multilabel_streaming, true_labels, pred_labels,
                                       expected_intermediate, expected_final):
    """Test streaming accuracy calculation for multi-label classification 
        with different input types."""
    assert accuracy_multilabel_streaming.do_score(true_labels, pred_labels) == \
        pytest.approx(expected_intermediate)

    # Second batch
    new_true_labels = np.array([[1, 0, 1], [1, 0, 0]])
    new_pred_labels = np.array([[1, 0, 1], [1, 1, 0]])
    assert accuracy_multilabel_streaming.do_score(new_true_labels, new_pred_labels) == \
        pytest.approx(expected_final)

def test_empty_inputs(accuracy_multilabel_batch):
    """Test behavior when empty inputs are provided for multi-label."""
    empty_inputs = [np.array([]), [], pd.DataFrame({"label1": [], "label2": []})]

    for empty in empty_inputs:
        with pytest.warns(Warning,
                          match="Either true_labels or pred_labels is empty. Returning 0.0."):
            assert accuracy_multilabel_batch.do_score(empty, empty) == 0.0

def test_mismatched_lengths(accuracy_multilabel_batch):
    """Test that ValueError is raised for mismatched input lengths in multi-label classification."""
    mismatched_cases = [
        (np.array([[1, 0], [1, 1]]), np.array([[1, 0]])),
        ([[1, 0], [1, 1]], [[1, 0]]),
        (pd.DataFrame({"label1": [1, 0]}), pd.DataFrame({"label1": [1]})),
    ]

    for true_labels, pred_labels in mismatched_cases:
        with pytest.raises(ValueError, match="The shape of true_labels and " + \
                           "pred_labels must be the same."):
            accuracy_multilabel_batch.do_score(true_labels, pred_labels)

@pytest.mark.parametrize(
    "true_labels, pred_labels",
    [
        ("invalid", [[1, 0, 1]]),
        ([[1, 0, 1]], "invalid"),
        (None, [[1, 0, 1]]),
        ([[1, 0, 1]], None),
        ({1, 0, 1}, [[1, 0, 1]]),  # Set type (unsupported)
    ],
)
def test_invalid_input_types(accuracy_multilabel_batch, true_labels, pred_labels):
    """Test that TypeError is raised for unsupported input types in multi-label classification."""
    with pytest.raises(TypeError):
        accuracy_multilabel_batch.do_score(true_labels, pred_labels)
