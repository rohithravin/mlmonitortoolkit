""" 
Author: Rohith Ravindranath
Jan 30th 2025
"""

import numpy as np
import pandas as pd
import warnings
from mlmonkit.metrics.utils import _convert_to_numpy
from mlmonkit.metrics.metric import BaseMetric

class AccuracyMultiLabel(BaseMetric):
    """
    Class to calculate accuracy for multi-label classification in both batch and 
    streaming evaluation modes.
    
    Attributes:
    use_streaming (bool): If True, streaming evaluation is enabled; otherwise, 
                            batch evaluation is used.
    correct (int): The number of correct predictions seen so far (used in streaming mode).
    total (int): The total number of predictions seen so far (used in streaming mode).
    """

    def __init__(self, name='AccuracyMultiLabel', use_streaming=False):
        """
        Initializes the AccuracyMultiLabel class.
        
        Parameters:
        name (str): Name of the metric.
        use_streaming (bool): If True, enables streaming (incremental) mode.
        """
        super().__init__(name)
        self.use_streaming = use_streaming
        if self.use_streaming:
            self.correct = 0  # Number of correct predictions
            self.total = 0  # Total number of predictions made
        else:
            pass

    def evaluate_batch(self, true_labels, pred_labels):
        """
        Evaluates accuracy in batch mode for multi-label classification.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels
                                                                            of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels 
                                                                        of the dataset.
        
        Returns:
        float: Accuracy score, calculated as the proportion of correctly predicted 
        labels for each instance.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same shape.
        """
        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        if true_labels.shape != pred_labels.shape:
            raise ValueError("The shape of true_labels and pred_labels must be the same.")

        if len(true_labels) == 0 or len(pred_labels) == 0:
            warnings.warn("Either true_labels or pred_labels is empty. Returning 0.0.")
            return 0.0

        # Compute the accuracy per instance, treating partial matches as 0
        accuracy_per_instance = np.all(true_labels == pred_labels, axis=1).astype(int)

        # Calculate overall accuracy as the mean of instance accuracies
        return np.mean(accuracy_per_instance)

    def evaluate_streaming(self, true_labels, pred_labels):
        """
        Evaluates accuracy incrementally in streaming mode for multi-label classification.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels of the 
                                                                        incoming data batch.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels of 
                                                                        the incoming data batch.

        Returns:
        float: Updated accuracy score, calculated incrementally.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same shape.
        TypeError: If true_labels or pred_labels are not of the expected types 
                    (list, numpy array, or pandas).
        """
        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        if len(true_labels) == 0 or len(pred_labels) == 0:
            warnings.warn("Either true_labels or pred_labels is empty")
            if self.total == 0:
                return 0.0
            return self.correct / self.total

        if true_labels.shape != pred_labels.shape:
            raise ValueError("The shape of true_labels and pred_labels must be the same.")

        # Incrementally update the correct predictions and total predictions
        correct_predictions = np.all(true_labels == pred_labels, axis=1).astype(int)
        self.correct += np.sum(correct_predictions)
        self.total += len(true_labels)

        print(f"correct: {self.correct}")
        print(f'total: {self.total}')

        # Handle potential divide-by-zero scenario
        if self.total == 0:
            return 0.0  # Prevent division by zero

        # Return the updated accuracy
        return self.correct / self.total

    def do_score(self, true_labels, pred_labels):
        """
        Updates the accuracy calculation for multi-label classification based on 
        the selected mode (batch or streaming).
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels 
                                                                            of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels 
                                                                            of the dataset.

        Returns:
        float: Accuracy score based on the mode selected (batch or streaming).

        Raises:
        ValueError: If true_labels and pred_labels do not have the same shape.
        TypeError: If true_labels or pred_labels are not of the expected types 
                    (list, numpy array, or pandas).
        """
        # Validate input types and check shapes
        if not isinstance(true_labels, (list, np.ndarray, pd.Series, pd.DataFrame)) \
            or not isinstance(pred_labels, (list, np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("Both true_labels and pred_labels should be of type list, \
                            pandas.Series, pandas.DataFrame, or numpy.ndarray.")

        if isinstance(true_labels, list) and isinstance(pred_labels, list):
            # Check if they have the same number of rows and columns
            shape1 = (len(true_labels), len(true_labels[0])) if true_labels else (0, 0)
            shape2 = (len(pred_labels), len(pred_labels[0])) if pred_labels else (0, 0)
            print(f'shape1: {shape1}, shape2: {shape2}')
            are_shapes_same = shape1 == shape2
            if not are_shapes_same:
                raise ValueError("The shape of true_labels and pred_labels must be the same.")
        elif true_labels.shape != pred_labels.shape:
            raise ValueError("The shape of true_labels and pred_labels must be the same.")

        if self.use_streaming:
            return self.evaluate_streaming(true_labels, pred_labels)
        else:
            return self.evaluate_batch(true_labels, pred_labels)
