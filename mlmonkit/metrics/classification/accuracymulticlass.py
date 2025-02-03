"""
Author: Rohith Ravindranath
Jan 30th 2025
"""

import numpy as np
import pandas as pd
import warnings
from mlmonkit.metrics.utils import _convert_to_numpy
from mlmonkit.metrics.metric import BaseMetric

class AccuracyMultiClass(BaseMetric):
    """
    Class to calculate accuracy for multi-class classification in both batch and 
    streaming evaluation modes.
    
    Attributes:
    use_streaming (bool): If True, streaming evaluation is enabled; otherwise, 
                            batch evaluation is used.
    correct (int): The number of correct predictions seen so far (used in streaming mode).
    total (int): The total number of predictions seen so far (used in streaming mode).
    """

    def __init__(self, name='AccuracyMultiClass', use_streaming=False):
        """
        Initializes the AccuracyMultiClass class.
        
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
        Evaluates accuracy in batch mode for multi-class classification.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels
                                                                            of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels 
                                                                        of the dataset.
        
        Returns:
        float: Accuracy score, calculated as the proportion of correct predictions.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same shape.
        """
        if true_labels.shape != pred_labels.shape:
            raise ValueError("The shape of true_labels and pred_labels must be the same.")

        if len(true_labels) == 0 or len(pred_labels) == 0:
            warnings.warn("Either true_labels or pred_labels is empty. Returning 0.0.")
            return 0.0

        # Compute the number of correct predictions
        correct_predictions = np.sum(true_labels == pred_labels)

        # Calculate overall accuracy
        return correct_predictions / len(true_labels)

    def evaluate_streaming(self, true_labels, pred_labels):
        """
        Evaluates accuracy incrementally in streaming mode for multi-class classification.
        
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
        if len(true_labels) == 0 or len(pred_labels) == 0:
            warnings.warn("Either true_labels or pred_labels is empty")
            if self.total == 0:
                return 0.0
            return self.correct / self.total

        if true_labels.shape != pred_labels.shape:
            raise ValueError("The shape of true_labels and pred_labels must be the same.")

        # Incrementally update the correct predictions and total predictions
        correct_predictions = np.sum(true_labels == pred_labels)
        self.correct += correct_predictions
        self.total += len(true_labels)

        # Handle potential divide-by-zero scenario
        if self.total == 0:
            return 0.0  # Prevent division by zero

        # Return the updated accuracy
        return self.correct / self.total

    def do_score(self, true_labels, pred_labels):
        """
        Updates the accuracy calculation for multi-class classification based on 
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

        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        if true_labels.shape != pred_labels.shape:
            raise ValueError("The shape of true_labels and pred_labels must be the same.")

        if self.use_streaming:
            return self.evaluate_streaming(true_labels, pred_labels)
        else:
            return self.evaluate_batch(true_labels, pred_labels)
