""" 
Author: Rohith Ravindranath
Jan 30th 2025
"""
import numpy as np
import pandas as pd
from mlmonkit.metrics.utils import _convert_to_numpy
from mlmonkit.metrics.metric import BaseMetric

class Accuracy(BaseMetric):
    """
    Class to calculate accuracy for both batch and streaming evaluation modes.
    
    Attributes:
    use_streaming (bool): If True, streaming evaluation is enabled; otherwise, 
                            batch evaluation is used.
    correct (int): The number of correct predictions seen so far (used in streaming mode).
    total (int): The total number of predictions seen so far (used in streaming mode).
    """

    def __init__(self, name = 'Accuracy', use_streaming=False):
        """
        Initializes the StandardizedAccuracy class.
        
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
        Evaluates accuracy in batch mode by comparing true and predicted labels.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels
                                                                            of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels 
                                                                        of the dataset.
        
        Returns:
        float: Accuracy score, calculated as the proportion of correct predictions.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        """
        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        if len(true_labels) == 0:
            return 0.0  # Prevent NaN result for empty input

        # Check for input length mismatch
        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        # Calculate accuracy as the proportion of correct predictions
        return np.mean(true_labels == pred_labels)

    def evaluate_streaming(self, true_labels, pred_labels):
        """
        Evaluates accuracy incrementally in streaming mode.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels of the 
                                                                        incoming data batch.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels of 
                                                                        the incoming data batch.

        Returns:
        float: Updated accuracy score, calculated incrementally.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        TypeError: If true_labels or pred_labels are not of the expected types 
                    (list, numpy array, or pandas).
        """
        # Convert labels to numpy arrays
        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        # Check for input length mismatch
        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        # Incrementally update the correct predictions and total predictions
        self.correct += np.sum(true_labels == pred_labels)
        self.total += len(true_labels)

        # Handle potential divide-by-zero scenario
        if self.total == 0:
            return 0.0  # Prevent division by zero

        # Return the updated accuracy
        return self.correct / self.total

    def do_score(self, true_labels, pred_labels):
        """
        Updates the accuracy calculation based on the selected mode (batch or streaming).
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels 
                                                                            of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels 
                                                                            of the dataset.

        Returns:
        float: Accuracy score based on the mode selected (batch or streaming).

        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        TypeError: If true_labels or pred_labels are not of the expected types 
                    (list, numpy array, or pandas).
        """
        # Validate input types and check lengths
        if not isinstance(true_labels, (list, np.ndarray, pd.Series, pd.DataFrame)) \
            or not isinstance(pred_labels, (list, np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("Both true_labels and pred_labels should be of type list, \
                            pandas.Series, pandas.DataFrame, or numpy.ndarray.")

        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        if self.use_streaming:
            return self.evaluate_streaming(true_labels, pred_labels)
        else:
            return self.evaluate_batch(true_labels, pred_labels)
