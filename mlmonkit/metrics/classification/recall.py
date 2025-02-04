""" 
Author: Rohith Ravindranath
Jan 30th 2025
"""
import numpy as np
import pandas as pd
from mlmonkit.metrics.utils import _convert_to_numpy
from mlmonkit.metrics.metric import BaseMetric

class Recall(BaseMetric):
    """
    Class to calculate recall for both batch and streaming evaluation modes.
    
    Attributes:
    use_streaming (bool): If True, streaming evaluation is enabled; otherwise, 
                          batch evaluation is used.
    true_positive (int): The number of true positive predictions seen so 
    far (used in streaming mode).
    total_positive (int): The total number of actual positive cases seen so 
    far (used in streaming mode).
    """

    def __init__(self, name='Recall', use_streaming=False):
        """
        Initializes the Recall class.
        
        Parameters:
        name (str): Name of the metric.
        use_streaming (bool): If True, enables streaming (incremental) mode.
        """
        super().__init__(name)
        self.use_streaming = use_streaming
        if self.use_streaming:
            self.true_positive = 0  # Number of true positives
            self.total_positive = 0  # Total actual positives

    def evaluate_batch(self, true_labels, pred_labels):
        """
        Evaluates recall in batch mode by comparing true and predicted labels.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            True labels of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            Predicted labels of the dataset.
        
        Returns:
        float: Recall score, calculated as true positives divided by total actual positives.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        """
        if len(true_labels) == 0:
            return 0.0  # Prevent NaN result for empty input

        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        true_positive = np.sum((true_labels == 1) & (pred_labels == 1))
        total_positive = np.sum(true_labels == 1)

        return true_positive / total_positive if total_positive > 0 else 0.0

    def evaluate_streaming(self, true_labels, pred_labels):
        """
        Evaluates recall incrementally in streaming mode.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            True labels of incoming data batch.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            Predicted labels of incoming data batch.
        
        Returns:
        float: Updated recall score, calculated incrementally.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        """
        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        self.true_positive += np.sum((true_labels == 1) & (pred_labels == 1))
        self.total_positive += np.sum(true_labels == 1)

        return self.true_positive / self.total_positive if self.total_positive > 0 else 0.0

    def do_score(self, true_labels, pred_labels):
        """
        Updates the recall calculation based on the selected mode (batch or streaming).
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            True labels of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            Predicted labels of the dataset.
        
        Returns:
        float: Recall score based on the mode selected (batch or streaming).
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        TypeError: If true_labels or pred_labels are not of the expected types.
        """
        if not isinstance(true_labels, (list, np.ndarray, pd.Series, pd.DataFrame)) \
            or not isinstance(pred_labels, (list, np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("Both true_labels and pred_labels should be of type list, "+\
                            "pandas.Series, pandas.DataFrame, or numpy.ndarray.")

        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        if self.use_streaming:
            return self.evaluate_streaming(true_labels, pred_labels)
        else:
            return self.evaluate_batch(true_labels, pred_labels)
