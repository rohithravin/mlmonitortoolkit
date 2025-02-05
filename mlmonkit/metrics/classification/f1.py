""" 
Author: Rohith Ravindranath
Jan 30th 2025
"""

import numpy as np
import pandas as pd
from mlmonkit.metrics.utils import _convert_to_numpy
from mlmonkit.metrics.metric import BaseMetric
from sklearn.metrics import f1_score

class F1Score(BaseMetric):
    """
    Class to calculate F1 score for both batch and streaming evaluation modes.
    
    Attributes:
    use_streaming (bool): If True, streaming evaluation is enabled; otherwise, 
                            batch evaluation is used.
    TP (int): The number of true positives seen so far (used in streaming mode).
    FP (int): The number of false positives seen so far (used in streaming mode).
    FN (int): The number of false negatives seen so far (used in streaming mode).
    """

    def __init__(self, name='F1 Score', use_streaming=False):
        """
        Initializes the F1Score class.
        
        Parameters:
        name (str): Name of the metric.
        use_streaming (bool): If True, enables streaming (incremental) mode.
        """
        super().__init__(name)
        self.use_streaming = use_streaming
        if self.use_streaming:
            self.TP = 0  # True Positives
            self.FP = 0  # False Positives
            self.FN = 0  # False Negatives
        else:
            pass

    def evaluate_batch(self, true_labels, pred_labels, average='binary'):
        """
        Evaluates F1 score in batch mode by comparing true and predicted labels.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels
                                                                            of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels 
                                                                        of the dataset.
        average (str): The type of averaging performed on the data
        ('binary', 'micro', 'macro', or 'weighted').
        
        Returns:
        float: F1 score, calculated based on the chosen average.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        """
        if len(true_labels) == 0:
            return 0.0  # Prevent NaN result for empty input

        # Check for input length mismatch
        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        # Convert labels to numpy arrays
        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        # Calculate precision, recall, and F1 score
        return f1_score(true_labels, pred_labels, average=average)

    def evaluate_streaming(self, true_labels, pred_labels, average='binary'):
        """
        Evaluates F1 score incrementally in streaming mode.
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels of the 
                                                                        incoming data batch.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels of 
                                                                        the incoming data batch.
        average (str): The type of averaging performed on the data
        ('binary', 'micro', 'macro', or 'weighted').
        
        Returns:
        float: Updated F1 score, calculated incrementally.
        
        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        TypeError: If true_labels or pred_labels are not of the expected types 
                    (list, numpy array, or pandas).
        """
        # Check for input length mismatch
        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        # Convert labels to numpy arrays
        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        # Calculate the number of true positives, false positives, and false negatives
        TP = np.sum((true_labels == 1) & (pred_labels == 1))
        FP = np.sum((true_labels == 0) & (pred_labels == 1))
        FN = np.sum((true_labels == 1) & (pred_labels == 0))

        # Incrementally update the counts
        self.TP += TP
        self.FP += FP
        self.FN += FN

        # Compute precision, recall, and F1 score
        precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0
        recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0

        # Handle potential divide-by-zero for F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def do_score(self, true_labels, pred_labels, average='binary'):
        """
        Updates the F1 score calculation based on the selected mode (batch or streaming).
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels 
                                                                            of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels 
                                                                            of the dataset.
        average (str): The type of averaging performed on the data
        ('binary', 'micro', 'macro', or 'weighted').
        
        Returns:
        float: F1 score based on the mode selected (batch or streaming).
        
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

        # Convert labels to numpy arrays
        true_labels = _convert_to_numpy(true_labels)
        pred_labels = _convert_to_numpy(pred_labels)

        if len(true_labels) != len(pred_labels):
            raise ValueError("The length of true_labels and pred_labels must be the same.")

        if self.use_streaming:
            return self.evaluate_streaming(true_labels, pred_labels, average)
        else:
            return self.evaluate_batch(true_labels, pred_labels, average)
