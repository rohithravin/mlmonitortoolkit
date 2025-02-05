""" 
Author: Rohith Ravindranath
Jan 30th 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from mlmonkit.metrics.utils import _convert_to_numpy
from mlmonkit.metrics.metric import BaseMetric

class AUCROC(BaseMetric):
    """
    Class to calculate AUC-ROC for both batch and streaming evaluation modes.

    Attributes:
    use_streaming (bool): If True, streaming evaluation is enabled; otherwise,
                          batch evaluation is used.
    true_labels_stream (list): A list to store true labels for streaming evaluation.
    pred_labels_stream (list): A list to store predicted probabilities for streaming evaluation.
    """

    def __init__(self, name='AUCROC', use_streaming=False):
        """
        Initializes the AUCROC class.

        Parameters:
        name (str): Name of the metric.
        use_streaming (bool): If True, enables streaming (incremental) mode.
        """
        super().__init__(name)
        self.use_streaming = use_streaming
        if self.use_streaming:
            self.true_labels_stream = []  # List to store true labels in streaming mode
            self.pred_labels_stream = []  # List to store predicted probabilities in streaming mode

    def evaluate_batch(self, true_labels, pred_probs):
        """
        Evaluates AUC-ROC in batch mode by comparing true labels and predicted probabilities.

        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            True labels of the dataset.
        pred_probs (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            Predicted probabilities (for positive class) of the dataset.

        Returns:
        float: AUC-ROC score, calculated using the roc_auc_score function.
        
        Raises:
        ValueError: If true_labels and pred_probs do not have the same length.
        """
        if len(true_labels) == 0:
            return 0.0  # Prevent NaN result for empty input

        if len(true_labels) != len(pred_probs):
            raise ValueError("The length of true_labels and pred_probs must be the same.")

        print(f'true_labels: {true_labels}')
        print(f'pred_probs: {roc_auc_score(true_labels, pred_probs)}')
        return roc_auc_score(true_labels, pred_probs)

    def evaluate_streaming(self, true_labels, pred_probs):
        """
        Evaluates AUC-ROC incrementally in streaming mode.

        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            True labels of incoming data batch.
        pred_probs (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            Predicted probabilities of incoming data batch.

        Returns:
        float: Updated AUC-ROC score, calculated incrementally.

        Raises:
        ValueError: If true_labels and pred_probs do not have the same length.
        """
        if len(true_labels) != len(pred_probs):
            raise ValueError("The length of true_labels and pred_probs must be the same.")

        self.true_labels_stream.extend(true_labels)
        self.pred_labels_stream.extend(pred_probs)

        return roc_auc_score(self.true_labels_stream, self.pred_labels_stream)

    def do_score(self, true_labels, pred_probs):
        """
        Updates the AUC-ROC calculation based on the selected mode (batch or streaming).

        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            True labels of the dataset.
        pred_probs (list, Pandas Series, Pandas DataFrame, or NumPy array): 
            Predicted probabilities of the dataset.

        Returns:
        float: AUC-ROC score based on the mode selected (batch or streaming).

        Raises:
        ValueError: If true_labels and pred_probs do not have the same length.
        TypeError: If true_labels or pred_probs are not of the expected types.
        """
        if not isinstance(true_labels, (list, np.ndarray, pd.Series, pd.DataFrame)) \
            or not isinstance(pred_probs, (list, np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("Both true_labels and pred_probs should be of type list, "+\
                            "pandas.Series, pandas.DataFrame, or numpy.ndarray.")

        true_labels = _convert_to_numpy(true_labels)
        pred_probs = _convert_to_numpy(pred_probs)

        if len(true_labels) != len(pred_probs):
            raise ValueError("The length of true_labels and pred_probs must be the same.")

        if self.use_streaming:
            return self.evaluate_streaming(true_labels, pred_probs)
        else:
            return self.evaluate_batch(true_labels, pred_probs)
