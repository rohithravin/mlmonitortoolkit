""" 
Author: Rohith Ravindranath
Jan 30th 2025
"""

from abc import ABC, abstractmethod

class BaseMetric:
    """
    Base class for calculating various scoring metrics (e.g., accuracy, precision, recall).
    Provides a utility function to handle conversion of inputs to numpy arrays, etc.
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def do_score(self, true_labels, pred_labels):
        """
        Abstract method to calculate the score based on the selected mode (batch or streaming).
        
        Parameters:
        true_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): True labels 
                                                                            of the dataset.
        pred_labels (list, Pandas Series, Pandas DataFrame, or NumPy array): Predicted labels 
                                                                            of the dataset.

        Returns:
        float: The accuracy (or any other metric) based on the mode selected (batch or streaming).

        Raises:
        ValueError: If true_labels and pred_labels do not have the same length.
        TypeError: If true_labels or pred_labels are not of the expected types 
                    (list, numpy array, or pandas).
        """
