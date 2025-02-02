""" 
Author: Rohith Ravindranath
Jan 30th 2025
"""

class BaseMetric:
    """
    Base class for calculating various scoring metrics (e.g., accuracy, precision, recall).
    Provides a utility function to handle conversion of inputs to numpy arrays, etc.
    """

    def __init__(self, name):
        self.name = name
