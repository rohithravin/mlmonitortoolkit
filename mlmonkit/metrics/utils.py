""" 
Author: Rohith Ravindranath
Jan 30th 2
"""
import pandas as pd
import numpy as np

def _convert_to_numpy(data):
    """
    Converts the input data (either a Python list, Pandas Series/DataFrame, or NumPy array) 
        to a NumPy array.

    Parameters:
    data (list, Pandas DataFrame, Pandas Series, or NumPy array): The input data to convert.

    Returns:
    numpy.ndarray: The converted NumPy array.

    Raises:
    TypeError: If the input data is not of type list, Pandas DataFrame, Pandas Series,
        or NumPy ndarray.
    """
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise TypeError("Input data must be of type list, Pandas DataFrame, \
                        Pandas Series, or NumPy ndarray.")
