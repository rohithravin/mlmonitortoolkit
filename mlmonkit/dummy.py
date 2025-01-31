"""Dummy module for demonstration."""

import numpy as np

def cool_numpy_function(arr):
    """Computes the determinant of a given square matrix.
    
    Args:
        arr (numpy.ndarray): A square NumPy array.
    
    Returns:
        float: The determinant of the input matrix.
    """
    return np.linalg.det(arr)

def cool_pandas_function(df):
    """Calculates the sum of each column in a Pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): A DataFrame containing numerical values.
    
    Returns:
        pandas.Series: A Series with the sum of each column.
    """
    return df.sum()
