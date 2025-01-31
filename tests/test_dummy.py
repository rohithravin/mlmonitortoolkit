"""Test module for dummy.py."""

import numpy as np
import pandas as pd
from mlmonkit.dummy import cool_numpy_function, cool_pandas_function

def test_cool_numpy_function():
    """Tests cool_numpy_function by verifying the determinant of a 2x2 matrix."""
    matrix = np.array([[2, 3], [1, 4]])
    determinant = cool_numpy_function(matrix)
    assert np.isclose(determinant, 5)

def test_cool_pandas_function():
    """Tests cool_pandas_function by verifying column-wise sum of a DataFrame."""
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = pd.DataFrame(data)
    column_sums = cool_pandas_function(df)
    assert column_sums['A'] == 6
    assert column_sums['B'] == 15
