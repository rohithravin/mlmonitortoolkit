""" 
Author: Rohith Ravindranath
Jan 30th 2
"""

import pytest
import numpy as np
import pandas as pd
from mlmonkit.metrics.utils import _convert_to_numpy 

@pytest.mark.parametrize("input_data, expected_output", [
    ([1, 2, 3], np.array([1, 2, 3])),
    (np.array([1, 2, 3]), np.array([1, 2, 3])),
    (pd.Series([1, 2, 3]), np.array([1, 2, 3])),
    (pd.DataFrame([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
])
def test_convert_to_numpy(input_data, expected_output):
    """
    Tests the _convert_to_numpy function with various input types (list, NumPy array, 
    Pandas Series, Pandas DataFrame) and ensures that the output is correctly converted 
    to a NumPy array.
    """
    result = _convert_to_numpy(input_data)
    np.testing.assert_array_equal(result, expected_output)

def test_convert_to_numpy_invalid():
    """
    Tests the _convert_to_numpy function with an invalid input type (e.g., a string),
    expecting a TypeError to be raised.
    """
    with pytest.raises(TypeError):
        _convert_to_numpy("Invalid input")
