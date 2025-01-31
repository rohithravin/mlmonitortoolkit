"""Test module for dummy.py."""

from mlmonkit.dummy import add_numbers

def test_add_numbers():
    """Tests the add_numbers function."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
