"""Test functional utilities."""

from bkd.base.utils.functional.operator import identity
from hypothesis import given
from hypothesis import strategies as st


@given(num=st.integers(min_value=-1000, max_value=1000))
def test_identity_int(num: int) -> None:
    """Test identity function with integers."""
    assert num == identity(num)


@given(text=st.text(min_size=1, max_size=100))
def test_identity_str(text: str) -> None:
    """Test identity function with integers."""
    assert text == identity(text)
