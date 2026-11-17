"""Tests for the util module."""

import pytest

from katsdpingest.utils import Range


class TestRange:
    """Tests for :class:`katsdpingest.sigproc.Range`."""
    def test_init(self):
        r = Range(3, 5)
        assert 3 == r.start
        assert 5 == r.stop
        r = Range(10, 10)
        assert 10 == r.start
        assert 10 == r.stop

    def test_str(self):
        assert '3:5' == str(Range(3, 5))
        assert '2:2' == str(Range(2, 2))

    def test_repr(self):
        assert 'Range(3, 5)' == repr(Range(3, 5))
        assert 'Range(10, 10)' == repr(Range(10, 10))

    def test_contains(self):
        r = Range(3, 5)
        assert 3 in r
        assert 4 in r
        assert 2 not in r
        assert 5 not in r

    def test_issubset(self):
        assert Range(3, 5).issubset(Range(3, 5))
        assert Range(3, 5).issubset(Range(0, 10))
        # Empty range if always a subset, even if the start value is outside
        assert Range(6, 6).issubset(Range(3, 4))
        # Disjoint
        assert not Range(1, 3).issubset(Range(6, 8))
        assert not Range(6, 8).issubset(Range(1, 3))
        # Partial overlap
        assert not Range(3, 8).issubset(Range(6, 10))
        assert not Range(6, 10).issubset(Range(3, 8))
        # Superset
        assert not Range(0, 10).issubset(Range(1, 10))

    def test_issuperset(self):
        # It's implemented by issubset, so just a quick test for coverage
        assert Range(3, 5).issuperset(Range(3, 5))
        assert not Range(3, 5).issuperset(Range(3, 6))

    def test_isaligned(self):
        assert Range(2, 12).isaligned(2)
        assert not Range(2, 12).isaligned(4)
        assert not Range(5, 11).isaligned(5)

    def test_alignto(self):
        assert Range(-10, 15) == Range(-8, 13).alignto(5)
        # Empty range case
        assert 0 == len(Range(9, 9).alignto(5))

    def test_intersection(self):
        assert Range(3, 7) == Range(-5, 7).intersection(Range(3, 10))
        assert 0 == len(Range(3, 7).intersection(Range(7, 10)))

    def test_union(self):
        # Overlapping
        assert Range(-5, 10) == Range(-5, 7).union(Range(3, 10))
        # Disjoint
        assert Range(-5, 10) == Range(8, 10).union(Range(-5, 0))
        # First one empty
        assert Range(-5, 10) == Range(100, 100).union(Range(-5, 10))
        # Second one empty
        assert Range(-5, 10) == Range(-5, 10).union(Range(-10, -10))
        # Both empty
        assert 0 == len(Range(5, 5).union(Range(10, 10)))

    def test_len(self):
        assert 4 == len(Range(3, 7))
        assert 0 == len(Range(2, 2))

    def test_nonzero(self):
        assert Range(3, 4)
        assert not Range(3, 3)

    def test_iter(self):
        assert [3, 4, 5] == list(Range(3, 6))

    def test_relative_to(self):
        assert Range(3, 5) == Range(8, 10).relative_to(Range(5, 10))
        with pytest.raises(ValueError):
            Range(8, 10).relative_to(Range(5, 9))

    def test_split(self):
        assert Range(14, 16) == Range(10, 20).split(5, 2)
        assert Range(0, 0) == Range(10, 10).split(5, 2)
        with pytest.raises(ValueError):
            Range(10, 20).split(6, 3)
        with pytest.raises(ValueError):
            Range(10, 20).split(5, -2)
        with pytest.raises(ValueError):
            Range(10, 20).split(5, 5)
