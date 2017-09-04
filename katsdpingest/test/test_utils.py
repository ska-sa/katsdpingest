"""Tests for the util module."""
from katsdpingest.utils import Range
from nose.tools import *


class TestRange(object):
    """Tests for :class:`katsdpingest.sigproc.Range`."""
    def test_init(self):
        r = Range(3, 5)
        assert_equal(3, r.start)
        assert_equal(5, r.stop)
        r = Range(10, 10)
        assert_equal(10, r.start)
        assert_equal(10, r.stop)

    def test_str(self):
        assert_equal('3:5', str(Range(3, 5)))
        assert_equal('2:2', str(Range(2, 2)))

    def test_repr(self):
        assert_equal('Range(3, 5)', repr(Range(3, 5)))
        assert_equal('Range(10, 10)', repr(Range(10, 10)))

    def test_contains(self):
        r = Range(3, 5)
        assert_in(3, r)
        assert_in(4, r)
        assert_not_in(2, r)
        assert_not_in(5, r)

    def test_issubset(self):
        assert_true(Range(3, 5).issubset(Range(3, 5)))
        assert_true(Range(3, 5).issubset(Range(0, 10)))
        # Empty range if always a subset, even if the start value is outside
        assert_true(Range(6, 6).issubset(Range(3, 4)))
        # Disjoint
        assert_false(Range(1, 3).issubset(Range(6, 8)))
        assert_false(Range(6, 8).issubset(Range(1, 3)))
        # Partial overlap
        assert_false(Range(3, 8).issubset(Range(6, 10)))
        assert_false(Range(6, 10).issubset(Range(3, 8)))
        # Superset
        assert_false(Range(0, 10).issubset(Range(1, 10)))

    def test_issuperset(self):
        # It's implemented by issubset, so just a quick test for coverage
        assert_true(Range(3, 5).issuperset(Range(3, 5)))
        assert_false(Range(3, 5).issuperset(Range(3, 6)))

    def test_isaligned(self):
        assert_true(Range(2, 12).isaligned(2))
        assert_false(Range(2, 12).isaligned(4))
        assert_false(Range(5, 11).isaligned(5))

    def test_alignto(self):
        assert_equal(Range(-10, 15), Range(-8, 13).alignto(5))
        # Empty range case
        assert_equal(0, len(Range(9, 9).alignto(5)))

    def test_intersection(self):
        assert_equal(Range(3, 7), Range(-5, 7).intersection(Range(3, 10)))
        assert_equal(0, len(Range(3, 7).intersection(Range(7, 10))))

    def test_union(self):
        # Overlapping
        assert_equal(Range(-5, 10), Range(-5, 7).union(Range(3, 10)))
        # Disjoint
        assert_equal(Range(-5, 10), Range(8, 10).union(Range(-5, 0)))
        # First one empty
        assert_equal(Range(-5, 10), Range(100, 100).union(Range(-5, 10)))
        # Second one empty
        assert_equal(Range(-5, 10), Range(-5, 10).union(Range(-10, -10)))
        # Both empty
        assert_equal(0, len(Range(5, 5).union(Range(10, 10))))

    def test_len(self):
        assert_equal(4, len(Range(3, 7)))
        assert_equal(0, len(Range(2, 2)))

    def test_nonzero(self):
        assert_true(Range(3, 4))
        assert_false(Range(3, 3))

    def test_iter(self):
        assert_equal([3, 4, 5], list(Range(3, 6)))

    def test_relative_to(self):
        assert_equal(Range(3, 5), Range(8, 10).relative_to(Range(5, 10)))
        assert_raises(ValueError, Range(8, 10).relative_to, Range(5, 9))

    def test_split(self):
        assert_equal(Range(14, 16), Range(10, 20).split(5, 2))
        assert_equal(Range(0, 0), Range(10, 10).split(5, 2))
        assert_raises(ValueError, Range(10, 20).split, 6, 3)
        assert_raises(ValueError, Range(10, 20).split, 5, -2)
        assert_raises(ValueError, Range(10, 20).split, 5, 5)
