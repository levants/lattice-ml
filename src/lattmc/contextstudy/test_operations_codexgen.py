"""Independent finite examples for the lattice-operation study.

These tests include empty extents, distributed witnesses, redundant joins,
and held-out failure of a training-fitted transport inclusion.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from lattmc.contextstudy.operations_codexgen import (
    extent, intent, project, relation, witnesses,
)


class OperationChecks(unittest.TestCase):
    """Check order identities and experimental edge cases independently."""

    def test_dense_identity_and_closure(self: OperationChecks) -> None:
        """Check exact sparse retrieval against dense inequalities."""
        rng = np.random.default_rng(7)
        values = rng.integers(0, 5, size=(30, 8)).astype(float)
        csr, csc = sparse.csr_matrix(values), sparse.csc_matrix(values)
        top = values.max(axis=0)
        for _ in range(40):
            u, v = rng.integers(0, 6, size=(2, 8)).astype(float)
            a, b = extent(csc, u), extent(csc, v)
            np.testing.assert_array_equal(a, (values >= u).all(axis=1))
            np.testing.assert_array_equal(
                extent(csc, np.maximum(u, v)), a & b)
            self.assertTrue(np.all(~(a | b) |
                                   extent(csc, np.minimum(u, v))))
            if np.all(u <= top):
                closed = intent(csr, a, top)
                np.testing.assert_array_equal(extent(csc, closed), a)

    def test_projection_ties_and_redundancy(self: OperationChecks) -> None:
        """Keep original amplitudes and deterministic coordinate ties."""
        u = np.array([0., 3., 3., 2.])
        np.testing.assert_array_equal(project(u, (1,)), [0, 3, 0, 0])
        self.assertIsNone(project(u, (4,)))
        values = sparse.csc_matrix([[0., 3.], [0., 2.], [0., 0.]])
        left, right = extent(values, np.array([0, 3.])), extent(
            values, np.array([0, 2.]))
        self.assertEqual(relation(left, right)['rho_left'], 1)
        self.assertFalse(relation(left, right)['strict_both'])

    def test_distributed_and_zero(self: OperationChecks) -> None:
        """Distinguish two token witnesses from one whole-query witness."""
        trace = np.array([[2., 0.], [0., 3.]])
        self.assertEqual(witnesses(trace, np.array([2., 3.]))['status'], 'D')
        self.assertEqual(witnesses(trace, np.array([2., 0.]))['status'], 'S')
        self.assertEqual(witnesses(trace, np.array([3., 3.]))['status'], 'R')
        self.assertEqual(witnesses(trace, np.zeros(2))['status'], 'N')

    def test_transport_does_not_guarantee_test_inclusion(
        self: OperationChecks,
    ) -> None:
        """A fitting-set inclusion can fail for a new shared object."""
        left = sparse.csc_matrix([[2.], [0.], [3.]])
        right = sparse.csr_matrix([[4.], [1.], [2.]])
        before = extent(left, np.array([2.]))
        query = intent(right[:2], before[:2], np.array([4.]))
        after = extent(right.tocsc(), query)
        self.assertTrue(np.all(~before[:2] | after[:2]))
        self.assertTrue(before[2] and not after[2])


if __name__ == '__main__':
    unittest.main()
