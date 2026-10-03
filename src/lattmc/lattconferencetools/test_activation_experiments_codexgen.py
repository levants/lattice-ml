"""Independent dense checks for the sparse graded-query implementation."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from .activation_experiments_codexgen import coincidence, extent, intent


class GradedQueries(unittest.TestCase):
    def test_sparse_dense_and_closure(self: GradedQueries) -> None:
        """Verify sparse dense and closure."""
        rng = np.random.default_rng(2026)
        for _ in range(12):
            dense = rng.integers(0, 8, size=(17, 11)).astype(np.float32)
            dense[rng.random(dense.shape) < 0.7] = 0
            matrix = sparse.csc_matrix(dense)
            query = rng.integers(0, 6, size=11).astype(np.float32)
            for alpha in [0.25, 0.5, 0.75, 1.0]:
                expected = np.all(dense >= alpha * query, axis=1)
                actual = extent(matrix, query, alpha)
                np.testing.assert_array_equal(actual, expected)
                closed = intent(matrix.tocsr(), actual, dense.max(axis=0),
                                dense.min(axis=0))
                # Test the closed extent only for queries in the declared box.
                if np.all(alpha * query <= dense.max(axis=0)):
                    np.testing.assert_array_equal(
                        extent(matrix, closed), actual,
                    )
                if actual.any():
                    np.testing.assert_array_equal(
                        closed, dense[actual].min(axis=0),
                    )
                else:
                    np.testing.assert_array_equal(closed, dense.max(axis=0))
            np.testing.assert_array_equal(
                extent(matrix, query, None),
                np.all(dense[:, query > 0] > 0, axis=1),
            )

    def test_zero_query_and_coincidence(self: GradedQueries) -> None:
        """Verify zero query and coincidence."""
        matrix = sparse.csc_matrix([[0., 2.], [3., 0.], [3., 2.]])
        self.assertTrue(extent(matrix, np.zeros(2)).all())
        left = np.array([True, False, True, False])
        right = np.array([False, True, True, False])
        self.assertEqual(
            coincidence(left, right),
            dict(both=1, only_left=1, only_right=1, neither=1, jaccard=1 / 3),
        )


if __name__ == '__main__':
    unittest.main()
