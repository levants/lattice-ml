"""Independent checks of retrieval, ties, and split isolation."""

from __future__ import annotations

import itertools
import unittest

import numpy as np
from scipy import sparse

from .conference_experiments_codexgen import (
    choose, expected_precision, full_scores, phrases, split_tokens,
)


class RetrievalChecks(unittest.TestCase):
    def test_sparse_scores_against_dense_definition(
        self: RetrievalChecks,
    ) -> None:
        """Verify sparse scores against dense definition."""
        rng = np.random.default_rng(923)
        for _ in range(40):
            values = rng.integers(0, 6, (31, 9)).astype(float)
            query = rng.integers(0, 5, 9).astype(float)
            active = query > 0
            expected = (values[:, active] / query[active]).min(axis=1)
            actual = full_scores(sparse.csc_matrix(values), query)
            np.testing.assert_array_equal(actual, expected)
            for alpha in (.05, .5, 1.):
                extent = np.all(values >= alpha * query, axis=1)
                np.testing.assert_array_equal(actual >= alpha, extent)
        np.testing.assert_array_equal(
            full_scores(sparse.csc_matrix(values), np.zeros(9)),
            np.ones(len(values)),
        )

    def test_cutoff_ties_against_enumeration(self: RetrievalChecks) -> None:
        """Verify cutoff ties against enumeration."""
        labels = np.array([True, False, True, False, False])
        scores = np.array([3., 2., 2., 2., 1.])
        empirical = np.mean([
            labels[[0, order[0]]].mean()
            for order in itertools.permutations([1, 2, 3])
        ])
        self.assertAlmostEqual(expected_precision(labels, scores, 2),
                               empirical)

    def test_duplicates_do_not_cross_splits(self: RetrievalChecks) -> None:
        """Verify duplicates do not cross splits."""
        tokens = np.arange(120, dtype=np.int64).reshape(40, 3)
        tokens[39] = tokens[0]
        splits, unique = split_tokens(tokens)
        self.assertEqual(unique, 39)
        membership = {row: key for key, rows in splits.items() for row in rows}
        self.assertEqual(membership[0], membership[39])
        self.assertEqual(len(membership), 40)
        self.assertEqual(sum(map(len, splits.values())), 40)

    def test_selection_ignores_test_scores_and_labels(
        self: RetrievalChecks,
    ) -> None:
        """Verify selection ignores test scores and labels."""
        scores = [np.array([3., 2., 1., 0.]),
                  np.array([2., 3., 0., 1.])]
        labels = np.array([True, False, True, False])
        self.assertEqual(choose(scores, np.array([0, 1]), labels), 0)
        labels[2:] = ~labels[2:]
        scores[1][2:] = 1000
        self.assertEqual(choose(scores, np.array([0, 1]), labels), 0)

    def test_phrase_boundary_and_stopwords(self: RetrievalChecks) -> None:
        """Verify phrase boundary and stopwords."""
        self.assertEqual(phrases('New York; social media, the world'),
                         {'new york', 'social media'})


if __name__ == '__main__':
    unittest.main()
