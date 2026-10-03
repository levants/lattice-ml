"""Independent small-array checks for the external retrieval protocol."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from .common_codexgen import full_scores, metrics, queries, threshold_for
from .data_codexgen import deduplicate


class ExternalChecks(unittest.TestCase):
    def test_sparse_dominance_matches_dense(self: ExternalChecks) -> None:
        """Verify sparse dominance matches dense."""
        rng = np.random.default_rng(81)
        x = rng.integers(0, 5, (40, 12)).astype(float)
        for _ in range(20):
            query = rng.integers(0, 5, 12).astype(float)
            expected = (x[:, query > 0] / query[query > 0]).min(axis=1)
            np.testing.assert_allclose(full_scores(sparse.csc_matrix(x),
                                                   query), expected)

    def test_zero_query_is_universal(self: ExternalChecks) -> None:
        """Verify zero query is universal."""
        x = sparse.csc_matrix(np.eye(3))
        np.testing.assert_array_equal(full_scores(x, np.zeros(3)), 1.)

    def test_budgets_match_literal_conjunction(self: ExternalChecks) -> None:
        """Verify budgets match literal conjunction."""
        x = np.array([[2., 4., 0.], [1., 3., 4.], [3., 0., 2.]])
        query = np.array([1., 2., 0.])
        scores, order = queries(sparse.csc_matrix(x), query, x.max(0),
                                 (x > 0).sum(0), len(x))
        for score, count in zip(scores, (1, 4, 16, 64, len(order))):
            selected = order[:count]
            expected = (x[:, selected] / query[selected]).min(1)
            np.testing.assert_allclose(score, expected)

    def test_tie_aware_precision_and_confusion(self: ExternalChecks) -> None:
        """Verify tie aware precision and confusion."""
        y = np.array([True] * 10 + [False] * 10)
        result = metrics(y, np.ones(20), 1.)
        self.assertEqual(result['p10'], .5)
        self.assertEqual(result['ap'], .5)
        self.assertEqual([result[k] for k in ('tp', 'fp', 'fn', 'tn')],
                         [10, 10, 0, 0])

    def test_calibration_tie_prefers_larger_alpha(
        self: ExternalChecks,
    ) -> None:
        """Verify calibration tie prefers larger alpha."""
        y = np.array([True, False])
        self.assertEqual(threshold_for(y, np.array([2., 0.]), True), 1.)

    def test_dedup_preserves_official_test(self: ExternalChecks) -> None:
        """Verify dedup preserves official test."""
        records = [dict(text='One repeated document about a football match',
                        split=split, label=0)
                   for split in ('train', 'calibration', 'test')]
        tokens = dict(gpt2=dict(input_ids=np.tile([1, 2], (3, 1)),
                               attention_mask=np.ones((3, 2), bool)))
        kept, removed = deduplicate(records, tokens)
        self.assertEqual(kept, [2])
        self.assertEqual(removed[0]['rows'], [0, 1])

    def test_dedup_excludes_conflicting_labels(self: ExternalChecks) -> None:
        """Verify dedup excludes conflicting labels."""
        records = [dict(text='An identical text with incompatible labels',
                        split='train', label=i) for i in (0, 1)]
        tokens = dict(gpt2=dict(input_ids=np.tile([1, 2], (2, 1)),
                               attention_mask=np.ones((2, 2), bool)))
        self.assertEqual(deduplicate(records, tokens)[0], [])


if __name__ == '__main__':
    unittest.main()
