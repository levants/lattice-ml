"""Exhaustive small-model checks, independent of learned image results."""

from itertools import product

import numpy as np

from lattmc.vision.contexts_codexgen import VectorContext, spatial_extents


def audit():
    counts = {"contexts": 0, "adjunctions": 0, "spatial_queries": 0}
    masks = [np.array(bits, dtype=bool) for bits in product([0, 1], repeat=3)]
    for entries in product(range(3), repeat=6):
        codes = np.array(entries).reshape(3, 2)
        context = VectorContext(codes)
        queries = list(product(*(range(int(t) + 1) for t in context.top)))
        for mask in masks:
            closed = context.close_rows(mask)
            assert np.all(~mask | closed)
            assert np.array_equal(context.close_rows(closed), closed)
            # Independent ordinal scaling: all observed positive thresholds.
            attributes = [
                codes[:, j] >= level
                for j in range(2)
                for level in sorted(set(codes[:, j])) if level > 0
            ]
            incidence = np.array(attributes, dtype=bool).T
            if not attributes:
                incidence = np.ones((3, 0), dtype=bool)
            common = incidence[mask].all(axis=0)
            ordinal_closed = incidence[:, common].all(axis=1)
            assert np.array_equal(closed, ordinal_closed)
            for query in queries:
                query = np.array(query)
                extent = context.extent(query)
                assert bool(np.all(~mask | extent)) == bool(
                    np.all(query <= context.intent(mask))
                )
                counts["adjunctions"] += 1
        for query in queries:
            closed = context.close_query(query)
            assert np.array_equal(
                context.extent(query), context.extent(closed))
            for other in queries:
                other = np.array(other)
                assert np.array_equal(
                    context.extent(np.maximum(query, other)),
                    context.extent(query) & context.extent(other),
                )
        counts["contexts"] += 1
    for entries in product(range(3), repeat=4):
        patches = np.array(entries).reshape(1, 2, 2)
        for query in product(range(3), repeat=2):
            image, site = spatial_extents(patches, query)
            assert np.all(~site | image)
            counts["spatial_queries"] += 1
    patches = np.array([[[2, 0], [0, 2]], [[2, 2], [0, 0]]])
    image, site = spatial_extents(patches, [2, 2])
    assert image.tolist() == [True, True]
    assert site.tolist() == [False, True]
    # A zero coordinate and an empty extent are legitimate boundary cases.
    context = VectorContext([[2, 0, 0], [0, 2, 0]])
    assert not context.extent([2, 2, 0]).any()
    assert np.array_equal(context.close_query([2, 2, 0]), [2, 2, 0])
    for invalid in ([[np.nan, 0]], [[-1, 0]], np.empty((0, 2))):
        try:
            VectorContext(invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid input accepted")
    return counts


if __name__ == "__main__":
    print(audit())
