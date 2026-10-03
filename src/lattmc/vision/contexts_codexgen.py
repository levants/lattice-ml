"""Finite graded contexts and spatial retrieval for nonnegative codes."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

from dataclasses import dataclass

import numpy as np


def nonnegative(values: ArrayLike, ndim: int) -> np.ndarray:
    """Reject invalid data rather than silently clipping sparse codes."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError("Expected a finite array of the specified rank")
    if np.any(array < 0):
        raise ValueError("Codes must be nonnegative")
    return array


@dataclass
class VectorContext:
    """A finite context in the data-bounded product of real intervals."""

    codes: np.ndarray

    def __post_init__(self: VectorContext) -> None:
        """Copy and validate the code matrix and compute its coordinatewise
        top.
        """
        self.codes = nonnegative(self.codes, 2).copy()
        if 0 in self.codes.shape:
            raise ValueError("Context needs at least one row and coordinate")
        self.top = self.codes.max(axis=0)

    def extent(self: VectorContext, query: ArrayLike) -> np.ndarray:
        """Return the Boolean mask of rows dominating the query."""
        query = nonnegative(query, 1)
        if query.shape != self.top.shape or np.any(query > self.top):
            raise ValueError("Query must lie in the context's product lattice")
        return np.all(self.codes >= query, axis=1)

    def intent(self: VectorContext, rows: ArrayLike) -> np.ndarray:
        """Return the common row intent, using the context top for an empty
        mask.
        """
        rows = np.asarray(rows)
        if rows.dtype != np.bool_ or rows.shape != (len(self.codes),):
            raise ValueError("Rows must be a Boolean mask of context length")
        return self.codes[rows].min(axis=0) if rows.any() else self.top.copy()

    def close_query(self: VectorContext, query: ArrayLike) -> np.ndarray:
        """Close a query by applying extent followed by intent."""
        return self.intent(self.extent(query))

    def close_rows(self: VectorContext, rows: ArrayLike) -> np.ndarray:
        """Close a row mask by applying intent followed by extent."""
        return self.extent(self.intent(rows))


def pooled_codes(patches: ArrayLike) -> np.ndarray:
    """Max over spatial sites; a site is not necessarily a local cause."""
    patches = nonnegative(patches, 3)
    if 0 in patches.shape:
        raise ValueError("Need nonempty images, sites, and coordinates")
    return patches.max(axis=1)


def spatial_extents(
    patches: ArrayLike,
    query: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return image-level and same-site satisfaction masks."""
    patches = nonnegative(patches, 3)
    query = nonnegative(query, 1)
    pooled = pooled_codes(patches)
    if query.shape != pooled.shape[1:]:
        raise ValueError("Query dimension differs from code dimension")
    image = np.all(pooled >= query, axis=1)
    same_site = np.any(np.all(patches >= query, axis=2), axis=1)
    return image, same_site


def graded_score(codes: ArrayLike, query: ArrayLike) -> np.ndarray:
    """Minimum satisfaction ratio; zero queries receive a constant score."""
    codes = nonnegative(codes, 2)
    query = nonnegative(query, 1)
    if query.shape != codes.shape[1:]:
        raise ValueError("Query dimension differs from code dimension")
    active = query > 0
    if not active.any():
        return np.ones(len(codes))
    return np.min(codes[:, active] / query[active], axis=1)


def select_query(
    sources: ArrayLike,
    scale: ArrayLike,
    budget: int,
    alpha: float,
) -> np.ndarray:
    """Meet of positive sources; rank coordinates by training RMS units."""
    sources = nonnegative(sources, 2)
    scale = nonnegative(scale, 1)
    if not len(sources) or scale.shape != sources.shape[1:]:
        raise ValueError("Need sources and matching training scales")
    if budget < 1 or not 0 < alpha <= 1:
        raise ValueError("Invalid coordinate budget or threshold multiplier")
    shared = sources.min(axis=0)
    order = np.argsort(-shared / np.maximum(scale, 1e-12), kind="stable")
    chosen = order[:budget]
    query = np.zeros_like(shared)
    query[chosen] = alpha * shared[chosen]
    return query
