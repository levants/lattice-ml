"""Exact lattice operations and auditable measurements on sparse codes.

Arrays are finite nonnegative activation values. Extents use exact stored
thresholds, with float64 comparisons and no membership epsilon. Helpers
separate corpus membership, category labels, and token-level witnesses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

Vector: TypeAlias = NDArray[np.floating]
Mask: TypeAlias = NDArray[np.bool_]
Record: TypeAlias = dict[str, object]


def digest(path: Path) -> str:
    """Return SHA-256 of a file, reading it in bounded chunks."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: object) -> None:
    """Write JSON without NaN; create missing parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def extent(matrix: sparse.csc_matrix, query: Vector) -> Mask:
    """Return exact dominance of a nonnegative query for every matrix row.

    Args:
        matrix: Nonnegative CSC matrix, items by dictionary coordinates.
        query: One finite nonnegative requirement per matrix column.

    Returns:
        Boolean item mask; the zero query accepts all items.

    Raises:
        ValueError: If query shape, finiteness, or sign is invalid.
    """
    query = np.asarray(query, dtype=np.float64)
    if (query.shape != (matrix.shape[1],) or
            not np.isfinite(query).all() or (query < 0).any()):
        raise ValueError('Query must be finite, nonnegative and conformable')
    active = np.flatnonzero(query)
    sizes = np.diff(matrix.indptr)[active]
    result = np.ones(matrix.shape[0], dtype=bool)
    for j in active[np.argsort(sizes, kind='stable')]:
        lo, hi = matrix.indptr[j:j + 2]
        mask = np.zeros(matrix.shape[0], dtype=bool)
        values = matrix.data[lo:hi].astype(np.float64)
        mask[matrix.indices[lo:hi][values >= query[j]]] = True
        result &= mask
        if not result.any():
            break
    return result


def intent(matrix: sparse.csr_matrix, members: Mask,
           top: Vector) -> Vector:
    """Meet selected item rows, using the explicit top for an empty set."""
    ids = np.flatnonzero(members)
    value = top.copy()
    for start in range(0, len(ids), 64):
        part = matrix[ids[start:start + 64]].toarray()
        value = np.minimum(value, part.min(axis=0))
    return value


def project(vector: Vector, ranks: tuple[int, ...]) -> Vector | None:
    """Select one-based amplitude ranks, breaking ties by coordinate ID.

    Returns the original amplitudes on selected coordinates and zero
    elsewhere. Return None when the code has too few positive entries.
    """
    active = np.flatnonzero(vector > 0)
    order = active[np.lexsort((active, -vector[active]))]
    if len(order) < max(ranks):
        return None
    result = np.zeros_like(vector)
    chosen = order[np.array(ranks) - 1]
    result[chosen] = vector[chosen]
    return result


def encoded(query: Vector) -> Record:
    """Serialize nonzero coordinates and their unchanged amplitudes."""
    ids = np.flatnonzero(query)
    return dict(coordinates=ids.tolist(), values=query[ids].tolist())


def metrics(mask: Mask, truth: Mask) -> Record:
    """Return confusion counts and undefined-aware precision and recall."""
    tp = int((mask & truth).sum())
    fp = int((mask & ~truth).sum())
    fn = int((~mask & truth).sum())
    tn = int((~mask & ~truth).sum())
    return dict(n=int(mask.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None)


def relation(left: Mask, right: Mask) -> Record:
    """Measure a join's extent as the intersection of constituent extents."""
    both = int((left & right).sum())
    return dict(both=both, left_only=int((left & ~right).sum()),
                right_only=int((right & ~left).sum()),
                neither=int((~left & ~right).sum()),
                rho_left=both / int(left.sum()) if left.any() else None,
                rho_right=both / int(right.sum()) if right.any() else None,
                strict_both=bool(both < left.sum() and both < right.sum()))


def witnesses(trace: Vector, query: Vector) -> Record:
    """Classify exact token witnesses; yellow D denotes distributed evidence.

    Args:
        trace: Token by coordinate nonnegative matrix, all valid positions.
        query: Full code-space requirement, with the same coordinate order.

    Returns:
        S (single-token), D (distributed), R (rejected), or N (zero query),
        plus positions meeting any positive requirement or the entire query.
    """
    active = np.flatnonzero(query)
    if not len(active):
        return dict(status='N', whole=list(range(len(trace))), partial=[],
                    member=True)
    values = trace[:, active].astype(np.float64)
    matches = values >= query[active].astype(np.float64)
    whole = matches.all(axis=1)
    member = bool(matches.any(axis=0).all())
    return dict(status='S' if whole.any() else 'D' if member else 'R',
                whole=np.flatnonzero(whole).tolist(),
                partial=np.flatnonzero(matches.any(axis=1)).tolist(),
                member=member)


GRID = [((i,), (j,)) for i in (1, 2, 3) for j in (1, 2, 3)]
GRID += [((1, 2), (1, 2)), ((2, 3), (2, 3))]
