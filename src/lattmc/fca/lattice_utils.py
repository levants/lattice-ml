"""Lattice utilities for Lattice-theoretic Formal Concept Analysis (FCA)."""

from functools import reduce
from typing import List, Tuple, Union

import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from src.lattmc.fca.utils import is_empty, not_empty, to_numpy


def le(
    u: np.ndarray,
    v: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> bool:
    """Less than or equal one vector than another, u <= v.

    Args:
        u (np.ndarray): The first array.
        v (np.ndarray): The second array.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        bool: True if the first vector is less or equal
            to the second vector, False otherwise.
    """
    if is_empty(pos_idx) and is_empty(neg_idx):
        r = np.all(u <= v)
    elif is_empty(pos_idx) and not is_empty(neg_idx):
        r = np.all(v <= u)
    else:
        r = np.all(
            u[pos_idx] <= v[pos_idx]
        ) and np.all(
            v[neg_idx] <= u[neg_idx]
        )

    return r


# Vectorized less than or equal
vectorized_le = np.vectorize(le, signature='(n),(n)->()')


def le_all(
    U: np.ndarray,
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> bool:
    """Less or equals all vectors than another (element-wise)

    Args:
        U (np.ndarray): The first array.
        V (np.ndarray): The second array.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        bool: True if all vectors in U are less or
            equal to the corresponding vectors in V, False otherwise.
    """
    return np.all(
        vectorized_le(U, V, pos_idx=pos_idx, neg_idx=neg_idx)
    )


def upper_mask(
    u: np.ndarray,
    vs: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Get mask for vectors in vs that are upper than u (i.e., u <= v).

    Args:
        u (np.ndarray): The reference vector.
        vs (np.ndarray): 2D array of vectors to filter.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: Mask for vectors in vs that are upper than u.
    """
    pos_empty = is_empty(pos_idx)
    neg_empty = is_empty(neg_idx)
    if pos_empty and neg_empty:
        mask = np.all(u <= vs, axis=1)
    elif pos_empty and not neg_empty:
        mask = np.all(vs <= u, axis=1)
    else:
        mask = np.all(u[pos_idx] <= vs[:, pos_idx], axis=1)
        if not neg_empty:
            mask &= np.all(vs[:, neg_idx] <= u[neg_idx], axis=1)

    return mask


def upper_indices(
    u: np.ndarray,
    vs: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Get indices for vectors in vs that are upper than u (i.e., u <= v).

    Args:
        u (np.ndarray): The reference vector.
        vs (np.ndarray): 2D array of vectors to filter.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: Indices of vectors in vs that are upper than u.
    """
    return np.where(upper_mask(u, vs, pos_idx, neg_idx))[0]


def filter_upper(
    u: np.ndarray,
    vs: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Filter vectors in vs that are upper than u (i.e., u <= v).

    Args:
        u (np.ndarray): The reference vector.
        vs (np.ndarray): 2D array of vectors to filter.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: Filtered array containing only vectors v where u <= v.
    """
    return vs[upper_mask(u, vs, pos_idx, neg_idx)]


def lower_mask(
    u: np.ndarray,
    vs: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Filter vectors in vs that are lower than u (i.e., u >= v).

    Args:
        u (np.ndarray): The reference vector.
        vs (np.ndarray): 2D array of vectors to filter.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: Filtered array containing only vectors v where u >= v.
    """
    pos_empty = is_empty(pos_idx)
    neg_empty = is_empty(neg_idx)
    if pos_empty and neg_empty:
        mask = np.all(u >= vs, axis=1)
    elif pos_empty and not neg_empty:
        mask = np.all(vs >= u, axis=1)
    else:
        mask = np.all(u[pos_idx] >= vs[:, pos_idx], axis=1)
        if not neg_empty:
            mask &= np.all(vs[:, neg_idx] >= u[neg_idx], axis=1)

    return mask


def filter_lower(
    u: np.ndarray,
    vs: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Filter vectors in vs that are lower than u (i.e., u >= v).

    Args:
        u (np.ndarray): The reference vector.
        vs (np.ndarray): 2D array of vectors to filter.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: Filtered array containing only vectors v where u >= v.
    """
    return vs[lower_mask(u, vs, pos_idx, neg_idx)]


def subset(A: Union[List, np.ndarray], B: Union[List, np.ndarray]) -> bool:
    """Check if A is a subset of B
    Args:
        A (Union[List, np.ndarray]): The first array (potential subset).
        B (Union[List, np.ndarray]): The second array (potential superset).
    Returns:
        bool: True if A is a subset of B, False otherwise.
    """
    return np.all(np.isin(to_numpy(A), to_numpy(B)))


def intersect(*arrs: np.ndarray) -> np.ndarray:
    """Intersection of arrays

    Args:
        *arrs (np.ndarray): The arrays to intersect.
    Returns:
        np.ndarray: The intersection of the arrays.
    """
    return reduce(np.intersect1d, (arrs))


def intersect_xd(*arrs: np.ndarray) -> np.ndarray:
    """Intersection of arrays elementwise

    Args:
        *arrs (np.ndarray): The arrays to intersect.
    Returns:
        np.ndarray: The intersection of the arrays elementwise.
    """
    return np.minimum.reduce(arrs)


def union(*arrs: np.ndarray) -> np.ndarray:
    """Union of arrays as set union

    Args:
        *arrs (np.ndarray): The arrays to union.
    Returns:
        np.ndarray: The union of the arrays.
    """
    return reduce(np.union1d, (arrs))


def diff(*arrs: np.ndarray) -> np.ndarray:
    """Difference of arrays as set difference

    Args:
        *arrs (np.ndarray): The arrays to difference.
    Returns:
        np.ndarray: The difference of the arrays.
    """
    return reduce(np.setdiff1d, (arrs))


def diff_idcs(
    V: np.ndarray,
    neg_idx: Union[List, np.ndarray]
) -> np.ndarray:
    """Gets positive indices from denative indices

    Args:
        V (np.ndarray): The input array.
        neg_idx (Union[List, np.ndarray]): The negative indices.
    Returns:
        np.ndarray: The positive indices.
    """
    dm = V.shape[1] if len(V.shape) > 1 else V.shape[0]
    all_idx = np.arange(dm)
    pos_idx = np.setdiff1d(all_idx, neg_idx)

    return pos_idx


def init_indices(
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray],
    neg_idx: Union[List, np.ndarray],
) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """
    Initialize positive and negative indices for the given array.
    Args:
        V (np.ndarray): The input array.
        pos_idx (Union[List, np.ndarray]): The positive indices.
        neg_idx (Union[List, np.ndarray]): The negative indices.
    Returns:
        tuple: A tuple containing the positive and negative indices.
    """
    if not_empty(pos_idx) and not_empty(neg_idx):
        pos_idcs = np.array(pos_idx)
        neg_idcs = np.array(neg_idx)
    elif is_empty(pos_idx) and not_empty(neg_idx):
        neg_idcs = np.array(neg_idx)
        pos_idcs = diff_idcs(V, neg_idx)
    elif not_empty(pos_idx) and is_empty(neg_idx):
        pos_idcs = np.array(pos_idx)
        neg_idcs = diff_idcs(V, pos_idx)
    else:
        pos_idcs = pos_idx
        neg_idcs = neg_idx

    return pos_idcs, neg_idcs


def _meet_pos_neg(
    u: np.ndarray,
    v: np.ndarray,
    pos_idx: Union[List, np.ndarray],
    neg_idx: Union[List, np.ndarray]
) -> np.ndarray:
    """Meet operation on two arrays (vectors), considering directions
        by positive and negative indices.

    Args:
        u (np.ndarray): The first array.
        v (np.ndarray): The second array.
        pos_idx (Union[List, np.ndarray]): The positive indices.
        neg_idx (Union[List, np.ndarray]): The negative indices.
    Returns:
        np.ndarray: The met array.
    """
    r = np.empty_like(u, dtype=u.dtype)
    r[pos_idx] = np.minimum(u[pos_idx], v[pos_idx])
    r[neg_idx] = np.maximum(u[neg_idx], v[neg_idx])

    return r


def _meet_all_pos_neg(
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray],
    neg_idx: Union[List, np.ndarray]
) -> np.ndarray:
    """Meet operation on all the arrays (vectors) in the given list,
        considering directions by positive and negative indices.

    Args:
        V (np.ndarray): The arrays to meet.
        pos_idx (Union[List, np.ndarray]): The positive indices.
        neg_idx (Union[List, np.ndarray]): The negative indices.
    Returns:
        np.ndarray: The met array.
    """
    V_arr = to_numpy(V)
    r = np.empty_like(V_arr[0], dtype=V_arr.dtype)
    r[pos_idx] = np.min(V_arr[:, pos_idx], axis=0)
    r[neg_idx] = np.max(V_arr[:, neg_idx], axis=0)

    return r


def meet(
    u: np.ndarray,
    v: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Meet operation on two arrays (vectors), considering directions
        by positive and negative indices.

    Args:
        u (np.ndarray): The first array.
        v (np.ndarray): The second array.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: The met array.
    """
    if is_empty(pos_idx) and is_empty(neg_idx):
        r = np.minimum(u, v)
    elif is_empty(pos_idx) and not_empty(neg_idx):
        pos_idx = diff_idcs(v, neg_idx)
        r = _meet_pos_neg(u, v, pos_idx, neg_idx)
    else:
        r = _meet_pos_neg(u, v, pos_idx, neg_idx)

    return r


def meet_all(
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Meet operation on all the arrays (vectors) in the given list,
        considering directions by positive and negative indices.

    Args:
        V (np.ndarray): The arrays to meet.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: The met array.
    """
    if is_empty(pos_idx) and is_empty(neg_idx):
        r = np.min(to_numpy(V), axis=0)
    elif is_empty(pos_idx) and not is_empty(neg_idx):
        pos_idx = diff_idcs(V, neg_idx)
        r = _meet_all_pos_neg(V, pos_idx, neg_idx)
    else:
        r = _meet_all_pos_neg(V, pos_idx, neg_idx)

    return r


def _join_pos_neg(
    u: np.ndarray,
    v: np.ndarray,
    pos_idx: Union[List, np.ndarray],
    neg_idx: Union[List, np.ndarray]
) -> np.ndarray:
    """Join operation on two arrays (vectors), considering directions
        by positive and negative indices.

    Args:
        u (np.ndarray): The first array.
        v (np.ndarray): The second array.
        pos_idx (Union[List, np.ndarray]): The positive indices.
        neg_idx (Union[List, np.ndarray]): The negative indices.
    Returns:
        np.ndarray: The joined array.
    """
    r = np.empty_like(u, dtype=u.dtype)
    r[pos_idx] = np.maximum(u[pos_idx], v[pos_idx])
    r[neg_idx] = np.minimum(u[neg_idx], v[neg_idx])

    return r


def _join_all_pos_neg(
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray],
    neg_idx: Union[List, np.ndarray]
) -> np.ndarray:
    """Join operation on all the arrays (vectors) in the given list,
        considering directions by positive and negative indices.

    Args:
        V (np.ndarray): The arrays to join.
        pos_idx (Union[List, np.ndarray]): The positive indices.
        neg_idx (Union[List, np.ndarray]): The negative indices.
    Returns:
        np.ndarray: The joined array.
    """
    V_arr = to_numpy(V)
    r = np.empty_like(V_arr[0], dtype=V_arr.dtype)
    r[pos_idx] = np.max(V_arr[:, pos_idx], axis=0)
    r[neg_idx] = np.min(V_arr[:, neg_idx], axis=0)

    return r


def join(
    u: np.ndarray,
    v: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Join operation on two arrays (vectors) considering directions
        by positive and negative indices.

    Args:
        u (np.ndarray): The first array.
        v (np.ndarray): The second array.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: The joined array.
    """
    if is_empty(pos_idx) and is_empty(neg_idx):
        r = np.maximum(u, v)
    elif is_empty(pos_idx) and not is_empty(neg_idx):
        pos_idx = diff_idcs(u, neg_idx)
        r = _join_pos_neg(u, v, pos_idx, neg_idx)
    else:
        r = _join_pos_neg(u, v, pos_idx, neg_idx)

    return r


def join_all(
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Join operation on all the arrays (vectors), in the given list
        considering directions by positive and negative indices.

    Args:
        V (np.ndarray): The arrays to join.
        pos_idx (Union[List, np.ndarray]): The positive indices. 
            Default is None.
        neg_idx (Union[List, np.ndarray]): The negative indices. 
            Default is None.
    Returns:
        np.ndarray: The joined array.
    """
    if is_empty(pos_idx) and is_empty(neg_idx):
        r = np.max(to_numpy(V), axis=0)
    elif is_empty(pos_idx) and not is_empty(neg_idx):
        pos_idx = diff_idcs(V, neg_idx)
        r = _join_all_pos_neg(V, pos_idx, neg_idx)
    else:
        r = _join_all_pos_neg(V, pos_idx, neg_idx)

    return r


if _HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _min_nonzeros_numba(
        V: np.ndarray,
        pos_idcs: np.ndarray,
        neg_idcs: np.ndarray
    ) -> np.ndarray:
        """Numba-accelerated min_nonzeros for 2D arrays with pos/neg indices.

        Args:
            V (np.ndarray): The array / matrix.
            pos_idcs (np.ndarray): The positive indices.
            neg_idcs (np.ndarray): The negative indices.
        Returns:
            np.ndarray: The minimum nonzero values of each column.
        """
        rows, cols = V.shape
        result = np.empty(cols, dtype=np.float64)

        # Process positive indices (min nonzero)
        for j in prange(pos_idcs.shape[0]):
            col = pos_idcs[j]
            min_val = np.inf
            for i in range(rows):
                val = V[i, col]
                if val != 0.0 and val < min_val:
                    min_val = val
            result[col] = 0.0 if min_val == np.inf else min_val

        # Process negative indices (max nonzero)
        for j in prange(neg_idcs.shape[0]):
            col = neg_idcs[j]
            max_val = -np.inf
            for i in range(rows):
                val = V[i, col]
                if val != 0.0 and val > max_val:
                    max_val = val
            result[col] = 0.0 if max_val == -np.inf else max_val

        return result


def min_nonzeros(
    V: np.ndarray,
    pos_idcs: Union[List, np.ndarray] = None,
    neg_idcs: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Compute the minimum nonzero values of each column in the array / matrix.

    Args:
        V (np.ndarray): The array / matrix.
        pos_idcs (Union[List, np.ndarray]): The positive indices (use min). 
            Default is None.
        neg_idcs (Union[List, np.ndarray]): The negative indices (use max). 
            Default is None.
    Returns:
        np.ndarray: The minimum nonzero values of each column.
    """
    V = np.asarray(V)
    pos_idcs, neg_idcs = init_indices(V, pos_idcs, neg_idcs)

    # All positive: simple case
    if is_empty(pos_idcs) and is_empty(neg_idcs):
        pos_idcs = np.arange(V.shape[-1])
        neg_idcs = np.array([], dtype=np.intp)
    else:
        pos_idcs = np.asarray(pos_idcs, dtype=np.intp)
        neg_idcs = np.asarray(neg_idcs, dtype=np.intp)

    if _HAS_NUMBA and V.ndim == 2:
        Vf = np.ascontiguousarray(V, dtype=np.float64)
        return _min_nonzeros_numba(Vf, pos_idcs, neg_idcs)

    # Fallback to optimized numpy
    Vf = V.astype(np.float64, copy=False)
    result = np.zeros(V.shape[-1], dtype=np.float64)

    if pos_idcs.size > 0:
        mins = np.min(
            Vf[..., pos_idcs], axis=0, where=(
                Vf[..., pos_idcs] != 0
            ), initial=np.inf
        )
        mins = np.asarray(mins).copy()
        mins[mins == np.inf] = 0.0
        result[pos_idcs] = mins

    if neg_idcs.size > 0:
        maxs = np.max(
            Vf[..., neg_idcs], axis=0, where=(
                Vf[..., neg_idcs] != 0
            ), initial=-np.inf
        )
        maxs = np.asarray(maxs).copy()
        maxs[maxs == -np.inf] = 0.0
        result[neg_idcs] = maxs

    return result
