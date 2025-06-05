from functools import reduce
from typing import List, Tuple, Union

import numpy as np

from src.lattmc.fca.utils import is_empty, not_empty, to_numpy


def le(
    u: np.ndarray,
    v: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> bool:
    """Less or equals one vector than another"""
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


vectorized_le = np.vectorize(le, signature='(n),(n)->()')


def le_all(
    U: np.ndarray,
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> bool:
    return np.all(
        vectorized_le(U, V, pos_idx=pos_idx, neg_idx=neg_idx)
    )


def intersect(*arrs: np.ndarray) -> np.ndarray:
    return reduce(np.intersect1d, (arrs))


def intersect_xd(*arrs: np.ndarray) -> np.ndarray:
    return np.minimum.reduce(arrs)


def union(*arrs: np.ndarray) -> np.ndarray:
    return reduce(np.union1d, (arrs))


def diff_idcs(V: np.ndarray, neg_idx: Union[List, np.ndarray]) -> np.ndarray:
    """Gets positive indices from denative indices"""
    dm = V.shape[1] if len(V.shape) > 1 else V.shape[0]
    all_idx = np.arange(dm)
    pos_idx = np.setdiff1d(all_idx, neg_idx)

    return pos_idx


def init_indices(
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray],
    neg_idx: Union[List, np.ndarray]
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
    r = np.empty_like(u, dtype=u.dtype)
    r[pos_idx] = np.minimum(u[pos_idx], v[pos_idx])
    r[neg_idx] = np.maximum(u[neg_idx], v[neg_idx])

    return r


def _meet_all_pos_neg(
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray],
    neg_idx: Union[List, np.ndarray]
) -> np.ndarray:
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
    r = np.empty_like(u, dtype=u.dtype)
    r[pos_idx] = np.maximum(u[pos_idx], v[pos_idx])
    r[neg_idx] = np.minimum(u[neg_idx], v[neg_idx])

    return r


def _join_all_pos_neg(
    V: np.ndarray,
    pos_idx: Union[List, np.ndarray],
    neg_idx: Union[List, np.ndarray]
) -> np.ndarray:
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
    if is_empty(pos_idx) and is_empty(neg_idx):
        r = np.max(to_numpy(V), axis=0)
    elif is_empty(pos_idx) and not is_empty(neg_idx):
        pos_idx = diff_idcs(V, neg_idx)
        r = _join_all_pos_neg(V, pos_idx, neg_idx)
    else:
        r = _join_all_pos_neg(V, pos_idx, neg_idx)

    return r
