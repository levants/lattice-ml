import logging
from functools import reduce
from typing import Any, List, Union

import numpy as np
import torch

logger = logging.getLogger(name=__file__)


def convert_to_array(a: Any) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    if isinstance(a, list):
        a = reduce(lambda x, y: x + y, a)
    if isinstance(a, np.ndarray):
        a = a.flatten()
    if isinstance(a, tuple):
        a = np.array(a).flatten()
    if isinstance(a, dict):
        a = np.array(list(a.values())).flatten()
    if isinstance(a, set):
        a = np.array(list(a)).flatten()
    if isinstance(a, str):
        a = np.array(list(a)).flatten()
    if isinstance(a, bytes):
        a = np.array(list(a)).flatten()
    if isinstance(a, range):
        a = np.array(list(a)).flatten()
    if isinstance(a, memoryview):
        a = np.array(list(a)).flatten()
    if isinstance(a, complex):
        a = np.array(list(a)).flatten()
    if isinstance(a, bool):
        a = np.array(list(a)).flatten()
    if isinstance(a, float):
        a = np.array(list(a)).flatten()
    if isinstance(a, int):
        a = np.array(list(a)).flatten()
    if isinstance(a, np.generic):
        a = np.array(list(a)).flatten()
    else:
        a = np.array([a]).flatten()

    return a


def _arr_is_empty(a: np.ndarray) -> bool:
    """
    Check if the given array is empty.
    Args:
        a (np.ndarray): The array to check.
    Returns:
        bool: True if the array is empty, False otherwise.
    """
    return a is None or len(a) == 0 or a.size == 0


def is_empty(a: Union[List[Any], np.ndarray]) -> bool:
    """
    Check if the given array is empty.
    Args:
        a (np.ndarray): The array to check.
    Returns:
        bool: True if the array is empty, False otherwise.
    """
    return a is None or len(a) == 0 or _arr_is_empty(to_numpy(a))


def not_empty(a: np.ndarray) -> bool:
    """
    Check if the given array is not empty.
    Args:
        a (np.ndarray): The array to check.
    Returns:
        bool: True if the array is not empty, False otherwise.
    """
    return not is_empty(a)


def _in_idcs(idx: int, idcs: Union[List[int], List[np.ndarray], np.ndarray]) -> bool:
    """Check if the index is in the list of indices."""
    return idx in idcs if isinstance(idcs, (
        np.ndarray,
        list
    )
    ) else int(idx) == int(idcs)


def in_any(
    idx: int,
    idcs: Union[List[int], List[np.ndarray], np.ndarray]
) -> bool:
    """Check if the index is in the list of indices."""
    return any(_in_idcs(idx, idc) for idc in idcs)


def truncate(arr, decimals=0):
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor


def argmax_kd(v: np.ndarray) -> tuple:
    return np.unravel_index(np.argmax(v), v.shape)


def argmax_kd_val(v):
    max_idxs = argmax_kd(v)
    max_vals = v[max_idxs]

    return max_idxs, max_vals


def topK(a: Any, k: int) -> tuple:
    """
    Get the top K values and their indices from the array.
    Args:
        a (Any): The input array.
        k (int): The number of top values to retrieve.
    Returns:
        tuple: A tuple containing the top K values and their indices.
    """
    a = to_numpy(a)
    idcs = np.argsort(a)[-k:][::-1]

    return a[idcs], idcs


def topKrange(a: Any, range: int) -> tuple:
    """
    Get the top K values and their indices from the array.
    Args:
        a (Any): The input array.
        range (int): The number of top values to retrieve.
    Returns:
        tuple: A tuple containing the top K values and their indices.
    """
    a = to_numpy(a)
    vals, idcs = topK(a, a.shape[0])
    vals = vals[:range]
    idcs = idcs[:range]

    return vals, idcs


def printTopK(a, k=20):
    vals, idxs = topK(a, k)
    logger.info(f'\n{repr(vals)}\n {repr(idxs)}')
    index_vals = '\n'.join(f'{val} {idx}' for val, idx in zip(vals, idxs))
    logger.info(f'\n{index_vals}')


def asort(a, idx):
    return np.argsort(a[:, idx])


def dsort(a, idx):
    return np.argsort(a[:, idx])[::-1]


def set_v(v_X, denm=4.0, val_th=0.2, asgn_max=True, verbose=logging.INFO):
    max_index, max_val = argmax_kd_val(v_X)
    logger.log(verbose, f'{max_index}, {max_index[0]}, {max_val}')
    neurons = np.zeros(v_X.shape)
    if asgn_max:
        neurons[max_index] = max_val
    else:
        fl = max_index[0]
        th = max_val - max_val / denm
        idxs = np.where(v_X[fl] >= th)
        neurons[fl][idxs] = val_th
    v = np.copy(neurons)

    return max_index, max_val, v


def set_vs(*v_Xs, denm=4.0, val_th=0.2, asgn_max=True, verbose=logging.INFO):
    v = None
    max_vals = list()
    max_indices = list()
    for v_X in v_Xs:
        max_index, max_val, _ = set_v(
            v_X,
            denm=denm,
            val_th=val_th,
            asgn_max=asgn_max,
            verbose=verbose,
        )
        v = np.zeros(v_X.shape, dtype=float) if v is None else v
        max_vals.append(max_val)
        max_indices.append(max_index)
    max_val_min = np.min(np.array(max_vals)) / denm
    for max_index in max_indices:
        v[max_index] = max_val_min

    return max_indices, max_vals, v


def to_numpy(v: Any) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.
    Args:
        v (Any): The input value, which can be a PyTorch tensor or other types.
    Returns:
        np.ndarray: The converted NumPy array.
    """

    if isinstance(
        v,
        (torch.Tensor,)
    ):
        arr = v.to('cpu').detach().numpy()
    elif isinstance(v, np.ndarray):
        arr = v
    else:
        arr = np.array(v)

    return arr


def le(v1, v2):
    v1_np = to_numpy(v1)
    v2_np = to_numpy(v2)

    return np.all(v1_np <= v2_np)


def ge(v1, v2):
    v1_np = to_numpy(v1)
    v2_np = to_numpy(v2)

    return np.all(v1_np >= v2_np)
