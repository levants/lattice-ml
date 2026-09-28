"""Utilities for Arrays, Sparse Arrays, and 
Lattice-theoretic Formal Concept Analysis (FCA)."""

import logging
from functools import reduce
from typing import Any, List, Set, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(name=__file__)


def convert_to_array(a: Any) -> np.ndarray:
    """
    Converts the input into a NumPy array depending on the type
    of the input.

    Args:
        a (Any): The input to convert.
    Returns:
        np.ndarray: The converted NumPy array.
    """
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    elif isinstance(a, list):
        a = reduce(lambda x, y: x + y, a)
    elif isinstance(a, np.ndarray):
        a = a.flatten()
    elif isinstance(a, tuple):
        a = np.array(a).flatten()
    elif isinstance(a, dict):
        a = np.array(list(a.values())).flatten()
    elif isinstance(
        a,
        (set,
         str,
         bytes,
         range,
         memoryview,
         complex,
         bool,
         float,
         int,
         np.generic,)
    ):
        a = np.array(list(a)).flatten()
    else:
        a = np.array([a]).flatten()

    return a


def convert_to_set(a: Any) -> Set[int]:
    """
    Converts the input into a Set of integers depending on the type
    of the input.

    Args:
        a (Any): The input to convert.
    Returns:
        Set[int]: The converted Set of integers.
    """
    if isinstance(a, torch.Tensor):
        a = set(a.cpu().detach().tolist())
    elif isinstance(a, (list, tuple, set)):
        a = set(a)
    elif isinstance(a, np.ndarray):
        a = set(a.flatten().tolist())
    elif isinstance(a, dict):
        a = set(a.values())
    elif isinstance(
        a,
        (str,
         bytes,
         range,
         complex,
         bool,
         float,
         int,)
    ):
        a = set(list(a))
    else:
        a = set(list(a))

    return a


def powerset_bit(S: Iterable[Any]) -> List[Set[Any]]:
    """
    Generate the power set of the given iterable.

    Args:
        S (Iterable[Any]): The iterable to generate the power set of.
    Returns:
        List[Set[Any]]: The power set of the given iterable.
    """
    elements = list(S)
    n = len(elements)
    ps = []
    for i in range(1 << n):
        subset = {elements[j] for j in range(n) if (i & (1 << j))}
        ps.append(subset)

    return ps


def _arr_is_empty(a: np.ndarray) -> bool:
    """
    Check if the given array is empty.

    Args:
        a (np.ndarray): The array to check.
    Returns:
        bool: True if the array is empty, False otherwise.
    """
    return a is None or len(a) == 0 or a.size == 0


def is_empty(a: Union[List[Any], Set[Any], np.ndarray]) -> bool:
    """
    Check if the given array is None or empty.

    Args:
        a (np.ndarray): The array to check.
    Returns:
        bool: True if the array is empty, False otherwise.
    """
    return (
        a is None
        or (
            len(a) == 0
            if isinstance(a, (List, Tuple, Set, np.ndarray))
            else _arr_is_empty(convert_to_array(a))
        )
    )


def not_empty(a: np.ndarray) -> bool:
    """
    Check if the given array is not empty.

    Args:
        a (np.ndarray): The array to check.
    Returns:
        bool: True if the array is not empty, False otherwise.
    """
    return not is_empty(a)


def _in_idcs(
    idx: int,
    idcs: Union[List[int], List[np.ndarray], np.ndarray]
) -> bool:
    """Check if the index is in the list of indices.

    Args:
        idx (int): The index to check.
        idcs (Union[List[int], List[np.ndarray], np.ndarray]):
            The list of indices to check.
    Returns:
        bool: True if the index is in the list of indices,
            False otherwise.
    """
    return idx in idcs if isinstance(
        idcs, (np.ndarray, list)
    ) else int(idx) == int(idcs)


def in_any(
    idx: int,
    idcs: Union[List[int], List[np.ndarray], np.ndarray]
) -> bool:
    """Check if the index is in the list of indices.

    Args:
        idx (int): The index to check.
        idcs (Union[List[int], List[np.ndarray], np.ndarray]):
            The list of indices to check.
    Returns:
        bool: True if the index is in the list of indices,
            False otherwise.
    """
    return any(_in_idcs(idx, idc) for idc in idcs)


def truncate(arr: np.ndarray, decimals: int = 0) -> np.ndarray:
    """Truncate the array to the given number of decimal places.

    Args:
        arr (np.ndarray): The array to truncate.
        decimals (int): The number of decimal places to truncate to.
            Default is 0.
    Returns:
        np.ndarray: The truncated array.
    """
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor


def argmax_kd(v: np.ndarray) -> Tuple[int, ...]:
    """Get the indices of the maximum values in a multi-dimensional array.

    Args:
        v (np.ndarray): The array to get the indices of the maximum
            values from.
    Returns:
        tuple: The indices of the maximum values.
    """
    return np.unravel_index(np.argmax(v), v.shape)


def argmax_kd_val(v: np.ndarray) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Get the indices and values of the maximum values in a
        multi-dimensional array.

    Args:
        v (np.ndarray): The array to get the indices
            and values of the maximum values from.
    Returns:
        tuple: The indices and values of the maximum values.
    """
    max_idxs = argmax_kd(v)
    max_vals = v[max_idxs]

    return max_idxs, max_vals


def topK(a: Any, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Get the top K values and their indices from the array.

    Args:
        a (Any): The input array.
        k (int): The number of top values to retrieve.
            Default is None, which means all values are retrieved.
    Returns:
        tuple: A tuple containing the top K values and their indices.
    """
    k_rng = a.shape[0] if k is None else k
    a = to_numpy(a)
    idcs = np.argsort(a)[-k_rng:][::-1]

    return a[idcs], idcs


def topKrange(a: Any, range: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get the top K values and their indices from the array.

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


def topKProjedct(a: Any, k: int = None) -> np.ndarray:
    """Get the top K values and their indices from the array.

    Args:
        a (Any): The input array.
        k (int): The number of top values to retrieve. 
            Default is None, which means all values are retrieved.
    Returns:
        np.ndarray: The top K values projection vector.
    """
    arr = to_numpy(a)
    k_rng = arr.shape[0] if k is None else k
    vals, idcs = topK(arr, k_rng)
    p_k = np.zeros_like(arr)
    p_k[idcs] = vals

    return p_k


def project_on(a: Any, idx: Union[int, List[int], np.ndarray]) -> np.ndarray:
    """Get the top K values and their indices from the array.

    Args:
        a (Any): The input array.
        idx (int): The index to project the array on.
    Returns:
        np.ndarray: The projected array.
    """
    arr = to_numpy(a)
    p_k = np.zeros_like(arr)
    p_k[idx] = arr[idx]

    return p_k


def topRangeProject(
    a: Any,
    from_idx: int = 0,
    to_idx: int = None
) -> np.ndarray:
    """Get the top K values and their indices from the array.

    Args:
        a (Any): The input array.
        from_idx (int): The starting index. Default is 0.
        to_idx (int): The ending index. 
            Default is None, which means all values are retrieved from 
            from_idx to the end.
    Returns:
        np.ndarray: The top K values projection vector.
    """
    arr = to_numpy(a)
    k_rng = arr.shape[0] if to_idx is None else to_idx - from_idx
    vals, idcs = topK(arr, arr.shape[0])
    p_k = np.zeros_like(arr)
    p_k[idcs[from_idx:to_idx]] = vals[from_idx:to_idx]

    return p_k


def topNonZeros(a: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Get the top non-zero values and their indices from the array.

    Args:
        a (Any): The input array.
    Returns:
        tuple: A tuple containing the top non-zero values 
            and their indices.
    """
    a = to_numpy(a)
    idcs = np.nonzero(a)[0]
    vals = a[idcs]

    return vals, idcs


def printTopK(a: Any, k: int = 20) -> None:
    """Print the top K values and their indices from the array.

    Args:
        a (Any): The input array.
        k (int): The number of top values to retrieve. Default is 20.
    """
    vals, idxs = topK(a, k)
    logger.info(f'\n{repr(vals)}\n {repr(idxs)}')
    index_vals = '\n'.join(f'{val} {idx}' for val, idx in zip(vals, idxs))
    logger.info(f'\n{index_vals}')


def asort(a: np.ndarray, idx: int) -> np.ndarray:
    """Sort the array in ascending order based on the given index.

    Args:
        a (np.ndarray): The array to sort.
        idx (int): The index to sort the array by.
    Returns:
        np.ndarray: The sorted array.
    """
    return np.argsort(a[:, idx])


def dsort(a: np.ndarray, idx: int) -> np.ndarray:
    """Sort the array in descending order based on the given index.

    Args:
        a (np.ndarray): The array to sort.
        idx (int): The index to sort the array by.
    Returns:
        np.ndarray: The sorted array.
    """
    return np.argsort(a[:, idx])[::-1]


def set_v(
    v_X: np.ndarray,
    denm: float = 4.0,
    val_th: float = 0.2,
    asgn_max: bool = True,
    verbose: int = logging.INFO
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Set the values of the array based on the maximum value.

    Args:
        v_X (np.ndarray): The array to set the values of.
        denm (float): The denominator for the threshold.
            Default is 4.0.
        val_th (float): The threshold value. Default is 0.2.
        asgn_max (bool): Whether to assign the maximum value.
            Default is True.
        verbose (int): The logging level. Default is logging.INFO.
    Returns:
        Tuple[np.ndarray, float, np.ndarray]: The maximum index,
            maximum value, and the array with the values set.
    """
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


def set_vs(
    *v_Xs: np.ndarray,
    denm: float = 4.0,
    val_th: float = 0.2,
    asgn_max: bool = True,
    verbose: int = logging.INFO
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Set the values of the arrays based on the maximum value.

    Args:
        v_Xs (tuple): The arrays to set the values of.
        denm (float): The denominator for the threshold.
            Default is 4.0.
        val_th (float): The threshold value. Default is 0.2.
        asgn_max (bool): Whether to assign the maximum value.
            Default is True.
        verbose (int): The logging level. Default is logging.INFO.
    Returns:
        Tuple[np.ndarray, float, np.ndarray]: The maximum indices,
            maximum values, and the array with the values set.
    """
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
        v (Any): The input value, which can be a PyTorch tensor 
            or other types.
    Returns:
        np.ndarray: The converted NumPy array.
    """
    return v.cpu().detach().numpy() if isinstance(
        v,
        (torch.Tensor,)
    ) else v if isinstance(
        v,
        np.ndarray
    ) else np.array(v)


def le(v1: Any, v2: Any) -> Union[bool, np.ndarray]:
    """Converts inputs to numpy arrays and checks if the first array
       is less than or equal to the second array (element-wise).
       If the second array is a matrix, compare the first array with
       every row and return a boolean mask.

    Args:
        v1 (Any): The first array.
        v2 (Any): The second array.
    Returns:
        Union[bool, np.ndarray]: True if the first array is less than or
            equal to the second array, False otherwise. If the second array
            is two-dimensional, returns one boolean value for each row.
    """
    v1_np = to_numpy(v1)
    v2_np = to_numpy(v2)
    axis_idx = 0 if len(v2_np.shape) == 1 else 1

    return np.all(v1_np <= v2_np, axis=axis_idx)


def ge(v1: Any, v2: Any) -> bool:
    """Check if the first array is greater than or equal to the second array
       using the element-wise (Cartesian product) comparison.

    Args:
        v1 (Any): The first array.
        v2 (Any): The second array.
    Returns:
        bool: True if the first array is greater than or 
            equal to the second array, False otherwise.
    """
    v1_np = to_numpy(v1)
    v2_np = to_numpy(v2)
    axis_idx = 0 if len(v2_np.shape) == 1 else 1

    return np.all(v1_np >= v2_np, axis=axis_idx)
