"""Array utilities for Lattice-theoretic Formal Concept Analysis (FCA)."""

from typing import Union

import numpy as np
import torch

def contains_torch_array(
    arr: torch.Tensor, 
    arrs: torch.Tensor) -> torch.Tensor:
    """Returns indices of rows in `arrs` that contain all elements of `arr`.
    
    Args:
        arr (torch.Tensor): 1-D array of elements to look for.
        arrs (torch.Tensor): 2-D array of rows to search in.

    Returns:
        torch.Tensor: 1-D index array of rows where every element of `arr` 
        is present.
    """
    arr_t = torch.as_tensor(arr, device=arrs.device).reshape(-1)
    arr_t = torch.unique(arr_t)
    mask = torch.ones(arrs.shape[0], dtype=torch.bool, device=arrs.device)
    for a in arr_t:
        mask &= (arrs == a).any(dim=1)
        if not mask.any().item():
            break
    res = mask.nonzero(as_tuple=True)[0]

    return res

def contains_numpy_array(
    arr: np.ndarray, 
    arrs: np.ndarray) -> np.ndarray:
    """Returns indices of rows in `arrs` that contain all elements of `arr`.
    
    Args:
        arr (np.ndarray): 1-D array of elements to look for.
        arrs (np.ndarray): 2-D array of rows to search in.

    Returns:
        np.ndarray: 1-D index array of rows where every element of `arr` 
        is present.
    """
    arr_np = np.unique(np.asarray(arr).reshape(-1))
    arrs_np = np.asarray(arrs)
    mask = np.ones(arrs_np.shape[0], dtype=bool)
    for a in arr_np:
        mask &= (arrs_np == a).any(axis=1)
        if not mask.any():
            break
    res = np.flatnonzero(mask)

    return res

def contains_array(
    arr: Union[np.ndarray, torch.Tensor],
    arrs: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Return indices of rows in `arrs` that contain all elements of `arr`.

    Args:
        arr (Union[np.ndarray, torch.Tensor]): 1-D array of elements 
            to look for.
        arrs (Union[np.ndarray, torch.Tensor]): 2-D array of rows 
            to search in.

    Returns:
        Union[np.ndarray, torch.Tensor]: 1-D index array of rows where every
            element of `arr` is present.
    """
    return contains_torch_array(
        arr, 
        arrs
    ) if isinstance(arrs, torch.Tensor) else contains_numpy_array(
        arr, 
        arrs
    )
    
