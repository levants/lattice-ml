"""Utilities for Lattice-theoretic Formal Concept Analysis (FCA)."""

from __future__ import annotations
from typing import Set

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.lattmc.fca.data_utils import layer_hist
from src.lattmc.fca.lattice_utils import (init_indices, intersect, join,
                                          join_all, le, meet, meet_all,
                                          min_nonzeros)
from src.lattmc.fca.utils import is_empty, not_empty, powerset_bit, to_numpy

logger = logging.getLogger(__name__)


def layer_V(
    data: Union[List, np.ndarray],
    net: Callable,
    k: int = 5,
    bs: int = 1
) -> Tuple[np.ndarray, List]:
    """Compute the concept values for the given data.

    Args:
        data (Union[List, np.ndarray]): The data.
        net (Callable): The neural network.
        k (int, optional): The number of layers. Defaults to 5.
        bs (int, optional): The batch size. Defaults to 1.

    Returns:
        Tuple[np.ndarray, List]: The concept values and the data.
    """
    V = list()
    X = list()
    with tqdm(list(range(0, len(data), bs))) as ds:
        for bi in ds:
            xs = [data[batch][0] for batch in range(bi, bi + bs)]
            vs = net(*xs, k=k)
            V.append(vs)
            X.extend(xs)

    return np.vstack(V), X


def loop_maxes(
    V: Union[np.ndarray, List],
    func: Callable,
    *args: Any,
    **kwargs: Any
) -> None:
    """Loop through the concept values and apply the function.

    Args:
        V (Union[np.ndarray, List]): The concept values.
        func (Callable): The function to apply.
        *args (Any): The arguments to pass to the function.
        **kwargs (Any): The keyword arguments to pass to the function.
    """
    with tqdm(V) as mstml:
        for i, v in enumerate(mstml):
            func(i, v, *args, **kwargs)


def select_top(
    V: Union[np.ndarray, List],
    idx: int,
    thresh: float
) -> List[int]:
    """Select the top indices based on the threshold.

    Args:
        V (Union[np.ndarray, List]): The concept values.
        idx (int): The attribute index.
        thresh (float): The threshold.

    Returns:
        List[int]: The top indices.
    """
    tops = list()

    def add_to_top(i: int, v: np.ndarray) -> None:
        """Add the index to the top if the threshold is met.

        Args:
            i (int): The index.
            v (np.ndarray): The concept value.
        """
        if thresh <= v[idx]:
            tops.append(i)
    loop_maxes(V, lambda i, v: add_to_top(i, v))

    return tops


def find_v_x(
    V: np.ndarray,
    mrng: Union[List, np.ndarray],
    idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the concept value and index for a given attribute index.

    Args:
        V (np.ndarray): The concept values.
        mrng (Union[List, np.ndarray]): The range of indices.
        idx (int): The attribute index.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The concept value and index.
    """
    mid = np.argmin(to_numpy(V)[mrng], axis=0)[idx]
    x_id = mrng[mid]
    v_x = V[x_id]

    return v_x, x_id


def find_v_A(
    V: np.ndarray,
    mrng: Union[List, np.ndarray],
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Find the concept value for a given attribute indices.

    Args:
        V (np.ndarray): The concept values.
        mrng (Union[List, np.ndarray]): The range of indices.
        pos_idx (Union[List, np.ndarray], optional): The positive indices.
            Defaults to None.
        neg_idx (Union[List, np.ndarray], optional): The negative indices.
            Defaults to None.

    Returns:
        np.ndarray: The concept value.
    """
    if is_empty(mrng):
        v_A = join_all(V, pos_idx=pos_idx, neg_idx=neg_idx)
    else:
        v_A = meet_all(to_numpy(V)[mrng], pos_idx=pos_idx, neg_idx=neg_idx)

    return v_A


def _separate_v_xs(
    V: np.ndarray,
    idxs: Union[List, np.ndarray],
    model: Callable[..., Any],
    y: int,
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None,
) -> Tuple[np.ndarray, List, List]:
    """Separate the concept values based on the model.

    Args:
        V (np.ndarray): The concept values.
        idxs (Union[List, np.ndarray]): The indices.
        model (callable): The model.
        y (int): The target value.
        pos_idx (Union[List, np.ndarray], optional): The positive indices.
            Defaults to None.
        neg_idx (Union[List, np.ndarray], optional): The negative indices.
            Defaults to None.

    Returns:
        Tuple[np.ndarray, List, List]: The concept value,
            cluster 1, and cluster 2.
    """
    clust1 = list()
    clust2 = list()
    v_A = join_all(V, pos_idx=pos_idx, neg_idx=neg_idx)
    v_C = None
    for idx, i in enumerate(idxs):
        v = V[i]
        if idx == 0:
            v_C = v
        v_A = meet(v_A, v, pos_idx=pos_idx, neg_idx=neg_idx)
        if model(v_A) == y:
            clust1.append(i)
            v_C = np.copy(v_A)
        else:
            clust2.append(i)

    return v_C, clust1, clust2


def find_v_A_model(
    V: np.ndarray,
    mrng: Union[List, np.ndarray],
    pos_idx: Union[List, np.ndarray] = None,
    neg_idx: Union[List, np.ndarray] = None,
    model: Callable[..., Any] = None,
    y: int = None,
) -> Tuple[np.ndarray, List]:
    """Find the concept value for a given attribute indices.

    Args:
        V (np.ndarray): The concept values.
        mrng (Union[List, np.ndarray]): The range of indices.
        pos_idx (Union[List, np.ndarray], optional): The positive indices.
            Defaults to None.
        neg_idx (Union[List, np.ndarray], optional): The negative indices.
            Defaults to None.
        model (callable, optional): The model. Defaults to None.
        y (int, optional): The target value. Defaults to None.

    Returns:
        Tuple[np.ndarray, List]: The concept value and the clusters.
    """
    v_As = list()
    clusters = list()
    if model is None or y is None:
        v_A = find_v_A(V, mrng, pos_idx=pos_idx, neg_idx=neg_idx)
        v_As.append(v_A)
    else:
        V_arr = to_numpy(V)
        idxs = mrng.tolist() if isinstance(mrng, np.ndarray) else list(mrng)
        while len(idxs) > 0:
            v_C, clust1, clust2 = _separate_v_xs(
                V_arr,
                idxs,
                model,
                y,
                pos_idx=pos_idx,
                neg_idx=neg_idx,
            )
            clusters.append(to_numpy(clust1))
            v_As.append(v_C)
            idxs = clust2

    return to_numpy(v_As), clusters


def find_G_x(
    V: Union[np.ndarray, List],
    v_x: np.ndarray,
    pos_idx: Union[np.ndarray, List[int]] = None,  # type: ignore
    neg_idx: Union[np.ndarray, List[int]] = None,  # type: ignore
    disable_progress: bool = False,
) -> np.ndarray:
    """Find the G_x for a given v_x.

    Args:
        V (Union[np.ndarray, List]): The concept values.
        v_x (np.ndarray): The concept value.
        pos_idx (Union[np.ndarray, List[int]], optional): The positive indices.
            Defaults to None.
        neg_idx (Union[np.ndarray, List[int]], optional): The negative indices.
            Defaults to None.
        disable_progress (bool, optional): Disable the progress bar.
            Defaults to False.

    Returns:
        np.ndarray: The G_x.
    """
    v_x = to_numpy(v_x)
    with tqdm(V, disable=disable_progress) as mstm:
        G_x = np.array(
            [i for i, v in enumerate(mstm) if le(
                v_x,
                v,
                pos_idx=pos_idx,
                neg_idx=neg_idx
            )]
        )

    return G_x


def find_G_xs(
    V: Union[np.ndarray, List],
    V_As: np.ndarray,
    pos_idx: Union[np.ndarray, List[int]] = None,  # type: ignore
    neg_idx: Union[np.ndarray, List[int]] = None,  # type: ignore
) -> List[np.ndarray]:
    """Find the G_xs for a given V_As.

    Args:
        V (Union[np.ndarray, List]): The concept values.
        V_As (np.ndarray): The concept values.
        pos_idx (Union[np.ndarray, List[int]], optional): The positive indices.
            Defaults to None.
        neg_idx (Union[np.ndarray, List[int]], optional): The negative indices.
            Defaults to None.

    Returns:
        List[np.ndarray]: The G_xs.
    """
    G_As = list()
    for v_x in V_As:
        G_A = find_G_x(V, v_x, pos_idx=pos_idx, neg_idx=neg_idx)
        if G_A is not None and G_A.shape[0] > 0:
            G_As.append(G_A)

    return G_As


def find_V_X_digits(
    V_X: np.ndarray,
    data: Union[np.ndarray, List],
) -> List[np.ndarray]:
    """Find the V_X digits for a given data.

    Args:
        V_X (np.ndarray): The concept values.
        data (Union[np.ndarray, List]): The data.

    Returns:
        List[np.ndarray]: The V_X digits.
    """
    return [
        layer_hist(data, V_X, y=k) for k in range(10)
    ]


def sort_V(*V_Xs: np.ndarray) -> List[np.ndarray]:
    """Sorts the V_Xs by their values.

    Args:
        *V_Xs (np.ndarray): The concept values.

    Returns:
        List[np.ndarray]: The sorted concept values.
    """
    with tqdm(V_Xs) as pV_Xs:
        V_X_sr = [np.sort(V_X_d, axis=0) for V_X_d in V_Xs]

    return V_X_sr


def sort_V_X(
    V_X: np.ndarray,
    data: Union[np.ndarray, List],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Sort the V_X by their values.

    Args:
        V_X (np.ndarray): The concept values.
        data (Union[np.ndarray, List]): The data.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: The V_X digits
            and the sorted V_X digits.
    """
    V_X_ds = find_V_X_digits(V_X, data)
    V_X_sr = sort_V(*V_X_ds)

    return V_X_ds, V_X_sr


def features_hist(
    *n_Fs: int,
    V: np.ndarray = np.zeros((1, 16)),
) -> None:
    """Plot the histogram of the features.

    Args:
        *n_Fs (int): The feature indices.
        V (np.ndarray, optional): The concept values.
            Defaults to np.zeros((1, 16)).
    """
    rows = len(n_Fs)
    vs_ls = list()
    with tqdm(n_Fs) as pn_Fs:
        for n_F in pn_Fs:
            vs = [v[n_F] for v in V]
            vs_ls.append(vs)
    fig, axs = plt.subplots(rows, 1, sharey=True,
                            tight_layout=True, figsize=(8 * rows, 32))
    for r in range(rows):
        vs_h = vs_ls[r]
        axs[r].hist(vs_h)
        # Set the X-axis limit if you want a specific range
        axs[r].set_xlim(0, 32)
        # Ensure the ticks match the new range
        axs[r].set_xticks(np.arange(0, 32, 0.5))
        # axs[r].set_title(str(vs_h))


class Concept(object):
    """Formal concept class.

    Attributes:
        A (np.ndarray): The concept attributes.
        v (np.ndarray): The concept values.
        V (np.ndarray): The concept values.
        pos_idx (np.ndarray): The positive indices. Default is None.
        neg_idx (np.ndarray): The negative indices. Default is None.
    """

    def __init__(
        self: Concept,
        A: np.ndarray,
        v: np.ndarray,
        V: np.ndarray,
        pos_idx: Union[List, np.ndarray] = None,  # type: ignore
        neg_idx: Union[List, np.ndarray] = None  # type: ignore
    ) -> None:
        """Initialize Concept and its required state."""
        self._A = A
        self._v = v
        self._V = to_numpy(V)
        self._pos_idx = pos_idx
        self._neg_idx = neg_idx

    @property
    def V(self: Concept) -> np.ndarray:
        """Get the concept values.

        Returns:
            np.ndarray: The concept values.
        """
        return self._V

    @property
    def A(self: Concept) -> np.ndarray:
        """Get the concept attributes.

        Returns:
            np.ndarray: The concept attributes.
        """
        return self._A

    @A.setter
    def A(self: Concept, other_A: np.ndarray) -> None:
        """Set the concept attributes.

        Args:
            other_A (np.ndarray): The concept attributes.
        """
        self._A = other_A

    @property
    def v(self: Concept) -> np.ndarray:
        """Get the concept value.

        Returns:
            np.ndarray: The concept value.
        """
        return self._v

    @v.setter
    def v(self: Concept, other_v: np.ndarray) -> None:
        """Set the concept value.

        Args:
            other_v (np.ndarray): The concept value.
        """
        self._v = other_v

    @property
    def pos_idcs(self: Concept) -> Union[List, np.ndarray]:
        """Get the positive indices.

        Returns:
            Union[List, np.ndarray]: The positive indices.
        """
        return self._pos_idx

    @property
    def pos_idx(self: Concept) -> Union[List, np.ndarray]:
        """Get the positive indices.

        Returns:
            Union[List, np.ndarray]: The positive indices.
        """
        return self._pos_idx

    @property
    def neg_idcs(self: Concept) -> Union[List, np.ndarray]:
        """Get the negative indices.

        Returns:
            Union[List, np.ndarray]: The negative indices.
        """
        return self._neg_idx

    @property
    def neg_idx(self: Concept) -> Union[List, np.ndarray]:
        """Get the negative indices.

        Returns:
            Union[List, np.ndarray]: The negative indices.
        """
        return self._neg_idx

    def __eq__(self: Concept, value: object) -> bool:
        """Equal to.

        Args:
            value (object): The other concept.

        Returns:
            bool: True if self is equal to other, False otherwise.
        """
        return np.all(self.v == value.v)  # type: ignore

    def __ne__(self: Concept, value: object) -> bool:
        """Not equal to.

        Args:
            value (object): The other concept.

        Returns:
            bool: True if self is not equal to other, False otherwise.
        """
        return np.any(self.v != value.v)  # type: ignore

    def __lt__(self: Concept, other: object) -> bool:
        """Less than.

        Args:
            other (object): The other concept.

        Returns:
            bool: True if self is less than other, False otherwise.
        """
        return not self.__eq__(other) and le(
            self.v,
            other.v,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

    def __le__(self: Concept, other: object) -> bool:
        """Less than or equal to.

        Args:
            other (object): The other concept.

        Returns:
            bool: True if self is less than or equal to other, False otherwise.
        """
        return le(self.v, other.v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def __gt__(self: Concept, other: object) -> bool:
        """Greater than.

        Args:
            other (object): The other concept.

        Returns:
            bool: True if self is greater than other, False otherwise.
        """
        return not self.__eq__(other) and le(
            other.v,
            self.v,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

    def __ge__(self: Concept, other: object) -> bool:
        """Greater than or equal to.

        Args:
            other (object): The other concept.

        Returns:
            bool: True if self is greater than or equal to other,
                False otherwise.
        """
        return le(other.v, self.v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def __and__(self: Concept, other: Concept) -> 'Concept':
        """Intersection of two concepts.

        Args:
            other (object): The other concept.

        Returns:
            object: The intersection of self and other.
        """
        B = intersect(self.A, other.A)
        v = join(self.v, other.v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)
        G_v = find_G_x(self.V, v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)
        F_G_v = find_v_A(
            self.V,
            G_v,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )
        int_concpt = Concept(
            B,
            F_G_v,
            self.V,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

        return int_concpt

    def __mul__(self: Concept, other: Concept) -> 'Concept':
        """Multiplication of two concepts.

        Args:
            other (object): The other concept.

        Returns:
            object: The multiplication of self and other.
        """
        return self.__and__(other)

    def __or__(self: Concept, other: Concept) -> 'Concept':
        """Union of two concepts.

        Args:
            other (object): The other concept.

        Returns:
            object: The union of self and other.
        """
        B = np.union1d(self.A, other.A)
        v = meet(self.v, other.v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)
        F_B = find_v_A(self.V, B, pos_idx=self.pos_idx, neg_idx=self.neg_idx)
        G_F_B = find_G_x(
            self.V,
            F_B,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )
        uni_concept = Concept(
            G_F_B,
            v,
            self.V,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

        return uni_concept

    def __add__(self: Concept, other: Concept) -> 'Concept':
        """Addition of two concepts.

        Args:
            other (object): The other concept.

        Returns:
            object: The addition of self and other.
        """
        return self.__or__(other)

    def __repr__(self: Concept) -> str:
        """Representation of the concept.

        Returns:
            str: The representation of the concept.
        """
        return f'{self.__class__.__name__}(A = ' \
            f'{self.A.shape}, v = {self.v.shape})'

    def __str__(self: Concept) -> str:
        """String representation of the concept.

        Returns:
            str: The string representation of the concept.
        """
        return self.__repr__()


class FCA(object):
    """The Formal Context (FC) class for operation
        on Concept objects.

    Attributes:
        V (np.ndarray): The attribute values.
        pos_idx (np.ndarray): The positive indices. Default is None.
        neg_idx (np.ndarray): The negative indices. Default is None.
        v_min (np.ndarray): The minimum value. Default is None.
        v_max (np.ndarray): The maximum value. Default is None.
        v_min_nonzeros (np.ndarray): The minimum nonzero values.
            Default is None.
    """

    def __init__(
        self: FCA,
        V: Union[np.ndarray, List],
        pos_idx: Union[np.ndarray, List] = None,  # type: ignore
        neg_idx: Union[np.ndarray, List] = None,  # type: ignore
        min_val: Union[np.ndarray, List, float] = None,  # type: ignore
        max_val: Union[np.ndarray, List, float] = None,  # type: ignore
        v_min_nonzeros: Union[np.ndarray, List] = None,  # type: ignore
    ) -> None:
        """Initialize FCA and its required state."""
        self._V = to_numpy(V)
        self._pos_idx, self._neg_idx = init_indices(
            self._V,
            pos_idx=pos_idx,
            neg_idx=neg_idx
        )
        self._v_min = meet_all(
            self._V,
            pos_idx=self._pos_idx,
            neg_idx=self._neg_idx,
        ) if min_val is None else to_numpy(min_val) if isinstance(
            min_val, (np.ndarray, List)
        ) else np.full_like(self._V[0], min_val)
        self._v_max = join_all(
            self._V,
            pos_idx=self._pos_idx,
            neg_idx=self._neg_idx,
        ) if max_val is None else to_numpy(max_val) if isinstance(
            max_val, (np.ndarray, List)
        ) else np.full_like(self._V[0], max_val)
        self._v_min_nonzeros = min_nonzeros(
            self._V,
            pos_idcs=self._pos_idx,
            neg_idcs=self._neg_idx,
        ) if v_min_nonzeros is None else to_numpy(v_min_nonzeros)

    @property
    def V(self: FCA) -> np.ndarray:
        """Get the concept attributes.

        Returns:
            np.ndarray: The concept attributes.
        """
        return self._V

    @property
    def pos_idcs(self: FCA) -> Union[List, np.ndarray]:
        """Get the positive indices.

        Returns:
            Union[List, np.ndarray]: The positive indices.
        """
        return self._pos_idx

    @pos_idcs.setter
    def pos_idcs(self: FCA, other_idcs: Union[List, np.ndarray]) -> None:
        """Set the positive indices.

        Args:
            other_idcs (Union[List, np.ndarray]): The positive indices.
        """
        self._pos_idx = other_idcs

    @property
    def pos_idx(self: FCA) -> Union[List, np.ndarray]:
        """Get the positive indices.

        Returns:
            Union[List, np.ndarray]: The positive indices.
        """
        return self._pos_idx

    @pos_idx.setter
    def pos_idx(self: FCA, other_idx: Union[List, np.ndarray]) -> None:
        """Set the positive indices.

        Args:
            other_idx (Union[List, np.ndarray]): The positive indices.
        """
        self._pos_idx = other_idx

    @property
    def neg_idcs(self: FCA) -> Union[List, np.ndarray]:
        """Get the negative indices.

        Returns:
            Union[List, np.ndarray]: The negative indices.
        """
        return self._neg_idx

    @neg_idcs.setter
    def neg_idcs(self: FCA, other_idx: Union[List, np.ndarray]) -> None:
        """Set the negative indices.

        Args:
            other_idx (Union[List, np.ndarray]): The negative indices.
        """
        self._neg_idx = other_idx

    @property
    def neg_idx(self: FCA) -> Union[List, np.ndarray]:
        """Get the negative indices.

        Returns:
            Union[List, np.ndarray]: The negative indices.
        """
        return self._neg_idx

    @neg_idx.setter
    def neg_idx(self: FCA, other_idx: Union[List, np.ndarray]) -> None:
        """Set the negative indices.

        Args:
            other_idx (Union[List, np.ndarray]): The negative indices.
        """
        self._neg_idx = other_idx

    @property
    def v_max(self: FCA) -> np.ndarray:
        """Get the maximum value of the concept.

        Returns:
            np.ndarray: The maximum value of the concept.
        """
        return self._v_max

    @property
    def v_min(self: FCA) -> np.ndarray:
        """Get the minimum value of the concept.

        Returns:
            np.ndarray: The minimum value of the concept.
        """
        return self._v_min

    @property
    def v_min_nonzeros(self: FCA) -> np.ndarray:
        """Get the minimum nonzero value of the concept.

        Returns:
            np.ndarray: The minimum nonzero value of the concept.
        """
        return self._v_min_nonzeros

    @property
    def shape(self: FCA) -> Tuple[int, int]:
        """Get the shape of the concept value.

        Returns:
            Tuple[int, int]: The shape of the concept value.
        """
        return self.V.shape

    def min(self: FCA, i: Union[int, List[int], np.ndarray]) -> float:
        """Get the minimum value of the concept.

        Args:
            i (int): The index of the value.
        Returns:
            float: The minimum value of the concept.
        """
        sub_array = self.V[self.V[:, i] > 0]
        if not_empty(sub_array):
            _min_val = np.min(sub_array[:, i])
        else:
            _min_val = self.v_max[i]

        return _min_val

    def mins(self: FCA, *idxs: int) -> List[float]:
        """Get the minimum values of the concept on the given indices.

        Args:
            *idxs (int): The indices.

        Returns:
            List[float]: The minimum values
                of the concept on the given indices.
        """
        return [self.min(i) for i in idxs]

    def min_vals(self: FCA, idcs: Union[List[int], np.ndarray]) -> np.ndarray:
        """Get the minimum values of the concept on the given indices.

        Args:
            idcs (Union[List[int], np.ndarray]): The indices.

        Returns:
            np.ndarray: The minimum values
                of the concept on the given indices.
        """
        v_min = np.zeros(self.shape[1])
        v_min[idcs] = self.v_min_nonzeros[idcs]

        return v_min

    def le_att(self: FCA, u: np.ndarray, v: np.ndarray) -> bool:
        """Check if u is less than or equal to v.

        Args:
            u (np.ndarray): The first array.
            v (np.ndarray): The second array.

        Returns:
            bool: True if u is less than or equal to v, False otherwise.
        """
        return le(u, v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def meet_att(self: FCA, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute the meet of two vectors.

        Args:
            u (np.ndarray): The first vector.
            v (np.ndarray): The second vector.

        Returns:
            np.ndarray: The meet of the two vectors.
        """
        return meet(u, v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def join_att(self: FCA, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute the join of two vectors.

        Args:
            u (np.ndarray): The first vector.
            v (np.ndarray): The second vector.

        Returns:
            np.ndarray: The join of the two vectors.
        """
        return join(u, v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def meet_all_att(self: FCA, *args: Union[List, np.ndarray]) -> np.ndarray:
        """Compute the meet of all vectors.

        Args:
            *args (Union[List, np.ndarray]): The vectors.

        Returns:
            np.ndarray: The meet of all vectors.
        """
        return meet_all(*args, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def join_all_att(self: FCA, *args: Union[List, np.ndarray]) -> np.ndarray:
        """Compute the join of all vectors.

        Args:
            *args (Union[List, np.ndarray]): The vectors.

        Returns:
            np.ndarray: The join of all vectors.
        """
        return join_all(*args, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def F(self: FCA, idxs: Union[List, np.ndarray]) -> np.ndarray:
        """Objects on the indices to the attributed Galios mapping

        Args:
            idxs (Union[List, np.ndarray]): The object indices.

        Returns:
            np.ndarray: The attributed Galios mapping.
        """
        if is_empty(idxs):
            return self.v_max.copy()
        return find_v_A(
            self.V,
            idxs,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

    def G(self: FCA, v: np.ndarray) -> np.ndarray:
        """Attributes to the object Galios mapping

        Args:
            v (np.ndarray): The attribute indices.

        Returns:
            np.ndarray: The object Galios mapping.
        """
        return find_G_x(
            self.V,
            v,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

    def FG(self: FCA, u: np.ndarray) -> np.ndarray:
        """Embed attributes on a given indices into the concepts.

        Args:
            u (np.ndarray): The attribute indices.

        Returns:
            np.ndarray: The embedded concept.
        """
        return self.F(self.G(u))

    def GF(self: FCA, A: Union[List, np.ndarray]) -> np.ndarray:
        """Embed objects on a given indices into the concepts.

        Args:
            A (Union[List, np.ndarray]): The object indices.

        Returns:
            np.ndarray: The embedded concept.
        """
        return self.G(self.F(A))

    def map_A(self: FCA, idxs: Union[List, np.ndarray]) -> Concept:
        """Map the attributes into the concept.

        Args:
            idxs (Union[List, np.ndarray]): The attribute indices.

        Returns:
            Concept: The concept.
        """
        idxs = to_numpy(idxs)
        v_A = self.F(idxs)
        G_v_A = self.G(v_A)
        concpt = Concept(
            G_v_A,
            v_A,
            self.V,
            pos_idx=self.pos_idx,  # type: ignore
            neg_idx=self.neg_idx  # type: ignore
        )

        return concpt

    def map_v(self: FCA, v: Union[List, np.ndarray]) -> Concept:
        """Map the values into the concept.

        Args:
            v (Union[List, np.ndarray]): The concept values.

        Returns:
            Concept: The concept.
        """
        v = to_numpy(v)
        G_v = self.G(v)
        # An empty extent still has the closed intent F(empty).
        F_G_v = self.F(G_v)
        concpt = Concept(
            np.asarray(G_v, dtype=int),
            F_G_v,
            self.V,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx,
        )

        return concpt

    def GF_F(self: FCA, idxs: Union[List, np.ndarray]) -> Concept:
        """Embed objects on a given indices into the concepts.

        Args:
            idxs (Union[List, np.ndarray]): The objects.

        Returns:
            Concept: The concept.
        """
        return self.map_A(idxs)

    def G_FG(self: FCA, v: Union[List, np.ndarray]) -> Concept:
        """Embed attribute in the concepts.

        Args:
            v (Union[List, np.ndarray]): The attribute.

        Returns:
            Concept: The concept.
        """
        return self.map_v(v)

    def save(self: FCA, path: Path) -> None:
        """Save the FCA to a file.

        Args:
            path (Path): The path to save the FCA.
        """
        joblib.dump((self.V, self.pos_idx, self.neg_idx), path)

    @staticmethod
    def load(path: Path) -> FCA:
        """Load the FCA from a file.

        Args:
            path (Path): The path to load the FCA.

        Returns:
            FCA: The FCA.
        """
        V, pos_idx, neg_idx = joblib.load(path)
        fca = FCA(V, pos_idx=pos_idx, neg_idx=neg_idx)

        return fca

    def forward(self: FCA, v: Union[np.ndarray, List]) -> np.ndarray:
        """Get the FCA-specific vector for the given source vector.

        Args:
            v (Union[np.ndarray, List]): The source vector.

        Returns:
            np.ndarray: The FCA vector.
        """
        return v

    def __call__(self: FCA, v: Union[np.ndarray, List]) -> np.ndarray:
        """Get the FCA vector for the given values.

        Args:
            v (Union[np.ndarray, List]): The values.

        Returns:
            np.ndarray: The FCA vector.
        """
        return self.forward(v)


class NonzeroFCA(FCA):
    """Formal Context with only minimal nonzero activations, more like to
        classical formal context with attributes on sparse vectors.

    Attributes:
        V (np.ndarray): The attribute values.
        v_min_nonzeros (np.ndarray): The minimum nonzero values.
        pos_idx (np.ndarray): The positive indices.
        neg_idx (np.ndarray): The negative indices.
        v_min (np.ndarray): The minimum values.
    """

    def __init__(
        self: NonzeroFCA,
        V: Union[np.ndarray, List],
        v_bottom: Union[np.ndarray, List],
        pos_idx: Union[np.ndarray, List] = None,  # type: ignore
        neg_idx: Union[np.ndarray, List] = None,  # type: ignore
        min_val: Any = None,
    ) -> None:
        """Initialize NonzeroFCA and its required state."""
        V_nz = meet(
            V,
            v_bottom,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
        )
        super().__init__(
            V_nz,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            min_val=min_val,
            max_val=v_bottom,
            v_min_nonzeros=v_bottom,
        )

    @classmethod
    def fromFCA(cls: type[NonzeroFCA], fca: FCA) -> NonzeroFCA:
        """Create a NonzeroFCA from an FCA.

        Args:
            fca (FCA): The FCA.

        Returns:
            NonzeroFCA: The NonzeroFCA.
        """
        return NonzeroFCA(
            fca.V,
            fca.v_min_nonzeros,
            pos_idx=fca.pos_idx,
            neg_idx=fca.neg_idx,
            min_val=fca.v_min,
        )

    def forward(self: NonzeroFCA, v: Union[np.ndarray, List]) -> np.ndarray:
        """Get the FCA-specific (meet with minimal non-zero values vector)
            vector for the given vector.

        Args:
            v (Union[np.ndarray, List]): The source vector.

        Returns:
            np.ndarray: The FCA vector.
        """
        v = to_numpy(v)
        v_nz = meet(
            v,
            self.v_min_nonzeros,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx,
        )

        return v_nz


class TokenFCA(object):
    """Formal context for tokens in a corpus and a set of tokens.

    Attributes:
        corpus (Iterable): The tokens.
        T (Set[int]): The object values.
        X (np.ndarray): The attribute values.
        powS (np.ndarray): The powerset of T.
    """

    def __init__(
        self: TokenFCA,
        corpus: Iterable,
        T: Set[int],
    ) -> None:
        """
        Initialize the TokenFCA.

        Args:
            corpus (Iterable): The tokens.
            T (Set[int]): The object values.
        """
        self._corpus = np.array([
            set(ts.to('cpu').detach().tolist()) for ts in corpus
        ], dtype=set) if isinstance(
            corpus[0],
            torch.Tensor
        ) else np.array(
            corpus, dtype=set
        )
        self._T = T
        self._X = np.vstack(list(range(len(self._corpus))))
        self._powS = to_numpy(powerset_bit(T))

    @property
    def corpus(self: TokenFCA) -> List[Set[int]]:
        """Get the corpus."""
        return self._corpus

    @corpus.setter
    def corpus(self: TokenFCA, other_corpus: List[Set[int]]) -> None:
        """Set the corpus."""
        self._corpus = other_corpus
        self._X = np.vstack(list(range(len(other_corpus))))

    @property
    def T(self: TokenFCA) -> Set[int]:
        """Get the T."""
        return self._T

    @T.setter
    def T(self: TokenFCA, other_T: Set[int]) -> None:
        """Set the T."""
        self._T = other_T

    @property
    def X(self: TokenFCA) -> np.ndarray:
        """Get the X."""
        return self._X

    @X.setter
    def X(self: TokenFCA, other_X: np.ndarray) -> None:
        """Set the X."""
        self._X = other_X

    @property
    def powS(self: TokenFCA) -> np.ndarray:
        """Get the powerset of T."""
        return self._powS

    @powS.setter
    def powS(self: TokenFCA, other_powS: np.ndarray) -> None:
        """Set the powerset of T."""
        self._powS = other_powS

    def F(self: TokenFCA, A: Union[np.ndarray, List]) -> Set[int]:
        """Get the F for the given values.

        Args:
            A (Union[np.ndarray, List]): The attribute values.

        Returns:
            np.ndarray: The F for the given values.
        """
        if is_empty(A):
            F_A = self.T
        else:
            A = to_numpy(A)
            with tqdm(self.corpus[A], desc='F') as porpus:
                F_A = set.intersection(*[
                    set([
                        t for t in self.T if t in x
                    ]) for x in porpus
                ])

        return F_A

    def G(self: TokenFCA, S: Union[np.ndarray, List, Set]) -> np.ndarray:
        """Get the G for the given values.

        Args:
            S (Set[int]): The object values.

        Returns:
            np.ndarray: The G for the given values.
        """
        if is_empty(S):
            G_S = np.array(list(range(len(self.X))))
        else:
            S_set = set(S)
            with tqdm(self.corpus, desc='G') as porpus:
                G_S = np.array(list(
                    {i for i, x in enumerate(porpus) if S_set <= x}
                ))

        return G_S

    def GF(self: TokenFCA, A: Union[np.ndarray, List]) -> np.ndarray:
        """Get the G for the given values.

        Args:
            A (Union[np.ndarray, List]): The attribute values.

        Returns:
            np.ndarray: The G for the given values.
        """
        return self.G(self.F(A))

    def FG(self: TokenFCA, S: Set[int]) -> np.ndarray:
        """Get the F for the given values.

        Args:
            S (Set[int]): The object values.

        Returns:
            np.ndarray: The F for the given values.
        """
        return self.F(self.G(S))

    def is_closed_in_X(self: TokenFCA, A: Union[np.ndarray, List]) -> bool:
        """Check if the given values are closed in X.

        Args:
            A (Union[np.ndarray, List]): The attribute values.

        Returns:
            bool: True if the given values are closed in X, False otherwise.
        """
        return np.array_equal(self.GF(A), A)

    def is_closed_in_T(self: TokenFCA, S: Set[int]) -> bool:
        """Check if the given values are closed in T.

        Args:
            S (Set[int]): The object values.

        Returns:
            bool: True if the given values are closed, False otherwise.
        """
        return self.FG(S) == set(S)

    def check_attributes_closed_in_FCA(
        self: TokenFCA,
        logg_att: bool = True,
    ) -> List[bool]:
        """
        Check if the attribute sets in $T$ are closed in $L(X, T, R)$.

        Args:
            logg_att (bool, optional): Whether to log the attribute sets.
                Defaults to True.

        Returns:
            List[bool]: List of booleans indicating whether each attribute
                set in $T$ is closed in $L(X, T, R)$.
        """
        cl_pows = list()
        for ps in self.powS:
            is_cl = self.is_closed_in_T(ps)
            cl_pows.append(is_cl)
            if logg_att:
                logger.info(f'{ps} is closed in FCA: {is_cl}')

        return cl_pows

    def are_attributes_closed(
        self: TokenFCA,
        logg_att: bool = True,
    ) -> bool:
        """
        Check if the attribute sets in $T$ are closed in $L(X, T, R)$.

        Args:
            logg_att (bool, optional): Whether to log the attribute sets.
                Defaults to True.

        Returns:
            bool: Boolean indicating whether all attribute sets in $T$ are
                closed in $L(X, T, R)$.
        """
        cl_pows = self.check_attributes_closed_in_FCA(logg_att=logg_att)
        att_closed = all(cl_pows)
        logger.info(f"Are attributes closed in FCA: {att_closed}")

        return att_closed


class LayerFCA(object):
    """Formal Context for activations in model layers.

    Attributes:
        V_X (np.ndarray): The attribute values.
        U_X (np.ndarray): The object values.
        data (Union[np.ndarray, List]): The data.
        G_As (List[np.ndarray]): The FCA for the values.
        v_As (List[np.ndarray]): The FCA for the values.
        D (np.ndarray): The FCA for the values.
        v_D (np.ndarray): The FCA for the values.
        U_D (np.ndarray): The FCA for the values.
        G_U_D (np.ndarray): The FCA for the values.
        find_G_x (Callable): The function to find the FCA for the values.
        find_v_A (Callable): The function to find the FCA for the values.
    """

    def __init__(
        self: LayerFCA,
        V_X: np.ndarray,
        U_X: np.ndarray,
        data: Union[np.ndarray, List],
    ) -> None:
        """Initialize LayerFCA and its required state."""
        self.V_X = V_X
        self.U_X = U_X
        self.data = data
        self.G_As = list()
        self.v_As = list()
        self.D = None
        self.v_D = None
        self.U_D = None
        self.G_U_D = None
        self.find_G_x = find_G_x
        self.find_v_A = find_v_A

    def fca_v(
        self: LayerFCA,
        ns: List[int],
        ths: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the FCA for the values.

        Args:
            ns (List[int]): The number of attributes.
            ths (List[float]): The thresholds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The FCA and the concept values.
        """
        for n_A, th_A in zip(ns, ths):
            G_A_v_A = select_top(self.V_X, n_A, th_A)
            v_A = find_v_A(self.V_X, G_A_v_A)
            self.v_As.append(v_A)
            G_A = find_G_x(self.V_X, v_A)
            self.G_As.append(G_A)
        self.D = intersect(*self.G_As) if self.G_As else []
        self.v_D = np.maximum.reduce(self.v_As)

        return self.D, self.v_D

    def fca_u(
        self: LayerFCA,
        ns: List[int],
        ths: List[float],
    ) -> np.ndarray:
        """Compute the FCA for the values.

        Args:
            ns (List[int]): The number of attributes.
            ths (List[float]): The thresholds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The FCA and the concept values.
        """
        D, _ = self.fca_v(ns, ths)
        self.u_D = find_v_A(
            self.U_X, D
        ) if np.any(D) else np.zeros(
            (16,), dtype=float
        )
        self.G_u_D = find_G_x(self.U_X, self.u_D)

        return self.G_u_D

    @staticmethod
    def count_ys(ys: np.ndarray) -> np.ndarray:
        """Count the number of unique values.

        Args:
            ys (np.ndarray): The values.

        Returns:
            np.ndarray: The unique values and their counts.
        """
        un, cn = np.unique(ys, return_counts=True)
        uncn = np.array([un, cn])

        return uncn

    def _report_u(
        self: LayerFCA,
        G_u_D: np.ndarray,
        data: Union[np.ndarray, List] = None,
    ) -> np.ndarray:
        """Report the unique values.

        Args:
            G_u_D (np.ndarray): The unique values.
            data (Union[np.ndarray, List], optional): The data. 
                Defaults to None.

        Returns:
            np.ndarray: The unique values and their counts.
        """
        data_ls = self.data if data is None else data
        ys = np.array([data_ls[idx][1] for idx in G_u_D])
        uncn = self.count_ys(ys)
        if data is None:
            self.uncn = uncn

        return uncn

    def report(
        self: LayerFCA,
        G_u_D: np.ndarray,
        data: Union[np.ndarray, List],
    ) -> np.ndarray:
        """Report the unique values.

        Args:
            G_u_D (np.ndarray): The unique values.
            data (Union[np.ndarray, List]): The data.

        Returns:
            np.ndarray: The unique values and their counts.
        """
        return self._report_u(G_u_D, data=data)

    def fca_u_arr(
        self: LayerFCA,
        ns_arr: np.ndarray,
        neur_idx: int,
    ) -> np.ndarray:
        """Compute the FCA for the values.

        Args:
            ns_arr (np.ndarray): The number of attributes.
            neur_idx (int): The neuron index.

        Returns:
            np.ndarray: The FCA.
        """
        ns = [nr[0] for nr in ns_arr[neur_idx]]
        ts = [nr[1] for nr in ns_arr[neur_idx]]
        G_U_D = self.fca_u(ns, ts)
        self._report_u(G_U_D)

        return self.G_u_D

    @staticmethod
    def G_U(
        U_X: np.ndarray,
        u_D: np.ndarray
    ) -> np.ndarray:
        """Compute the FCA for the values.

        Args:
            U_X (np.ndarray): The values.
            u_D (np.ndarray): The values.

        Returns:
            np.ndarray: The FCA.
        """
        return find_G_x(U_X, u_D)

    def find_u_G_u(
        self: LayerFCA,
        v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the FCA for the values.

        Args:
            v (np.ndarray): The values.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The FCA 
                and the concept values.
        """
        G_v = find_G_x(self.V_X, v)
        u_D = find_v_A(self.U_X, G_v)
        G_u = find_G_x(self.U_X, u_D)
        self._report_u(G_u)

        return G_v, u_D, G_u

    def find_G_u(
        self: LayerFCA,
        u: np.ndarray,
        U: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Embeds an element u into the formal concept.

        Args:
            u (np.ndarray): The values.
            U (np.ndarray): The values.
            X (np.ndarray): The values.

        Returns:
            np.ndarray: The FCA.
        """
        G_v, u_D, G_u = self.find_u_G_u(u)
        G_rest = find_G_x(U, u_D)

        return G_rest

    def find_G_v_us(
        self: LayerFCA,
        v: np.ndarray,
        V_X: np.ndarray,
        U_X: np.ndarray,
        data: Union[np.ndarray, List],
    ) -> (
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]
    ):
        """Compute the FCA for the values.

        Args:
            v (np.ndarray): The values.
            V_X (np.ndarray): The values.
            U_X (np.ndarray): The values.
            data (Union[np.ndarray, List]): The data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                np.ndarray, np.ndarray, np.ndarray]: The FCA 
                    and the concept values.
        """
        G_v, u_D, G_u = self.find_u_G_u(v)
        v_D = self.find_v_A(
            self.V_X,
            G_v
        ) if np.any(G_v) else np.array([], dtype=float)
        G_v_test = self.find_G_x(V_X, v)
        G_u_test = self.find_G_x(U_X, u_D)
        uncn_test = self.report(G_u_test, data)
        uncn_reps = [
            uncn_test,
            np.round(
                uncn_test[1] / np.sum(uncn_test[1]), decimals=4
            )
        ]

        return G_v, v_D, u_D, G_u, G_v_test, G_u_test, uncn_reps
