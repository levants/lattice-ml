from pathlib import Path
from typing import Callable, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.lattmc.fca.data_utils import layer_hist
from src.lattmc.fca.lattice_utils import (init_indices, intersect, join,
                                          join_all, le, meet, meet_all)
from src.lattmc.fca.utils import is_empty, not_empty, to_numpy


def layer_V(
    data: Union[List, np.ndarray],
    net: Callable,
    k: int = 5,
    bs: int = 1
) -> Tuple[np.ndarray, List]:
    V = list()
    X = list()
    with tqdm(list(range(0, len(data), bs))) as ds:
        for bi in ds:
            xs = [data[batch][0] for batch in range(bi, bi + bs)]
            vs = net(*xs, k=k)
            V.append(vs)
            X.extend(xs)

    return np.vstack(V), X


def loop_maxes(V: Union[np.ndarray, List], func: Callable, *args, **kwargs):
    with tqdm(V) as mstml:
        for i, v in enumerate(mstml):
            func(i, v, *args, **kwargs)


def select_top(
    V: Union[np.ndarray, List],
    idx: int,
    thresh: float
) -> List[int]:
    tops = list()

    def add_to_top(i, v):
        if thresh <= v[idx]:
            tops.append(i)
    loop_maxes(V, lambda i, v: add_to_top(i, v))

    return tops


def find_v_x(
    V: np.ndarray,
    mrng: Union[List, np.ndarray],
    idx: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    if is_empty(mrng):
        v_A = join_all(V, pos_idx=pos_idx, neg_idx=neg_idx)
    else:
        v_A = meet_all(to_numpy(V)[mrng], pos_idx=pos_idx, neg_idx=neg_idx)

    return v_A


def _separate_v_xs(V, idxs, model, y, pos_idx=None, neg_idx=None):
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


def find_v_A_model(V, mrng, pos_idx=None, neg_idx=None, model=None, y=None):
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
    disable_progress: bool = False
) -> np.ndarray:
    """Find the G_x for a given v_x."""
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
    neg_idx: Union[np.ndarray, List[int]] = None  # type: ignore
) -> List[np.ndarray]:
    G_As = list()
    for v_x in V_As:
        G_A = find_G_x(V, v_x, pos_idx=pos_idx, neg_idx=neg_idx)
        if G_A is not None and G_A.shape[0] > 0:
            G_As.append(G_A)

    return G_As


def find_V_X_digits(V_X, data):
    return [
        layer_hist(data, V_X, y=k) for k in range(10)
    ]


def sort_V(*V_Xs: np.ndarray) -> List[np.ndarray]:
    """Sorts the V_Xs by their values."""
    with tqdm(V_Xs) as pV_Xs:
        V_X_sr = [np.sort(V_X_d, axis=0) for V_X_d in V_Xs]

    return V_X_sr


def sort_V_X(V_X, data):
    V_X_ds = find_V_X_digits(V_X, data)
    V_X_sr = sort_V(*V_X_ds)

    return V_X_ds, V_X_sr


def features_hist(*n_Fs, V=np.zeros((1, 16))):
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
    """Formal concept"""

    def __init__(
        self,
        A: np.ndarray,
        v: np.ndarray,
        V: np.ndarray,
        pos_idx: Union[List, np.ndarray] = None,  # type: ignore
        neg_idx: Union[List, np.ndarray] = None  # type: ignore
    ) -> None:
        self._A = A
        self._v = v
        self._V = to_numpy(V)
        self._pos_idx = pos_idx
        self._neg_idx = neg_idx

    @property
    def V(self) -> np.ndarray:
        return self._V

    @property
    def A(self) -> np.ndarray:
        return self._A

    @A.setter
    def A(self, other_A: np.ndarray):
        self._A = other_A

    @property
    def v(self) -> np.ndarray:
        """Get the concept value."""
        return self._v

    @v.setter
    def v(self, other_v: np.ndarray):
        self._v = other_v

    @property
    def pos_idcs(self) -> Union[List, np.ndarray]:
        """Get the positive indices."""
        return self._pos_idx

    @property
    def pos_idx(self) -> Union[List, np.ndarray]:
        """Get the positive indices."""
        return self._pos_idx

    @property
    def neg_idcs(self) -> Union[List, np.ndarray]:
        """Get the negative indices."""
        return self._neg_idx

    @property
    def neg_idx(self) -> Union[List, np.ndarray]:
        """Get the negative indices."""
        return self._neg_idx

    def __eq__(self, value: object) -> bool:
        return np.all(self.v == value.v)  # type: ignore

    def __ne__(self, value: object) -> bool:
        return np.any(self.v != value.v)  # type: ignore

    def __lt__(self, other: object) -> bool:
        return not self.__eq__(other) and le(
            self.v,
            other.v,  # type: ignore
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

    def __le__(self, other: object) -> bool:
        # type: ignore
        return le(self.v, other.v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def __gt__(self, other: object) -> bool:
        return not self.__eq__(other) and le(
            other.v,  # type: ignore
            self.v,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

    def __ge__(self, other: object) -> bool:
        # type: ignore
        return le(other.v, self.v, pos_idx=self.pos_idx, neg_idx=self.neg_idx)

    def __and__(self, other):
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

    def __mul__(self, other):
        return self.__and__(other)

    def __or__(self, other):
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

    def __add__(self, other):
        return self.__or__(other)

    def __repr__(self):
        return f'{self.__class__.__name__}(A = ' \
            f'{self.A.shape}, v = {self.v.shape})'

    def __str__(self):
        return self.__repr__()


class FCA(object):
    """FCA implementation"""

    def __init__(
        self,
        V: Union[np.ndarray, List],
        pos_idx: Union[np.ndarray, List] = None,  # type: ignore
        neg_idx: Union[np.ndarray, List] = None  # type: ignore
    ) -> None:
        self._V = to_numpy(V)
        self._pos_idx, self._neg_idx = init_indices(
            self._V,
            pos_idx=pos_idx,
            neg_idx=neg_idx
        )
        # self._v_max = join_all(
        #     self._V,
        #     pos_idx=self._pos_idx,
        #     neg_idx=self._neg_idx,
        # )
        # self._v_min = meet_all(
        #     self._V,
        #     pos_idx=self._pos_idx,
        #     neg_idx=self._neg_idx,
        # )
        self._max = np.full_like(self._V[0], 1000.)
        self._min = np.zeros_like(self._V[0])

    @property
    def V(self) -> np.ndarray:
        """Get the concept value."""
        return self._V

    @property
    def pos_idcs(self) -> Union[List, np.ndarray]:
        """Get the positive indices."""
        return self._pos_idx

    @pos_idcs.setter
    def pos_idcs(self, other_idcs: Union[List, np.ndarray]):
        """Get the positive indices."""
        self._pos_idx = other_idcs

    @property
    def pos_idx(self) -> Union[List, np.ndarray]:
        """Get the positive indices."""
        return self._pos_idx

    @pos_idx.setter
    def pos_idx(self, other_idx: Union[List, np.ndarray]):
        """Get the positive indices."""
        self._pos_idx = other_idx

    @property
    def neg_idcs(self) -> Union[List, np.ndarray]:
        """Get the negative indices."""
        return self._neg_idx

    @neg_idcs.setter
    def neg_idcs(self, other_idx: Union[List, np.ndarray]):
        """Get the positive indices."""
        self._neg_idx = other_idx

    @property
    def neg_idx(self) -> Union[List, np.ndarray]:
        """Get the negative indices."""
        return self._neg_idx

    @neg_idx.setter
    def neg_idx(self, other_idx: Union[List, np.ndarray]):
        """Get the positive indices."""
        self._neg_idx = other_idx

    @property
    def v_max(self) -> np.ndarray:
        """Get the maximum value of the concept."""
        return self._v_max

    @property
    def v_min(self) -> np.ndarray:
        """Get the minimum value of the concept."""
        return self._v_min

    @property
    def shape(self) -> tuple:
        """Get the shape of the concept value."""
        return self.V.shape

    def F(self, idxs: Union[List, np.ndarray]) -> np.ndarray:
        return find_v_A(
            self.V,
            idxs,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

    def G(self, v: np.ndarray) -> np.ndarray:
        return find_G_x(
            self.V,
            v,
            pos_idx=self.pos_idx,
            neg_idx=self.neg_idx
        )

    def map_A(self, idxs: Union[List, np.ndarray]) -> Concept:
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

    def map_v(self, v: Union[List, np.ndarray]) -> Concept:
        v = to_numpy(v)
        G_v = self.G(v)
        if is_empty(G_v):
            concpt = Concept(
                np.array([], dtype=float),
                v,
                self.V,
                pos_idx=self.pos_idx,  # type: ignore
                neg_idx=self.neg_idx  # type: ignore
            )
        else:
            F_G_v = self.F(G_v)
            concpt = Concept(
                G_v,
                F_G_v,
                self.V,
                pos_idx=self.pos_idx,  # type: ignore
                neg_idx=self.neg_idx  # type: ignore
            )

        return concpt

    def GF_F(self, idxs: Union[List, np.ndarray]) -> Concept:
        return self.map_A(idxs)

    def G_FG(self, v: Union[List, np.ndarray]) -> Concept:
        return self.map_v(v)

    def save(self, path: Path):
        joblib.dump((self.V, self.pos_idx, self.neg_idx), path)

    @staticmethod
    def load(path):
        V, pos_idx, neg_idx = joblib.load(path)
        fca = FCA(V, pos_idx=pos_idx, neg_idx=neg_idx)

        return fca


class LayerFCA(object):
    """FCA classes for model layers"""

    def __init__(self, V_X, U_X, data):
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
        self.find_G_x = find_G_x

    def fca_v(self, ns, ths):
        for n_A, th_A in zip(ns, ths):
            G_A_v_A = select_top(self.V_X, n_A, th_A)
            v_A = find_v_A(self.V_X, G_A_v_A)
            self.v_As.append(v_A)
            G_A = find_G_x(self.V_X, v_A)
            self.G_As.append(G_A)
        self.D = intersect(*self.G_As) if self.G_As else []
        self.v_D = np.maximum.reduce(self.v_As)

        return self.D, self.v_D

    def fca_u(self, ns, ths):
        D, _ = self.fca_v(ns, ths)
        self.u_D = find_v_A(
            self.U_X, D
        ) if np.any(D) else np.zeros(
            (16,), dtype=float
        )
        self.G_u_D = find_G_x(self.U_X, self.u_D)

        return self.G_u_D

    @staticmethod
    def count_ys(ys):
        un, cn = np.unique(ys, return_counts=True)
        uncn = np.array([un, cn])

        return uncn

    def _report_u(self, G_u_D, data=None):
        data_ls = self.data if data is None else data
        ys = np.array([data_ls[idx][1] for idx in G_u_D])
        uncn = self.count_ys(ys)
        if data is None:
            self.uncn = uncn

        return uncn

    def report(self, G_u_D, data):
        return self._report_u(G_u_D, data=data)

    def fca_u_arr(self, ns_arr, neur_idx):
        ns = [nr[0] for nr in ns_arr[neur_idx]]
        ts = [nr[1] for nr in ns_arr[neur_idx]]
        G_U_D = self.fca_u(ns, ts)
        self._report_u(G_U_D)

        return self.G_u_D

    @staticmethod
    def G_U(U_X, u_D):
        return find_G_x(U_X, u_D)

    def find_u_G_u(self, v):
        G_v = find_G_x(self.V_X, v)
        u_D = find_v_A(self.U_X, G_v)
        G_u = find_G_x(self.U_X, u_D)
        self._report_u(G_u)

        return G_v, u_D, G_u

    def find_G_u(self, v, U, X):
        G_v, u_D, G_u = self.find_u_G_u(v)
        G_rest = find_G_x(U, u_D)

        return G_rest

    def find_G_v_us(self, v, V_X, U_X, data):
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
