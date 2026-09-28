"""Utilities for infomorhisms on lattice-theoretic contexts."""

from typing import Any, Set

import numpy as np

from src.lattmc.fca.fca_utils import FCA, TokenFCA
from src.lattmc.fca.lattice_utils import lower_mask
from src.lattmc.fca.utils import to_numpy


class InfoMorphism(object):
    """
    An abstract class for infomorphism between two Lattice-theoretic 
    Formal Contexts.

    Attributes:
        f (np.ndarray): The object mapping.
        g (np.ndarray): The attribute mapping.
    """

    def f(self, A: np.ndarray) -> np.ndarray:
        """
        The Object mapping of the infomorphism.

        Args:
            A (np.ndarray): The input objects.
        Returns:
            np.ndarray: The mapped objects.
        """
        raise NotImplementedError

    def g(self, u: Any) -> Any:
        """
        The Attribute mapping of the infomorphism.

        Args:
            u (Any): The input attributes.
        Returns:
            Any: The mapped attributes.
        """
        raise NotImplementedError


class FCAtoPCA(InfoMorphism):
    """
    A class for infomorphism from FCA to PCA.

    Attributes:
        fca (FCA): The FCA.
        pca (TokenFCA): The PCA.
        powS (np.ndarray): The power set of PCA attributes.
    """

    def __init__(
        self,
        fca: FCA,
        pca: TokenFCA,
    ):
        self._fca = fca
        self._pca = pca
        self._powS = pca.powS
        self._g_st = np.array([self.g(S) for S in self._powS])

    @property
    def fca(self):
        return self._fca

    @property
    def pca(self):
        return self._pca

    @property
    def powS(self):
        return self._powS

    @property
    def g_st(self):
        return self._g_st

    def _filter_lower(self, u: np.ndarray) -> List[np.ndarray]:
        """
        The lower filter of the FCA-to-PCA infomorphism.

        Args:
            u (np.ndarray): The input attributes.
        Returns:
            List[np.ndarray]: The lower filter of the FCA-to-PCA infomorphism.
        """
        mask = lower_mask(
            u,
            self.g_st,
            pos_idx=self.fca.pos_idcs,
            neg_idx=self.fca.neg_idcs
        )
        return self.powS[mask]

    def g_rev(self, u: np.ndarray) -> np.ndarray:
        """
        The reverse attribute mapping of the FCA-to-PCA infomorphism.

        Args:
            u (np.ndarray): The input attributes.
        Returns:
            np.ndarray: The reverse mapped attributes.
        """
        return set.union(*self._filter_lower(u))

    def pre_f(self, A: np.ndarray) -> np.ndarray:
        """
        The Pre Object mapping of the FCA-to-PCA infomorphism.

        Args:
            A (np.ndarray): The input (set of) objects.
        Returns:
            np.ndarray: The pre mapped (set of) objects.
        """
        return self.g_rev(self.fca.F(A))

    def f(self, A: np.ndarray) -> np.ndarray:
        """
        The Object mapping of the FCA-to-PCA infomorphism.

        Args:
            A (np.ndarray): The input (set of) objects.
        Returns:
            np.ndarray: The mapped (set of) objects.
        """
        g_st = self.pre_f(A)
        f_A = self.pca.G(g_st)

        return f_A

    def g(self, u: Set[int]) -> np.ndarray:
        """
        The Attribute mapping of the FCA-to-PCA infomorphism.

        Args:
            u (Set[int]): The input (set of) attributes.
        Returns:
            np.ndarray: The mapped (set of) attributes.
        """
        return self.fca.F(self.pca.G(u))
