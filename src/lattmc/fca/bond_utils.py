""" Utilities for bounds on lattice-theoretic contexts."""


from typing import Any, Set, Tuple

import numpy as np

from src.lattmc.fca.fca_utils import FCA, TokenFCA
from src.lattmc.fca.lattice_utils import lower_mask
from src.lattmc.fca.utils import to_numpy


class BondOnLayers:
    """
    Bonds on different data items and sparse surrogate activations 
    lattice-theoretic formal contexts and layers on the same dataset.

    Attributes:
        fca_from (FCA): The FCA from which the bond is defined.
        fca_to (FCA): The FCA to which the bond is defined.
    """

    def __init__(self, fca_from: FCA, fca_to: FCA):
        self.fca_from = fca_from
        self.fca_to = fca_to

    def closures(
        self,
        A: np.ndarray,
        v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The closures of the bond.

        Args:
            A (np.ndarray): The input (set of) objects.
            v (np.ndarray): The input (set of) concepts.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The closures of the bond.
        """
        return self.fca_from.GF(A), self.fca_to.FG(v)

    def G_F(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The upper bound of the bond.

        Args:
            A (np.ndarray): The input (set of) objects.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The upper bound of the bond.
        """
        return self.fca_from.GF(A), self.fca_to.F(A)

    def F_G(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The upper bound of the bond.

        Args:
            v (np.ndarray): The input (set of) concepts.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The upper bound of the bond.
        """
        return self.fca_from.GF(v), self.fca_to.FG(v)

    def R_GF_GF(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The upper bound of the bond.

        Args:
            A (np.ndarray): The input (set of) objects.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The upper bound of the bond.
        """
        return (self.fca_from.GF(A), self.fca_to.GF(A))

    def R_FG_FG(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The lower bound of the bond.

        Args:
            v (np.ndarray): The input (set of) concepts.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The lower bound of the bond.
        """
        return (self.fca_from.FG(v), self.fca_to.FG(v))
