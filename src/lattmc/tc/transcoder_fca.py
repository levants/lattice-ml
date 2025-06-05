import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.lattmc.fca.fca_utils import FCA
from src.lattmc.sae.nlp_sae_utils import join_all, load_vectors, save_vectors
from src.lattmc.tc.transcoders_utils import Transcoder

logger = logging.getLogger(__name__)


class TranscoderUtils(object):
    """Utility class to serialize and load transcoder activations"""

    def __init__(
        self,
        transcoder: Transcoder,
        tokens: np.ndarray,
        path: Path,
        pos_idxs: Dict[int, Union[List[int], np.ndarray]] = None,
        neg_idxs: Dict[int, Union[List[int], np.ndarray]] = None,
    ):
        super().__init__()
        self._transcoder = transcoder
        self._tokens = tokens
        self._path = path
        self._pos_idxs = pos_idxs if pos_idxs else {}
        self._neg_idxs = neg_idxs if neg_idxs else {}

    @property
    def transcoder(self) -> Transcoder:
        """Get the transcoder."""
        return self._transcoder

    @property
    def tokens(self) -> np.ndarray:
        """Get the tokens."""
        return self._tokens

    @property
    def path(self) -> Path:
        """Get the path."""
        return self._path

    @property
    def pos_idxs(self) -> Dict[int, Union[List[int], np.ndarray]]:
        """Get the positive indices."""
        return self._pos_idxs

    @property
    def neg_idxs(self) -> Dict[int, Union[List[int], np.ndarray]]:
        """Get the negative indices."""
        return self._neg_idxs

    def create_V(self, layer: int) -> csr_matrix:
        """Create embedding for layer"""
        Vs = []
        with tqdm(self.tokens) as ptokens:
            for tk in ptokens:
                vs = self.transcoder(tk, layer)[0]
                v = join_all(vs)
                Vs.append(v)
        V = np.array(Vs)
        V_sparse = csr_matrix(V)

        return V_sparse

    def load_V(self, V_path: Path) -> np.ndarray:
        """Load the V matrix from the given path."""
        V_sparse = load_vectors(V_path)
        V = V_sparse.toarray()

        return V

    def create_and_save(self, layer: int) -> np.ndarray:
        """Create and save the V matrix for the given layer."""
        V_path = self.path / f'V{layer}.npz'
        if V_path.exists():
            logger.info(f'{V_path} exists')
            V = self.load_V(V_path)
            logger.info(f'{V_path} loaded')
        else:
            logger.info(f'{V_path} does not exist')
            logger.info(f'Creating {V_path}')
            V_sparse = self.create_V(layer)
            save_vectors(V_path, V_sparse)
            V = self.load_V(V_path)
            logger.info(f'{V_path} created and loaded')

        return V

    def create_layers(self, layers: List[int]) -> Dict[int, np.ndarray]:
        """Create and save the V matrix for the given layers."""
        VS = {}
        for layer in layers:
            logger.info(f'Creating V for layer {layer}')
            V = self.create_and_save(layer)
            VS[layer] = V

        return VS

    def init_fcas(self, layers: List[int]) -> Dict[int, FCA]:
        """Initialize the FCA for the given layers."""
        VS = self.create_layers(layers)
        fcas = {}
        for layer in layers:
            V = VS[layer]
            fca = FCA(
                V,
                pos_idx=self.pos_idxs.get(layer, None),
                neg_idx=self.neg_idxs.get(layer, None),
            )
            fcas[layer] = fca

        return fcas

    def run_transcoders(
        self,
        prompt: str,
        layers: List[int],
    ) -> Dict[int, np.ndarray]:
        """Run the transcoder for the given layers."""
        return self.transcoder.run_layers(prompt, layers)

    def print_tokens(self, prompt: str):
        """Print the tokens."""
        self.transcoder.print_text_tokens(prompt)
