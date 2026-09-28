"""Transcoder FCA analysis and representation for 
Lattice-theoretic Formal Concept Analysis (FCA) on 
SAE / transcoder activations."""

import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.lattmc.fca.fca_utils import FCA
from src.lattmc.sae.nlp_sae_utils import join_all, load_vectors, save_vectors
from src.lattmc.tc.transcoder_utils import Transcoder

logger = logging.getLogger(__name__)


class FCAVectorSerializer(object):
    """Utility class to serialize and load FCA vectors.

    Attributes:
        path (Path): The path.
    """

    MIN_VAL_NAME = 'v_min'
    MAX_VAL_NAME = 'v_max'
    MIN_NONZERO_VAL_NAME = 'v_min_nonzero'
    V_VAL_NAME = 'V'

    def __init__(self, path: Path, ext: str = 'npz'):
        self._path = path
        self._ext = ext

    @property
    def path(self) -> Path:
        """Get the path.

        Returns:
            Path: The path.
        """
        return self._path

    @property
    def ext(self) -> str:
        """Get the extension.

        Returns:
            str: The extension.
        """
        return self._ext

    @staticmethod
    def load_V(V_path: Path) -> np.ndarray:
        """Load the V matrix from the given path.

        Args:
            V_path: (Path) The path to the V matrix.

        Returns:
            (np.ndarray) The V matrix.
        """
        V_sparse = load_vectors(V_path)
        V = V_sparse.toarray()

        return V

    def _init_v_val_path(
        self,
        layer: int,
        value_name: str = 'min_nonzeros',
    ) -> Path:
        """Initialize the path to the V matrix, minimum, maximim 
        or min_nonzero values path.

        Args:
            layer: (int) The layer of the model.
            value_name: (str) The name of the value.

        Returns:
            (Path) The path to the V matrix.
        """
        return self.path / f'{value_name}{layer}.{self.ext}'

    def _load_array_if_exists(
        self,
        layer: int,
        value_name: str = 'min_nonzeros'
    ) -> np.ndarray:
        """Load the V matrix from the given path if it exists.

        Args:
            layer: (int) The layer of the model.
            value_name: (str) The name of the value.

        Returns:
            (np.ndarray) The V vector, matrix or min/max/min_nonzero values.
        """
        v_val_path = self._init_v_val_path(layer, value_name)
        if v_val_path.exists():
            logger.info(f'{v_val_path} exists')
            v_val = self.load_V(v_val_path)
            logger.info(f'{v_val_path} loaded')
        else:
            logger.info(f'{v_val_path} does not exist')
            v_val = None

        return v_val

    def _save_array_if_not_exists(
        self,
        v_val: np.ndarray,
        layer: int,
        value_name: str = 'min_nonzeros'
    ) -> None:
        """Save the V matrix to the given path if it does not exist.

        Args:
            v_val: (np.ndarray) The V vector, matrix or min/max/min_nonzero 
                values.
            layer: (int) The layer of the model.
            value_name: (str) The name of the value / file to be saved.
        """
        v_val_path = self._init_v_val_path(layer, value_name)
        if v_val_path.exists():
            logger.info(f'{v_val_path} exists')
        else:
            logger.info(f'{v_val_path} does not exist')
            logger.info(f'Creating {v_val_path}')
            V_sparse = csr_matrix(v_val)
            save_vectors(v_val_path, V_sparse)
            logger.info(f'{v_val_path} created')

    def _extract_min_max_nonzeros(
        self,
        layer: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract min, max, and min_nonzeros values for the given layer.

        Args:
            layer: (int) The layer of the model.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]) The min, max, 
                and min_nonzeros values.
        """
        v_min = self._load_array_if_exists(layer, self.MIN_VAL_NAME)
        v_max = self._load_array_if_exists(layer, self.MAX_VAL_NAME)
        v_min_nonzeros = self._load_array_if_exists(
            layer,
            self.MIN_NONZERO_VAL_NAME,
        )

        return v_min, v_max, v_min_nonzeros

    def _save_min_max_nonzeros(
        self,
        v_min: np.ndarray,
        v_max: np.ndarray,
        v_min_nonzeros: np.ndarray,
        layer: int,
    ) -> None:
        """Save min, max, and min_nonzeros values for the given layer.

        Args:
            v_min: (np.ndarray) The min values.
            v_max: (np.ndarray) The max values.
            v_min_nonzeros: (np.ndarray) The min_nonzeros values.
            layer: (int) The layer of the model.
        """
        self._save_array_if_not_exists(
            v_min, layer, self.MIN_VAL_NAME)
        self._save_array_if_not_exists(
            v_max, layer, self.MAX_VAL_NAME)
        self._save_array_if_not_exists(
            v_min_nonzeros,
            layer,
            self.MIN_NONZERO_VAL_NAME,
        )


class TranscoderUtils(FCAVectorSerializer):
    """Utility class to serialize and load transcoder activations and 
        compute lattice-theoretic FCA on those latent activations.

    Attributes:
        transcoder (Transcoder): The transcoder.
        tokens (np.ndarray): The tokens.
        path (Path): The path.
        ext (str): The extension of the file. Default is 'npz'.
        pos_idxs (Dict[int, Union[List[int], np.ndarray]]): 
            The positive indices. Default is None.
        neg_idxs (Dict[int, Union[List[int], np.ndarray]]): 
            The negative indices. Default is None.
    """

    def __init__(
        self,
        transcoder: Transcoder,
        tokens: np.ndarray,
        path: Path,
        ext: str = 'npz',
        pos_idxs: Dict[int, Union[List[int], np.ndarray]] = None,
        neg_idxs: Dict[int, Union[List[int], np.ndarray]] = None,
    ):
        super().__init__(path, ext)
        self._transcoder = transcoder
        self._tokens = tokens
        self._pos_idxs = pos_idxs if pos_idxs else dict()
        self._neg_idxs = neg_idxs if neg_idxs else dict()

    @property
    def transcoder(self) -> Transcoder:
        """Get the transcoder.

        Returns:
            Transcoder: The transcoder.
        """
        return self._transcoder

    @property
    def tokens(self) -> np.ndarray:
        """Get the tokens (corpus).

        Returns:
            np.ndarray: The tokens.
        """
        return self._tokens

    @property
    def pos_idxs(self) -> Dict[int, Union[List[int], np.ndarray]]:
        """Get the positive indices.

        Returns:
            Dict[int, Union[List[int], np.ndarray]]: The positive indices.
        """
        return self._pos_idxs

    @property
    def neg_idxs(self) -> Dict[int, Union[List[int], np.ndarray]]:
        """Get the negative indices.

        Returns:
            Dict[int, Union[List[int], np.ndarray]]: The negative indices.
        """
        return self._neg_idxs

    def _generate_joint_values(self, layer: int) -> List[np.ndarray]:
        """Generate joint values for the given layer.

        Args:
            layer: (int) The layer of the model.

        Returns:
            (List[np.ndarray]) The V matrix.
        """
        Vs = []
        with tqdm(self.tokens) as ptokens:
            for tk in ptokens:
                vs = self.transcoder(tk, layer)[0]
                v = join_all(vs)
                Vs.append(v)

        return Vs

    def create_V(self, layer: int) -> csr_matrix:
        """Create embedding for layer.

        Args:
            layer: (int) The layer of the model.

        Returns:
            (csr_matrix) The V matrix.
        """
        Vs = self._generate_joint_values(layer)
        V = np.array(Vs)
        V_sparse = csr_matrix(V)

        return V_sparse

    def create_and_save(self, layer: int) -> np.ndarray:
        """Create, serialize and save the V (joint per token latent vectors 
            for each text item) matrix for the given layer.

        Args:
            layer: (int) The layer of the model to compute the vectors.

        Returns:
            (np.ndarray) The V matrix.
        """
        V_path = self._init_v_val_path(layer, value_name='V')
        if V_path.exists():
            logger.info(f'{V_path} exists')
            V = self.load_V(V_path)
            logger.info(f'{V_path} loaded')
        else:
            logger.info(f'{V_path} does not exist')
            logger.info(f'Creating {V_path}')
            V_sparse = self.create_V(layer)
            save_vectors(V_path, V_sparse)
            logger.info(f'{V_path} serialized and saved')
            V = self.load_V(V_path)
            logger.info(f'{V_path} created and loaded')

        return V

    def create_layers(self, layers: List[int]) -> Dict[int, np.ndarray]:
        """Create and save the V matrix for the given layers.

        Args:
            layers: (List[int]) The layers.

        Returns:
            (Dict[int, np.ndarray]) The V matrix.
        """
        VS = dict()
        for layer in layers:
            logger.info(f'Creating V for layer {layer}')
            V = self.create_and_save(layer)
            VS[layer] = V

        return VS

    def init_fcas(self, layers: List[int]) -> Dict[int, FCA]:
        """Initialize the FCA from the sparse activations for the 
            given layers.

        Args:
            layers: (List[int]) The layers.

        Returns:
            (Dict[int, FCA]) The FCA.
        """
        VS = self.create_layers(layers)
        fcas = {}
        for layer in layers:
            V = VS[layer]
            v_min, v_max, v_min_nonzeros = self._extract_min_max_nonzeros(
                layer)
            fca = FCA(
                V,
                pos_idx=self.pos_idxs.get(layer, None),
                neg_idx=self.neg_idxs.get(layer, None),
                min_val=v_min,
                max_val=v_max,
                v_min_nonzeros=v_min_nonzeros,
            )
            self._save_min_max_nonzeros(
                fca.v_min,
                fca.v_max,
                fca.v_min_nonzeros,
                layer,
            )
            fcas[layer] = fca

        return fcas

    def run_transcoders(
        self,
        prompt: str,
        layers: List[int],
    ) -> Dict[int, np.ndarray]:
        """Run the transcoders on the given prompt for the given layers.

        Args:
            prompt: (str) The prompt.
            layers: (List[int]) The layers.

        Returns:
            (Dict[int, np.ndarray]) The transcoder outputs.
        """
        return self.transcoder.run_layers(prompt, layers)

    def print_tokens(self, prompt: str):
        """Print the text tokens from the given prompt.

        Args:
            prompt: (str) The prompt.
        """
        self.transcoder.print_text_tokens(prompt)
