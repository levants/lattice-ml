"""NLP SAE / Transcoder utilities for 
Lattice-theoretic Formal Concept Analysis (FCA)."""

import gc
import logging
import os
import pprint
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple, Union

import joblib
import numpy as np
import torch
from regex import D
from sae_lens import SAE
from scipy.sparse import csr_matrix, isspmatrix_csr, load_npz, save_npz
from torch import nn
from tqdm import tqdm
from transformer_lens import HookedTransformer
from zmq import device

from src.lattmc.fca.fca_utils import FCA, Concept, find_G_x
from src.lattmc.fca.file_utils import not_exists
from src.lattmc.fca.lattice_utils import join_all, meet_all
from src.lattmc.fca.utils import is_empty, not_empty, to_numpy

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True


def init_device():
    """
    Initialize the device.

    Returns:
        str: The device.
    """
    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    logger.info(f'Device: {device}')

    return device


def empty_cache():
    """
    Empty the cache.

    Returns:
        Any: The garbage collected objects size.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gc.collect()


def normalize_sae_device(device: Union[str, torch.device]) -> str:
    """Convert torch.device objects into the string form sae_lens expects."""
    return str(device) if isinstance(device, torch.device) else device


def add_library_level(level: int = 4):
    """
    Add the library level to the sys.path.

    Args:
        level (int): The level. Defaults to 4.
    """
    suf_path = ['..']
    path = '..'
    for i in range(0, level):
        join_path = suf_path * i
        path = '/'.join(join_path)
        module_path = os.path.abspath(os.path.join(path))
        if module_path not in sys.path:
            sys.path.append(module_path)
            logger.info(f'Appendeding {path}')


def mkdirs(*dirs: Path):
    """
    Make directories.

    Args:
        *dirs (Path): The directories.
    """
    for dir in dirs:
        dir.mkdir(exist_ok=True, parents=True)


def clean_eof(text: str) -> str:
    """Remove <|endoftext|> from the text

    Args:
        text (str): The text.

    Returns:
        str: The text without <|endoftext|> from the end.
    """

    return text.replace('�', '')


class Text2Latent(object):
    """
    Text to latent.

    Args:
        model_name (str): The model name.
        release (str): The release.
        sae_id (str): The sae id.
        device (Union[str, torch.device]): The device.
        model (HookedTransformer, optional): The model. Defaults to None.
    """

    METADATA = 'metadata'
    HOOK_NAME = 'hook_name'

    def __init__(
        self,
        model_name: str,
        release: str,
        sae_id: str,
        device: Union[str, torch.device],
        model: HookedTransformer = None
    ):
        self._model_name = model_name
        self._sae_id = sae_id
        self._release = release
        self._device = device
        _model, sae, cfg_dict, _ = self._init_model_and_sae(model)
        self._model = _model.eval()
        self._sae = sae.eval()
        self._cfg_dict = cfg_dict
        self._hook_point = self._init_hook_name()

    def _init_sae(self) -> SAE:
        """Initialize the SAE.

        Returns:
            SAE: The SAE.
        """
        sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=self.release,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=self.sae_id,  # won't always be a hook point
            device=normalize_sae_device(self.device),
        )
        sae = sae.to(self.device)
        logger.info(f'cfg_dict:\n{pprint.pformat(cfg_dict)}')
        logger.info(f'{sparsity=}')

        return sae, cfg_dict, sparsity

    def _init_model(self) -> Tuple[HookedTransformer, SAE, Dict, torch.Tensor]:
        """Initialize the model.

        Returns:
            Tuple[HookedTransformer, SAE, Dict, torch.Tensor]: The model,
                SAE, config dict, and sparsity.
        """
        sae, cfg_dict, sparsity = self._init_sae()
        model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device
        )

        return model, sae, cfg_dict, sparsity

    def _init_model_and_sae(
        self, _model: HookedTransformer
    ) -> Tuple[HookedTransformer, SAE, Dict, torch.Tensor]:
        """Initialize the model and SAE.

        Args:
            _model (HookedTransformer): The model.

        Returns:
            Tuple[HookedTransformer, SAE, Dict, torch.Tensor]: The model,
                SAE, config dict, and sparsity.
        """
        sae, cfg_dict, sparsity = self._init_sae()
        model = self._init_model() if _model is None else _model

        return model, sae, cfg_dict, sparsity

    @property
    def model_name(self) -> str:
        """Get the model name.

        Returns:
            str: The model name.
        """
        return self._model_name

    @property
    def release(self) -> str:
        """Get the release.

        Returns:
            str: The release.
        """
        return self._release

    @property
    def sae_id(self) -> str:
        """Get the SAE ID.

        Returns:
            str: The SAE ID.
        """
        return self._sae_id

    @property
    def device(self) -> Union[str, torch.device]:
        """Get the device.

        Returns:
            Union[str, torch.device]: The device.
        """
        return self._device

    @property
    def model(self) -> HookedTransformer:
        """Get the model.

        Returns:
            HookedTransformer: The model.
        """
        return self._model

    @property
    def sae(self) -> SAE:
        """Get the SAE.

        Returns:
            SAE: The SAE.
        """
        return self._sae

    @property
    def cfg_dict(self) -> Dict:
        """Get the config dict.

        Returns:
            Dict: The config dict.
        """
        return self._cfg_dict

    @property
    def hook_point(self) -> str:
        """Get the hook point.

        Returns:
            str: The hook point.
        """
        return self._hook_point

    def _init_hook_name(self) -> str:
        """Initialize the hook name.

        Returns:
            str: The hook name.
        """
        if hasattr(self.sae.cfg, self.HOOK_NAME):
            hook_point = self.sae.cfg.hook_name
            logger.info(f'{hook_point=}')
        elif (
            self.METADATA in self.cfg_dict and
            self.HOOK_NAME in self.cfg_dict[self.METADATA]
        ):
            hook_point = self.cfg_dict[self.METADATA][self.HOOK_NAME]
            logger.info(f'{hook_point=}')
        else:
            hook_point = self.sae_id
            logger.info(f'Inferred {hook_point=}')

        return hook_point

    def to(self, device: Union[str, torch.device]) -> Any:
        """Move the model and SAE to the given device.

        Args:
            device (Union[str, torch.device]): The device.

        Returns:
            Any: The model and SAE.
        """
        self.model.to(device)
        self.sae.to(device)

        return self

    def eval(self) -> Any:
        """Set the model and SAE to evaluation mode.

        Returns:
            Any: The model and SAE.
        """
        self.model.eval()
        self.sae.eval()

        return self

    @torch.inference_mode()
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize the text.

        Args:
            text (str): The text.

        Returns:
            torch.Tensor: The tokens.
        """
        return self.model.to_tokens(text)

    @torch.inference_mode()
    def to_string(self, tokens: torch.Tensor) -> str:
        """Convert tokens to string.

        Args:
            tokens (torch.Tensor): The tokens.

        Returns:
            str: The string.
        """
        return self.model.to_string(tokens)

    @torch.inference_mode()
    def encode(self, text: Union[str, torch.Tensor]) -> torch.Tensor:
        """Encode the text.

        Args:
            text (Union[str, torch.Tensor]): The text.

        Returns:
            torch.Tensor: The encoded text.
        """
        _, cache = self.model.run_with_cache(text, prepend_bos=True)
        # get the feature activations from our SAE
        z = self.sae.encode(cache[self.hook_point])

        return z

    @torch.inference_mode()
    def embed(self, text: Union[str, torch.Tensor]) -> np.ndarray:
        """Embed the text.

        Args:
            text (Union[str, torch.Tensor]): The text.

        Returns:
            np.ndarray: The embedded text.
        """
        h = self.encode(text)
        z = to_numpy(h)

        return z

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the SAE activations.

        Args:
            z (torch.Tensor): The SAE activations.

        Returns:
            torch.Tensor: The decoded activations.
        """
        return self.sae.decode(z)

    @torch.inference_mode()
    def forward(self, text: Union[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the model and SAE.

        Args:
            text (Union[str, torch.Tensor]): The text.

        Returns:
            torch.Tensor: The decoded activations.
        """
        z = self.encode(text)
        r = self.decode(z)

        return r


class Text2Sae(Text2Latent):
    """
    Text to SAE.

    Args:
        model_name (str): The model name.
        release (str): The release.
        sae_id (str): The sae id.
        device (Union[str, torch.device]): The device.
    """

    def __init__(
        self,
        model_name: str,
        release: str,
        sae_id: str,
        device: Union[str, torch.device],
    ):
        self.model_name = model_name
        self.release = release
        self.sae_id = sae_id
        self.device = device
        model, sae, cfg_dict, _ = self._init_model()
        super().__init__(model, sae, cfg_dict, sae_id)

    def _init_model(self) -> Tuple[HookedTransformer, SAE, Dict, torch.Tensor]:
        """Initialize the model.

        Returns:
            Tuple[HookedTransformer, SAE, Dict, torch.Tensor]: The model,
                SAE, config dict, and sparsity.
        """
        sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=self.release,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=self.sae_id,  # won't always be a hook point
            device=normalize_sae_device(self.device),
        )
        sae = sae.to(self.device)
        logger.info(f'cfg_dict:\n{pprint.pformat(cfg_dict)}')
        logger.info(f'{sparsity=}')
        hook_point = sae.cfg.hook_name
        logger.info(f'{hook_point=}')
        model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device
        )

        return model, sae, cfg_dict, sparsity


class WordMapper(object):
    """
    Word mapper.

    Args:
        net (nn.Module): The net.
        dataset (Iterable): The dataset.
        matrix_dir (Union[str, Path]): The matrix directory.
        ext (str, optional): The extension. Defaults to 'npz'.
    """

    def __init__(
        self,
        net: nn.Module,
        dataset: Iterable,
        matrix_dir: Union[str, Path],
        ext: str = 'npz'
    ):
        self.net = net
        self.dataset = dataset
        self.matrix_dir = matrix_dir
        self.ext = ext

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize the text.

        Args:
            text (str): The text.

        Returns:
            torch.Tensor: The tokens.
        """
        return self.net.tokenize(text)

    def get(self, idx: int) -> Any:
        """Get the data at the given index.

        Args:
            idx (int): The index.

        Returns:
            Any: The data at the given index.
        """
        return self.dataset[idx]

    def to_string(self, tokens: torch.Tensor) -> str:
        """Convert tokens to string.

        Args:
            tokens (torch.Tensor): The tokens.

        Returns:
            str: The string.
        """
        return self.net.to_string(tokens)

    def select_ws(
        self,
        A: np.ndarray,
        v: np.ndarray,
        log=False
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Select the word vectors.

        Args:
            A (np.ndarray): The word vectors.
            v (np.ndarray): The target vector.
            log (bool, optional): Whether to log the results. 
                Defaults to False.

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]: The word 
                vectors and their indices.
        """
        idxs = {}
        vecs = {}
        with tqdm(A, desc='Searching through tokens', disable=log) as pA:
            for i in pA:
                vector_path = self.matrix_dir / f'{i}.{self.ext}'
                v_sparse = load_vectors(vector_path, ext=self.ext)
                if log:
                    logger.info(vector_path)
                else:
                    pA.set_postfix_str(f'Vector: {vector_path}')
                vs = v_sparse.toarray()
                gs = find_G_x(vs, v, disable_progress=True)
                idxs[i.item()] = gs
                v_idxs = meet_all(vs[gs]) if not_empty(
                    gs
                ) else np.full(vs.shape[1], 10000.0)
                vecs[i.item()] = v_idxs

        return idxs, vecs

    @staticmethod
    def _get_min_max(context: Union[bool, List] = False) -> Tuple[int, int]:
        """Get the min and max context.

        Args:
            context (Union[bool, List], optional): The context. 
                Defaults to False.

        Returns:
            Tuple[int, int]: The min and max context.
        """
        if context is None:
            cmin = 0
            cmax = 9
        elif isinstance(context, List) or isinstance(context, tuple):
            cmin, cmax = context
        elif isinstance(context, bool) and context:
            cmin = 2
            cmax = 2
        else:
            cmin = 0
            cmax = 0

        return cmin, cmax

    @staticmethod
    def _slide_dict(idxs: Dict[Any, Any], top_k: int = None) -> Dict[Any, Any]:
        """Slide the dictionary.

        Args:
            idxs (Dict[Any, Any]): The dictionary.
            top_k (int, optional): The top k. Defaults to None.

        Returns:
            Dict[Any, Any]: The slid dictionary.
        """
        if top_k is None:
            sidxs = idxs
        else:
            with tqdm(
                enumerate(idxs.items()), desc='Slicing words'
            ) as pslices:
                sidxs = {
                    ki: vi for nm, (ki, vi) in pslices if nm < top_k
                }
            logger.info(
                f'Selected {top_k=}, from {len(idxs)=} to {len(sidxs)=}'
            )

        return sidxs

    def find_words(
        self,
        idxs: np.ndarray,
        context: Union[bool, List[Any]] = None,
        top_k: int = None
    ) -> List[Any]:
        """Find the words.

        Args:
            idxs (np.ndarray): The indices.
            context (Union[bool, List[Any]], optional): The context. 
                Defaults to None.
            top_k (int, optional): The top k. Defaults to None.

        Returns:
            List[Any]: The words.
        """
        words = list()
        cmin, cmax = WordMapper._get_min_max(context)
        sidxs = WordMapper._slide_dict(idxs, top_k=top_k)
        with tqdm(
            sidxs.items(),
            total=len(sidxs),
            desc='Localizing words',
            position=0,
            leave=True,
        ) as pidxs:
            for k, gs in pidxs:
                if not_empty(gs):
                    ws = []
                    for idx in gs:
                        tkns = self.tokenize(self.get(k)['text'])[0]
                        w = self.to_string(tkns[idx])
                        if context:
                            start = max(idx - cmin, 0)
                            end = min(len(tkns), idx + cmax)
                            ctxt = self.to_string(tkns[start:end])
                            ctxc = ctxt.replace('<|endoftext|>', '')
                            item = [k, w, ctxc]
                        else:
                            item = [k, w]
                        ws.append(item)
                        pidxs.set_postfix_str(f'Word: {w}')
                    words.append(ws)

        return words

    def search_words(
        self,
        cn: np.ndarray,
        v: np.ndarray = None,
        log: bool = False,
        context: bool = False,
        top_k: int = None,
        indices_only: bool = False
    ) -> List[str]:
        """Search for the words.

        Args:
            cn (np.ndarray): The word vectors.
            v (np.ndarray, optional): The target vector. 
                Defaults to None.
            log (bool, optional): Whether to log the results. 
                Defaults to False.
            context (bool, optional): Whether to include the context. 
                Defaults to False.
            top_k (int, optional): The top k. Defaults to None.
            indices_only (bool, optional): Whether to return only the indices. 
                Defaults to False.

        Returns:
            List[str]: The words.
        """
        idxs, vecs = self.select_ws(cn.A, cn.v if is_empty(v) else v, log=log)
        v_idxs = meet_all(list(vecs.values()))
        if indices_only:
            words = SimpleNamespace(word_indices=idxs, vecs=vecs, v=v_idxs)
        else:
            words = SimpleNamespace(
                word_indices=idxs,
                vecs=vecs,
                v=v_idxs,
                words=self.find_words(idxs, context=context, top_k=top_k),
            )

        return words

    def forward(
        self,
        cn: np.ndarray,
        v: np.ndarray = None,
        context: bool = False,
        top_k: int = None,
        indices_only: bool = False
    ) -> Union[List[str], SimpleNamespace]:
        """Forward pass through the model and SAE.

        Args:
            cn (np.ndarray): The word vectors.
            v (np.ndarray, optional): The target vector. 
                Defaults to None.
            context (bool, optional): Whether to include the context. 
                Defaults to False.
            top_k (int, optional): The top k. Defaults to None.
            indices_only (bool, optional): Whether to return only the indices. 
                Defaults to False.

        Returns:
            Union[List[str], SimpleNamespace]: The words or a simple namespace.
        """
        return self.search_words(
            cn,
            v=v,
            context=context,
            top_k=top_k,
            indices_only=indices_only)

    def __call__(self, *args, **kwargs):
        """Call the forward method.

        Args:
            *args: The arguments.
            **kwargs: The keyword arguments.

        Returns:
            Union[List[str], SimpleNamespace]: The words or a simple namespace.
        """
        return self.forward(*args, **kwargs)


def to_sparse(v: np.ndarray) -> csr_matrix:
    """Convert a vector to a sparse vector.

    Args:
        v (np.ndarray): The vector.

    Returns:
        csr_matrix: The sparse vector.
    """
    return v if isspmatrix_csr(v) else csr_matrix(v)


def to_array(v: Union[np.ndarray, csr_matrix]) -> np.ndarray:
    """Convert a vector to an array.

    Args:
        v (Union[np.ndarray, csr_matrix]): The vector.

    Returns:
        np.ndarray: The array.
    """
    return v.toarray() if isspmatrix_csr(v) else v


def save_vectors(
    word_vec_path: Path,
    v_sparse: Union[np.ndarray, csr_matrix],
    ext: str = 'npz'
):
    """Serialize and save the vectors to the file.

    Args:
        word_vec_path (Path): The word vector path.
        v_sparse (Union[np.ndarray, csr_matrix]): The sparse vector.
        ext (str, optional): The extension. Defaults to 'npz'.
    """
    if ext == 'npz':
        save_npz(word_vec_path, v_sparse, compressed=True)
    else:
        joblib.dump(v_sparse, word_vec_path)


def load_vectors(
    word_vec_path: Path,
    ext: str = 'npz'
) -> Union[np.ndarray, csr_matrix]:
    """Load the vectors from the file.

    Args:
        word_vec_path (Path): The word vector path.
        ext (str, optional): The extension. Defaults to 'npz'.

    Returns:
        Union[np.ndarray, csr_matrix]: The sparse vector.
    """
    return load_npz(
        word_vec_path
    ) if ext == 'npz' else joblib.load(
        word_vec_path
    )


def init_matrices(
    matrix_dir: Path,
    dataset: Iterable,
    net: Text2Latent,
    ext: str = 'npz'
):
    """Initialize the matrices.

    Args:
        matrix_dir (Path): The matrix directory.
        dataset (Iterable): The dataset.
        net (Text2Latent): The network.
        ext (str, optional): The extension. Defaults to 'npz'.
    """
    if any(
        Path(matrix_dir).iterdir()
    ) and len(list(
        Path(matrix_dir).glob(f'*.{ext}')
    )) == len(dataset):
        logger.info(f'{matrix_dir} is not empty')
    else:
        with tqdm(dataset) as pdata:
            for idx, d in enumerate(pdata):
                word_vec_path = matrix_dir / f'{idx}.{ext}'
                if not_exists(word_vec_path):
                    t = d['text']
                    v = net.embed(t)
                    v_sparse = to_sparse(to_numpy(v)[0])
                    save_vectors(word_vec_path, v_sparse, ext=ext)


def convert_matrices(matrix_dir: Path, items_len: int, unlink_jb: bool = True):
    """Convert the matrices.

    Args:
        matrix_dir (Path): The matrix directory.
        items_len (int): The number of items.
        unlink_jb (bool, optional): Whether to unlink the joblib files. 
            Defaults to True.
    """
    if any(
        Path(matrix_dir).iterdir()
    ) and list(
        Path(matrix_dir).glob('*.npz')
    ) == items_len:
        logger.info(f'{matrix_dir} is not empty')
    else:
        vecs_joblib = list(Path(matrix_dir).glob('*.joblib'))
        with tqdm(vecs_joblib) as pdata:
            for _, d in enumerate(pdata):
                k = d.stem
                word_vec_path = matrix_dir / f'{k}.npz'
                if not_exists(word_vec_path):
                    v = joblib.load(d)
                    v_sparse = to_sparse(v)
                    save_npz(word_vec_path, v_sparse)
                    if unlink_jb:
                        d.unlink()


def convert_text(dataset, idx: int, net, matrix_dir, ext: str = 'npz'):
    """Convert the text to vectors.

    Args:
        dataset (Iterable): The dataset.
        idx (int): The index.
        net (Text2Latent): The network.
        matrix_dir (Path): The matrix directory.
        ext (str, optional): The extension. Defaults to 'npz'.
    """
    vs = net.embed(dataset[idx]['text'])[0]
    logger.info(f'{vs.shape=}')

    v_sparse = to_sparse(vs)
    logger.info(f'{vs.shape=}')

    save_vectors(matrix_dir / f'{idx}.{ext}', v_sparse)


def init_vectors(
    vectors_path: Path,
    matrix_dir: Path,
    segment: bool = False,
    pos_idx: np.ndarray = None,
    neg_idx: np.ndarray = None,
    ext: str = 'npz'
) -> Union[np.ndarray, csr_matrix]:
    """Initialize the vectors.

    Args:
        vectors_path (Path): The vectors path.
        matrix_dir (Path): The matrix directory.
        segment (bool, optional): Whether to segment the vectors. 
            Defaults to False.
        pos_idx (np.ndarray, optional): The positive indices. 
            Defaults to None.
        neg_idx (np.ndarray, optional): The negative indices. 
            Defaults to None.
        ext (str, optional): The extension. Defaults to 'npz'.
    """
    if vectors_path.exists():
        W = load_vectors(vectors_path)
        W = to_array(W)
        logger.info(f'Vectors are loaded from {vectors_path}')
    else:
        v_paths = list(matrix_dir.glob(f'*.{ext}'))
        V_dict = {}
        U_dict = {}
        V_list = []
        U_list = []
        max_idx = 0
        with tqdm(v_paths) as v_ppaths:
            for v_path in v_ppaths:
                try:
                    v_sparse = load_vectors(v_path, ext=ext)
                    vs = to_array(v_sparse)[1:]
                    v_idx = int(v_path.stem)
                    if segment:
                        u = meet_all(vs, pos_idx=pos_idx, neg_idx=neg_idx)
                        U_dict[v_idx] = u
                    v = join_all(vs, pos_idx=pos_idx, neg_idx=neg_idx)
                    V_dict[v_idx] = v
                    max_idx = max(v_idx, max_idx)
                except Exception as ex:
                    logger.error(f'{v_path=}')
                    logger.error(ex)
                    raise ex
        with tqdm(list(range(max_idx + 1))) as prange:
            for k in prange:
                if segment:
                    U_list.append(U_dict[k])
                V_list.append(V_dict[k])
        V = to_numpy(V_list)
        if segment:
            U = to_numpy(U_list)
            W = np.concatenate((U, V), axis=1)
        else:
            W = V
        W = to_sparse(W) if ext == 'npz' else W
        save_vectors(vectors_path, W, ext=ext)

    return W


def gen_vx(
    idx: Union[int, List[int], np.ndarray],
    val: Union[float, List[float], np.ndarray],
    fca: FCA
) -> np.ndarray:
    """Generate the vector.

    Args:
        idx (Union[int, List[int], np.ndarray]): The index.
        val (Union[float, List[float], np.ndarray]): The value.
        fca (FCA): The FCA.

    Returns:
        np.ndarray: The vector.
    """
    v_idx = np.zeros((fca.shape[1],), dtype=float)
    idxs = idx if isinstance(idx, Iterable) else [idx]
    vals = val if isinstance(val, Iterable) else [val]
    v_idx[idxs] = vals

    return v_idx


def gen_concept(
    idx: Union[int, List[int], np.ndarray],
    val: Union[float, List[float], np.ndarray],
    fca: Concept
) -> SimpleNamespace:
    """Generate the concept.

    Args:
        idx (Union[int, List[int], np.ndarray]): The index.
        val (Union[float, List[float], np.ndarray]): The value.
        fca (Concept): The FCA.

    Returns:
        SimpleNamespace: The concept.
    """
    v_idx = gen_vx(idx, val, fca)
    concept = fca.G_FG(v_idx)
    gen_result = SimpleNamespace(v=v_idx, c=concept)

    return gen_result


class ConceptUtils(object):
    """Concept utilities."""

    def __init__(self, fca: FCA, word_mapper: WordMapper):
        """Initialize the concept utilities.

        Args:
            fca (FCA): The FCA.
            word_mapper (WordMapper): The word mapper.
        """
        self._fca = fca
        self._mapper = word_mapper
        self._shape = fca.shape[1]

    @property
    def fca(self):
        """Get the FCA."""
        return self._fca

    @property
    def mapper(self):
        """Get the word mapper."""
        return self._mapper

    @property
    def shape(self):
        """Get the shape."""
        return self._shape

    @property
    def net(self):
        """Get the network."""
        return self.mapper.net

    def gen_concept(self, idx, val):
        """Generate the concept.

        Args:
            idx (Union[int, List[int], np.ndarray]): The index.
            val (Union[float, List[float], np.ndarray]): The value.

        Returns:
            SimpleNamespace: The concept.
        """
        return gen_concept(idx, val, self.fca)

    def gen_neughbors(
        self,
        idxs: Union[int, List[int], np.ndarray],
        vals: Union[float, List[float], np.ndarray],
        context: Union[bool, List[Any]] = [8, 8],
        top_k: int = None,
        indices_only: bool = False
    ) -> Tuple[np.ndarray, SimpleNamespace, List[Any]]:
        """Generate the neighbors.

        Args:
            idxs (Union[int, List[int], np.ndarray]): The indices.
            vals (Union[float, List[float], np.ndarray]): The values.
            context (Union[bool, List[Any]], optional): The context. 
                Defaults to [8, 8].
            top_k (int, optional): The top k. Defaults to None.
            indices_only (bool, optional): Whether to return only the indices. 
                Defaults to False.

        Returns:
            Tuple[np.ndarray, SimpleNamespace, List[Any]]: The vector, 
                the concept, and the words.
        """
        v_idx, c_idx = self.gen_concept(idxs, vals)
        words = self.mapper(
            c_idx,
            v=v_idx,
            context=context,
            top_k=top_k,
            indices_only=indices_only
        )

        return v_idx, c_idx, words

    def gen_print(
        self,
        idxs,
        vals,
        context=[8, 8],
        top_k: int = None,
        indices_only: bool = False
    ):
        """Generate and print the neighbors.

        Args:
            idxs (Union[int, List[int], np.ndarray]): The indices.
            vals (Union[float, List[float], np.ndarray]): The values.
            context (Union[bool, List[Any]], optional): The context. 
                Defaults to [8, 8].
            top_k (int, optional): The top k. Defaults to None.
            indices_only (bool, optional): Whether to return only the indices. 
                Defaults to False.

        Returns:
            Tuple[np.ndarray, SimpleNamespace, List[Any]]: The vector, 
                the concept, and the words.
        """
        logger.info(f'{top_k=}')
        v_idx, c_idx, words = self.gen_neughbors(
            idxs,
            vals,
            context=context,
            top_k=top_k,
            indices_only=indices_only
        )
        logger.info(f'{c_idx=}')
        if not indices_only:
            words_txt = '\n'.join(f'{wd}' for wd in enumerate(words.words))
            logger.info(f'\n{words_txt}')

        return v_idx, c_idx, words

    def print_tokens(self, tokens: Iterable[int]):
        """Decode tokens to string and print.

        Args:
            tokens (Iterable[int]): The tokens.
        """
        for idx, t in enumerate(tokens):
            logger.info(f'{idx} {self.net.to_string(t)} {t}')
