import logging
import os
import pprint
import sys
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

import joblib
import numpy as np
import torch
from sae_lens import SAE
from scipy.sparse import csr_matrix, isspmatrix_csr, load_npz, save_npz
from torch import nn
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.lattmc.fca.fca_utils import FCA, Concept, find_G_x
from src.lattmc.fca.file_utils import not_exists
from src.lattmc.fca.lattice_utils import join_all, meet_all
from src.lattmc.fca.utils import is_empty, not_empty, to_numpy

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True


def init_device():
    # For the most part I'll try to import functions and classes near
    # where they are used
    # to make it clear where they come from.
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f'Device: {device}')

    return device


def add_library_level(level=4):
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
    for dir in dirs:
        dir.mkdir(exist_ok=True, parents=True)


def clean_eof(text: str) -> str:
    return text.replace('<|endoftext|>', '')


class Text2Latent(object):

    def __init__(self, model: nn.Module, sae: nn.Module):
        self.model = model.eval()
        self.sae = sae.eval()
        self.hook_point = sae.cfg.hook_name

    @torch.inference_mode()
    def tokenize(self, text):
        return self.model.to_tokens(text)

    @torch.inference_mode()
    def to_string(self, tokens):
        return self.model.to_string(tokens)

    @torch.inference_mode()
    def encode(self, text):
        _, cache = self.model.run_with_cache(text, prepend_bos=True)
        # get the feature activations from our SAE
        z = self.sae.encode(cache[self.hook_point])

        return z

    @torch.inference_mode()
    def embed(self, text):
        h = self.encode(text)
        z = to_numpy(h)

        return z

    @torch.inference_mode()
    def decode(self, z):
        return self.sae.decode(z)

    @torch.inference_mode()
    def forward(self, text):
        z = self.encode(text)
        r = self.decode(z)

        return r


class Text2Sae(Text2Latent):

    def __init__(
        self,
        model_name: str,
        release: str,
        sae_id: str,
        device: str,
    ):
        self.model_name = model_name
        self.release = release
        self.sae_id = sae_id
        self.device = device
        model, sae = self._init_model()
        super().__init__(model, sae)

    def _init_model(self):
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=self.release,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=self.sae_id,  # won't always be a hook point
            device=self.device,
        )
        logger.info(f'cfg_dict:\n{pprint.pformat(cfg_dict)}')
        logger.info(f'{sparsity=}')
        hook_point = sae.cfg.hook_name
        logger.info(f'{hook_point=}')
        model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device
        )

        return model, sae


class WordMapper(object):

    def __init__(self, net, dataset, matrix_dir, ext: str = 'npz'):
        self.net = net
        self.dataset = dataset
        self.matrix_dir = matrix_dir
        self.ext = ext

    def tokenize(self, text):
        return self.net.tokenize(text)

    def get(self, idx):
        return self.dataset[idx]

    def to_string(self, tokens):
        return self.net.to_string(tokens)

    def select_ws(self, A, v, log=False):
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
    def _get_min_max(context: Union[bool, List] = False):
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
    def _slide_dict(idxs, top_k: int = None) -> Dict:
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
        context: Union[bool, List] = None,
        top_k: int = None
    ) -> List:
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
        cn,
        v=None,
        log=False,
        context: bool = False,
        top_k: int = None,
        indices_only: bool = False
    ) -> List:
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
        cn,
        v=None,
        context: bool = False,
        top_k: int = None,
        indices_only: bool = False
    ) -> Union[List, SimpleNamespace]:
        return self.search_words(
            cn,
            v=v,
            context=context,
            top_k=top_k,
            indices_only=indices_only)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def to_sparse(v):
    return v if isspmatrix_csr(v) else csr_matrix(v)


def to_array(v):
    return v.toarray() if isspmatrix_csr(v) else v


def save_vectors(
    word_vec_path,
    v_sparse: Union[np.ndarray, csr_matrix],
    ext: str = 'npz'
):
    if ext == 'npz':
        save_npz(word_vec_path, v_sparse, compressed=True)
    else:
        joblib.dump(v_sparse, word_vec_path)


def load_vectors(
    word_vec_path: Path,
    ext: str = 'npz'
):
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
):
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


def gen_vx(idx, val, fca):
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
    v_idx = gen_vx(idx, val, fca)
    concept = fca.G_FG(v_idx)
    gen_result = SimpleNamespace(v=v_idx, c=concept)

    return gen_result


class ConceptUtils(object):

    def __init__(self, fca: FCA, word_mapper: WordMapper):
        self._fca = fca
        self._mapper = word_mapper
        self._shape = fca.shape[1]

    @property
    def fca(self):
        return self._fca

    @property
    def mapper(self):
        return self._mapper

    @property
    def shape(self):
        return self._shape

    @property
    def net(self):
        return self.mapper.net

    def gen_concept(self, idx, val):
        return gen_concept(idx, val, self.fca)

    def gen_neughbors(
        self,
        idxs,
        vals,
        context=[8, 8],
        top_k: int = None,
        indices_only: bool = False
    ):
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

    def print_tokens(self, tokens):
        for idx, t in enumerate(tokens):
            logger.info(f'{idx} {self.net.to_string(t)} {t}')
