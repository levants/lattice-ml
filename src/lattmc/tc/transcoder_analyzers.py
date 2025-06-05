import csv
import gc
import logging
import token
from ast import Str
from math import e
from pathlib import Path
from sre_parse import Tokenizer
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from traitlets import Int

from src.lattmc.fca.fca_utils import FCA, Concept
from src.lattmc.fca.lattice_utils import join_all, meet_all
from src.lattmc.fca.utils import in_any, not_empty, topK, topKrange, truncate
from src.lattmc.fca.visualization_utils import with_background
from src.lattmc.sae.nlp_sae_utils import gen_concept, gen_vx
from src.lattmc.tc.data_utils import load_tokens
from src.lattmc.tc.transcoder_fca import TranscoderUtils
from src.lattmc.tc.transcoders_utils import Transcoder, init_transcoder

logger = logging.getLogger(__name__)


class TranscoderAnalyzer(object):
    """Utility class for transcoder analysis"""

    def __init__(
        self,
        transcoder: Transcoder = None,
        tokens: torch.Tensor = None,
        tr_utils: TranscoderUtils = None,
        fcas: Dict[int, FCA] = None,
        layers: List[int] = None,
        pad_token: str = None,
        pos_idxs: Union[List[int], np.ndarray] = None,
        neg_idxs: Union[List[int], np.ndarray] = None,
    ):
        self._transcoder = transcoder
        self._tokens = tokens
        self._tr_utils = tr_utils
        self._fcas = fcas
        self._layers = layers
        self._pad_token = pad_token if (
            pad_token
        ) else transcoder.tokenizer.pad_token
        self._pos_idxs = pos_idxs if pos_idxs else {}
        self._neg_idxs = neg_idxs if neg_idxs else {}

    @property
    def transcoder(self) -> Transcoder:
        return self._transcoder

    @property
    def tokenizer(self) -> Tokenizer:
        return self.transcoder.tokenizer

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def tokens(self) -> torch.Tensor:
        return self._tokens

    @property
    def trancoder_utils(self) -> TranscoderUtils:
        return self._tr_utils

    @property
    def tr_utils(self) -> TranscoderUtils:
        return self._tr_utils

    @property
    def fcas(self) -> Dict[int, FCA]:
        return self._fcas

    @property
    def layers(self) -> List[int]:
        return self._layers

    @property
    def pos_idxs(self) -> Dict[int, Union[List[int], np.ndarray]]:
        return self._pos_idxs

    @property
    def neg_idxs(self) -> Dict[int, Union[List[int], np.ndarray]]:
        return self._neg_idxs

    def get_idcs(self, layer: int) -> Union[List[int], np.ndarray]:
        return self.pos_idxs.get(layer, None), self.neg_idxs.get(layer, None)

    def encode(
        self,
        prompt: Union[str, torch.Tensor],
        layer: int
    ) -> np.ndarray:
        return self.transcoder(prompt, layer)

    def tokenize(self, prompt: str) -> torch.Tensor:
        return self.transcoder.tokenize(prompt)

    def clean_pad(self, prompt: str) -> str:
        return prompt.replace(self.pad_token, '')

    def to_string(self, prompt: torch.Tensor) -> str:
        return self.transcoder.to_string(prompt)

    def det_string(self, indcs: Union[List[int], np.ndarray]) -> str:
        return self.to_string(self.tokens[indcs])

    def to_clean(self, prompt: torch.Tensor) -> str:
        return self.clean_pad(self.to_string(prompt))

    def texts(self, idx: int) -> str:
        return self.to_clean(self.tokens[idx])

    def det_clean(self, indcs: Union[List[int], np.ndarray]) -> str:
        return [(pt, self.to_clean(pt)) for pt in self.tokens[indcs]]

    def detect_token(
        self,
        layer: int,
        prompt: torch.Tensor,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        return self.transcoder.detect_token(layer, prompt, u)

    def print_detected_tokens(
            self,
            layer: int,
            prompt: torch.Tensor,
            u: np.ndarray,
            with_text: bool = False
    ):
        self.transcoder.print_detected_tokens(
            layer,
            prompt,
            u,
            with_text=with_text
        )

    def print_all_detected_tokens(
        self,
        layer: int,
        prompts: torch.Tensor,
        u: np.ndarray,
        with_text: bool = False
    ) -> np.ndarray:
        return self.transcoder.print_all_detected_tokens(
            layer,
            prompts,
            u,
            with_text=with_text
        )

    def print_all_from_objects(
        self,
        layer: int,
        A: Union[np.ndarray, torch.Tensor],
        u: np.ndarray,
        with_text: bool = False,
        limit: int = None,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print all detected tokens from the concept objects."""
        if isinstance(A, torch.Tensor):
            A = A.numpy()
        v_FG, token_idcs = self.transcoder.print_all_detected_tokens(
            layer,
            self.tokens[A],
            u,
            with_text=with_text,
            indices=A,
            limit=limit,
        )

        return v_FG, token_idcs

    def print_all_from_concept(
        self,
        layer: int,
        c: SimpleNamespace,
        with_text: bool = False,
        limit: int = None,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print all detected tokens from the concept."""
        return self.print_all_from_objects(
            layer,
            c.c.A,
            c.v,
            with_text=with_text,
            limit=limit,
        )

    def gen_concept(
        self,
        idx: Union[int, List[int], np.ndarray],
        val: Union[float, List[float], np.ndarray],
        layer: int
    ) -> SimpleNamespace:
        return gen_concept(idx, val, self.fcas[layer])

    def gen_and_print_all(
        self,
        idx: Union[int, List[int], np.ndarray],
        val: Union[float, List[float], np.ndarray],
        layer: int,
        with_text: bool = False,
        limit: int = None,
    ) -> SimpleNamespace:
        """Print all detected tokens from the concept."""
        c = gen_concept(idx, val, self.fcas[layer])
        v_FG, token_idcs = self.print_all_from_concept(
            layer,
            c,
            with_text=with_text,
            limit=limit,
        )
        c.v_FG = v_FG
        c.token_idcs = token_idcs

        return c

    def dump_text(self, prompt: torch.Tensor, dest: Path):
        with dest.open('a') as fl:
            with (tqdm(prompt)) as pprmpt:
                for pr in pprmpt:
                    tx = self.to_clean(pr)
                    fl.write(f'{tx}\n')

    def dump_csv(self, prompt: torch.Tensor, dest: Path):
        with dest.open('w', newline='') as fl:
            csv_writer = csv.writer(fl, delimiter=',')
            with (tqdm(prompt)) as pprmpt:
                texts_head = ['idcs', 'text']
                texts_data = [
                    [idx, self.to_clean(pr)] for idx, pr in enumerate(pprmpt)
                ]
                texts = texts_head + texts_data
            csv_writer.writerows(texts)
        print(f'Text is written to {dest} as a CSV.')

    def dump_tokens(self, dest: Path):
        self.dump_text(self.tokens, dest)

    def tokens_to_csv(self, dest: Path):
        self.dump_csv(self.tokens, dest)


class ConceptAnalysis(object):
    """Class to analyze the concepts in the transcoder activations."""

    def __init__(
        self,
        prompt: str,
        tr_analyzer: TranscoderAnalyzer,
        trunc: int = None,
    ):
        self._tr_utils = tr_analyzer.tr_utils
        self._prompt = prompt
        self._layers = tr_analyzer.layers
        self._fcas = tr_analyzer.fcas
        self._corpus = tr_analyzer.tokens
        self._trunc = trunc
        self._vs: Dict[int, np.ndarray] = {}
        self._idcs: List[int] = []
        self._v_is: Dict[int, Dict[int, np.ndarray]] = {
            l: {} for l in self.layers
        }
        self._c_is: Dict[int, Concept] = {}  # type: ignore
        # Initialize detected tokens for each layer
        self._det_tokens: Dict[torch.Tensor] = {
            l: torch.tensor([]) for l in self.layers
        }
        # Initialize propagated tokens for each layer
        self._prop_tokens: Dict[torch.Tensor] = {
            l: torch.tensor([]) for l in self.layers
        }
        # Initialize detected values
        self._detected_vs = {
            l: {} for l in self.layers
        } if self.layers else {}
        # Initialize detected foreground values for each layer and token
        self._detected_v_FG = {
            l: {} for l in self.layers
        } if self.layers else {}
        # Initialize foreground values for each layer
        self._v_FG = {
            l: {} for l in self.layers
        } if self.layers else {}
        # Initialize positive and negative indices
        self._pos_idxs = tr_analyzer.pos_idxs
        self._neg_idxs = tr_analyzer.neg_idxs

    @property
    def tr_utils(self) -> TranscoderUtils:
        return self._tr_utils

    @property
    def transcoder(self) -> Transcoder:
        return self.tr_utils.transcoder

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def layers(self) -> List[int]:
        return self._layers

    @property
    def fcas(self) -> Dict[int, FCA]:
        return self._fcas

    @property
    def corpus(self) -> torch.Tensor:
        return self._corpus

    @property
    def trunc(self) -> int:
        return self._trunc if self._trunc is not None else 0

    @trunc.setter
    def trunc(self, value: int):
        """Set the truncation value."""
        if value is not None and value < 0:
            raise ValueError("Truncation value must be non-negative.")
        self._trunc = value

    @property
    def vs(self) -> Dict[int, np.ndarray]:
        return self._vs

    @vs.setter
    def vs(self, other_vs: Dict[int, np.ndarray]):
        self._vs = other_vs

    @property
    def idcs(self) -> List[int]:
        return self._idcs

    @idcs.setter
    def idcs(self, other_idcs: List[int]):
        """Set the indices."""
        self._idcs = other_idcs

    @property
    def v_is(self) -> Dict[int, np.ndarray]:
        return self._v_is

    @v_is.setter
    def v_is(self, other_v_is: Dict[int, np.ndarray]):
        self._v_is = other_v_is

    @property
    def c_is(self) -> Dict[int, Concept]:
        return self._c_is

    @c_is.setter
    def c_is(self, other_c_is: Dict[int, Concept]):
        """Set the concept."""
        self._c_is = other_c_is

    @property
    def det_tokens(self) -> Dict[int, torch.Tensor]:
        """Get the detected tokens."""
        return self._det_tokens

    @det_tokens.setter
    def det_tokens(self, other_det_tokens: Dict[int, torch.Tensor]):
        """Set the detected tokens."""
        self._det_tokens = other_det_tokens

    @property
    def prop_tokens(self) -> Dict[int, torch.Tensor]:
        """Get the propagated tokens."""
        return self._prop_tokens

    @prop_tokens.setter
    def prop_tokens(self, other_prop_tokens: Dict[int, torch.Tensor]):
        """Set the propagated tokens."""
        self._prop_tokens = other_prop_tokens

    @property
    def detected_vs(self) -> Dict[int, Union[torch.Tensor, np.ndarray]]:
        return self._detected_vs

    @property
    def detected_v_FG(self) -> Dict[
        int,
        Dict[int, Union[np.ndarray, torch.Tensor]]
    ]:
        """Get the detected foreground values."""
        return self._detected_v_FG

    @property
    def v_FG(self) -> Dict[int, Dict[int, np.ndarray]]:
        """Get the v_FG values."""
        return self._v_FG

    @property
    def pos_idxs(self) -> Union[List[int], np.ndarray]:
        return self._pos_idxs

    @property
    def neg_idxs(self) -> Union[List[int], np.ndarray]:
        return self._neg_idxs

    def add_detected_vs(
        self,
        idcs: int,
        vs: Union[np.ndarray, torch.Tensor],
        det_vs: Dict[int, Union[np.ndarray, torch.Tensor]]
    ):
        """Add detected values for the given layer and index."""
        for idx, v in zip(idcs, vs):
            if isinstance(idx, np.ndarray):
                for i in idx:
                    det_vs[i] = v
            else:
                det_vs[idx] = v

    def get_idcs(self, layer: int) -> Tuple[List[int], List[int]]:
        """Get the positive and negative indices for the given layer."""
        return self.pos_idxs.get(layer, None), self.neg_idxs.get(layer, None)

    def V(self, layer: int) -> np.ndarray:
        """Get the V matrix for the given layer."""
        return self.fcas[layer].V

    def init_fca(
        self,
        layer: int,
        neg_idxs: Union[List[int], np.ndarray] = None
    ) -> FCA:
        """Initialize the FCA for the given layer."""
        return FCA(self.V(layer), neg_idx=neg_idxs)

    def print_detected_tokens(
            self,
            layer: int,
            idx: int,
            u: np.ndarray,
            with_text: bool = False
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print detected tokens for the given layer and indices."""
        return self.transcoder.print_detected_tokens(
            layer,
            self.corpus[idx],
            u,
            with_text=with_text
        )

    def print_all_detected_tokens(
        self,
        layer: int,
        A: np.ndarray,
        u: np.ndarray,
        with_text: bool = False,
        limit: int = None
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print all detected tokens for the given layer and indices."""
        return self.transcoder.print_all_detected_tokens(
            layer,
            self.corpus[A],
            u,
            with_text=with_text,
            limit=limit,
            indices=A,
        )

    def gen_concept(
        self,
        idcs: Union[int, List[int], np.ndarray],
        vals: Union[float, List[float], np.ndarray],
        layer: int,
        neg_idxs: Union[List[int], np.ndarray] = None,
    ) -> SimpleNamespace:
        """Generate a concept for the given indices and values."""
        fcn = self.init_fca(layer, neg_idxs=neg_idxs)
        cn = gen_concept(idcs, vals, fcn)
        cn.fca = fcn

        return cn

    def G_FG(
        self,
        v: np.ndarray,
        layer: int,
        neg_idxs: Union[List[int], np.ndarray] = None
    ) -> Concept:
        """Generate a concept from the given values."""
        fca = self.init_fca(layer, neg_idxs=neg_idxs)
        c_v = fca.G_FG(v)

        return c_v

    def gen_and_print(
        self,
        idcs: Union[int, List[int], np.ndarray],
        vals: Union[float, List[float], np.ndarray],
        layer: int,
        neg_idxs: Union[List[int], np.ndarray] = None,
        with_text=True,
        limit: int = None
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Generate a concept and print it."""
        cn = self.gen_concept(idcs, vals, layer, neg_idxs=neg_idxs)
        logger.info(f'{cn=}')
        if limit:
            A = cn.c.A[:limit]
            logger.info(
                f'Actual detection is {cn.c.A.shape} but {limit} is shown'
            )
        else:
            A = cn.c.A
            logger.info(f'All detections are shown')
        return self.print_all_detected_tokens(
            layer,
            A,
            cn.v,
            with_text=with_text,
            limit=limit,
        )

    def topK_v(
        self,
        layer: int,
        idx: int,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the top K values and indices for the given layer and index."""
        return topK(self.v_is[layer][idx], k)

    def analyze_concepts(self):
        """Analyze the concepts for the given prompt and layers."""
        vs = self.tr_utils.run_transcoders(self.prompt, self.layers)
        self.tr_utils.print_tokens(self.prompt)
        self.vs = vs

    def _collect_vals(
        self,
        layer: int,
        rng: int = 1,
        red_val: float = 0.0,
        trunc: int = 0
    ):
        """Collect the values for the given layer and indices."""
        vals = {}
        idxs = {}
        v_is_layer = {}
        for i in self.idcs:
            val_i, idxs[i] = topKrange(self.vs[layer][i], rng)
            vals[i] = truncate(
                val_i,
                trunc if self.trunc is None else self.trunc
            )
            vals[i] = vals[i] - red_val
            logger.info(f'{vals[i]}, {idxs[i]}')
            v_is_layer[i] = gen_vx(idxs[i], vals[i], self.fcas[layer])
        self.v_is[layer] = v_is_layer

    def gen_concepts(
        self,
        idcs: List[int],
        layer: int,
        rng: int = 1,
        red_val: float = 0.0,
        trunc: int = 0
    ):
        """Generate the concepts for the given indices and layer."""
        self.idcs = idcs
        self._collect_vals(layer, rng, red_val, trunc)
        pos_idx, neg_idx = self.get_idcs(layer)
        v_join = join_all(
            np.array(list(self.v_is[layer].values())),
            pos_idx=pos_idx,
            neg_idx=neg_idx,
        )
        self.c_is[layer] = self.fcas[layer].G_FG(v_join)
        logger.info(f'{self.c_is[layer]=}')
        self.det_tokens[layer] = self.corpus[self.c_is[layer].A] if len(
            self.c_is[layer].A
        ) > 0 else torch.tensor([])

    def to_string(self, tokens: torch.Tensor) -> str:
        """Convert tokens to a string."""
        return self.transcoder.to_string(tokens)

    def analyze_text(self, layer: int, limit: int = None):
        """Analyze the text for the given layer and indices."""
        rn_limit = min(
            limit,
            self.det_tokens[layer].shape[0]
        ) if limit else self.det_tokens[layer].shape[0]
        v_FG_is = {}
        num_detects = {}
        for indx, i_A in zip(range(rn_limit), self.c_is[layer].A[:rn_limit]):
            dets_i = []
            idcs_i = []
            det_vs = {}
            det_v_FG = {}
            for i in self.idcs:
                idx_i, vs, det_i, v_FG_i = self.transcoder.detect_token(
                    layer,
                    self.det_tokens[layer][indx],
                    self.v_is[layer][i]
                )
                dets_i.append(det_i)
                idcs_i.append(idx_i)
                num_detects.setdefault(i, 0)
                num_detects[i] += idx_i.size
                det_i_vs = {t_idx: t_v for t_idx, t_v in zip(idx_i, vs)}
                det_vs[i] = det_i_vs
                det_v_FG[i] = v_FG_i
                v_FG_is.setdefault(i, [])
                v_FG_is[i].append(v_FG_i if not_empty(
                    v_FG_i
                ) else self.fcas[layer].v_max)
            self.detected_vs.setdefault(layer, {})
            self.detected_vs[layer][i_A] = det_vs
            self.detected_v_FG.setdefault(layer, {})
            self.detected_v_FG[layer][i_A] = det_v_FG
            dets_idcs = dets_i + idcs_i
            decoded_tokens = self.transcoder.assemble_text(
                self.det_tokens[layer][indx],
                idcs_i,
            )
            print(f'{i_A}', decoded_tokens)
            print(*dets_idcs)
            print()
        self.v_FG[layer] = {i: meet_all(
            np.array(v_FG_is[i]),
            neg_idx=self.neg_idxs.get(layer, None)
        ) if v_FG_is and not_empty(
            v_FG_is[i]
        ) else None for i in self.idcs}
        print(f'{num_detects=}')

    def gen_text(
        self,
        idcs: List[int],
        layer: int,
        rng: int = 1,
        red_val: float = 0.0,
        trunc: int = 0,
        limit: int = None,
    ):
        """Generate text for the given layer, indices, and values."""
        self.gen_concepts(idcs, layer, rng=rng, red_val=red_val, trunc=trunc)
        self.analyze_text(layer, limit=limit)


def init_analyzer(
    layers: List[int],
    tokens_path: Path,
    model_path: Path,
    device: Union[str, torch.device] = torch.device('cpu'),
    dataset_path: Path = None,
    columns: List[str] = None,
    vector_dir: Path = None,
    pos_idxs: Union[List[int], np.ndarray] = None,
    neg_idxs: Union[List[int], np.ndarray] = None,
) -> TranscoderAnalyzer:
    transcoder = init_transcoder(device=device)
    gc.collect()
    torch.cuda.empty_cache()

    owt_tokens_torch = load_tokens(
        transcoder,
        tokens_path,
        device=device,
        csv_path=dataset_path,
        columns=columns,
    )
    gc.collect()

    tr_utils = TranscoderUtils(
        transcoder,
        owt_tokens_torch,
        vector_dir if vector_dir else model_path,
        pos_idxs=pos_idxs,
        neg_idxs=neg_idxs,
    )
    gc.collect()

    fcas = tr_utils.init_fcas(layers)
    gc.collect()

    tr_analyzer = TranscoderAnalyzer(
        transcoder=transcoder,
        tokens=owt_tokens_torch,
        tr_utils=tr_utils,
        fcas=fcas,
        layers=layers,
        pos_idxs=pos_idxs,
        neg_idxs=neg_idxs,
    )

    return tr_analyzer
