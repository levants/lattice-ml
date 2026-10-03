"""Transcoder and SAE analyzer and mapping to the 
Lattice-theoretic Formal Concept Analysis (FCA)."""

from __future__ import annotations
from typing import Set

import csv
import gc
import logging
from pathlib import Path
from sre_parse import Tokenizer
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from gradio import Dataset
from tqdm import tqdm

from src.lattmc.fca.fca_utils_codexgen import FCA, Concept
from src.lattmc.fca.lattice_utils import join_all, meet, meet_all
from src.lattmc.fca.utils import not_empty, to_numpy, topK, topKrange, truncate
from src.lattmc.sae.nlp_sae_utils import gen_concept, gen_vx
from src.lattmc.sae.sae_utils import init_sae
from src.lattmc.tc.data_utils import load_tokens
from src.lattmc.tc.transcoder_fca_codexgen import TranscoderUtils
from src.lattmc.tc.transcoder_utils import Transcoder, init_transcoder
from src.lattmc.utils import empty_cache, init_device

logger = logging.getLogger(__name__)


class IndexableCorpus:
    """
        Holder class for tokens that converts tokens to text using transcoder

        Attributes:
            corpus (torch.Tensor): The tokens.
            transcoder (Transcoder): The transcoder.
            pad_token (str): The pad token. Default is None.
    """

    def __init__(
        self: IndexableCorpus,
        corpus: torch.Tensor,
        transcoder: Transcoder,
        pad_token: str = None,
    ) -> None:
        """Initialize IndexableCorpus and its required state."""
        self.corpus = corpus
        self.transcoder = transcoder
        self.pad_token = pad_token if (
            pad_token
        ) else transcoder.tokenizer.pad_token

    def to_string(self: IndexableCorpus, corpus: torch.Tensor) -> str:
        """Convert tokens to text using transcoder.

        Args:
            corpus: (torch.Tensor) The tokens.

        Returns:
            (str) The text.
        """
        return self.transcoder.to_string(corpus).replace(self.pad_token, '')

    def __len__(self: IndexableCorpus) -> int:
        """Return the length of the corpus.

        Returns:
            (int) The length of the corpus.
        """
        return len(self.corpus)

    def __get_slices(self: IndexableCorpus, key: slice) -> List[str]:
        """Get slices of the corpus.

        Args:
            key: (slice) The slice.

        Returns:
            (List[str]) The slices.
        """
        start, stop, step = key.indices(len(self))
        result = []
        for i in range(start, stop, step):
            val = self.to_string(self.corpus[i])
            result.append(val)

        return result

    def __getitem__(
        self: IndexableCorpus,
        key: int | slice | list[int] | tuple[int, ...] | np.ndarray,
    ) -> Union[str, List[str]]:
        """Get items from the corpus.

        Args:
            key: (slice) The slice.

        Returns:
            (List[str]) The items.
        """
        return self.__get_slices(key) if isinstance(
            key,
            slice,
        ) else [self.to_string(self.corpus[i]) for i in key] if isinstance(
            key,
            (list, tuple, np.ndarray),
        ) else self.to_string(self.corpus[key])


class TranscoderAnalyzer(object):
    """Utility class for transcoder and SAE latent activations analysis.

    Attributes:
        transcoder (Transcoder): The transcoder. Default is None.
        tokens (torch.Tensor): The tokens. Default is None.
        tr_utils (TranscoderUtils): The transcoder utils. Default is None.
        fcas (Dict[int, FCA]): The FCAs. Default is None.
        layers (List[int]): The layers. Default is None.
        pad_token (str): The pad token. Default is None.
        pos_idxs (Union[List[int], np.ndarray]): The positive indices. 
            Default is None.
        neg_idxs (Union[List[int], np.ndarray]): The negative indices. 
            Default is None.
        texts (IndexableCorpus): The texts. Default is None.
    """

    def __init__(
        self: TranscoderAnalyzer,
        transcoder: Transcoder = None,
        tokens: torch.Tensor = None,
        tr_utils: TranscoderUtils = None,
        fcas: Dict[int, FCA] = None,
        layers: List[int] = None,
        pad_token: str = None,
        pos_idxs: Union[List[int], np.ndarray] = None,
        neg_idxs: Union[List[int], np.ndarray] = None,
        texts: IndexableCorpus = None,
    ) -> None:
        """Initialize TranscoderAnalyzer and its required state."""
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
        self._texts = IndexableCorpus(
            tokens,
            transcoder,
            pad_token,
        ) if texts is None else texts

    @classmethod
    def fromAnalyzer(
        cls: type[TranscoderAnalyzer],
        analyzer: 'TranscoderAnalyzer',
        fcas: Dict[int, FCA] = None,
    ) -> 'TranscoderAnalyzer':
        """Create a new analyzer from an existing analyzer.

        Args:
            analyzer (TranscoderAnalyzer): The analyzer to create a 
                new analyzer from.
            fcas (Dict[int, FCA], optional): The FCAs. Defaults to None. 
                Default is None.

        Returns:    
            TranscoderAnalyzer: The new analyzer.
        """
        if fcas is None:
            res_analyzer = analyzer
        else:
            res_analyzer = cls(
                transcoder=analyzer.transcoder,
                tokens=analyzer.tokens,
                tr_utils=analyzer.tr_utils,
                fcas=fcas,
                layers=analyzer.layers,
                pad_token=analyzer.pad_token,
                pos_idxs=analyzer.pos_idxs,
                neg_idxs=analyzer.neg_idxs,
                texts=analyzer.texts,
            )

        return res_analyzer

    @property
    def transcoder(self: TranscoderAnalyzer) -> Transcoder:
        """Return the transcoder.

        Returns:
            (Transcoder) The transcoder.
        """
        return self._transcoder

    @property
    def tokenizer(self: TranscoderAnalyzer) -> Tokenizer:
        """Return the tokenizer.

        Returns:
            (Tokenizer) The tokenizer.
        """
        return self.transcoder.tokenizer

    @property
    def pad_token(self: TranscoderAnalyzer) -> str:
        """Return the pad token.

        Returns:
            (str) The pad token.
        """
        return self._pad_token

    @property
    def tokens(self: TranscoderAnalyzer) -> torch.Tensor:
        """Return the tokens.

        Returns:
            (torch.Tensor) The tokens.
        """
        return self._tokens

    @property
    def corpus(self: TranscoderAnalyzer) -> torch.Tensor:
        """Return the corpus.

        Returns:
            (torch.Tensor) The corpus.
        """
        return self._tokens

    @property
    def trancoder_utils(self: TranscoderAnalyzer) -> TranscoderUtils:
        """Return the transcoder utils.

        Returns:
            (TranscoderUtils) The transcoder utils.
        """
        return self._tr_utils

    @property
    def tr_utils(self: TranscoderAnalyzer) -> TranscoderUtils:
        """Return the transcoder utils.

        Returns:
            (TranscoderUtils) The transcoder utils.
        """
        return self._tr_utils

    @property
    def fcas(self: TranscoderAnalyzer) -> Dict[int, FCA]:
        """Return the FCAs.

        Returns:
            (Dict[int, FCA]) The FCAs.
        """
        return self._fcas

    @property
    def layers(self: TranscoderAnalyzer) -> List[int]:
        """Return the layers.

        Returns:
            (List[int]) The layers.
        """
        return self._layers

    @property
    def pos_idxs(
        self: TranscoderAnalyzer,
    ) -> Dict[int, Union[List[int], np.ndarray]]:
        """Return the positive indices.

        Returns:
            (Dict[int, Union[List[int], np.ndarray]]) The positive indices.
        """
        return self._pos_idxs

    @property
    def neg_idxs(
        self: TranscoderAnalyzer,
    ) -> Dict[int, Union[List[int], np.ndarray]]:
        """Return the negative indices.

        Returns:
            (Dict[int, Union[List[int], np.ndarray]]) The negative indices.
        """
        return self._neg_idxs

    def get_idcs(
        self: TranscoderAnalyzer,
        layer: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get all positive and negative indices of FCA for the given layer.

        Args:
            layer: (int) The layer of the model.

        Returns:
            (Tuple[np.ndarray, np.ndarray]) Positive and negative indices.
        """
        return self.pos_idxs.get(layer, None), self.neg_idxs.get(layer, None)

    def min_vals(
        self: TranscoderAnalyzer,
        layer: int,
        idcs: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """Get the minimum nonzero values of the concept on the given indices.

        Args:
            layer: (int) The layer of the model.
            idcs: (Union[List[int], np.ndarray]) The indices.

        Returns:
            (np.ndarray) The minimum nonzero values of the concept 
                on the given indices.
        """
        return self.fcas[layer].min_vals(idcs)

    def min_nonzero_vals(
        self: TranscoderAnalyzer,
        layer: int,
        u: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """Get the minimum nonzero values of the concept on the given indices.

        Args:
            layer: (int) The layer of the model.
            u: (Union[List[int], np.ndarray]) vector of concept values.

        Returns:
            (np.ndarray) The minimum nonzero values of the concept 
                on the given indices.
        """
        vals, idcs = topK(u, k=u.shape[0])
        nonz_idcs = idcs[vals > 0]
        min_nonz_vals = self.min_vals(layer, nonz_idcs)

        return min_nonz_vals

    def encode(
        self: TranscoderAnalyzer,
        prompt: Union[str, torch.Tensor],
        layer: int,
    ) -> np.ndarray:
        """Encode the prompt for the given layer.

        Args:
            prompt: (Union[str, torch.Tensor]) The prompt.
            layer: (int) The layer of the model.

        Returns:
            (np.ndarray) The encoded prompt.
        """
        return self.transcoder(prompt, layer)

    def tokenize(self: TranscoderAnalyzer, prompt: str) -> torch.Tensor:
        """Tokenize the prompt.

        Args:
            prompt: (str) The prompt.

        Returns:
            (torch.Tensor) The tokenized prompt.
        """
        return self.transcoder.tokenize(prompt)

    def clean_pad(self: TranscoderAnalyzer, prompt: str) -> str:
        """Remove the padding tokens from the prompt.

        Args:
            prompt: (str) The prompt.

        Returns:
            (str) The prompt without padding tokens.
        """
        return prompt.replace(self.pad_token, '')

    def to_string(self: TranscoderAnalyzer, prompt: torch.Tensor) -> str:
        """Convert the prompt to a string.

        Args:
            prompt: (torch.Tensor) The prompt.

        Returns:
            (str) The prompt as a string.
        """
        return self.transcoder.to_string(prompt)

    def det_string(
        self: TranscoderAnalyzer,
        indcs: Union[List[int], np.ndarray],
    ) -> str:
        """Convert the indices to a string.

        Args:
            indcs: (Union[List[int], np.ndarray]) The indices.

        Returns:
            (str) The indices as a string.
        """
        return self.to_string(self.tokens[indcs])

    def to_clean(self: TranscoderAnalyzer, prompt: torch.Tensor) -> str:
        """Remove the padding tokens from the prompt.

        Args:
            prompt: (torch.Tensor) The prompt.

        Returns:
            (str) The prompt without padding tokens.
        """
        return self.clean_pad(self.to_string(prompt))

    @property
    def texts(self: TranscoderAnalyzer) -> IndexableCorpus:
        """Get the text corpus item by index.

        Returns:
            (IndexableCorpus) The text corpus item by index.
        """
        return self._texts

    def det_clean(
        self: TranscoderAnalyzer,
        indcs: Union[List[int], np.ndarray],
    ) -> str:
        """Convert the indices to a string.

        Args:
            indcs: (Union[List[int], np.ndarray]) The indices.

        Returns:
            (str) The indices as a string.
        """
        return [(pt, self.to_clean(pt)) for pt in self.tokens[indcs]]

    def detect_token(
        self: TranscoderAnalyzer,
        layer: int,
        prompt: torch.Tensor,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Detect the tokens in the prompt for each feature 
            of the given vector.

        Args:
            layer: (int) The layer of the model.
            prompt: (torch.Tensor) The prompt.
            u: (np.ndarray) The concept vector.

        Returns:
            (Tuple[np.ndarray, np.ndarray, List[str]]) The detected tokens.
        """
        return self.transcoder.detect_token(layer, prompt, u)

    def print_detected_tokens(
        self: TranscoderAnalyzer,
        layer: int,
        prompt: torch.Tensor,
        u: np.ndarray,
        with_text: bool = False,
    ) -> None:
        """Print the detected tokens in the prompt for each feature
            of the given vector.

        Args:
            layer: (int) The layer of the model.
            prompt: (torch.Tensor) The prompt.
            u: (np.ndarray) The concept vector.
            with_text: (bool) Whether to print the text of the tokens. 
                Default is False.
        """
        self.transcoder.print_detected_tokens(
            layer,
            prompt,
            u,
            with_text=with_text
        )

    def print_all_detected_tokens(
        self: TranscoderAnalyzer,
        layer: int,
        prompts: torch.Tensor,
        u: np.ndarray,
        with_text: bool = False,
    ) -> np.ndarray:
        """Prints all detected tokens in the prompts for each feature
            of the given vector.

        Args:
            layer: (int) The layer of the model.
            prompts: (torch.Tensor) The prompts.
            u: (np.ndarray) The concept vector.
            with_text: (bool) Whether to print the text of the tokens. 
                Default is False.

        Returns:
            (np.ndarray) The detected tokens.
        """
        return self.transcoder.print_all_detected_tokens(
            layer,
            prompts,
            u,
            with_text=with_text
        )

    def print_all_from_objects(
        self: TranscoderAnalyzer,
        layer: int,
        A: Union[np.ndarray, torch.Tensor],
        u: np.ndarray,
        with_text: bool = False,
        limit: int = None,
        log_tokens_and_idxs: bool = True,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print all detected tokens from the concept objects for each 
            feature of the given vector.

        Args:
            layer: (int) The layer of the model.
            A: (Union[np.ndarray, torch.Tensor]) The concept objects.
            u: (np.ndarray) The concept vector.
            with_text: (bool) Whether to print the text of the tokens. 
                Default is False.
            limit: (int) The number of tokens to print. Default is None.
            log_tokens_and_idxs: (bool) Whether to log the tokens and indices. 
                Default is True.

        Returns:
            (Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]) 
                The detected tokens.
        """
        A_np = to_numpy(A)
        v_FG, token_idcs = self.transcoder.print_all_detected_tokens(
            layer,
            self.tokens[A_np],
            u,
            with_text=with_text,
            indices=A_np,
            limit=limit,
            log_tokens_and_idxs=log_tokens_and_idxs,
        )

        return v_FG, token_idcs

    def print_multitexts_from_objects(
        self: TranscoderAnalyzer,
        layer: int,
        A: Union[np.ndarray, torch.Tensor],
        u: np.ndarray,
        limit: int = None,
    ) -> Dict[int, str]:
        """Print all detected tokens in the items with the given indices 
            for each feature of the given vector.

        Args:
            layer: (int) The layer of the model.
            A: (Union[np.ndarray, torch.Tensor]) The source vector.
            u: (np.ndarray) The concept object.
            limit: (int) The number of tokens to print. Default is None.
        Returns:
            Dict[int, str]: dictionary indexed by prompt indices with 
                text with multiply tokens highlighted as values
        """
        A_np = to_numpy(A)
        return self.transcoder.print_detected_multitokens(
            layer,
            self.tokens[A_np],
            A_np,
            u,
            limit=limit,
        )

    def print_all_from_concept(
        self: TranscoderAnalyzer,
        layer: int,
        c: SimpleNamespace,
        with_text: bool = False,
        limit: int = None,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print all detected tokens from the concept object for each
            feature of the concept vector.

        Args:
            layer: (int) The layer of the model.
            c: (SimpleNamespace) The concept object.
            with_text: (bool) Whether to print the text of the tokens. 
                Default is False.
            limit: (int) The number of tokens to print. Default is None.

        Returns:
            (Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]) 
                The detected tokens.
        """
        return self.print_all_from_objects(
            layer,
            c.c.A,
            c.v,
            with_text=with_text,
            limit=limit,
        )

    def match_tokens_in_prompts_and_print(
        self: TranscoderAnalyzer,
        A: np.ndarray,
        S: Union[Set[int], List[int], np.ndarray, List[torch.Tensor],
        torch.Tensor],
        with_text: bool = True,
        color_idx: int = 35,
    ) -> Dict[int, str]:
        """Match the given tokens S in the prompts with indices A and 
            print them as a highlighted text
        Args:
            A (np.ndarray): indices of the prompts.
            S (Union[Set[int], List[int], np.ndarray, 
                List[torch.Tensor], torch.Tensor]):
                set of token indices to match.
            with_text (bool): whether to print the tokens as text.
                Default is True.
            color_idx (int): index of the color code in self.bg_codes.
                Default is 35.
        Returns:
            text_dict (Dict[int, str]): dictionary indexed by prompt 
                identifiers with 
                latex multitexts as values
        """
        return self.transcoder.match_tokens_in_prompts_and_print(
            A,
            S,
            corpus=self.corpus,
            with_text=with_text,
            color_idx=color_idx,
        )

    def gen_concept(
        self: TranscoderAnalyzer,
        idx: Union[int, List[int], np.ndarray],
        val: Union[float, List[float], np.ndarray],
        layer: int,
    ) -> SimpleNamespace:
        """Generate a concept from the given index and value.

        Args:
            idx: (Union[int, List[int], np.ndarray]) The index of the concept.
            val: (Union[float, List[float], np.ndarray]) 
                The value of the concept.
            layer: (int) The layer of the model.

        Returns:
            (SimpleNamespace) The concept.
        """
        return gen_concept(idx, val, self.fcas[layer])

    def gen_and_print_all(
        self: TranscoderAnalyzer,
        idx: Union[int, List[int], np.ndarray],
        val: Union[float, List[float], np.ndarray],
        layer: int,
        with_text: bool = False,
        limit: int = None,
    ) -> SimpleNamespace:
        """Print all detected tokens from the concept.

        Args:
            idx: (Union[int, List[int], np.ndarray]) The index of the concept.
            val: (Union[float, List[float], np.ndarray]) 
                The value of the concept.
            layer: (int) The layer of the model.
            with_text: (bool) Whether to print the text of the tokens. 
                Default is False.
            limit: (int) The number of tokens to print. Default is None.

        Returns:
            (SimpleNamespace) The concept.
        """
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

    def dump_text(
        self: TranscoderAnalyzer,
        prompt: torch.Tensor,
        dest: Path,
    ) -> None:
        """Dump the text of the prompt to the destination file.

        Args:
            prompt: (torch.Tensor) The prompt.
            dest: (Path) The destination file.
        """
        with dest.open('a') as fl:
            with (tqdm(prompt)) as pprmpt:
                for pr in pprmpt:
                    tx = self.to_clean(pr)
                    fl.write(f'{tx}\n')

    def dump_csv(
        self: TranscoderAnalyzer,
        prompt: torch.Tensor,
        dest: Path,
    ) -> None:
        """Dump the text of the prompt to the destination file.

        Args:
            prompt: (torch.Tensor) The prompt.
            dest: (Path) The destination file.
        """
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

    def dump_tokens(self: TranscoderAnalyzer, dest: Path) -> None:
        """Dump the tokens to the destination file.

        Args:
            dest: (Path) The destination file.
        """
        self.dump_text(self.tokens, dest)

    def tokens_to_csv(self: TranscoderAnalyzer, dest: Path) -> None:
        """Dump the tokens to the destination file.

        Args:
            dest: (Path) The destination file.
        """
        self.dump_csv(self.tokens, dest)


class ConceptAnalysis(object):
    """Class to analyze the concepts in the transcoder activations.

    Attributes:
        prompt: (str) The prompt.
        tr_analyzer: (TranscoderAnalyzer) The transcoder analyzer.
        trunc: (int) The truncation value. Default is None.
    """

    def __init__(
        self: ConceptAnalysis,
        prompt: str,
        tr_analyzer: TranscoderAnalyzer,
        trunc: int = None,
    ) -> None:
        """Initialize ConceptAnalysis and its required state."""
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
        self._texts = tr_analyzer.texts

    @property
    def tr_utils(self: ConceptAnalysis) -> TranscoderUtils:
        """Get the transcoder utils.

        Returns:
            TranscoderUtils: The transcoder utils.
        """
        return self._tr_utils

    @property
    def transcoder(self: ConceptAnalysis) -> Transcoder:
        """Get the transcoder.

        Returns:
            Transcoder: The transcoder.
        """
        return self.tr_utils.transcoder

    @property
    def prompt(self: ConceptAnalysis) -> str:
        """Get the prompt.

        Returns:
            str: The prompt.
        """
        return self._prompt

    @property
    def layers(self: ConceptAnalysis) -> List[int]:
        """Get the layers.

        Returns:
            List[int]: The layers.
        """
        return self._layers

    @property
    def fcas(self: ConceptAnalysis) -> Dict[int, FCA]:
        """Get the FCA objects.

        Returns:
            Dict[int, FCA]: The FCA objects.
        """
        return self._fcas

    @property
    def corpus(self: ConceptAnalysis) -> torch.Tensor:
        """Get the corpus as a matrix of tokens.

        Returns:
            torch.Tensor: The corpus.
        """
        return self._corpus

    @property
    def texts(self: ConceptAnalysis) -> IndexableCorpus:
        """Get the corpus as texts.

        Returns:
            IndexableCorpus: The texts.
        """
        return self._texts

    @property
    def trunc(self: ConceptAnalysis) -> int:
        """Get the truncation value.

        Returns:
            int: The truncation value.
        """
        return self._trunc if self._trunc is not None else 0

    @trunc.setter
    def trunc(self: ConceptAnalysis, value: int) -> None:
        """Set the truncation value.

        Args:
            value: (int) The truncation value.
        """
        if value is not None and value < 0:
            raise ValueError("Truncation value must be non-negative.")
        self._trunc = value

    @property
    def vs(self: ConceptAnalysis) -> Dict[int, np.ndarray]:
        """Get the V vectors.

        Returns:
            Dict[int, np.ndarray]: The V vectors.
        """
        return self._vs

    @vs.setter
    def vs(self: ConceptAnalysis, other_vs: Dict[int, np.ndarray]) -> None:
        """Set the V vectors.

        Args:
            other_vs: (Dict[int, np.ndarray]) The V vectors.
        """
        self._vs = other_vs

    @property
    def idcs(self: ConceptAnalysis) -> List[int]:
        """Get the indices.

        Returns:
            List[int]: The indices.
        """
        return self._idcs

    @idcs.setter
    def idcs(self: ConceptAnalysis, other_idcs: List[int]) -> None:
        """Set the indices.

        Args:
            other_idcs: (List[int]) The indices.
        """
        self._idcs = other_idcs

    @property
    def v_is(self: ConceptAnalysis) -> Dict[int, np.ndarray]:
        """Get the V vectors.

        Returns:
            Dict[int, np.ndarray]: The V vectors.
        """
        return self._v_is

    @v_is.setter
    def v_is(self: ConceptAnalysis, other_v_is: Dict[int, np.ndarray]) -> None:
        """Set the V vectors.

        Args:
            other_v_is: (Dict[int, np.ndarray]) The V vectors.
        """
        self._v_is = other_v_is

    @property
    def c_is(self: ConceptAnalysis) -> Dict[int, Concept]:
        """Get the concept.

        Returns:
            Dict[int, Concept]: The concept.
        """
        return self._c_is

    @c_is.setter
    def c_is(self: ConceptAnalysis, other_c_is: Dict[int, Concept]) -> None:
        """Set the concept.

        Args:
            other_c_is: (Dict[int, Concept]) The concept.
        """
        self._c_is = other_c_is

    @property
    def prop_tokens(self: ConceptAnalysis) -> Dict[int, torch.Tensor]:
        """Get the propagated tokens.

        Returns:
            Dict[int, torch.Tensor]: The propagated tokens.
        """
        return self._prop_tokens

    @prop_tokens.setter
    def prop_tokens(
        self: ConceptAnalysis,
        other_prop_tokens: Dict[int, torch.Tensor],
    ) -> None:
        """Set the propagated tokens.

        Args:
            other_prop_tokens: (Dict[int, torch.Tensor]) The propagated tokens.
        """
        self._prop_tokens = other_prop_tokens

    @property
    def det_tokens(self: ConceptAnalysis) -> Dict[int, torch.Tensor]:
        """Get the detected tokens.

        Returns:
            Dict[int, torch.Tensor]: The detected tokens.
        """
        return self._det_tokens

    @det_tokens.setter
    def det_tokens(
        self: ConceptAnalysis,
        other_det_tokens: Dict[int, torch.Tensor],
    ) -> None:
        """Set the detected tokens.

        Args:
            other_det_tokens: (Dict[int, torch.Tensor]) The detected tokens.
        """
        self._det_tokens = other_det_tokens

    @property
    def prop_tokens(self: ConceptAnalysis) -> Dict[int, torch.Tensor]:
        """Get the propagated tokens.

        Returns:
            Dict[int, torch.Tensor]: The propagated tokens.
        """
        return self._prop_tokens

    @prop_tokens.setter
    def prop_tokens(
        self: ConceptAnalysis,
        other_prop_tokens: Dict[int, torch.Tensor],
    ) -> None:
        """Set the propagated tokens.

        Args:
            other_prop_tokens: (Dict[int, torch.Tensor]) The propagated tokens.
        """
        self._prop_tokens = other_prop_tokens

    @property
    def detected_vs(
        self: ConceptAnalysis,
    ) -> Dict[int, Union[torch.Tensor, np.ndarray]]:
        """Get the detected values.

        Returns:
            Dict[int, Union[torch.Tensor, np.ndarray]]: The detected values.
        """
        return self._detected_vs

    @property
    def detected_v_FG(
        self: ConceptAnalysis,
    ) -> Dict[int, Dict[int, Union[np.ndarray, torch.Tensor]]]:
        """Get the detected foreground values.

        Returns:
            Dict[int, Dict[int, Union[np.ndarray, torch.Tensor]]]: The 
                detected foreground values.
        """
        return self._detected_v_FG

    @property
    def v_FG(self: ConceptAnalysis) -> Dict[int, Dict[int, np.ndarray]]:
        """Get the v_FG values.

        Returns:
            Dict[int, Dict[int, np.ndarray]]: The v_FG values.
        """
        return self._v_FG

    @property
    def pos_idxs(self: ConceptAnalysis) -> Union[List[int], np.ndarray]:
        """Get the positive indices.

        Returns:
            Union[List[int], np.ndarray]: The positive indices.
        """
        return self._pos_idxs

    @property
    def neg_idxs(self: ConceptAnalysis) -> Union[List[int], np.ndarray]:
        """Get the negative indices.

        Returns:
            Union[List[int], np.ndarray]: The negative indices.
        """
        return self._neg_idxs

    def add_detected_vs(
        self: ConceptAnalysis,
        idcs: int,
        vs: Union[np.ndarray, torch.Tensor],
        det_vs: Dict[int, Union[np.ndarray, torch.Tensor]],
    ) -> None:
        """Add detected values for the given layer and index.

        Args:
            idcs: (int) The index of the concept.
            vs: (Union[np.ndarray, torch.Tensor]) The values of the concept.
            det_vs: (Dict[int, Union[np.ndarray, torch.Tensor]]) 
                    The detected values.
        """
        for idx, v in zip(idcs, vs):
            if isinstance(idx, np.ndarray):
                for i in idx:
                    det_vs[i] = v
            else:
                det_vs[idx] = v

    def get_idcs(
        self: ConceptAnalysis,
        layer: int,
    ) -> Tuple[List[int], List[int]]:
        """Get the positive and negative indices for the given layer.

        Args:
            layer: (int) The layer of the model.

        Returns:
            (Tuple[List[int], List[int]]) The positive and negative indices.
        """
        return self.pos_idxs.get(layer, None), self.neg_idxs.get(layer, None)

    def V(self: ConceptAnalysis, layer: int) -> np.ndarray:
        """Get the V matrix for the given layer.

        Args:
            layer: (int) The layer of the model.

        Returns:
            (np.ndarray) The V matrix.
        """
        return self.fcas[layer].V

    def init_fca(
        self: ConceptAnalysis,
        layer: int,
        neg_idxs: Union[List[int], np.ndarray] = None,
    ) -> FCA:
        """Initialize the FCA for the given layer.

        Args:
            layer: (int) The layer of the model.
            neg_idxs: (Union[List[int], np.ndarray]) The negative indices. 
                Default is None.

        Returns:
            (FCA) The initialized FCA.
        """
        return FCA(self.V(layer), neg_idx=neg_idxs)

    def run_model(self: ConceptAnalysis, idx: int, layer: int) -> np.ndarray:
        """Run the model for the given index and layer.

        Args:
            idx: (int) The index of the corpus.
            layer: (int) The layer of the model.

        Returns:
            (np.ndarray) The output of the model.
        """
        return self.transcoder(self.corpus[idx], layer)[0]

    def print_detected_tokens(
        self: ConceptAnalysis,
        layer: int,
        idx: int,
        u: np.ndarray,
        with_text: bool = False,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print detected tokens for the given layer and indices.

        Args:
            layer: (int) The layer of the model.
            idx: (int) The index of the corpus.
            u: (np.ndarray) The output of the model.
            with_text: (bool) Whether to include the text. Default is False.

        Returns:
            Union[np.ndarray, List[List[int]]]: The detected tokens.
        """
        return self.transcoder.print_detected_tokens(
            layer,
            self.texts[idx],
            u,
            with_text=with_text
        )

    def print_all_detected_tokens(
        self: ConceptAnalysis,
        layer: int,
        A: np.ndarray,
        u: np.ndarray,
        with_text: bool = False,
        limit: int = None,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print all detected tokens for the given layer and indices.

        Args:
            layer: (int) The layer of the model.
            A: (np.ndarray) The indices of the corpus.
            u: (np.ndarray) The output of the model.
            with_text: (bool) Whether to include the text. Default is False.
            limit: (int) The limit of the number of tokens to print. 
                Default is None.

        Returns:
            Union[np.ndarray, List[List[int]]]: The detected tokens.
        """
        return self.transcoder.print_all_detected_tokens(
            layer,
            self.corpus[A],
            u,
            with_text=with_text,
            limit=limit,
            indices=A,
        )

    def gen_concept(
        self: ConceptAnalysis,
        idcs: Union[int, List[int], np.ndarray],
        vals: Union[float, List[float], np.ndarray],
        layer: int,
        neg_idxs: Union[List[int], np.ndarray] = None,
    ) -> SimpleNamespace:
        """Generate a concept for the given indices and values.

        Args:
            idcs: (Union[int, List[int], np.ndarray]) The indices.
            vals: (Union[float, List[float], np.ndarray]) The values.
            layer: (int) The layer of the model.
            neg_idxs: (Union[List[int], np.ndarray]) The negative indices. 
                Default is None.

        Returns:
            (SimpleNamespace) The concept.
        """
        fcn = self.init_fca(layer, neg_idxs=neg_idxs)
        cn = gen_concept(idcs, vals, fcn)
        cn.fca = fcn

        return cn

    def G_FG(
        self: ConceptAnalysis,
        v: np.ndarray,
        layer: int,
        neg_idxs: Union[List[int], np.ndarray] = None,
    ) -> Concept:
        """Generate a concept from the given values.

        Args:
            v: (np.ndarray) The values.
            layer: (int) The layer of the model.
            neg_idxs: (Union[List[int], np.ndarray]) The negative indices. 
                Default is None.

        Returns:
            (Concept) The concept.
        """
        fca = self.init_fca(layer, neg_idxs=neg_idxs)
        c_v = fca.G_FG(v)

        return c_v

    def gen_and_print(
        self: ConceptAnalysis,
        idcs: Union[int, List[int], np.ndarray],
        vals: Union[float, List[float], np.ndarray],
        layer: int,
        neg_idxs: Union[List[int], np.ndarray] = None,
        with_text: bool = True,
        limit: int = None,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray], Concept]:
        """Generate a concept and print it.

        Args:
            idcs: (Union[int, List[int], np.ndarray]) The indices.
            vals: (Union[float, List[float], np.ndarray]) The values.
            layer: (int) The layer of the model.
            neg_idxs: (Union[List[int], np.ndarray]) The negative indices. 
                Default is None.
            with_text: (bool) Whether to include the text. Default is True.
            limit: (int) The limit of the number of tokens to print. 
                Default is None.

        Returns:
            (Tuple[np.ndarray, Union[List[List[int]], np.ndarray], Concept]) 
                The detected tokens and their indices.
        """
        cn = self.gen_concept(idcs, vals, layer, neg_idxs=neg_idxs)
        logger.info(f'{cn=}')
        if limit:
            A = cn.c.A[:limit]
            logg_txt = f'but {limit}' if (
                limit < cn.c.A.shape[0]
            ) else f'and all {cn.c.A.shape[0]}'
            logger.info(
                f'Actual detection is {cn.c.A.shape} {logg_txt} are shown'
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
        ), cn

    def topK_v(
        self: ConceptAnalysis,
        layer: int,
        idx: int,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the top K values and indices for the given layer and index.

        Args:
            layer: (int) The layer of the model.
            idx: (int) The index of the corpus.
            k: (int) The number of top values to return. Default is 10.

        Returns:
            (Tuple[np.ndarray, np.ndarray]) The top K values and indices.
        """
        return topK(self.v_is[layer][idx], k)

    def analyze_concepts(self: ConceptAnalysis) -> None:
        """Analyze the concepts for the given prompt and layers."""
        vs = self.tr_utils.run_transcoders(self.prompt, self.layers)
        self.tr_utils.print_tokens(self.prompt)
        self.vs = vs

    def _set_vals(
        self: ConceptAnalysis,
        vals: dict[int, np.ndarray],
        set_val: float | list[float] | dict[int, float] | None,
        red_val: float | list[float] | dict[int, float],
        i: int,
        i_idx: int,
    ) -> None:
        """Set the values for the given indices.

        Args:
            vals: (Dict[int, np.ndarray]) The values.
            set_val: (Union[float, List[float], Dict[int, float]]) 
                The set value.
            red_val: (Union[float, List[float], Dict[int, float]]) 
                The reduction value.
            i: (int) The index.
            i_idx: (int) The index of the corpus.
        """
        if set_val:
            act_val = set_val if isinstance(
                set_val, (float, int)
            ) else set_val[i] if isinstance(
                set_val, Dict
            ) else set_val[i_idx]
            vals[i].fill(act_val)
        else:
            vals[i] -= red_val if isinstance(
                red_val, (float, int)
            ) else red_val[i] if isinstance(
                red_val, Dict
            ) else red_val[i_idx]

    def _collect_vals(
        self: ConceptAnalysis,
        layer: int,
        rng: int = 1,
        red_val: Union[float, List[float], Dict[int, float]] = 0.0,
        set_val: Union[float, List[float], Dict[int, float]] = None,
        trunc: int = 0,
        min_vals: bool = False,
    ) -> None:
        """Collect the values for the given layer and indices.

        Args:
            layer: (int) The layer of the model.
            rng: (int) The range of values to collect. Default is 1.
            red_val: (Union[float, List[float], Dict[int, float]]) 
                The reduction value. Default is 0.0.
            set_val: (Union[float, List[float], Dict[int, float]]) 
                The set value. Default is None.
            trunc: (int) The truncation value. Default is 0.
            min_vals: (bool) Whether to collect minimum values. 
                Default is False.
        """
        vals = {}
        idxs = {}
        v_is_layer = {}
        v_min_layer = {}
        for i_idx, i in enumerate(self.idcs):
            if min_vals and not np.any(self.vs[layer][i] > 0):
                raise ValueError(
                    "A support probe needs a positive source coordinate."
                )
            val_i, idxs[i] = topKrange(
                self.vs[layer][i],
                1 if min_vals else rng
            )
            logger.info(f'{idxs[i]=}')
            vals[i] = truncate(
                val_i,
                trunc if self.trunc is None else self.trunc
            )
            min_val_i = self.fcas[layer].min(idxs[i][0])
            if min_vals and min_val_i <= 0:
                raise ValueError(
                    "The selected coordinate has no positive corpus support."
                )
            logger.info(f'{min_val_i=}')
            logger.info(f'{val_i}')
            logger.info(f'{vals[i]}')
            self._set_vals(vals, set_val, red_val, i, i_idx)
            vals[i][vals[i] < 0] = 0
            logger.info(f'{i_idx=}, {i=}, {red_val=} {set_val=} {vals[i]=}')
            logger.info(f'{vals[i]}, {idxs[i]}')
            v_is_layer[i] = gen_vx(
                idxs[i],
                min_val_i,
                self.fcas[layer]
            ) if min_vals else gen_vx(
                idxs[i],
                vals[i],
                self.fcas[layer]
            )
            v_min_layer[i] = min_val_i
        self.v_is[layer] = v_is_layer

    def gen_concepts(
        self: ConceptAnalysis,
        idcs: List[int],
        layer: int,
        rng: int = 1,
        red_val: Union[float, List[float], Dict[int, float]] = 0.0,
        set_val: Union[float, List[float], Dict[int, float]] = None,
        trunc: int = 0,
        min_vals: bool = False,
    ) -> None:
        """Generate the concepts for the given indices and layer.

        Args:
            idcs: (List[int]) The indices.
            layer: (int) The layer of the model.
            rng: (int) The range of values to collect. Default is 1.
            red_val: (Union[float, List[float], Dict[int, float]]) 
                The reduction value. Default is 0.0.
            set_val: (Union[float, List[float], Dict[int, float]]) 
                The set value. Default is None.
            trunc: (int) The truncation value. Default is 0.
            min_vals: (bool) Whether to collect minimum values. 
                Default is False.
        """
        self.idcs = idcs
        self._collect_vals(
            layer,
            rng=rng,
            red_val=red_val,
            set_val=set_val,
            trunc=trunc,
            min_vals=min_vals,
        )
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

    def to_string(self: ConceptAnalysis, tokens: torch.Tensor) -> str:
        """Convert tokens to a string.

        Args:
            tokens: (torch.Tensor) The tokens.

        Returns:
            (str) The converted text.
        """
        return self.transcoder.to_string(tokens)

    def _analyze_limited_text(
        self: ConceptAnalysis,
        layer: int,
        limit: int = None,
        full_tokens: bool = False,
    ) -> int:
        """Analyze the text for the given layer and indices.

        Args:
            layer: (int) The layer of the model.
            limit: (int) The limit of the number of tokens to analyze.
            full_tokens: (bool) Whether to collect full detected tokens.

        Returns:
            (int) The number of tokens analyzed.
        """
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
                    self.v_is[layer][i],
                    pos_idx=self.pos_idxs.get(layer, None),
                    neg_idx=self.neg_idxs.get(layer, None),
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

        return rn_limit

    def _analyze_rest_tokens(
        self: ConceptAnalysis,
        layer: int,
        rn_limit: int,
    ) -> None:
        """Analyze the text for the given layer and indices.

        Args:
            layer: (int) The layer of the model.
            rn_limit: (int) The limit of the number of tokens to analyze.
        """
        logger.info(
            f'Rest of the tokens:{self.c_is[layer].A.shape[0] - rn_limit} '
            f'for the layer {layer}'
        )
        with tqdm(self.c_is[layer].A[rn_limit:]) as pbar:
            for idx, i_A in enumerate(pbar):
                for i in self.idcs:
                    _, v_FG = self.transcoder.detect_tokens_upper_than_u(
                        layer,
                        self.det_tokens[layer][rn_limit + idx],
                        self.v_is[layer][i],
                        pos_idx=self.pos_idxs.get(layer, None),
                        neg_idx=self.neg_idxs.get(layer, None),
                    )
                    v_FG_ext = self.v_FG[layer][i]
                    self.v_FG[layer][i] = meet(v_FG_ext, v_FG) if not_empty(
                        v_FG_ext
                    ) and not_empty(v_FG) else v_FG_ext if not_empty(
                        v_FG_ext
                    ) else v_FG if not_empty(v_FG) else None

    def analyze_text(
        self: ConceptAnalysis,
        layer: int,
        limit: int = None,
        full_tokens: bool = False,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Analyze the text for the given layer and indices.

        Args:
            layer: (int) The layer of the model.
            limit: (int) The limit of the number of tokens to analyze.
            full_tokens: (bool) Whether to collect full detected tokens.

        Returns:
            (Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]) 
                The detected tokens.
        """
        rn_limit = self._analyze_limited_text(layer, limit, full_tokens)
        if full_tokens and not_empty(self.c_is[layer].A[rn_limit:]):
            logger.info(f'Analyzing rest of the tokens for layer {layer}')
            self._analyze_rest_tokens(layer, rn_limit)
        else:
            logger.info(f'Not analyzing rest of the tokens for layer {layer}')

        return self.det_tokens[layer], self.detected_vs[layer]

    def gen_text(
        self: ConceptAnalysis,
        idcs: List[int],
        layer: int,
        rng: int = 1,
        red_val: Union[float, List[float], Dict[int, float]] = 0.0,
        set_val: Union[float, List[float], Dict[int, float]] = None,
        trunc: int = 0,
        limit: int = None,
        min_vals: bool = False,
        full_tokens: bool = False,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Generate text for the given layer, indices, and values.

        Args:
            idcs: (List[int]) The indices.
            layer: (int) The layer of the model.
            rng: (int) The range of values to collect.
            red_val: (Union[float, List[float], Dict[int, float]]) 
                The reduction value.
            set_val: (Union[float, List[float], Dict[int, float]]) 
                The set value.
            trunc: (int) The truncation value.
            limit: (int) The limit of the number of tokens to analyze.
            min_vals: (bool) Whether to collect minimum values.
            full_tokens: (bool) Whether to collect full detected tokens.

        Returns:
            (Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]) 
                The detected tokens.
        """
        self.gen_concepts(
            idcs,
            layer,
            rng=rng,
            red_val=red_val,
            set_val=set_val,
            trunc=trunc,
            min_vals=min_vals,
        )
        dets, vs = self.analyze_text(
            layer,
            limit=limit,
            full_tokens=full_tokens,
        )

        return dets, vs


def init_from_model(
    transcoder: Transcoder,
    layers: List[int],
    tokens_path: Path,
    model_path: Union[Path, str] = None,
    device: Union[str, torch.device] = torch.device('cpu'),
    dataset_path: Path = None,
    columns: List[str] = None,
    vector_dir: Path = None,
    pos_idxs: Union[List[int], np.ndarray] = None,
    neg_idxs: Union[List[int], np.ndarray] = None,
    dataset: Dataset = None,
) -> TranscoderAnalyzer:
    """Initialize the transcoder analyzer from the model.

    Args:
        transcoder: (Transcoder) The transcoder.
        layers: (List[int]) The layers of the model.
        tokens_path: (Path) The path to the tokens.
        model_path: (Union[Path, str]) The path to the model.
        device: (Union[str, torch.device]) The device to use.
        dataset_path: (Path) The path to the dataset.
        columns: (List[str]) The columns of the dataset.
        vector_dir: (Path) The directory to store the vectors.
        pos_idxs: (Union[List[int], np.ndarray]) The positive indices.
        neg_idxs: (Union[List[int], np.ndarray]) The negative indices.
        dataset: (Dataset) The dataset.

    Returns:
        (TranscoderAnalyzer) The transcoder analyzer.
    """

    owt_tokens_torch = load_tokens(
        transcoder,
        tokens_path,
        device=device,
        csv_path=dataset_path,
        columns=columns,
        dataset=dataset,
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


def _init_release_and_sae_id(
    model_name: str = None,
    release: str = None,
    sae_id: str = None
) -> Dict[str, str]:
    """Initialize the release and sae id.

    Args:
        model_name: (str) The name of the model.
        release: (str) The release of the model.
        sae_id: (str) The sae id of the model.

    Returns:
        (Dict[str, str]) The release and sae id.
    """
    kwargs = dict()
    if model_name:
        kwargs['model_name'] = model_name
    if release:
        kwargs['release'] = release
    if sae_id:
        kwargs['sae_id'] = sae_id

    return kwargs


def init_transcoder_or_sae(
    model_name: str = None,
    release: str = None,
    sae_id: str = None,
    device: Union[str, torch.device] = torch.device('cpu'),
    tr_or_sae: bool = True,
    layers: List[int] = None,
) -> Transcoder:
    """Initialize the sparse surrogate models on a given layers 
        (transcoder or SAEs).

    Args:
        model_name: (str) The name of the model.
        release: (str) The release of the model.
        sae_id: (str) The sae id of the model.
        device: (Union[str, torch.device]) The device to use.
        tr_or_sae: (bool) Whether to use the transcoder or the SAE.
        layers: (List[int]) The layers of the model.

    Returns:
        (Transcoder) The sparse surrogates on layers.
    """

    kwargs = _init_release_and_sae_id(
        model_name=model_name,
        release=release,
        sae_id=sae_id,
    )
    transcoder = init_transcoder(
        model_name=model_name,
        layers=layers,
        device=device,
    ) if tr_or_sae else init_sae(
        layers=layers,
        device=device,
        **kwargs,
    )

    return transcoder


def init_analyzer(
    layers: List[int],
    tokens_path: Path,
    model_path: Path,
    model_name: str = 'gpt2',
    release: str = None,
    sae_id: str = None,
    device: Union[str, torch.device] = torch.device('cpu'),
    dataset_path: Path = None,
    columns: List[str] = None,
    vector_dir: Path = None,
    pos_idxs: Union[List[int], np.ndarray] = None,
    neg_idxs: Union[List[int], np.ndarray] = None,
    tr_or_sae: bool = True,
    dataset: Dataset = None,
) -> TranscoderAnalyzer:
    """Initialize the transcoder analyzer.

    Args:
        layers: (List[int]) The layers of the model.
        tokens_path: (Path) The path to the tokens.
        model_path: (Path) The path to the model.
        model_name: (str) The name of the model.
        release: (str) The release of the model.
        sae_id: (str) The sae id of the model.
        device: (Union[str, torch.device]) The device to use.
        dataset_path: (Path) The path to the dataset.
        columns: (List[str]) The columns of the dataset.
        vector_dir: (Path) The directory to store the vectors.
        pos_idxs: (Union[List[int], np.ndarray]) The positive indices.
        neg_idxs: (Union[List[int], np.ndarray]) The negative indices.
        tr_or_sae: (bool) Whether to use the transcoder or the SAE.
        dataset: (Dataset) The dataset.

    Returns:
        (TranscoderAnalyzer) The transcoder analyzer.
    """
    transcoder = init_transcoder_or_sae(
        model_name=model_name,
        release=release,
        sae_id=sae_id,
        device=device,
        tr_or_sae=tr_or_sae,
        layers=layers,
    )
    empty_cache()

    tr_analyzer = init_from_model(
        transcoder,
        layers,
        tokens_path,
        model_path=model_path,
        device=device,
        dataset_path=dataset_path,
        dataset=dataset,
        columns=columns,
        vector_dir=vector_dir,
        pos_idxs=pos_idxs,
        neg_idxs=neg_idxs,
    )
    empty_cache()

    return tr_analyzer
