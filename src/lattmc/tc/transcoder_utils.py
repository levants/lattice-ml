"""Transcoder / FCA utilities for 
Lattice-theoretic Formal Concept Analysis (FCA) on 
SAE / transcoder activations."""

import logging
import re
import sys
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.lattmc.fca.utils import convert_to_array, project_on, topNonZeros
from src.lattmc.fca.fca_utils import (le, not_empty, to_numpy)
from src.lattmc.fca.lattice_utils import filter_upper, meet_all, upper_indices
from src.lattmc.fca.visualization_utils import with_background
from src.lattmc.tc.sparse_surrogates import SparseSurrogates
from src.lattmc.tc.tcirc import sae_training
from src.lattmc.tc.tcirc.sae_training.sparse_autoencoder import \
    SparseAutoencoder

logger = logging.getLogger(__name__)

TC_PREFIX = 'final_sparse_autoencoder_gpt2-small_blocks'
TC_SUFFIX = 'ln2.hook_normalized_24576.pt'

LOG_PREFIX = 'Base GPT-2 and transcoder'

REPO_ID = 'pchlenski/gpt2-transcoders'
LOG_SUFFIX = 'weights loaded successfully.'


class Transcoder(SparseSurrogates):
    """Transcoder class for the GPT-2 model.
    Attributes:
        model (HookedTransformer): model
        transcoders (Dict[int, SparseAutoencoder]): transcoders
        device (torch.device): device
        background_dets (int): background detection code. 
            Default is None.
        texgraph_dets (str): texgraph detection code. 
            Default is None.
    """

    def __init__(
        self,
        model: HookedTransformer,
        transcoders: Dict[int, SparseAutoencoder],
        device: torch.device = torch.device('cpu'),
        background_dets: int = None,
        texgraph_dets: str = None,
    ):
        super().__init__(
            model,
            transcoders,
            device,
            background_dets=background_dets,
            texgraph_dets=texgraph_dets
        )

    def _generate_joint_activations(
        self,
        idxs_vs: List[Tuple[int, np.ndarray, str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Generate joint activations from the given indices and vectors.
        Args:
            idxs_vs (List[Tuple[int, np.ndarray, str]]): list of indices, 
                vectors and words
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]: 
                tuple of indices, vectors, words and joint activations
        """
        idxl, vl, ws = zip(*idxs_vs) if idxs_vs else ([], [], [])
        idxs = np.array(
            idxl if isinstance(
                idxl, (list, tuple)
            ) else [idxl]
        )
        vs = np.array(vl)
        v_FG = meet_all(vs) if not_empty(vs) else None

        return idxs, vs, ws, v_FG

    def _loop_and_filter_features(
        self,
        vs: Iterable[np.ndarray],
        u: np.ndarray,
        str_tokens: List[str],
        pos_idx: Union[List, np.ndarray] = None,
        neg_idx: Union[List, np.ndarray] = None,
        bg_code: Union[int, str] = None,
    ) -> List[Tuple[int, np.ndarray, str]]:
        """Loop through the transcoder activations and filter the features 
            by comparing with u.
        Args:
            vs (np.ndarray): transcoder activations
            u (np.ndarray): target vector to compare with
            str_tokens (List[str]): list of tokens
            pos_idx (Union[List, np.ndarray]): positive indices. 
                Default is None.
            neg_idx (Union[List, np.ndarray]): negative indices. 
                Default is None.
            bg_code (Union[int, str]): background code. Default is None.
        Returns:
            List[Tuple[int, np.ndarray, str]]: list of indices, 
                vectors and words
        """
        masked_idcs = upper_indices(u, vs, pos_idx=pos_idx, neg_idx=neg_idx)
        text_idcs = [
            (
                idx,
                vs[idx],
                self._generate_text_with_background(
                    str_tokens,
                    idx,
                    bg_code=bg_code,
                )
            )
            for idx in masked_idcs
        ]

        return text_idcs

    def _progress_and_filter_features(
        self,
        vs: np.ndarray,
        u: np.ndarray,
        str_tokens: List[str],
        pos_idx: Union[List, np.ndarray] = None,
        neg_idx: Union[List, np.ndarray] = None,
        bg_code: Union[int, str] = None,
    ) -> List[Tuple[int, np.ndarray, str]]:
        """Loop through the transcoder activations and filter the features 
            by comparing with u.
        Args:
            vs (np.ndarray): transcoder activations
            u (np.ndarray): target vector to compare with
            str_tokens (List[str]): list of tokens
            pos_idx (Union[List, np.ndarray]): positive indices. 
                Default is None.
            neg_idx (Union[List, np.ndarray]): negative indices. 
                Default is None.
            bg_code (Union[int, str]): background code. Default is None.
        Returns:
            List[Tuple[int, np.ndarray, str]]: list of indices, vectors and words
        """
        with tqdm(vs) as vp:
            idxs_vs = self._loop_and_filter_features(
                vp,
                u,
                str_tokens,
                pos_idx=pos_idx,
                neg_idx=neg_idx,
                bg_code=bg_code,
            )
        return idxs_vs

    @torch.inference_mode()
    def detect_tokens(
        self,
        layer: int,
        prompt: torch.Tensor,
        u: np.ndarray,
        pos_idx: Union[List, np.ndarray] = None,
        neg_idx: Union[List, np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Detect the token in the transcoder activations.

        Args:
            layer (int): layer to transcode
            prompt (torch.Tensor): input prompt
            u (np.ndarray): transcoder activations
            pos_idx (Union[List, np.ndarray]): positive indices. 
                Default is None.
            neg_idx (Union[List, np.ndarray]): negative indices. 
                Default is None.
        Returns:
            idxs_list (List[np.ndarray]): list of indices of the tokens
            bg_tokens (List[str]): list of background tokens
            color_codes (List[str]): list of color codes for detected tokens
        """
        vs = self(prompt, layer)[0]
        str_tokens = self.to_str_tokens(prompt)
        _, idxs_nonz = topNonZeros(u)
        idxs_list = list()
        prev_idxs_vs = set()
        bg_tokens_list = list()
        color_codes = list()
        with tqdm(idxs_nonz, desc='Detecting tokens') as p_nonz:
            for dim_idx, dim_i in enumerate(p_nonz):
                p_k = project_on(u, dim_i)
                color_code = self.bg_codes[dim_idx]
                idx_vs_text = self._loop_and_filter_features(
                    vs,
                    p_k,
                    str_tokens,
                    pos_idx=pos_idx,
                    neg_idx=neg_idx,
                    bg_code=color_code,
                )
                idxs_vs_list, _, bg_tokens = zip(
                    *idx_vs_text
                ) if idx_vs_text else ([], [], [])
                idxs_vs = set(idxs_vs_list)
                diff_ids_vs = idxs_vs - prev_idxs_vs
                if diff_ids_vs:
                    idxs_list.append(diff_ids_vs)
                    prev_idxs_vs.update(diff_ids_vs)
                    bg_tokens_list.append(bg_tokens)
                    color_codes.append(color_code)

        return idxs_list, bg_tokens_list, color_codes

    @torch.inference_mode()
    def detect_token(
        self,
        layer: int,
        prompt: torch.Tensor,
        u: np.ndarray,
        pos_idx: Union[List, np.ndarray] = None,
        neg_idx: Union[List, np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Detect the token in the transcoder activations.

        Args:
            layer (int): layer to transcode
            prompt (torch.Tensor): input prompt
            u (np.ndarray): transcoder activations
            pos_idx (Union[List, np.ndarray]): positive indices. 
                Default is None.
            neg_idx (Union[List, np.ndarray]): negative indices. 
                Default is None.
        Returns:
            idxs (np.ndarray): indices of the tokens
            vs (np.ndarray): transcoded output
            ws (List[str]): words
            v_FG (np.ndarray): transcoded output
        """
        vs = self(prompt, layer)[0]
        str_tokens = self.to_str_tokens(prompt)
        idxs_vs = self._loop_and_filter_features(
            vs, u, str_tokens, pos_idx=pos_idx, neg_idx=neg_idx
        )
        idxs, vs, ws, v_FG = self._generate_joint_activations(idxs_vs)

        return idxs, vs, ws, v_FG

    @torch.inference_mode()
    def detect_tokens_upper_than_u(
        self,
        layer: int,
        prompt: torch.Tensor,
        u: np.ndarray,
        pos_idx: Union[List, np.ndarray] = None,
        neg_idx: Union[List, np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect the token in the transcoder activations
        upper than u

        Args:
            layer (int): layer to transcode
            prompt (torch.Tensor): input prompt
            u (np.ndarray): target vector to compare with
            pos_idx (Union[List, np.ndarray]): positive indices. 
                Default is None.
            neg_idx (Union[List, np.ndarray]): negative indices. 
                Default is None.
        Returns:
            vs (np.ndarray): upper vectors
            v_FG (np.ndarray): meet of upper vectors
        """
        vs = self(prompt, layer)[0]
        vs = filter_upper(
            u, vs,
            pos_idx=pos_idx,
            neg_idx=neg_idx
        )
        v_FG = meet_all(vs) if not_empty(vs) else None

        return vs, v_FG

    def print_detected_tokens(
            self,
            layer: int,
            prompt: torch.Tensor,
            u: np.ndarray,
            with_text: bool = False,
            idx: int = None,
            log_tokens_and_idxs: bool = True,
    ) -> Tuple[np.ndarray, Union[List[int], np.ndarray]]:
        """Print the tokens in the transcoder activations.
        Args:
            layer (int): layer to print
            prompt (torch.Tensor): input prompt
            u (np.ndarray): transcoder activations
            with_text (bool): whether to print the text. Default is False.
            idx (int): index of the token. Default is None.
            log_tokens_and_idxs (bool): whether to log the tokens and indices. 
                Default is True.
        Returns:
            v_FG (np.ndarray): transcoded output
            idxs (np.ndarray): indices of the tokens
        """
        idxs, _, w_, v_FG = self.detect_token(layer, prompt, u)
        prefix = '' if idx is None else f'index {idx}: '
        if with_text:
            text_tokens = self.assemble_text(
                prompt, idxs
            ) if with_text else ''
            logger.info(f'Prompt: {prefix}{text_tokens}\n')
        if log_tokens_and_idxs:
            for idx in idxs:
                logger.info(f'Token: {self.to_string(prompt[idx])}')
                logger.info(f'Indices: {idxs}')

        return v_FG, idxs

    def print_all_detected_tokens(
        self,
        layer: int,
        prompts: torch.Tensor,
        u: np.ndarray,
        with_text: bool = False,
        limit: int = None,
        indices: Union[List[int], np.ndarray, torch.Tensor] = None,
        log_tokens_and_idxs: bool = True,
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print all the tokens in the transcoder activations.
        Args:
            layer (int): layer to print
            prompts (torch.Tensor): input prompts
            u (np.ndarray): transcoder activations
            with_text (bool): whether to print the text. Default is False.
            limit (int): limit the number of prompts. Default is None.
            indices (Union[List[int], np.ndarray, torch.Tensor]): indices 
                of the tokens. Default is None.
            log_tokens_and_idxs (bool): whether to log the tokens and indices. 
                Default is True.
        Returns:
            v_FG (np.ndarray): transcoded output
            idxs (np.ndarray): indices of the tokens 
        """
        max_text = min(prompts.shape[0], limit) if limit else prompts.shape[0]
        v_FG_list = []
        idxs_i_list = []
        for idx, prompt in enumerate(prompts[:max_text]):
            token_idx = None if indices is None else indices[idx]
            v_FG_i, idxs_i = self.print_detected_tokens(
                layer,
                prompt,
                u,
                with_text=with_text,
                idx=token_idx,
                log_tokens_and_idxs=log_tokens_and_idxs,
            )
            if not_empty(v_FG_i):
                v_FG_list.append(v_FG_i)
            cl_idxs_i = idxs_i if not_empty(idxs_i) else np.array([])
            idxs_i_list.append(cl_idxs_i)
        v_FGs = np.array(v_FG_list)
        v_FG_limit = meet_all(v_FGs) if not_empty(v_FG_list) else None
        token_idcs = idxs_i_list

        return v_FG_limit, token_idcs

    def print_detected_multitokens(
        self,
        layer: int,
        prompts: torch.Tensor,
        prompt_idxs: Union[List[int], np.ndarray, torch.Tensor],
        u: np.ndarray,
        limit: int = None,
    ) -> Dict[int, str]:
        """Print all the tokens in the transcoder activations.
        Args:
            layer (int): layer to print
            prompts (torch.Tensor): input prompts
            prompt_idxs (Union[List[int], np.ndarray, torch.Tensor]): indices 
                of the prompts.
            u (np.ndarray): transcoder activations
            limit (int): limit the number of prompts. Default is None.
        Returns:
            Dict[int, str]: dictionary indexed by prompt indices with 
                latex multitexts as values
        """
        max_text = min(prompts.shape[0], limit) if limit else prompts.shape[0]
        multitexts = dict()
        for prompt_idx, prompt in zip(
            prompt_idxs[:max_text],
            prompts[:max_text]
        ):
            indx_list, _, color_codes = self.detect_tokens(
                layer,
                prompt,
                u,
            )
            assembled_multitext = self.assemble_multitext(
                prompt,
                indx_list,
                color_codes,
            ) if not_empty(indx_list) else self.decode(prompt)
            logger.info(f'Prompt {prompt_idx}: {assembled_multitext}')
            multitexts[prompt_idx] = assembled_multitext

        return multitexts

    def match_tokens_in_prompts_and_print(
        self,
        A: np.ndarray,
        S: Union[
            Set[int],
            List[int],
            np.ndarray,
            List[torch.Tensor],
            torch.Tensor,
        ],
        corpus: torch.Tensor,
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
            corpus (torch.Tensor): corpus of tokens.
            with_text (bool): whether to print the tokens as text.
                Default is True.
            color_idx (int): index of the color code in self.bg_codes.
                Default is 35.
        Returns:
            text_dict (Dict[int, str]): dictionary indexed by prompt 
                identifiers with 
                latex multitexts as values
        """
        A_arr = convert_to_array(A)
        prompts = corpus[A_arr]
        text_dict = dict()
        S_list = list(S)
        for idx, prompt in zip(A_arr, prompts):
            idcs = self.match_tokens(prompt, S_list)
            text = self.assemble_text(
                prompt,
                idcs,
                color_idx=color_idx,
            )
            text_dict[idx] = text
            if with_text:
                logger.info(f'Prompt {idx}: {text}')

        return text_dict


def load_tc_state(layer: int = 0) -> Dict[str, Any]:
    """Load the transcoder state dict.

    Args:
        layer (int): trancoder layer (default: {0})
    Returns:
        tc_state (Dict[str, Any]): the transcoder state dict
    """
    # 1. Download transcoder weights for MLP block 0
    filename = f'{TC_PREFIX}.{layer}.{TC_SUFFIX}'
    tc_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename
    )
    logger.info(f'Loaded file {filename}')
    # 2. Load the transcoder state dict
    sys.modules['sae_training'] = sae_training
    tc_state = torch.load(tc_path, map_location='cpu', weights_only=False)

    return tc_state


def load_transcoder(
    layer: int = 0,
    device: torch.device = torch.device('cpu'),
) -> SparseAutoencoder:
    """Load the transcoder weights for the specified layer.

    Args:
        layer (int): trancoder layer (default: {0})
        device (torch.device): device to load the model on (default: {cpu})

    Returns:
        transcoder (SparseAutoencoder): the transcoder model
    """
    # 3. Load the transcoder state dict
    transcoder_state = load_tc_state(layer)

    # Change device
    cfg = transcoder_state['cfg']
    cfg.device = device

    # Handle deprecated torch_dtype -> dtype rename
    if hasattr(cfg, 'torch_dtype') and not hasattr(cfg, 'dtype'):
        cfg.dtype = cfg.torch_dtype

    # Initialize the transcoder model class
    transcoder = SparseAutoencoder(cfg)
    transcoder.load_state_dict(transcoder_state['state_dict'])
    transcoder = transcoder.eval()

    # 4. (Optional) Initialize your transcoder model class and load weights
    logger.info(
        f'{LOG_PREFIX} {cfg.is_transcoder} {layer=} {LOG_SUFFIX}'
    )

    return transcoder


def load_transcoders(
    layers: List[int] = list(range(11)),
    device: torch.device = torch.device('cpu'),
) -> Dict[int, SparseAutoencoder]:
    """Load the transcoder weights for the specified layers.

    Args:
        layers (List[int]): list of transcoder layers. 
            Default is list(range(11)).
        device (torch.device): device to load the model on (default: {cpu})
    Returns:
        transcoders (Dict[int, SparseAutoencoder]): Dictionary 
            of transcoders by layers
    """
    transcoders = {}
    for layer in layers:
        logger.info(f'Loading transcoder for layer {layer}')
        transcoders[layer] = load_transcoder(layer, device)
        logger.info(f'Transcoder for layer {layer} loaded')

    return transcoders


def init_transcoder(
    model_name: str = 'gpt2',
    layers: List[int] = list(range(12)),
    device: torch.device = torch.device('cpu'),
) -> Transcoder:
    """Initialize the transcoders for the specified layers.

    Args:
        model_name (str): The model name. Default is 'gpt2'.
        layers (List[int]): The layers. Default is list(range(12)).
        device (torch.device): The device. Default is 'cpu'.

    Returns:
        transcoder (Transcoder): The transcoder model.
    """
    transcoders = load_transcoders(layers, device)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    logger.info('Initializing Transcoder')
    transcoder = Transcoder(
        model=model,
        transcoders=transcoders,
        device=device
    )
    logger.info('Transcoder initialized')

    return transcoder
