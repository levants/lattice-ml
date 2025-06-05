import gc
import logging
import re
import sys
from cmd import IDENTCHARS
from operator import index
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sympy import limit
from torch import nn
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformers import PreTrainedTokenizer

from src.lattmc.fca.fca_utils import le
from src.lattmc.fca.lattice_utils import meet, meet_all
from src.lattmc.fca.utils import in_any, not_empty
from src.lattmc.fca.visualization_utils import with_background
from src.lattmc.tc.tcirc import sae_training
from src.lattmc.tc.tcirc.sae_training.sparse_autoencoder import \
    SparseAutoencoder

logger = logging.getLogger(__name__)

TC_PREFIX = 'final_sparse_autoencoder_gpt2-small_blocks'
TC_SUFFIX = 'ln2.hook_normalized_24576.pt'

LOG_PREFIX = 'Base GPT-2 and transcoder'

REPO_ID = 'pchlenski/gpt2-transcoders'
LOG_SUFFIX = 'weights loaded successfully.'


class Transcoder(object):
    """Transcoder class for the GPT-2 model."""

    ACT_PREFIX = 'normalized'
    ACT_SUFFIX = 'ln2'

    def __init__(
        self,
        model: HookedTransformer,
        transcoders: Dict[int, SparseAutoencoder],
        device: torch.device = torch.device('cpu'),
        background_dets: int = None,
    ):
        self._model = model.to(device).eval()
        self._device = device
        self._transcoders = {
            k: tc.to(device).eval() for
            k, tc in transcoders.items()
        }
        self._background_dets = background_dets

    @property
    def model(self) -> HookedTransformer:
        """Get the model."""
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        return self.model.tokenizer

    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self._device

    @device.setter
    def device(self, other_device: torch.device):
        """Set the device."""
        self._device = other_device
        self.model.to(other_device)
        self.transcoders = {
            k: tc.to(other_device) for k, tc in self.transcoders.items()
        }

    @property
    def transcoders(self) -> Dict[int, SparseAutoencoder]:
        """Get the transcoders."""
        return self._transcoders

    @transcoders.setter
    def transcoders(self, other_transcoders: Dict[int, SparseAutoencoder]):
        """Set the transcoders."""
        self._transcoders = {
            k: tc.to(
                self.device
            ).eval() for k, tc in other_transcoders.items()
        }

    @property
    def background_dets(self) -> int:
        """Get the background detection code."""
        return self._background_dets

    @background_dets.setter
    def background_dets(self, bg_code: int):
        """Set the background detection code."""
        if bg_code is None or isinstance(bg_code, int):
            self._background_dets = bg_code
        else:
            raise ValueError(
                f'Background detection code must be an integer, '
                f'got {bg_code} of type {type(bg_code)}.'
            )

    def __getitem__(self, layer: int) -> SparseAutoencoder:
        """Get the transcoder for the given layer."""
        if layer not in self.transcoders:
            raise ValueError(
                f'Layer {layer} not found in transcoders.'
            )
        return self.transcoders[layer]

    def _get_act_name(self, layer: int) -> str:
        """Get the activation name for the given layer."""
        return get_act_name(
            self.ACT_PREFIX,
            layer,
            self.ACT_SUFFIX
        )

    def to(self, device: torch.device):
        """Move the model to the specified device."""
        self.device = device
        self.model.to(device)
        self.transcoders = {
            k: tc.to(device) for k, tc in self.transcoders.items()
        }
        return self

    def _add_padding_token(self):
        """Add a padding token to the tokenizer."""
        if self.tokenizer.pad_token is None:
            # We add a padding token, purely to implement the tokenizer.
            # This will be removed before inputting tokens to the model,
            # so we do not need to increment d_vocab in the model.
            self.tokenizer.add_special_tokens(  # type: ignore
                {"pad_token": "<PAD>"}
            )

    def tokenize(
        self,
        prompt: str,
        return_tensors: str = 'pt',
        padding: bool = True
    ) -> torch.Tensor:
        """Tokenize the input prompt."""
        # Check if the tokenizer has a padding token
        self._add_padding_token()
        # Tokenize the input prompt
        text = f'{self.tokenizer.eos_token}{prompt}'  # type: ignore
        # Tokenize the text and return the tensor
        tokens = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding
        ).to(self.device)  # type: ignore

        input_ids = tokens['input_ids']

        return input_ids

    def _check_and_tokenize(
        self,
        prompt: Union[str, torch.Tensor],
        return_tensors: str = 'pt',
        padding: bool = True
    ) -> torch.Tensor:
        """Check the type of the input prompt and tokenize it."""
        if isinstance(prompt, str):
            tokens = self.tokenize(prompt, return_tensors, padding)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.to(self.device)
        else:
            raise ValueError(
                f'Invalid prompt type: {type(prompt)}. '
                'Expected str or torch.Tensor.'
            )
        return tokens

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, torch.Tensor],
        layer: int
    ) -> np.ndarray:
        """Forward pass through the model and transcoder.
        Args:
            prompt -- input prompt
            layer -- layer to transcode
        Returns:
            v -- transcoded output
        """
        # 1. Tokenize the input prompt
        tokens = self._check_and_tokenize(prompt)
        # 2 Get the activations cache from the model
        _, cache = self.model.run_with_cache(tokens)  # type: ignore
        # 3 Get the activations from cache
        z = cache[self._get_act_name(layer)]
        # 4 Get the transcoder activations for the layer
        t = self.transcoders[layer](z)
        # 5 Get the transcoded output
        v = t[1].to('cpu').detach().numpy()

        return v

    def run_layers(
        self,
        prompt: str,
        layers: List[int]
    ) -> Dict[int, np.ndarray]:
        """Run the transcoder for the given layers."""
        vs = {}
        tokens = self.tokenize(prompt)
        for layer in layers:
            vs[layer] = self(tokens, layer)[0]

        return vs

    def text_tokens(self, prompt: str) -> List[Tuple[torch.Tensor, str]]:
        """Get the text tokens for the input prompt."""
        tokens = self.tokenize(prompt)[0] if isinstance(
            prompt,
            str
        ) else prompt
        text_tokens = [(tk, self.to_string(tk)) for tk in tokens]

        return text_tokens

    def print_text_tokens(self, prompt: str):
        """Print the text tokens for the input prompt."""
        text_tokens = self.text_tokens(prompt)
        logger.info(f'Text tokens for prompt: {prompt}')
        for idx, (tk, t) in enumerate(text_tokens):
            logger.info(f'{idx} {tk}: {t}')

    @torch.inference_mode()
    def detect_token(
        self,
        layer: int,
        prompt: torch.Tensor,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Detect the token in the transcoder activations."""
        vs = self(prompt, layer)[0]
        with tqdm(vs) as vp:
            idxs_vs = [(
                idx,
                v,
                with_background(
                    self.to_string(prompt[idx]),
                    bg_code=self.background_dets
                ) if self.background_dets else self.to_string(prompt[idx])
            )
                for idx, v in enumerate(vp) if le(u, v)]
            idxl, vl, ws = zip(*idxs_vs) if idxs_vs else ([], [], [])
            idxs = np.array(
                idxl if isinstance(
                    idxl, (list, tuple)
                ) else [idxl]
            )
            vs = np.array(vl)
            v_FG = meet_all(vs) if not_empty(vs) else None

        return idxs, vs, list(ws), v_FG

    def assemble_text(
        self,
        prompt: torch.Tensor,
        idxs: np.ndarray,
    ) -> str:
        """Assemble the text from the tokens."""
        tokens = self.tokenizer.convert_ids_to_tokens(prompt)
        decoded_tokens = [
            self.to_background_string(
                token,
                bg_code=41
            ) if in_any(
                idx,
                idxs
            ) else self.to_clean_string(
                token
            ) for idx, token in enumerate(tokens)
        ]
        text_tokens = ''.join(decoded_tokens)

        return text_tokens

    def print_detected_tokens(
            self,
            layer: int,
            prompt: torch.Tensor,
            u: np.ndarray,
            with_text: bool = False,
            idx: int = None,
    ) -> Tuple[np.ndarray, Union[List[int], np.ndarray]]:
        """Print the tokens in the transcoder activations."""
        idxs, _, w_, v_FG = self.detect_token(layer, prompt, u)
        prefix = '' if idx is None else f'index {idx}: '
        if with_text:
            text_tokens = self.assemble_text(
                prompt, idxs
            ) if with_text else ''
            logger.info(f'Prompt: {prefix}{text_tokens}\n')
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
        indices: Union[List[int], np.ndarray, torch.Tensor] = None
    ) -> Tuple[np.ndarray, Union[List[List[int]], np.ndarray]]:
        """Print all the tokens in the transcoder activations."""
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
                idx=token_idx
            )
            if not_empty(v_FG_i):
                v_FG_list.append(v_FG_i)
            cl_idxs_i = idxs_i if not_empty(idxs_i) else np.array([])
            idxs_i_list.append(cl_idxs_i)
        v_FGs = np.array(v_FG_list)
        v_FG = meet_all(v_FGs) if not_empty(v_FG_list) else None
        token_idcs = idxs_i_list

        return v_FG, token_idcs

    def to_string(self, prompt: torch.Tensor) -> str:
        return self.model.to_string(prompt)  # type: ignore

    def decode(
        self,
        prompt: torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return self.tokenizer.decode(
            prompt,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def _clean_each_token(self, token: str) -> str:
        text = self.tokenizer.convert_tokens_to_string([token])
        text = self.tokenizer.clean_up_tokenization(text)
        text = re.sub(r'\uFFFD+', "'", text)

        return text

    def to_clean_string(self, token: str) -> str:
        """Convert a token to a clean string."""
        ctokn = self._clean_each_token(token)
        ctokn = ' ' + ctokn[1:] if ctokn.startswith('Ġ') else ctokn

        return ctokn

    def to_space_string(self, token: str) -> str:
        """Convert a token to a space and string."""
        ctokn = self._clean_each_token(token)
        return (' ',  ctokn[1:]) if ctokn.startswith('Ġ') else ('', ctokn)

    def to_background_string(self, token: str, bg_code: int) -> str:
        """Convert a token to a background string."""
        space, ctokn = self.to_space_string(token)
        btokn = with_background(ctokn, bg_code=bg_code)
        btext = f'{space}{btokn}'

        return btext

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def load_tc_state(layer: int = 0) -> dict:
    """Load the transcoder state dict.

    Args:
        layer -- trancoder layer (default: {0})
    Returns:
        tc_state --  the transcoder state dict
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
    device: torch.device = torch.device('cpu')
) -> SparseAutoencoder:
    """Load the transcoder weights for the specified layer.

    Keyword Arguments:
        layer -- trancoder layer (default: {0})
        device -- device to load the model on (default: {cpu})

    Returns:
        transcoder --  the transcoder model
    """
    # 3. Load the transcoder state dict
    transcoder_state = load_tc_state(layer)

    # Change device
    cfg = transcoder_state['cfg']
    cfg.device = device

    # Initialize the transcoder model class
    transcoder = SparseAutoencoder(cfg)
    transcoder.load_state_dict(transcoder_state['state_dict'])

    # 4. (Optional) Initialize your transcoder model class and load weights
    logger.info(
        f'{LOG_PREFIX} {cfg.is_transcoder} {layer=} {LOG_SUFFIX}'
    )

    return transcoder


def load_transcoders(
    layers: List[int] = list(range(11)),
    device: torch.device = torch.device('cpu')
) -> Dict[int, SparseAutoencoder]:
    """Load the transcoder weights for the specified layers.

    Keyword Arguments:
        layers -- list of transcoder layers
        device -- device to load the model on (default: {cpu})
    Returns:
        DIctionary of transcoders by layers --  the transcoder models
    """
    transcoders = {}
    for layer in layers:
        logger.info(f'Loading transcoder for layer {layer}')
        transcoders[layer] = load_transcoder(layer, device)

    return transcoders


def init_transcoder(
    layers: List[int] = list(range(12)),
    device: torch.device = torch.device('cpu')
) -> Transcoder:
    """Initialize the transcoders for the specified layers.

    Keyword Arguments:
        layers -- list of transcoder layers
        device -- device to load the model on (default: {cpu})
    Returns:
        transcoder --  the transcoder model
    """
    transcoders = load_transcoders(layers, device)
    model = HookedTransformer.from_pretrained('gpt2', device=device)
    logger.info('Initializing Transcoder')
    transcoder = Transcoder(
        model=model,
        transcoders=transcoders,
        device=device
    )
    logger.info('Transcoder initialized')

    return transcoder
