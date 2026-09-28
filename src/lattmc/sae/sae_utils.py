"""SAE / Transcoder utilities for 
Lattice-theoretic Formal Concept Analysis (FCA)."""

import logging
import pprint
import re
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from pyexpat import model
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformers import PreTrainedTokenizer

from src.lattmc import sae
from src.lattmc.fca.fca_utils import le
from src.lattmc.fca.lattice_utils import meet_all
from src.lattmc.fca.utils import in_any, not_empty
from src.lattmc.fca.visualization_utils import with_background
from src.lattmc.sae.nlp_sae_utils import Text2Latent, Text2Sae
from src.lattmc.tc.tcirc import sae_training
from src.lattmc.tc.tcirc.sae_training.sparse_autoencoder import \
    SparseAutoencoder
from src.lattmc.tc.transcoder_utils import Transcoder

logger = logging.getLogger(__name__)

MODEL_NAME = 'gpt2-small'
RELEASE = 'gpt2-small-res-jb'
SAE_ID = 'blocks.{}.hook_resid_pre'

SAE_PREFIX = 'blocks'
SAE_SUFFIX = 'hook_resid_pre'

LOG_PREFIX = 'Base GPT-2 and transcoder'

REPO_ID = 'pchlenski/gpt2-transcoders'
LOG_SUFFIX = 'weights loaded successfully.'


class SAECoder(Text2Latent):
    """SAE encoder class for the GPT-2 model."""

    def __init__(
        self,
        model_name: str,
        release: str,
        sae_id: str,
        device: str = 'cpu',
        model: HookedTransformer = None
    ):
        """Initialize the SAECoder encoder.

        Args:
            model_name (str): name of the model.
            release (str): release of the model.
            sae_id (str): id of the SAE.
            device (str): device to bind module. Default is 'cpu'.
            model (HookedTransformer): HookedTransformer model.
                Default is None.
        """
        super().__init__(model_name, release, sae_id, device, model)


class SAETranscoder(Transcoder):
    """Sparse AutoEncoder / Transcoder class for the GPT-2 model.

    Attributes:
        model (HookedTransformer): HookedTransformer model.
        transcoders (Dict[int, SAE]): dictionary of transcoders by layers.
        device (torch.device): device to bind module. Default is 'cpu'.
        background_dets (int): number of background detectors.
            Default is None.
        texgraph_dets (str): path to the texgraph detectors.
            Default is None.
    """

    def __init__(
        self,
        model: HookedTransformer,
        transcoders: Dict[int, SAE],
        device: torch.device = torch.device('cpu'),
        background_dets: int = None,
        texgraph_dets: str = None,
    ):
        """Initialize the SAETranscoder.

        Args:
            model (HookedTransformer): HookedTransformer model.
            transcoders (Dict[int, SAE]): dictionary of transcoders by layers.
            device (torch.device): device to bind module. Default is 'cpu'.
            background_dets (int): number of background detectors.
                Default is None.
            texgraph_dets (str): path to the texgraph detectors.
                Default is None.
        """
        super().__init__(
            model,
            transcoders,
            device=device,
            background_dets=background_dets,
            texgraph_dets=texgraph_dets,
        )

    def forward(self, prompt: str | torch.Tensor, layer: int) -> np.ndarray:
        """Forward pass through the transcoder.

        Args:
            prompt (str | torch.Tensor): The prompt.
            layer (int): The layer.

        Returns:
            np.ndarray: The activations.
        """
        sae = self.transcoders[layer]
        z = sae.embed(prompt)

        return z


def load_sae(
    model_name: str = MODEL_NAME,
    release: str = RELEASE,
    sae_id: str = SAE_ID,
    layer: int = 0,
    device: Union[str, torch.device] = torch.device('cpu'),
    model: HookedTransformer = None,
) -> SAECoder:
    """Load Sparse Autoencoder.

    Args:
        model_name (str): name of the model. Default is 'gpt2-small'.
        release (str): release of the model. Default is 'gpt2-small-res-jb'.
        sae_id (str): id of the SAE. Default is 'blocks.{}.hook_resid_pre'.
        layer (int): trancoder layer (default: {0})
        device (Union[str, torch.device]): device to bind module. 
            Default is torch.device('cpu').
        model (HookedTransformer): HookedTransformer model. Default is None.
    Returns:
        SAECoder: the SAE model
    """
    # 1. Download and load SAE model and weights
    sae_id = sae_id.format(layer)
    sae = SAECoder(
        model_name,
        release,
        sae_id,
        device,
        model=model,
    )
    logger.info(f'Loaded model {sae_id}')

    return sae


def load_transcoder(
    model_name: str = MODEL_NAME,
    release: str = RELEASE,
    sae_id: str = SAE_ID,
    layer: int = 0,
    device: torch.device = torch.device('cpu'),
    model: HookedTransformer = None,
) -> SAECoder:
    """Load the transcoder weights for the specified layer.

    Args:
        model_name (str): name of the model. Default is 'gpt2-small'.
        release (str): release of the model. Default is 'gpt2-small-res-jb'.
        sae_id (str): id of the SAE. Default is 'blocks.{}.hook_resid_pre'.
        layer (int): trancoder layer (default: {0})
        device (Union[str, torch.device]): device to bind module. 
            Default is torch.device('cpu').
        model (HookedTransformer): HookedTransformer model. Default is None.
    Returns:
        SAECoder: the SAE model
    """
    transcoder = load_sae(
        model_name=model_name,
        release=release,
        sae_id=sae_id,
        layer=layer,
        device=device,
        model=model
    )
    logger.info(
        f'{LOG_PREFIX} Sparse AutoEncoder {layer=} {LOG_SUFFIX}'
    )

    return transcoder


def load_transcoders(
    model_name: str = MODEL_NAME,
    release: str = RELEASE,
    sae_id: str = SAE_ID,
    layers: List[int] = list(range(11)),
    device: torch.device = torch.device('cpu'),
    model: HookedTransformer = None,
) -> Dict[int, SAECoder]:
    """Load the transcoder weights for the specified layers.

    Args:
        model_name (str): name of the model. Default is 'gpt2-small'.
        release (str): release of the model. Default is 'gpt2-small-res-jb'.
        sae_id (str): id of the SAE. Default is 'blocks.{}.hook_resid_pre'.
        layers (List[int]): list of transcoder layers 
                (default: {list(range(11))})
        device (Union[str, torch.device]): device to bind module. 
            Default is torch.device('cpu').
        model (HookedTransformer): HookedTransformer model. Default is None.
    Returns:
        Dict[int, SAECoder]: dictionary of transcoders by layers
    """
    transcoders = {}
    for layer in layers:
        logger.info(f'Loading transcoder for layer {layer}')
        transcoders[layer] = load_transcoder(
            model_name=model_name,
            release=release,
            sae_id=sae_id,
            layer=layer,
            device=device,
            model=model
        )

    return transcoders


def init_sae_from_hooked_transformer(
    model_name: str = MODEL_NAME,
    release: str = RELEASE,
    sae_id: str = SAE_ID,
    layers: List[int] = list(range(12)),
    device: torch.device = torch.device('cpu'),
) -> Transcoder:
    """Initialize the transcoders for the specified layers.

    Args:
        model_name (str): name of the model. Default is 'gpt2-small'.
        release (str): release of the model. Default is 'gpt2-small-res-jb'.
        sae_id (str): id of the SAE. Default is 'blocks.{}.hook_resid_pre'.
        layers (List[int]): list of transcoder layers 
                    (default: {list(range(12))})
        device (Union[str, torch.device]): device to bind module. 
            Default is torch.device('cpu').
    Returns:
        Transcoder: the transcoder / SAE model wrapper
    """
    model = HookedTransformer.from_pretrained(model_name, device=device)
    transcoders = load_transcoders(
        model_name,
        release,
        sae_id,
        layers=layers,
        device=device,
        model=model
    )
    logger.info('Initializing Transcoder')
    transcoder = SAETranscoder(
        model,
        transcoders=transcoders,
        device=device
    )
    logger.info('Transcoder initialized')

    return transcoder


def init_sae(
    model_name: str = MODEL_NAME,
    release: str = RELEASE,
    sae_id: str = SAE_ID,
    layers: List[int] = list(range(12)),
    device: torch.device = torch.device('cpu')
) -> Transcoder:
    """Initialize the transcoders for the specified layers.

    Args:
        model_name (str): name of the model. Default is 'gpt2-small'.
        release (str): release of the model. Default is 'gpt2-small-res-jb'.
        sae_id (str): id of the SAE. Default is 'blocks.{}.hook_resid_pre'.
        layers (List[int]): list of transcoder layers 
                (default: {list(range(12))})
        device (Union[str, torch.device]): device to bind module. 
            Default is torch.device('cpu').
    Returns:
        Transcoder: the transcoder model
    """
    transcoder = init_sae_from_hooked_transformer(
        model_name=model_name,
        release=release,
        sae_id=sae_id,
        layers=layers,
        device=device
    )
    logger.info(
        'Transcoder initialized from pretrained HookedTransformer'
    )

    return transcoder
