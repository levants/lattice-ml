"""Sparse surrogate models such as Transcoder and SAE utility functions."""

from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.lattmc.fca.utils import to_numpy
from src.lattmc.tc.tcirc.sae_training.sparse_autoencoder import \
    SparseAutoencoder
from src.lattmc.tc.tokenization_utils import TokenizationUtils


class SparseSurrogates(TokenizationUtils):
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
        """Initialize the Transcoder.

        Args:
            model (HookedTransformer): HookedTransformer model
            transcoders (Dict[int, SparseAutoencoder]): dictionary of 
                transcoders by layers
            device (torch.device): device to bind module. Default is 'cpu'.
            background_dets (int): number of background detectors. 
                Default is None.
            texgraph_dets (str): path to the texgraph detectors. 
                Default is None.
        """
        super().__init__(
            model,
            device,
            background_dets=background_dets,
            texgraph_dets=texgraph_dets
        )
        self._transcoders = {
            k: tc.to(device).eval() for
            k, tc in transcoders.items()
        }

    @property
    def device(self) -> torch.device:
        """Get the device.
        Returns:
            torch.device: device
        """
        return super().device

    @device.setter
    def device(self, other_device: torch.device):
        """Set the device.
        Args:
            other_device (torch.device): device to move the model to
        """
        super().device = other_device
        self.transcoders = {
            k: tc.to(other_device) for k, tc in self.transcoders.items()
        }

    @property
    def transcoders(self) -> Dict[int, SparseAutoencoder]:
        """Get the transcoders.
        Returns:
            Dict[int, SparseAutoencoder]: transcoders
        """
        return self._transcoders

    @transcoders.setter
    def transcoders(self, other_transcoders: Dict[int, SparseAutoencoder]):
        """Set the transcoders.
        Args:
            other_transcoders (Dict[int, SparseAutoencoder]): transcoders
        """
        self._transcoders = {
            k: tc.to(
                self.device
            ).eval() for k, tc in other_transcoders.items()
        }

    def __getitem__(self, layer: int) -> SparseAutoencoder:
        """Get the transcoder for the given layer.
        Args:
            layer (int): layer number
        Returns:
            SparseAutoencoder: transcoder for the given layer
        """
        if layer not in self.transcoders:
            raise ValueError(
                f'Layer {layer} not found in transcoders.'
            )
        return self.transcoders[layer]

    def to(self, device: torch.device):
        """Move the model to the specified device.
        Args:
            device (torch.device): device to move the model to
        Returns:
            Transcoder: the transcoder model
        """
        super().to(device)
        self.transcoders = {
            k: tc.to(device) for k, tc in self.transcoders.items()
        }
        return self

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, torch.Tensor],
        layer: int
    ) -> np.ndarray:
        """Forward pass through the model and transcoder.
        Args:
            prompt (Union[str, torch.Tensor]): input prompt
            layer (int): layer to transcode
        Returns:
            v (np.ndarray): transcoded output
        """
        z = super().forward(prompt, layer)
        # 4 Get the transcoder activations for the layer
        t = self.transcoders[layer](z)
        # 5 Get the transcoded output
        v = to_numpy(t[1])

        return v

    @torch.inference_mode()
    def run_layers(
        self,
        prompt: Union[str, torch.Tensor],
        layers: List[int]
    ) -> Dict[int, np.ndarray]:
        """Run the transcoder for the given layers.
        Args:
            prompt (Union[str, torch.Tensor]): input prompt
            layers (List[int]): layers to run the transcoder on
        Returns:
            vs (Dict[int, np.ndarray]): dictionary of transcoded outputs
        """
        tokens = self._check_and_tokenize(prompt)
        vs = {layer: self(tokens, layer)[0] for layer in layers}

        return vs

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Call the transcoder.
        Args:
            *args (Any): positional arguments
            **kwargs (Any): keyword arguments
        Returns:
            torch.Tensor: transcoded output
        """
        return self.forward(*args, **kwargs)
