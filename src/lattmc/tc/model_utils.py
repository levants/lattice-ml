"""Model utilities for the Transcoder class."""


import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name


class ModelUtils(object):
    """Model utilities for the Transcoder class.

    Attributes:
        model (HookedTransformer): model
        device (torch.device): device
    """

    ACT_PREFIX = 'normalized'
    ACT_SUFFIX = 'ln2'

    def __init__(
        self,
        model: HookedTransformer,
        device: torch.device = torch.device('cpu'),
    ):
        self._model = model.to(device).eval()
        self._device = device

    @property
    def model(self) -> HookedTransformer:
        """Get the model.
        Returns:
            HookedTransformer: model
        """
        return self._model

    @property
    def device(self) -> torch.device:
        """Get the device.
        Returns:
            torch.device: device
        """
        return self._device

    @device.setter
    def device(self, other_device: torch.device):
        """Set the device.
        Args:
            other_device (torch.device): device to move the model to
        """
        self._device = other_device
        self.model.to(other_device)

    def _get_act_name(self, layer: int) -> str:
        """Get the activation name for the given layer.
        Args:
            layer (int): layer number
        Returns:
            str: activation name
        """
        return get_act_name(
            self.ACT_PREFIX,
            layer,
            self.ACT_SUFFIX
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, torch.Tensor],
        layer: int
    ) -> torch.Tensor:
        """Forward pass through the model and transcoder.
        Args:
            prompt (Union[str, torch.Tensor]): input prompt
            layer (int): layer to transcode
        Returns:
            v (torch.Tensor): activations for the given layer
        """
        # 1. Tokenize the input prompt
        tokens = self._check_and_tokenize(prompt)
        # 2 Get the activations cache from the model
        _, cache = self.model.run_with_cache(tokens)  # type: ignore
        # 3 Get the activations from cache
        z = cache[self._get_act_name(layer)]

        return z

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Call the transcoder.
        Args:
            *args (Any): positional arguments
            **kwargs (Any): keyword arguments
        Returns:
            torch.Tensor: activations for the given layer
        """
        return self.forward(*args, **kwargs)
