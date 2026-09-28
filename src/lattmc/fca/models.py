"""Models for Lattice-theoretic Formal Concept Analysis (FCA)."""

import logging
from typing import Callable, Union

import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


def find_device() -> torch.device:
    """Find the device to run the model on.

    Returns:
        torch.device: The device to run the model on.
    """
    device_type = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device_type)
    logger.info(f'Using device: {device}')

    return device


class ToTensor(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def convert(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert a PIL image or numpy array to a PyTorch tensor.

        Args:
            x (Union[np.ndarray, torch.Tensor]): The input data.

        Returns:
            torch.Tensor: The output tensor.
        """
        return x if isinstance(x, torch.Tensor) else self.to_tensor(x)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert a PIL image or numpy array to a PyTorch tensor.

        Args:
            x (Union[np.ndarray, torch.Tensor]): The input data.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.convert(x)


class NetWrapper(object):
    """Inference wrapper for PyTorch model.

    Attributes:
        net (torch.nn.Module): The PyTorch model.
        transform (Union[transforms.Compose, Callable]): The transform to
            apply to the input data.
        device (torch.device): The device to run the model on.
        cpu (torch.device): The CPU device.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        transform: Union[transforms.Compose, Callable],
        device: Union[str, torch.device] = None
    ):
        self._net = net.eval()
        self.transform = transform
        self._device = torch.device(device) if device else find_device()
        self._net.to(self._device)
        self.cpu = torch.device('cpu')

    @property
    def net(self):
        """Get the PyTorch model.

        Returns:
            torch.nn.Module: The PyTorch model.
        """
        return self._net

    @property
    def device(self):
        """Get the device.

        Returns:
            torch.device: The device.
        """
        return self._device

    def __getitem__(self, i):
        """Get the i-th layer of the network.

        Args:
            i (int): The index of the layer.

        Returns:
            torch.nn.Module: The i-th layer of the network.
        """
        return self.net[i]

    def __len__(self):
        """Get the number of layers in the network.

        Returns:
            int: The number of layers in the network.
        """
        return len(self.net)

    @torch.inference_mode()
    def forward(self, *xs: torch.Tensor, k: int = 6) -> np.ndarray:
        """Forward pass through the network.

        Args:
            *xs (torch.Tensor): The input data.
            k (int, optional): The number of layers to use. Defaults to 6.

        Returns:
            np.ndarray: The output of the network.
        """
        ts = torch.stack(
            [self.transform(x) for x in xs],
            dim=0
        )
        ts = ts.to(self.device)
        rs = self[: k](ts) if k else self.net(ts)
        rs = rs.to(self.cpu).detach().numpy()

        return rs

    def to(self, device: Union[str, torch.device] = None):
        """Device setter to attach the model.

        Args:
            device (Union[str, torch.device], optional): The device to move
                the network to. Defaults to None.
        """
        dvc = torch.device(device) if device else find_device()
        self._device = dvc
        self.net.to(self._device)

    def __call__(self, *xs: torch.Tensor, k: int = 6) -> np.ndarray:
        """Forward pass through the network.

        Args:
            *xs (torch.Tensor): The input data.
            k (int, optional): The number of layers to use. Defaults to 6.

        Returns:
            np.ndarray: The output of the network.
        """
        return self.forward(*xs, k=k)
