"""Utilities for model initialization, device management, 
and garbage collection."""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def init_device() -> torch.device:
    """
    Initialize the device CUDA or MPS or CPU

    Returns:
        torch.device: The device
    """
    device_type = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    device = torch.device(device_type)

    logger.info(f'Device: {device}')

    return device


def empty_cache() -> int:
    """
    Empty the CUDA and MPS cache and collect garbage

    Returns:
        int: The result of the garbage collection
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gc.collect()
