"""Image utilities for Lattice-theoretic Formal Concept Analysis (FCA)."""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.utils import make_grid


def show_img(ds: List[Tuple], idx: int):
    """Show image from dataset

    Args:
        ds (List[Tuple]): The dataset.
        idx (int): The index of the image.
    """
    plt.imshow(ds[idx][0])


def show(
    imgs: List[Union[torch.Tensor, np.ndarray]],
    h: int = 12,
    w: int = 12
):
    """Show images in a row

    Args:
        imgs (List[Union[torch.Tensor, np.ndarray]]): The images to show.
        h (int, optional): The height of the figure. Defaults to 12.
        w (int, optional): The width of the figure. Defaults to 12.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(
        ncols=len(imgs),
        figsize=(w, h),
        squeeze=False
    )
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.to('cpu').detach()
            img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_grid(
    G_A: np.ndarray,
    data: List[Tuple],
    nrow: int = 8,
    h: int = 12,
    w: int = 12,
    my: int = None
):
    """Show grid of images from dataset

    Args:
        G_A (np.ndarray): The grid of images.
        data (List[Tuple]): The dataset.
        nrow (int, optional): The number of rows. Defaults to 8.
        h (int, optional): The height of the figure. Defaults to 12.
        w (int, optional): The width of the figure. Defaults to 12.
        my (int, optional): The class to exclude. Defaults to None.
    """
    G_A_F = G_A.ravel()
    to_tensor = transforms.ToTensor()
    A_gr = [
        to_tensor(data[i][0]) for i in G_A_F
    ] if my is None else [
        to_tensor(data[i][0]) for i in G_A_F if data[i][1] != my
    ]
    grid = make_grid(A_gr, nrow=nrow)
    show(grid, h=h, w=w)
