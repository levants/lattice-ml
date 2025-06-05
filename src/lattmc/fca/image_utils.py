import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.utils import make_grid


def show_img(ds, idx):
    plt.imshow(ds[idx][0])


def show(imgs, h=12, w=12):
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


def show_grid(G_A, data, nrow=8, h=12, w=12, my=None):
    G_A_F = G_A.ravel()
    to_tensor = transforms.ToTensor()
    A_gr = [
        to_tensor(data[i][0]) for i in G_A_F
    ] if my is None else [
        to_tensor(data[i][0]) for i in G_A_F if data[i][1] != my
    ]
    grid = make_grid(A_gr, nrow=nrow)
    show(grid, h=h, w=w)
