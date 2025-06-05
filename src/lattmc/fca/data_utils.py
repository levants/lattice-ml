import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def layer_hist(X_y, V_X, y=None):
    if y is None:
        V_X_y = V_X
    else:
        with tqdm(V_X) as p_V_X:
            V_X_y = np.array(
                [v_x for x_y, v_x in zip(X_y, p_V_X) if x_y[1] == y]
            )

    return V_X_y


def get_digits(data):
    digits = dict()
    with tqdm(data) as pdata:
        for x, y in pdata:
            digits.setdefault(y, list())
            digits[y].append(x)

    return digits


def visualize_slices(activations, filters=32):
    for k in range(0, filters, 16):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < 16:
                activation = activations[k + i]
                activation = activation.cpu() if isinstance(
                    activation, torch.Tensor
                ) else activation
                im = ax.imshow(activation, cmap='viridis')
                ax.set_title(f'Conv1 - Filter {k + i}')
                ax.axis('off')
                # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.show()


# Function to visualize the activations
def visualize_activations(model, image, layers=[2, 4], hist=False):
    # Pass the image through the network
    activations = list()
    with torch.no_grad():
        for k in layers:
            output = model(image, k=k)
            activations.append(np.squeeze(output, axis=0))

    # Plot the activations
    for k in range(0, 32, 16):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < 16:
                activation = activations[0][k + i]
                if hist:
                    ax = fig.add_subplot(4, 4, i + 1, projection='3d')
                    activation = activation
                    x, y = np.meshgrid(
                        np.arange(activation.shape[1]),
                        np.arange(activation.shape[0])
                    )
                    ax.bar3d(
                        x.ravel(),
                        y.ravel(),
                        np.zeros_like(x.ravel()), 1, 1,
                        activation.ravel(),
                        shade=True
                    )
                else:
                    im = ax.imshow(activation, cmap='viridis')
                    ax.set_title(f'Conv1 - Filter {k + i}')
                    ax.axis('off')
                # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.show()
    print('======================')
    for k in range(0, 64, 16):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < 16:
                im = ax.imshow(activations[1][k + i], cmap='viridis')
                ax.set_title(f'Conv2 - Filter {k + i}')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.show()

    return activations


def show_activation(activation, layer_name='', filter_index=0):
    # Plot the activation as a grid of numbers
    fig, ax = plt.subplots(figsize=(16, 16))
    cax = ax.matshow(activation, cmap='viridis')

    # Customize the ticks to show each step
    ax.set_xticks(np.arange(activation.shape[1]))
    ax.set_yticks(np.arange(activation.shape[0]))

    # Show grid lines
    ax.grid(color='black', linestyle='-', linewidth=1, which='both')
    ax.xaxis.set_ticks_position('bottom')

    # Annotate the grid with the activation values
    for (i, j), val in np.ndenumerate(activation):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    fig.colorbar(cax)
    plt.title(f'Activation of {layer_name} - Filter {filter_index}')
    plt.show()


def visualize_weights(layer, num_filters=32):
    weights = layer.weight.data.cpu().numpy()
    for k in range(0, num_filters, 16):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                # Select the i-th filter and the first input channel
                img = weights[i, 0, :, :]
                im = ax.imshow(img, cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Filter {i}')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.show()


def compute_mean_std(dataset, workers=1):
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=workers
    )
    mean = 0.
    std = 0.
    with tqdm(loader) as ploader:
        for images, _ in ploader:
            # batch size (the last batch can have smaller size)
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std
