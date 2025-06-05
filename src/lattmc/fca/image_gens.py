import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


# Functions to generate images using PIL with variations
def generate_vertical_line_image(
        height,
        width,
        line_length=14,
        line_thickness=2,
        shift=0,
        intensity=255
):
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    x = width // 2 + shift
    start_y = (height - line_length) // 2
    end_y = start_y + line_length
    draw.line((x, start_y, x, end_y), fill=intensity, width=line_thickness)
    return np.array(image)


def generate_horizontal_line_image(
        height,
        width,
        line_length=14,
        line_thickness=2,
        shift=0,
        intensity=255
):
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    y = height // 2 + shift
    start_x = (width - line_length) // 2
    end_x = start_x + line_length
    draw.line((start_x, y, end_x, y), fill=intensity, width=line_thickness)
    return np.array(image)


def generate_stretched_ring_image(
        height,
        width,
        radius_x=None,
        radius_y=None,
        thickness=2,
        intensity=255
):
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    if radius_x is None:
        radius_x = width // 4
    if radius_y is None:
        radius_y = height // 8
    center = (width // 2, height // 2)
    draw.ellipse(
        (center[0] - radius_x, center[1] - radius_y, center[0] +
         radius_x, center[1] + radius_y),
        outline=intensity, width=thickness
    )
    return np.array(image)


def generate_normal_ring_image(
        height,
        width,
        radius_x=None,
        radius_y=None,
        thickness=2,
        intensity=255
):
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    rdn = np.random.choice(np.array([0, 1, 2, 3]))
    radius_denom = 4 + rdn
    if radius_x is None:
        radius_x = width // radius_denom
    if radius_y is None:
        radius_y = height // radius_denom
    eps = np.random.choice(np.array([-3, -2, -1, 1, 2, 3]), size=2)
    center = ((width + eps[0]) // 2, (height + eps[1]) // 2)
    draw.ellipse(
        (center[0] - radius_x, center[1] - radius_y, center[0] +
         radius_x, center[1] + radius_y),
        outline=intensity, width=thickness
    )
    return np.array(image)

# Custom PyTorch Dataset


class CustomShapeDataset(Dataset):
    def __init__(self, num_samples, height=28, width=28):
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.shapes = ['vertical_line', 'horizontal_line', 'sring', 'nring']

    def __len__(self):
        return self.num_samples

    def _gen_data(self):
        shape_type = np.random.choice(self.shapes)
        shift = np.random.randint(-5, 6)  # Shift lines by up to ±5 pixels
        # Random intensity between 50 and 255
        intensity = np.random.randint(16, 256)
        if shape_type == 'vertical_line':
            image = generate_vertical_line_image(
                self.height, self.width, shift=shift, intensity=intensity)
        elif shape_type == 'horizontal_line':
            image = generate_horizontal_line_image(
                self.height, self.width, shift=shift, intensity=intensity)
        elif shape_type == 'sring':
            image = generate_stretched_ring_image(
                self.height, self.width, intensity=intensity)
        elif shape_type == 'nring':
            image = generate_normal_ring_image(
                self.height, self.width, intensity=intensity)

        # Convert image to PyTorch tensor and normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, shape_type

    def __getitem__(self, idx):

        if idx < self.__len__():
            image, shape_type = self._gen_data()
        else:
            raise StopIteration

        return image, shape_type


def gen_line_idx(hv_shift=6, sid=4, eid=9, hv='h'):
    range_hv_a = np.array(list(range(sid, eid)))
    range_hv_b = np.array([hv_shift for _ in range(range_hv_a.shape[0])])
    nz_idx_hv = (
        range_hv_a,
        range_hv_b
    ) if hv == 'v' else (
        range_hv_b,
        range_hv_a
    )

    return nz_idx_hv


# Display some examples
def show_images(images, titles, ncols=4):
    nrows = len(images) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axs[i // ncols, i % ncols]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()
