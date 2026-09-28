"""Image generators for Lattice-theoretic Formal Concept Analysis (FCA)."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


# Functions to generate images using PIL with variations
def generate_vertical_line_image(
        height: int,
        width: int,
        line_length: int = 14,
        line_thickness: int = 2,
        shift: int = 0,
        intensity: int = 255
):
    """Generate a vertical line image.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        line_length (int, optional): The length of the line. Defaults to 14.
        line_thickness (int, optional): The thickness of the line. 
            Defaults to 2.
        shift (int, optional): The shift of the line. Defaults to 0.
        intensity (int, optional): The intensity of the line. Defaults to 255.

    Returns:
        np.ndarray: The vertical line image.
    """
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    x = width // 2 + shift
    start_y = (height - line_length) // 2
    end_y = start_y + line_length
    draw.line((x, start_y, x, end_y), fill=intensity, width=line_thickness)
    return np.array(image)


def generate_horizontal_line_image(
        height: int,
        width: int,
        line_length: int = 14,
        line_thickness: int = 2,
        shift: int = 0,
        intensity: int = 255
):
    """Generate a horizontal line image.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        line_length (int, optional): The length of the line. Defaults to 14.
        line_thickness (int, optional): The thickness of the line. 
            Defaults to 2.
        shift (int, optional): The shift of the line. Defaults to 0.
        intensity (int, optional): The intensity of the line. Defaults to 255.

    Returns:
        np.ndarray: The horizontal line image.
    """
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    y = height // 2 + shift
    start_x = (width - line_length) // 2
    end_x = start_x + line_length
    draw.line((start_x, y, end_x, y), fill=intensity, width=line_thickness)
    return np.array(image)


def generate_stretched_ring_image(
        height: int,
        width: int,
        radius_x: int = None,
        radius_y: int = None,
        thickness: int = 2,
        intensity: int = 255
):
    """Generate a stretched ring image.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        radius_x (int, optional): The radius of the ring. Defaults to None.
        radius_y (int, optional): The radius of the ring. Defaults to None.
        thickness (int, optional): The thickness of the ring. Defaults to 2.
        intensity (int, optional): The intensity of the ring. Defaults to 255.

    Returns:
        np.ndarray: The stretched ring image.
    """
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
        height: int,
        width: int,
        radius_x: int = None,
        radius_y: int = None,
        thickness: int = 2,
        intensity: int = 255
):
    """Generate a normal ring image.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        radius_x (int, optional): The radius of the ring. Defaults to None.
        radius_y (int, optional): The radius of the ring. Defaults to None.
        thickness (int, optional): The thickness of the ring. Defaults to 2.
        intensity (int, optional): The intensity of the ring. Defaults to 255.

    Returns:
        np.ndarray: The normal ring image.
    """
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
    def __init__(self, num_samples: int, height: int = 28, width: int = 28):
        """Initialize the dataset.

        Args:
            num_samples (int): The number of samples.
            height (int, optional): The height of the image. Defaults to 28.
            width (int, optional): The width of the image. Defaults to 28.
        """
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.shapes = ['vertical_line', 'horizontal_line', 'sring', 'nring']

    def __len__(self):
        """Return the number of samples.

        Returns:
            int: The number of samples.
        """
        return self.num_samples

    def _gen_data(self):
        """Generate a single data sample.

        Returns:
            Tuple[torch.Tensor, str]: The data sample.
        """
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

    def __getitem__(self, idx: int):
        """Get the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            Tuple[torch.Tensor, str]: The item.
        """
        if idx < self.__len__():
            image, shape_type = self._gen_data()
        else:
            raise StopIteration

        return image, shape_type


def gen_line_idx(hv_shift: int = 6, sid: int = 4, eid: int = 9, hv: str = 'h'):
    """Generate the indices for the line.

    Args:
        hv_shift (int, optional): The shift of the line. Defaults to 6.
        sid (int, optional): The start index. Defaults to 4.
        eid (int, optional): The end index. Defaults to 9.
        hv (str, optional): The orientation of the line. Defaults to 'h'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The indices.
    """
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
def show_images(
    images: List[np.ndarray],
    titles: List[str],
    ncols: int = 4
) -> None:
    """Show images in a grid.

    Args:
        images (List[np.ndarray]): The images to show.
        titles (List[str]): The titles of the images.
        ncols (int, optional): The number of columns. Defaults to 4.
    """
    nrows = len(images) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axs[i // ncols, i % ncols]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()
