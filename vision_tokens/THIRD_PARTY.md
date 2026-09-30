# Data and pretrained-weight provenance

## Handwritten digits

`digits/dataset/digits_codexgen.npz` records the 1,797 images returned by
scikit-learn's `load_digits`, in their original order, with the declared
split indices. Source and dataset documentation:

- https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset
- https://archive.ics.uci.edu/dataset/80/

The results manifest records the SHA-256 of the original feature and label
arrays and of each released artifact. Digit CNN/SAE weights were trained
locally for the paper.

## CIFAR-10

`cifar10_resnet34/dataset/cifar10_sample_codexgen.npz` is a documented subset
of CIFAR-10 by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
Original source: https://www.cs.toronto.edu/~kriz/cifar.html.

For each class, the first 125 official training images and first 25 official
test images are retained. The first 100 selected training rows per class
train the SAE; the remaining 25 are reserved for calibration. Original
split names and indices are stored next to the original uint8 RGB pixels.
The selected image pixels occupy about 4 MB. The complete CIFAR archive and
unrelated legacy datasets are not duplicated. Class labels describe the
dataset, not established semantic meanings of surrogate coordinates.

## Pretrained ResNet34

The backbone checkpoint is copied byte-for-byte from torchvision's public
IMAGENET1K_V1 ResNet34 artifact:

https://download.pytorch.org/models/resnet34-b627a593.pth

Its exact SHA-256 is in `cifar10_resnet34/results/natural_codexgen.json`.
The file is about 83.3 MiB and is included for offline inference. The
512-coordinate Top-16 SAE checkpoint was trained locally for this paper.
The source model is credited in the manuscript and is not represented as
an original model trained by the paper's author.

These datasets and pretrained weights retain their upstream attribution
and applicable terms; this directory does not assign them a new license.
The image galleries are computed from the attributed sample. No Distill
figures were copied: the cited work motivates the investigative approach.
