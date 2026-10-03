"""Image-grouped datasets for position-free sparse feature comparisons."""

from __future__ import annotations
from typing import Any
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile

import argparse
import io
import json
import tarfile
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tf

from lattmc.vision.overcomplete_fetch_codexgen import ROOT, digest
from lattmc.vision.paths_codexgen import experiment_root


def crop(image: Image.Image, mask: bool = False) -> Image.Image:
    """Resize and center-crop an image or label mask to 224 pixels."""
    mode = InterpolationMode.NEAREST if mask else InterpolationMode.BICUBIC
    return tf.center_crop(tf.resize(image, 256, mode), [224, 224])


def save(
    name: str,
    rows: Sequence[tuple[dict[str, Any], bytes, Image.Image | None]],
) -> None:
    """Save aligned cropped images, masks, and metadata records."""
    folder = ROOT / 'dataset'
    folder.mkdir(parents=True, exist_ok=True)
    images, masks, records = [], [], []
    for record, raw, mask in rows:
        image = Image.open(io.BytesIO(raw)).convert('RGB')
        images.append(np.asarray(crop(image)))
        if mask is None:
            mask = Image.new('L', image.size, 255)
        masks.append(np.asarray(crop(mask, True)))
        records.append(record)
    np.savez_compressed(folder / f'{name}_codexgen.npz',
                        images=np.stack(images), masks=np.stack(masks))
    (folder / f'{name}_codexgen.json').write_text(
        json.dumps(records, indent=2) + '\n')
    print(name, len(records), flush=True)


def original(name: str) -> None:
    """Prepare the cached original-image dataset with consistent split names.
    """
    folder = experiment_root('imagenette_imagewoof') / 'dataset'
    data = np.load(folder / f'{name}_codexgen.npz')
    rows = []
    with zipfile.ZipFile(folder / f'{name}_originals_codexgen.zip') as z:
        for i, source in enumerate(data['source_ids']):
            record = {'dataset': name, 'source': str(source),
                      'label': str(data['classes'][data['labels'][i]]),
                      'split': str(data['splits'][i])}
            if record['split'] == 'calibration':
                record['split'] = 'val'
            rows.append((record, z.read(source), None))
    save(name, rows)


def pets() -> None:
    """Prepare the fixed Oxford Pets subset and segmentation masks."""
    folder = ROOT / 'downloads'
    rows = []
    with tarfile.open(folder / 'pets_annotations.tar.gz') as ann:
        chosen = []
        for split in ['trainval', 'test']:
            lines = ann.extractfile(f'annotations/{split}.txt').read()
            groups = {}
            for line in sorted(lines.decode().splitlines()):
                name, label, species, breed = line.split()
                group = groups.setdefault(label, [])
                if len(group) < (4 if split == 'trainval' else 2):
                    part = ('train' if len(group) < 3 else 'val')
                    group.append((name, part if split == 'trainval'
                                  else 'test', species))
            chosen.extend(r for group in groups.values() for r in group)
        masks = {name: ann.extractfile(
            f'annotations/trimaps/{name}.png').read()
                 for name, _, _ in chosen}
    wanted = {f'images/{name}.jpg' for name, _, _ in chosen}
    with tarfile.open(folder / 'pets_images.tar.gz') as archive:
        raw = {m.name: archive.extractfile(m).read() for m in archive
               if m.name in wanted}
    for name, split, species in chosen:
        record = {'dataset': 'pets', 'source': name, 'split': split,
                  'label': name.rsplit('_', 1)[0], 'species': species}
        # 1 foreground, 2 background, 3 boundary in the source trimap.
        rows.append((record, raw[f'images/{name}.jpg'],
                     Image.open(io.BytesIO(masks[name]))))
    save('pets', rows)


def dtd() -> None:
    """Prepare the fixed texture dataset subset."""
    rows = []
    with tarfile.open(ROOT / 'downloads/dtd-r1.0.1.tar.gz') as archive:
        chosen = []
        for split in ['train', 'val', 'test']:
            text = archive.extractfile(f'dtd/labels/{split}1.txt').read()
            groups = {}
            for name in sorted(text.decode().splitlines()):
                group = groups.setdefault(name.split('/')[0], [])
                if len(group) < 2:
                    group.append((name, split))
            chosen.extend(r for group in groups.values() for r in group)
        wanted = {'dtd/images/' + name for name, _ in chosen}
        raw = {m.name: archive.extractfile(m).read() for m in archive
               if m.name in wanted}
    for name, split in chosen:
        record = {'dataset': 'dtd', 'source': name, 'split': split,
                  'label': name.split('/')[0]}
        rows.append((record, raw['dtd/images/' + name], None))
    save('dtd', rows)


def parts() -> None:
    """Prepare the selected PartImageNet images and part masks."""
    rows = []
    target = ROOT / 'downloads/partimagenet_subset_codexgen.zip'
    with zipfile.ZipFile(target) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.JPEG'):
                continue
            split = name.split('/')[-2]
            mask_name = name.replace('/images/', '/annotations/')
            mask_name = mask_name.rsplit('.', 1)[0] + '.png'
            mask = Image.open(io.BytesIO(archive.read(mask_name)))
            item = {'dataset': 'parts', 'source': name, 'split': split,
                    'label': name.split('/')[-1].split('_')[0]}
            rows.append((item, archive.read(name), mask))
    save('parts', rows)


def shapes() -> None:
    """Generate controlled synthetic images and segmentation masks."""
    rows = []
    for shape in ['line', 'arc', 'corner', 'circle']:
        for texture in ['plain', 'striped']:
            for angle in [0, 45, 90, 135]:
                image = Image.new('RGB', (256, 256), (128, 128, 128))
                draw = ImageDraw.Draw(image)
                color = (235, 235, 235)
                if shape == 'line':
                    draw.line((48, 128, 208, 128), fill=color, width=18)
                elif shape == 'arc':
                    draw.arc((48, 48, 208, 208), 0, 180, fill=color, width=18)
                elif shape == 'circle':
                    draw.ellipse((48, 48, 208, 208), outline=color, width=18)
                else:
                    draw.line((48, 128, 128, 128, 128, 48),
                              fill=color, width=18)
                if texture == 'striped':
                    pixels = np.array(image)
                    stripe = np.indices((256, 256))[1] % 12 < 6
                    pixels[(pixels[:, :, 0] > 200) & stripe] = 60
                    image = Image.fromarray(pixels)
                image = image.rotate(angle)
                stream = io.BytesIO()
                image.save(stream, format='PNG')
                source = f'{shape}_{texture}_{angle}'
                rows.append(({'dataset': 'shapes', 'source': source,
                              'label': shape, 'texture': texture,
                              'angle': angle, 'split': 'test'},
                             stream.getvalue(), None))
    save('shapes', rows)


def load(name: str) -> tuple[NpzFile, list[dict[str, Any]]]:
    """Open the cached dataset archive and load its metadata records."""
    folder = ROOT / 'dataset'
    data = np.load(folder / f'{name}_codexgen.npz')
    records = json.loads((folder / f'{name}_codexgen.json').read_text())
    return data, records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=['imagenette', 'imagewoof', 'pets',
                                        'dtd', 'parts', 'shapes'])
    name = parser.parse_args().name
    if name in ['imagenette', 'imagewoof']:
        original(name)
    else:
        globals()[name]()
