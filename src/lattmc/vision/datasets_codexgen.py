"""Prepare traceable Imagenette and Imagewoof subsets from fastai archives."""

import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tf

from lattmc.vision.paths_codexgen import experiment_root


NAMES = {
    'imagenette': ['tench', 'English springer', 'cassette player',
                  'chain saw', 'church', 'French horn', 'garbage truck',
                  'gas pump', 'golf ball', 'parachute'],
    'imagewoof': ['Shih-Tzu', 'Rhodesian ridgeback', 'Beagle',
                 'English foxhound', 'Border terrier', 'Australian terrier',
                 'Golden retriever', 'Old English sheepdog', 'Samoyed',
                 'Dingo'],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(name):
    root = experiment_root('imagenette_imagewoof')
    folder = root / 'dataset'
    folder.mkdir(parents=True, exist_ok=True)
    url = f'https://s3.amazonaws.com/fast-ai-imageclas/{name}2-320.tgz'
    archive = Path('/private/tmp') / f'{name}2-320.tgz'
    if not archive.exists():
        print('Downloading', url, flush=True)
        urllib.request.urlretrieve(url, str(archive) + '.partial')
        Path(str(archive) + '.partial').replace(archive)
    records, images, labels, splits = [], [], [], []
    original = folder / f'{name}_originals_codexgen.zip'
    with tarfile.open(archive) as tar, zipfile.ZipFile(original, 'w') as out:
        members = [m for m in tar.getmembers() if m.isfile()
                   and m.name.lower().endswith(('.jpeg', '.jpg'))
                   and not Path(m.name).name.startswith('._')]
        classes = sorted({Path(m.name).parent.name for m in members})
        assert len(classes) == 10
        selected = []
        splits_to_read = ['train', 'val'] if name == 'imagenette' else ['val']
        for split in splits_to_read:
            for synset in classes:
                group = sorted([m for m in members
                                if Path(m.name).parent.name == synset
                                and Path(m.name).parts[-3] == split],
                               key=lambda m: m.name)
                selected.extend(group[:25 if split == 'train' else 10])
        # Read gzip members in physical order to avoid repeated rewinds.
        raw_by_name = {m.name: tar.extractfile(m).read()
                       for m in sorted(selected, key=lambda m: m.offset)}
        for split in (['train', 'val'] if name == 'imagenette' else ['val']):
            for label, synset in enumerate(classes):
                candidates = sorted([m for m in members
                                     if Path(m.name).parent.name == synset
                                     and Path(m.name).parts[-3] == split],
                                    key=lambda m: m.name)
                for rank, member in enumerate(candidates[:25 if split
                                                         == 'train' else 10]):
                    raw = raw_by_name[member.name]
                    out.writestr(member.name, raw)
                    image = Image.open(io.BytesIO(raw)).convert('RGB')
                    image = tf.resize(image, 256,
                                      InterpolationMode.BICUBIC)
                    image = tf.center_crop(image, [224, 224])
                    images.append(np.asarray(image))
                    labels.append(label)
                    partition = ('train' if rank < 20 else 'calibration')
                    splits.append(partition if split == 'train' else 'test')
                    records.append({'archive_member': member.name,
                                    'original_sha256':
                                    hashlib.sha256(raw).hexdigest()})
    pixels = np.stack(images)
    split_array = np.array(splits)
    path = folder / f'{name}_codexgen.npz'
    np.savez_compressed(path, images=pixels, labels=labels, splits=split_array,
                        classes=np.array(NAMES[name]), synsets=classes,
                        source_ids=[r['archive_member'] for r in records])
    metadata = {'url': url, 'archive_sha256': sha256(archive),
                'selection': 'lexical member order, per class and split',
                'preprocessing': 'RGB; bicubic short edge 256; center 224',
                'classes_by_synset_order': dict(zip(classes, NAMES[name])),
                'counts': {s: int((split_array == s).sum())
                           for s in np.unique(split_array)},
                'records': records,
                'artifact_sha256': {path.name: sha256(path),
                                    original.name: sha256(original)}}
    result = root / 'results'
    result.mkdir(exist_ok=True)
    (result / f'{name}_codexgen.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    print(name, metadata['counts'], flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=NAMES)
    prepare(parser.parse_args().dataset)
