"""Fetch a small public Oxford Pet mirror subset and official trimaps."""

import io
import json
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from lattmc.vision.overcomplete_data_codexgen import save
from lattmc.vision.overcomplete_fetch_codexgen import ROOT, digest


def get(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def main():
    folder = ROOT / 'downloads'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'pets_subset_codexgen.zip'
    records_path = folder / 'pets_subset_codexgen.json'
    if not target.exists():
        entries, labels = [], None
        for offset in [0, 100]:
            url = ('https://datasets-server.huggingface.co/rows?dataset='
                   'timm%2Foxford-iiit-pet&config=default&split=test'
                   f'&offset={offset}&length=100')
            document = json.loads(get(url))
            labels = document['features'][1]['type']['names']
            entries.extend(document['rows'])
        records = []

        def read(entry):
            row = entry['row']
            return row['image_id'], get(row['image']['src'])

        with zipfile.ZipFile(target.with_suffix('.partial'), 'w') as out:
            with ThreadPoolExecutor(max_workers=8) as pool:
                fetched = pool.map(read, entries)
                for entry, (name, raw) in zip(entries, fetched):
                    out.writestr(name + '.jpg', raw)
                    row = entry['row']
                    records.append({'dataset': 'pets', 'source': name,
                                    'label': labels[row['label']],
                                    'split': 'test',
                                    'mirror_row': entry['row_idx']})
        target.with_suffix('.partial').replace(target)
        records_path.write_text(json.dumps(records, indent=2) + '\n')
    annotations = folder / 'pets_annotations.tar.gz'
    if not annotations.exists():
        url = 'https://thor.robots.ox.ac.uk/pets/annotations.tar.gz'
        annotations.write_bytes(get(url))
        annotations.with_suffix('.json').write_text(json.dumps(
            {'url': url, 'sha256': digest(annotations)}, indent=2))
    records = json.loads(records_path.read_text())
    rows = []
    with tarfile.open(annotations) as ann, zipfile.ZipFile(target) as z:
        lookup = {m.name.rsplit('/', 1)[-1].lower(): m for m in ann
                  if m.name.endswith('.png') and '/trimaps/' in m.name}
        for record in records:
            name = record['source']
            mask = Image.open(io.BytesIO(ann.extractfile(
                lookup[(name + '.png').lower()]).read()))
            raw = z.read(name + '.jpg')
            image = Image.open(io.BytesIO(raw))
            if mask.size != image.size:
                ratio = image.width / image.height
                assert abs(mask.width / mask.height - ratio) < 0.02
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
            rows.append((record, raw, mask))
    save('pets', rows)
    (folder / 'pets_mirror_provenance_codexgen.json').write_text(json.dumps({
        'dataset': 'timm/oxford-iiit-pet',
        'revision': '089695c834a7deb60505b7cc506672db1c31a6aa',
        'selection': 'first 200 test rows of public viewer; no labels used',
        'images': 'public viewer JPEG bytes; masks resized if necessary',
        'sha256': digest(target)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
