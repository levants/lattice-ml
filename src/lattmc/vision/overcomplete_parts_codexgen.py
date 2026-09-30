"""Read a small PartImageNet subset using validated HTTP byte ranges."""

import io
import json
import struct
import zlib
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from lattmc.vision.overcomplete_fetch_codexgen import ROOT, digest


REV = 'cfa2edf3cf5ffbede2bd1fc934edca0aba99f042'
URL = ('https://huggingface.co/datasets/turkeyju/PartImageNet/resolve/'
       + REV + '/PartImageNet_Seg.zip')
SIZE = 3124435169


class RemoteZip(io.RawIOBase):
    """Seekable public object; never confuse a full reply with a range."""

    def __init__(self):
        self.position = 0

    def seek(self, offset, whence=0):
        base = [0, self.position, SIZE][whence]
        self.position = base + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        end = SIZE if count < 0 else min(SIZE, self.position + count)
        if end <= self.position:
            return b''
        start = self.position
        req = urllib.request.Request(
            URL + f'?range={start}-{end}',
            headers={'Range': f'bytes={start}-{end - 1}'})
        with urllib.request.urlopen(req, timeout=120) as response:
            assert response.status == 206, response.status
            assert response.headers['Content-Range'].startswith(
                f'bytes {start}-{end - 1}/')
            value = response.read()
        assert len(value) == end - start
        self.position = end
        return value


def main():
    target = ROOT / 'downloads/partimagenet_subset_codexgen.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print('Existing subset:', digest(target))
        return
    index = ROOT / 'downloads/partimagenet_index_codexgen.json'
    if not index.exists():
        with zipfile.ZipFile(RemoteZip()) as archive:
            members = [{'name': m.filename, 'offset': m.header_offset,
                        'compressed': m.compress_size, 'crc': m.CRC,
                        'method': m.compress_type}
                       for m in archive.infolist()
                       if not m.is_dir()
                       and not m.filename.startswith('__MACOSX')]
        index.write_text(json.dumps(members))
    members = json.loads(index.read_text())
    lookup = {m['name']: m for m in members}
    images = [m for m in members if m['name'].endswith('.JPEG')]
    synsets = sorted({m['name'].split('/')[-1].split('_')[0]
                      for m in images})
    # Systematically span the synset list; no activation-based selection.
    synsets = set(synsets[::3])
    selected, groups = {}, {}
    for item in sorted(images, key=lambda m: m['name']):
        name = item['name']
        split = name.split('/')[-2]
        synset = name.split('/')[-1].split('_')[0]
        if synset not in synsets:
            continue
        key = (split, synset)
        quota = 2 if split == 'train' else 1
        if groups.get(key, 0) >= quota:
            continue
        groups[key] = groups.get(key, 0) + 1
        mask = name.replace('/images/', '/annotations/')
        mask = mask.rsplit('.', 1)[0] + '.png'
        assert mask in lookup, mask
        selected[name] = item
        selected[mask] = lookup[mask]
    print('Selected images', len(selected) // 2, flush=True)

    def read_member(info):
        source = RemoteZip()
        source.seek(info['offset'])
        raw = source.read(info['compressed'] + 2048)
        header = struct.unpack('<4s5H3I2H', raw[:30])
        assert header[0] == b'PK\x03\x04'
        start = 30 + header[-2] + header[-1]
        value = raw[start:start + info['compressed']]
        if info['method'] == 8:
            value = zlib.decompress(value, -15)
        else:
            assert info['method'] == 0
        assert zlib.crc32(value) == info['crc']
        return info['name'], value

    with zipfile.ZipFile(target.with_suffix('.partial'), 'w') as out:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for count, (name, raw) in enumerate(
                    pool.map(read_member, selected.values())):
                out.writestr(name, raw)
                if count % 40 == 0:
                    print('PartImageNet files', count, flush=True)
    target.with_suffix('.partial').replace(target)
    receipt = {'url': URL, 'revision': REV, 'source_bytes': SIZE,
               'selection': 'every third synset; lexical train 2, val/test 1',
               'images': len(selected) // 2, 'sha256': digest(target)}
    target.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
