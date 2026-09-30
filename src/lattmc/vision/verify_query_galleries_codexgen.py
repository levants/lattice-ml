"""Independently check all Imagenette gallery queries against cached sites."""

import json

import numpy as np

from lattmc.vision.backbones_codexgen import dataset, load_codes
from lattmc.vision.paths_codexgen import experiment_root


def verify():
    sample = dataset()
    train = np.flatnonzero(sample['splits'] == 'train')
    test = np.flatnonzero(sample['splits'] == 'test')
    checked = 0
    summary = {}
    for name in ['resnet34', 'dinov2']:
        root = experiment_root('imagenette_' + name)
        codes = load_codes(name)['codes']
        pooled = np.max(codes, axis=1)
        records = json.loads(
            (root / 'results/query_galleries_codexgen.json').read_text())
        summary[name] = []
        for target, slug, record in zip([1, 6], ['springer', 'truck'],
                                        records, strict=True):
            path = root / f'retrieval/query_{slug}_codexgen.npz'
            with np.load(path) as data:
                labels = sample['labels'][train]
                values = pooled[train].astype(np.float64)
                contrast = (values[labels == target].mean(0)
                            - values[labels != target].mean(0))
                contrast /= np.maximum(values.std(0), 1e-12)
                features = np.argsort(-contrast, kind='stable')[:2]
                np.testing.assert_array_equal(data['features'], features)
                selected = []
                for feature in features:
                    candidates = train[labels == target]
                    order = sorted(candidates,
                                   key=lambda i: (-pooled[i, feature], i))
                    selected.append(next(i for i in order
                                         if i not in selected))
                np.testing.assert_array_equal(selected, data['sources'])
                assert record['source_ids'] == (
                    sample['source_ids'][selected].tolist())
                base = pooled[selected][:, features] / 2
                expected = [base[0], base[1], np.minimum(*base),
                            np.maximum(*base)]
                np.testing.assert_array_equal(data['vectors'], expected)
                np.testing.assert_array_equal(data['rows'], test)
                masks = []
                for k, query in enumerate(expected):
                    mask, site, scores = [], [], []
                    for row in test:
                        sites = codes[row][:, features]
                        maxima = sites.max(0)
                        mask.append(all(maxima >= query))
                        site.append(any(np.all(sites >= query, axis=1)))
                        active = query > 0
                        scores.append(min(maxima[active].astype(float)
                                          / query[active])
                                      if active.any() else 1.0)
                    mask = np.array(mask)
                    np.testing.assert_array_equal(data['pooled'][k], mask)
                    np.testing.assert_array_equal(data['same_site'][k], site)
                    np.testing.assert_allclose(data['scores'][k], scores,
                                               rtol=1e-12, atol=0)
                    ranked = sorted(range(len(test)),
                                    key=lambda i: (-scores[i], i))
                    display = [int(test[i]) for i in ranked if mask[i]][:3]
                    display += [-1] * (3 - len(display))
                    np.testing.assert_array_equal(data['display_rows'][k],
                                                   display)
                    op = record['operations'][k]
                    assert op['pooled'] == int(mask.sum())
                    assert op['same_site'] == int(np.sum(site))
                    assert op['thresholds'] == query.tolist()
                    assert op['display_rows'] == display
                    assert op['class_counts'] == np.bincount(
                        sample['labels'][test[mask]], minlength=10).tolist()
                    masks.append(mask)
                    checked += 1
                assert np.array_equal(masks[3], masks[0] & masks[1])
                assert np.all(~(masks[0] | masks[1]) | masks[2])
                extra = masks[2] & ~(masks[0] | masks[1])
                np.testing.assert_array_equal(data['meet_extra'], extra)
                assert int(extra.sum()) == record['meet_extra_beyond_union']
                summary[name].append({
                    'case': slug, 'pooled': data['pooled'].sum(1).tolist(),
                    'same_site': data['same_site'].sum(1).tolist(),
                    'meet_extra_rows': test[extra].tolist()})
    return {'queries_checked': checked, 'results': summary}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
