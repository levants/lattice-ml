"""Evaluate lattice operations on fixed labeled activation caches.

This offline analysis preserves source amplitudes and existing corpus
splits. It records exact extents, category confusion counts, constituent
redundancy, and training-fitted description transport. Dataset labels are
category proxies, not independent annotations of feature semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from lattmc.contextstudy.operations_codexgen import (
    GRID, Mask, Record, Vector, digest, encoded, extent, intent, metrics,
    project, relation, save,
)

CONDITIONS = ('smol3', 'smol15', 'smol27', 'pythia_topk',
              'qwen_transcoder', 'gemma_matryoshka')
SEED = 20261003


def load_condition(root: Path, dataset: str,
                   condition: str) -> tuple[sparse.csr_matrix, Record]:
    """Read a matrix and verify its recorded extraction checksum."""
    base = root / 'data/activation_studies'
    if condition in ('smol3', 'smol27'):
        folder, stem = base / 'contextdepth_v1' / dataset, condition
        path = folder / f'{stem}_max.npz'
        record_path = folder / f'{stem}_extraction.json'
        record = json.loads(record_path.read_text())
        expected = record['files'][path.name]
    else:
        folder = base / 'families_v2' / dataset
        stem = 'smol_topk' if condition == 'smol15' else condition
        path = folder / f'{stem}_max.npz'
        record_path = folder / f'{stem}_extraction.json'
        record = json.loads(record_path.read_text())
        expected = record['files']['max']['sha256']
    actual = digest(path)
    assert actual == expected, path
    matrix = sparse.load_npz(path).tocsr()
    matrix.eliminate_zeros()
    assert np.isfinite(matrix.data).all() and (matrix.data >= 0).all()
    return matrix, dict(path=str(path.relative_to(root)), sha256=actual,
                        extraction=record, record_sha256=digest(record_path))


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for a conservative duplicate audit."""
    return ' '.join(re.findall(r'\w+', text.casefold()))


def design_for(root: Path, dataset: str) -> Record:
    """Freeze source groups and exclude detected training/test duplicates.

    Existing test IDs are retained except for recorded duplicate exclusions.
    Character similarity is a detector, not a proof of semantic independence.
    """
    base = root / 'data/activation_studies/families_v2' / dataset
    design = json.loads((base / 'design.json').read_text())
    texts = json.loads((base / 'texts.local.json').read_text())
    rows = design['rows']
    labels = np.array([r['label'] for r in rows])
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                                 min_df=2, max_features=60000)
    fitted = vectorizer.fit_transform([texts[i] for i in train])
    test_matrix = vectorizer.transform([texts[i] for i in test])
    nearest = (test_matrix @ fitted.T).max(axis=1).toarray().ravel()
    normalized = [normalize_text(t) for t in texts]
    train_texts = {normalized[i] for i in train}
    titles = [normalize_text(t.split('.  ', 1)[0]) for t in texts]
    train_titles = {titles[i] for i in train}
    excluded = []
    for offset, i in enumerate(test):
        reasons = []
        if normalized[i] in train_texts:
            reasons.append('normalized_duplicate')
        if dataset == 'dbpedia_14' and titles[i] in train_titles:
            reasons.append('repeated_title')
        if nearest[offset] >= .90:
            reasons.append('character_cosine_ge_0.90')
        if reasons:
            excluded.append(dict(row=int(i), reasons=reasons,
                                 nearest_similarity=float(nearest[offset])))
    excluded_ids = {r['row'] for r in excluded}
    test = np.array([i for i in test if i not in excluded_ids])
    classes = (0, 1, 2, 3) if dataset == 'ag_news' else (7, 8, 9, 10)
    rng = np.random.default_rng(SEED)
    groups = []
    for label in classes:
        candidates = train[labels[train] == label]
        # Existing preprocessing deduplicates text; audit selected titles too.
        selected = []
        seen = set()
        for i in rng.permutation(candidates):
            key = titles[i] if dataset == 'dbpedia_14' else normalized[i]
            if key not in seen:
                seen.add(key)
                selected.append(int(i))
            if len(selected) == 15:
                break
        assert len(selected) == 15
        for group in range(5):
            ids = selected[group * 3:group * 3 + 3]
            words = set()
            if dataset == 'dbpedia_14':
                for i in ids:
                    words.update(w for w in titles[i].split() if len(w) >= 4)
            clean = [int(i) for i in test if not words.intersection(
                normalized[i].split())]
            groups.append(dict(label=label, group=group, sources=ids,
                               name_words=sorted(words), lexical_clean=clean))
    return dict(dataset=dataset, groups=groups, labels=labels.tolist(),
                train=train.tolist(), test=test.tolist(), excluded=excluded,
                rows=rows, design_sha256=digest(base / 'design.json'),
                texts_sha256=digest(base / 'texts.local.json'),
                source=design['source'], pinned=design['pinned'])


def evaluate_masks(masks: dict[str, Mask], group: Record,
                   design: Record) -> Record:
    """Measure exact masks against held-out category and broad-union labels."""
    labels = np.array(design['labels'])
    truth = labels == group['label']
    test = np.array(design['test'], dtype=int)
    clean = np.array(group['lexical_clean'], dtype=int)
    result = {}
    broad = np.isin(labels, [7, 8] if group['label'] in (7, 8) else [9, 10])
    for name, mask in masks.items():
        result[name] = dict(test=metrics(mask[test], truth[test]),
                            corpus_count=int(mask.sum()))
        if design['dataset'] == 'dbpedia_14':
            result[name]['name_excluded'] = metrics(mask[clean], truth[clean])
            result[name]['broad_exploratory'] = metrics(
                mask[test], broad[test])
    return result


def match_controls(matrix: sparse.csr_matrix, design: Record) -> Vector:
    """Standardize training-only L0, norm and feature-frequency covariates."""
    train = np.array(design['train'])
    frequency = np.asarray((matrix[train] > 0).mean(axis=0)).ravel()
    l0 = np.diff(matrix.indptr)
    norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
    weighted = np.asarray((matrix > 0) @ frequency).ravel()
    weighted /= np.maximum(l0, 1)
    raw = np.log1p(np.column_stack((l0, norm, weighted)))
    scale = raw[train].std(axis=0)
    return (raw - raw[train].mean(axis=0)) / np.maximum(scale, 1e-12)


def evaluate_condition(matrix: sparse.csr_matrix, design: Record,
                       condition: str) -> list[Record]:
    """Execute frozen full and selected-component queries for every group."""
    csc = matrix.tocsc()
    covariates = match_controls(matrix, design)
    labels = np.array(design['labels'])
    train = np.array(design['train'])
    test = np.array(design['test'])
    top = matrix.max(axis=0).toarray().ravel()
    records = []
    for group in design['groups']:
        ids = group['sources']
        vectors = matrix[ids].toarray()
        u, v, w = vectors
        masks = {f'source{i + 1}': extent(csc, z)
                 for i, z in enumerate(vectors)}
        masks['pair_meet'] = extent(csc, np.minimum(u, v))
        masks['pair_join'] = extent(csc, np.maximum(u, v))
        masks['triple_meet'] = extent(csc, vectors.min(axis=0))
        masks['triple_join'] = extent(csc, vectors.max(axis=0))
        masks['extension'] = (masks['pair_meet'] &
                              ~(masks['source1'] | masks['source2']))
        assert np.array_equal(masks['pair_join'],
                              masks['source1'] & masks['source2'])
        assert np.all(~(masks['source1'] | masks['source2']) |
                      masks['pair_meet'])
        assert np.array_equal(masks['triple_join'],
                              masks['pair_join'] & masks['source3'])
        closed = intent(matrix, masks['pair_meet'], top)
        assert np.array_equal(extent(csc, closed), masks['pair_meet'])
        unrelated = train[labels[train] != group['label']]
        distances = ((covariates[unrelated] - covariates[ids[1]]) ** 2).sum(1)
        control = int(unrelated[np.lexsort((unrelated, distances))[0]])
        controls = []
        for pair_type, right in [('related', v),
                                 ('unrelated', matrix[control].toarray()[0])]:
            for lr, rr in GRID:
                left_q, right_q = project(u, lr), project(right, rr)
                row = dict(pair_type=pair_type, left_ranks=list(lr),
                           right_ranks=list(rr))
                if left_q is None or right_q is None:
                    controls.append(dict(**row, skipped='insufficient_rank'))
                    continue
                q = np.maximum(left_q, right_q)
                left, right_m = extent(csc, left_q), extent(csc, right_q)
                joined = extent(csc, q)
                assert np.array_equal(joined, left & right_m)
                common = np.intersect1d(np.flatnonzero(left_q),
                                        np.flatnonzero(right_q))
                row.update(left=encoded(left_q), right=encoded(right_q),
                           common_coordinates=common.tolist(),
                           relation=relation(left[test], right_m[test]),
                           results=evaluate_masks(
                               dict(left=left, right=right_m, join=joined),
                               group, design),
                           test_members=np.flatnonzero(joined & np.isin(
                               np.arange(len(labels)), test)).tolist())
                controls.append(row)
        record = dict(condition=condition, **group,
                      support_counts=dict(pair_meet=int(np.sum(
                          np.minimum(u, v) > 0)), triple_meet=int(np.sum(
                          vectors.min(axis=0) > 0))),
                      full=evaluate_masks(masks, group, design),
                      full_members={k: np.flatnonzero(m[test]).tolist()
                                    for k, m in masks.items()},
                      member_indexing='offsets in design.test for full_masks',
                      control=control, control_label=int(labels[control]),
                      control_distance=float(np.sqrt(distances.min())),
                      selected=controls, identity_checks=True,
                      closure_check=True)
        records.append(record)
    return records


def transports(matrices: dict[str, sparse.csr_matrix],
               design: Record) -> list[Record]:
    """Fit description transport on training rows and test it out of sample.

    Inclusion is a theorem on the fitting universe. It need not hold on
    held-out items, whose failures are therefore explicitly counted.
    """
    train = np.array(design['train'])
    test = np.array(design['test'])
    records = []
    for start, stop in ((3, 15), (15, 27), (3, 27)):
        left = matrices[f'smol{start}']
        right = matrices[f'smol{stop}']
        lc, rc = left.tocsc(), right.tocsc()
        target_train = right[train]
        top = right.max(axis=0).toarray().ravel()
        for group in design['groups']:
            u, v = left[group['sources'][:2]].toarray()
            queries = dict(pair_meet=np.minimum(u, v), selected_join=
                           np.maximum(project(u, (1,)), project(v, (1,))))
            for name, query in queries.items():
                before = extent(lc, query)
                transported = intent(target_train, before[train], top)
                after = extent(rc, transported)
                assert np.all(~before[train] | after[train])
                newly = after & ~before
                lost = before & ~after
                records.append(dict(
                    start=start, stop=stop, query_kind=name,
                    label=group['label'], group=group['group'],
                    source_extent=int(before[train].sum()),
                    source_query=encoded(query),
                    transported_query=encoded(transported),
                    training_members=train[before[train]].tolist(),
                    empty_reference=bool(not before[train].any()),
                    zero_transport=bool(not transported.any()),
                    positive_coordinates=int((transported > 0).sum()),
                    training_inclusion=True,
                    heldout_lost=int(lost[test].sum()),
                    heldout_relation=relation(before[test], after[test]),
                    results=evaluate_masks(dict(before=before, after=after,
                                                extension=newly),
                                           group, design)))
    return records


def main() -> None:
    """Read verified caches and write complete JSON records and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    for dataset in ('ag_news', 'dbpedia_14'):
        design = design_for(args.root, dataset)
        save(args.out / f'{dataset}_design.json', design)
        matrices, sources = {}, {}
        for condition in CONDITIONS:
            matrix, source = load_condition(args.root, dataset, condition)
            sources[condition] = source
            if condition.startswith('smol'):
                matrices[condition] = matrix
            results = evaluate_condition(matrix, design, condition)
            save(args.out / f'{dataset}_{condition}.json', dict(
                dataset=dataset, condition=condition, records=results,
                provenance=source, protocol_sha256=digest(args.protocol),
                source_sha256=digest(Path(__file__)), tolerance=0))
            print(dataset, condition, len(results), 'groups done', flush=True)
        save(args.out / f'{dataset}_transport.json', dict(
            records=transports(matrices, design),
            protocol_sha256=digest(args.protocol),
            source_sha256=digest(Path(__file__))))
        print(dataset, 'transport completed', flush=True)


if __name__ == '__main__':
    main()
