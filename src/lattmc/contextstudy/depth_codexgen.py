"""Matched-layer contextual retrieval on the existing external corpora."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import torch
from transformers import AutoModelForCausalLM

from lattmc.activationstudy.common_codexgen import (
    BUDGETS, metrics, queries, save_json, sha256, threshold_for,
)
from lattmc.activationstudy.families_adapters_codexgen import (
    attach, load_smol,
)
from lattmc.activationstudy.families_evaluate_codexgen import cosine, interval

ROOT = Path(os.environ.get('MY_PAPERS_REPOSITORY', Path.cwd())).resolve()
OLD = ROOT / 'data/activation_studies/families_v2'
OUT = ROOT / 'data/activation_studies/contextdepth_v1'
PROTOCOL = ROOT / 'experiments/activation_studies/contextdepth_v1/PROTOCOL.md'
MODEL = 'HuggingFaceTB/SmolLM2-135M'
RELEASE = 'EleutherAI/sae-SmolLM2-135M-64x'


def extract(layer: int) -> None:
    """Extract and cache activations for a matched-depth checkpoint."""
    torch.set_num_threads(4)
    cfg = dict(layer=layer, hook='mlp_output')
    design = json.loads((OLD / 'ag_news/design.json').read_text())
    pinned = design['pinned']
    path = Path.home() / '.cache/huggingface/hub'
    path /= 'models--EleutherAI--sae-SmolLM2-135M-64x/snapshots'
    path /= pinned[RELEASE]['revision']
    sae = load_smol(path / f'layers.{layer}.mlp', 'mps')
    sae.cfg.metadata.hook_name = f'blocks.{layer}.hook_mlp_out'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=pinned[MODEL]['revision'], local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').to('mps').eval()
    capture = attach(model, cfg)
    started = time.time()
    for dataset in ('ag_news', 'dbpedia_14'):
        dest = OUT / dataset
        dest.mkdir(parents=True, exist_ok=True)
        token_path = OLD / dataset / 'smol_topk_tokens.local.npz'
        tokens = np.load(token_path)
        design_path = OLD / dataset / 'design.json'
        rows = json.loads(design_path.read_text())['rows']
        values, dense = [], []
        squared_error = squared_target = active = ntokens = 0
        with torch.inference_mode():
            for start in range(0, len(rows), 4):
                ids = torch.tensor(tokens['input_ids'][start:start + 4],
                                   device='mps')
                mask = torch.tensor(tokens['attention_mask'][start:start + 4],
                                    device='mps')
                x, target = capture(ids, mask)
                z = sae.encode(x)
                assert torch.isfinite(z).all() and (z >= 0).all()
                valid = mask.clone()
                valid[:, 0] = False
                pooled = z.masked_fill(~valid[:, :, None], 0).amax(1)
                values.append(sparse.csr_matrix(pooled.cpu().numpy()))
                mean = x.masked_fill(~valid[:, :, None], 0).sum(1)
                mean /= valid.sum(-1)[:, None]
                dense.append(mean.cpu().numpy())
                test = torch.tensor([r['split'] == 'test'
                    for r in rows[start:start + 4]], device='mps')
                selected = valid & test[:, None]
                if selected.any():
                    code, y = z[selected], target[selected]
                    rec = sae.decode(code)
                    squared_error += float((rec - y).square().sum())
                    squared_target += float(y.square().sum())
                    active += int((code > 0).sum())
                    ntokens += len(y)
                if start % 400 == 0:
                    print(layer, dataset, start, 'seconds',
                          round(time.time() - started), flush=True)
        name = f'smol{layer}'
        sparse.save_npz(dest / f'{name}_max.npz', sparse.vstack(values))
        np.savez_compressed(dest / f'{name}_dense.npz',
                            values=np.vstack(dense))
        save_json(dest / f'{name}_extraction.json', dict(
            layer=layer, model=MODEL, release=RELEASE,
            backbone_revision=pinned[MODEL]['revision'],
            sae_revision=pinned[RELEASE]['revision'],
            sae_config=sae.cfg.to_dict(),
            weights_sha256=sha256(
                path / f'layers.{layer}.mlp/sae.safetensors'),
            design_sha256=sha256(design_path),
            tokens_sha256=sha256(token_path),
            protocol_sha256=sha256(PROTOCOL), source_sha256=sha256(__file__),
            mean_token_l0=active / ntokens,
            uncentered_nmse=squared_error / squared_target,
            files={p.name: sha256(p) for p in (
                dest / f'{name}_max.npz', dest / f'{name}_dense.npz')},
        ))


def evaluate() -> None:
    """Evaluate retrieval conditions across the registered model depths."""
    for dataset in ('ag_news', 'dbpedia_14'):
        source, dest = OLD / dataset, OUT / dataset
        dest.mkdir(parents=True, exist_ok=True)
        design = json.loads((source / 'design.json').read_text())
        texts = json.loads((source / 'texts.local.json').read_text())
        labels = np.array([r['label'] for r in design['rows']])
        train, cal, test = [np.array([i for i, r in enumerate(design['rows'])
                           if r['split'] == split])
                           for split in ('train', 'calibration', 'test')]
        old = json.loads((source / 'smol_topk_max_results.json').read_text())
        tasks = [r for r in old['records'] if r['shot'] == 3]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            max_features=30000)
        vectorizer.fit([texts[i] for i in train])
        tfidf = vectorizer.transform(texts)
        for layer in (3, 15, 27):
            feature = (source / 'smol_topk_max.npz' if layer == 15 else
                       dest / f'smol{layer}_max.npz')
            dense_path = (source / 'smol_topk_dense.npz' if layer == 15 else
                          dest / f'smol{layer}_dense.npz')
            matrix = sparse.load_npz(feature).astype(np.float64).tocsr()
            csc = matrix.tocsc()
            norm = normalize(matrix)
            dense = normalize(np.load(dense_path)['values'])
            maximum = matrix[train].max(axis=0).toarray().ravel()
            frequency = np.asarray((matrix[train] > 0).sum(0)).ravel()
            records, arrays = [], {}
            for task in tasks:
                y = labels == task['label']
                positive = np.array(task['positive'])
                query = matrix[positive].toarray().min(0)
                candidates, ordered = queries(
                    csc, query, maximum, frequency, len(train))
                chosen = int(np.argmax([average_precision_score(
                    y[cal], s[cal]) for s in candidates]))
                binary = int(np.argmax([average_precision_score(
                    y[cal], s[cal] > 0) for s in candidates]))
                singles = csc[:, ordered[:64]].toarray()
                singles /= query[ordered[:64]]
                if len(ordered):
                    best = int(np.argmax([average_precision_score(y[cal], s)
                                          for s in singles[cal].T]))
                    best_scores = singles[:, best]
                else:
                    best_scores = candidates[0]
                scores = dict(
                    graded=candidates[chosen],
                    support=(candidates[binary] > 0).astype(float),
                    best_single=best_scores,
                    sae_cosine=cosine(norm, positive),
                    dense_cosine=cosine(dense, positive),
                    tfidf_cosine=cosine(tfidf, positive),
                )
                low = scores['tfidf_cosine'][test] <= np.median(
                    scores['tfidf_cosine'][test])
                record = dict(label=task['label'], repeat=task['repeat'],
                              positive=task['positive'], metrics={}, low={},
                              low_ids=test[low].tolist(), thresholds={})
                for method, score in scores.items():
                    threshold = threshold_for(
                        y[cal], score[cal],
                        graded=method in ('graded', 'best_single'),
                        binary=method == 'support')
                    record['thresholds'][method] = threshold
                    record['metrics'][method] = metrics(
                        y[test], score[test], threshold)
                    if y[test][low].any() and (~y[test][low]).any():
                        record['low'][method] = metrics(
                            y[test][low], score[test][low], threshold)
                    arrays.setdefault(method, []).append(score[test])
                records.append(record)
            score_path = dest / f'smol{layer}_scores.npz'
            np.savez_compressed(score_path, **{k: np.array(v)
                                for k, v in arrays.items()})
            if layer == 15:
                for a, b in zip(records, tasks):
                    for method in arrays:
                        assert np.isclose(a['metrics'][method]['ap'],
                                          b['metrics'][method]['ap'])
            save_json(dest / f'smol{layer}_results.json', dict(
                layer=layer, records=records, test_ids=test.tolist(),
                test_labels=labels[test].tolist(),
                source_sha256=sha256(__file__),
                protocol_sha256=sha256(PROTOCOL),
                feature_sha256=sha256(feature),
                scores_sha256=sha256(score_path),
            ))
            print(dataset, layer, 'evaluated', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', type=int, choices=(3, 27))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.layer is None:
            evaluate()
        else:
            extract(args.layer)
