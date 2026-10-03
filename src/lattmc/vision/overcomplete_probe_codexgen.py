"""Fixed-contrast frequency probes for selected and control coordinates."""

from __future__ import annotations

import io
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lattmc.vision.overcomplete_data_codexgen import load, save
from lattmc.vision.overcomplete_evaluate_codexgen import pool, read_codes
from lattmc.vision.overcomplete_fetch_codexgen import ROOT
from lattmc.vision.overcomplete_figures_codexgen import foreground_features


def stimuli() -> None:
    """Generate controlled shape stimuli and cache their feature responses."""
    rows = []
    y, x = np.indices((256, 256)) / 224
    for frequency in [2, 4, 8, 16, 32, 56]:
        for angle in [0, 45, 90, 135]:
            for phase in [0, np.pi / 2]:
                theta = np.deg2rad(angle)
                axis = x * np.cos(theta) + y * np.sin(theta)
                pixels = .5 + .3 * np.cos(2 * np.pi * frequency * axis + phase)
                rgb = np.repeat((pixels * 255).astype('uint8')[:, :, None],
                                3, axis=2)
                stream = io.BytesIO()
                Image.fromarray(rgb).save(stream, format='PNG')
                record = {'dataset': 'gratings', 'source':
                          f'{frequency}_{angle}_{phase:.3f}',
                          'label': 'grating', 'split': 'test',
                          'frequency': frequency, 'angle': angle,
                          'phase': float(phase)}
                rows.append((record, stream.getvalue(), None))
    save('gratings', rows)


def plot() -> None:
    """Plot the controlled stimulus responses for selected sparse features."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    result = []
    for ax, name in zip(axes, ['pretrained_ra', 'prisma_transcoder']):
        definition = json.loads(
            (ROOT / f'results/{name}/queries_codexgen.json').read_text())
        features = (foreground_features(name) if name == 'pretrained_ra'
                    else definition['selected_features'][:3])
        training, _ = read_codes(name, 'imagenette')
        _, records = load('imagenette')
        sites = training.shape[0] // len(records)
        tr = np.array([r['split'] == 'train' for r in records])
        maxima = pool(training, sites)[tr]
        frequency = (maxima > 0).mean(0)
        rng = np.random.default_rng(319)
        controls = []
        for feature in features:
            distance = np.abs(frequency - frequency[feature])
            distance[features + controls] = np.inf
            closest = np.argsort(distance, kind='stable')[:20]
            controls.append(int(rng.choice(closest)))
        matrix, _ = read_codes(name, 'gratings')
        _, records = load('gratings')
        values = pool(matrix, sites)
        frequencies = np.array([r['frequency'] for r in records])
        for j, feature in enumerate(features + controls):
            scale = max(float(np.quantile(maxima[:, feature], .95)), 1e-8)
            response = values[:, feature] / scale
            average = [float(response[frequencies == f].mean())
                       for f in sorted(set(frequencies))]
            ax.plot(sorted(set(frequencies)), average,
                    '-' if j < 3 else '--', marker='o', markersize=3,
                    label=f'{feature}' + (' control' if j >= 3 else ''))
            result.append({'model': name, 'feature': feature,
                           'control': j >= 3, 'training_q95': scale,
                           'response': response.tolist()})
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Cycles per 224-pixel crop')
        ax.set_ylabel('Max code / training image-max 95th percentile',
                      fontsize=9)
        ax.set_title(name.replace('_', ' '))
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/frequency_probe_codexgen.png', dpi=180)
    plt.close(fig)
    (ROOT / 'results/frequency_probe_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    stimuli()
