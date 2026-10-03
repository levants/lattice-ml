"""Measured patch galleries and exact-pixel controls, with optional PDF."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from lattmc.vision.patch_contexts_codexgen import read
from lattmc.vision.patch_models_codexgen import ROOT


CASES = [(1, 'Springer-associated'), (4, 'Church-associated'),
         (6, 'Truck-associated')]


def save(fig: Figure, name: str, kind: str, paper: Path | None) -> None:
    """Save a figure to the experiment and optional paper directory."""
    folder = ROOT / name / 'figures'
    folder.mkdir(exist_ok=True)
    stem = f'{name}_patch_{kind}_codexgen'
    fig.savefig(folder / f'{stem}.png', dpi=180, bbox_inches='tight')
    if paper:
        fig.savefig(Path(paper) / 'figures' / f'{stem}.pdf',
                    bbox_inches='tight')
    plt.close(fig)


def photo(
    ax: Axes,
    image: np.ndarray,
    site: int | None = None,
    grid: int = 7,
) -> None:
    """Display an image and optionally outline a spatial grid cell."""
    ax.imshow(image)
    ax.set_axis_off()
    if site is not None:
        y, x = divmod(int(site), grid)
        pixel = 224 // grid
        ax.add_patch(Rectangle((x * pixel - .5, y * pixel - .5),
                               pixel, pixel, fill=False,
                               edgecolor='#00ffff', linewidth=1.6))


def render(name: str, paper: Path | None = None) -> None:
    """Render feature exemplars and query retrieval panels."""
    sample, codes, features, _ = read(name)
    root = ROOT / name
    test = np.flatnonzero(sample['splits'] == 'test')
    grid = int(np.sqrt(codes.shape[1]))
    fig, axes = plt.subplots(3, 6, figsize=(9.5, 6.3), layout='constrained')
    for row, (target, label) in enumerate(CASES):
        with np.load(root / f'contexts/class_{target:02d}_codexgen.npz') as c:
            c = dict(c)
        q = c['queries'][3]
        strength = c['scores'][3, test].max(1)
        ranked = test[np.argsort(-strength, kind='stable')[:2]]
        at = int(ranked[0])
        site = int(c['scores'][3, at].argmax())
        pair = c['pair']
        image = sample['images'][at]
        photo(axes[row, 0], image, site, grid)
        axes[row, 0].set_title(f'{sample["classes"][sample["labels"][at]]}\n'
                              f'row {sample["rows"][at]}', fontsize=9)
        for col, j in enumerate(pair, 1):
            ratio = codes[at, :, j].reshape(grid, grid) / q[j]
            axes[row, col].imshow(ratio, vmin=0, vmax=2, cmap='magma',
                                  interpolation='nearest')
            axes[row, col].set_title(
                f'Feature {features[j]}\nratio to threshold',
                                     fontsize=9)
            axes[row, col].set_axis_off()
        for col, operation in [(3, 2), (4, 3)]:
            ax = axes[row, col]
            mask = c['patch_extents'][operation, at].reshape(grid, grid)
            ax.imshow(image, alpha=.35)
            overlay = np.zeros((grid, grid, 4))
            overlay[mask] = [0, .55, .3, .75]
            ax.imshow(overlay, extent=(-.5, 223.5, 223.5, -.5),
                      interpolation='nearest')
            ax.set_axis_off()
            count = int(c['common_images'][operation, test].sum())
            op = 'Meet' if operation == 2 else 'Join'
            ax.set_title(f'{op}: common sites\n{count}/100 images', fontsize=9)
        second = int(ranked[1])
        second_site = int(c['scores'][3, second].argmax())
        photo(axes[row, 5], sample['images'][second], second_site, grid)
        axes[row, 5].set_title(
            f'{sample["classes"][sample["labels"][second]]}\n'
                              f'row {sample["rows"][second]}', fontsize=9)
    bar = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(0, 2), cmap='magma'), ax=axes[:, 1:3],
        orientation='horizontal', fraction=.025, pad=.01, aspect=35)
    bar.set_label('Activation / join threshold (display clipped at 2)',
                  fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.suptitle(f'{name.upper()}: shared coordinates '
                 'and common patch witnesses',
                 fontsize=12)
    save(fig, name, 'gallery', paper)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.7), layout='constrained')
    for row, (target, label) in enumerate(CASES):
        path = root / f'controls/context_{target:02d}_codexgen.npz'
        with np.load(path) as c:
            for col, condition in enumerate(['Original', '3 x 3 window',
                                              'Isolated', 'Relocated']):
                photo(axes[row, col], c['images'][col], c['sites'][col], grid)
                axes[row, col].set_title(
                    f'{condition}: score {c["ratios"][col]:.3f}', fontsize=10)
            axes[row, 0].text(-.05, .5, label, va='center', ha='right',
                              rotation=90, fontsize=10,
                              transform=axes[row, 0].transAxes)
    fig.suptitle(f'{name.upper()}: identical marked patch pixels; '
                 'changed context / position', fontsize=12)
    save(fig, name, 'controls', paper)


def downsets(paper: Path | None = None) -> None:
    """Draw the finite downset example used in the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout='constrained')
    for row, name in enumerate(['prisma', 'saev']):
        sample, codes, features, _ = read(name)
        for col, (target, label) in enumerate(CASES[1:]):
            with np.load(ROOT / name /
                         f'contexts/class_{target:02d}_codexgen.npz') as c:
                ax = axes[row, col]
                for k, source in enumerate(c['sources']):
                    points = codes[source][:, c['pair']]
                    ax.scatter(*points.T, s=12, alpha=.45,
                               label=f'Source {k + 1}')
                g = c['generators']
                points = np.vstack([[0, g[0, 1]], g,
                                     [g[-1, 0], 0]])
                points = points[np.argsort(points[:, 0], kind='stable')]
                # Rectangles are exact principal downsets; overlaps form
                # the intersection description, without interpolation.
                for x, y in g:
                    ax.add_patch(Rectangle((0, 0), x, y, color='green',
                                           alpha=.10, linewidth=0))
                ax.scatter(*g.T, marker='x', color='black', s=40,
                           label='Maximal intersection generators')
                for k, marker in enumerate(['<', '>', 'v', '^']):
                    q = c['queries'][k, c['pair']]
                    ax.scatter(*q, marker=marker, s=45,
                               label=['u', 'v', 'meet', 'join'][k])
                ax.set_title(f'{name.upper()} / {label}\n'
                              f'{len(g)} maximal generators', fontsize=10)
                ax.set_xlabel(f'Feature {features[c["pair"][0]]}')
                ax.set_ylabel(f'Feature {features[c["pair"][1]]}')
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4,
               fontsize=8)
    save(fig, 'prisma', 'downsets', paper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path)
    args = parser.parse_args()
    for model in ['prisma', 'saev']:
        render(model, args.paper)
    downsets(args.paper)
