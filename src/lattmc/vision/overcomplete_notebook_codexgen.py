"""Build and execute a lightweight notebook from the verified experiment."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[3]
    cells = [nbf.v4.new_markdown_cell(
        '# Sparse visual features across datasets\n\n'
        'Native Overcomplete models and pretrained RA-SAE / Prisma '
        'transcoder.\n'
        'All fitting and extraction live in `src/lattmc/vision`.\n'
        'Classes and patch positions do not constrain shared features.\n'
        'Exploratory annotation-guided images are separate from evaluation.'),
        nbf.v4.new_code_cell(
            'import json\nimport sys\nfrom pathlib import Path\n'
            'from IPython.display import Image, display\n'
            'import pandas as pd\n\n'
            'root = Path.cwd()\n'
            'sys.path.insert(0, str(root / "src"))\n'
            'folder = root / "vision_tokens/overcomplete"\n'
            'summary = json.loads(\n'
            '    (folder / "results/summary_codexgen.json").read_text())\n'
            'verification = json.loads(\n'
            '    (folder / "results/verification_codexgen.json")\n'
            '    .read_text())\n'
            'verification'),
        nbf.v4.new_markdown_cell(
            '## Reconstruction and measured sparsity\n\n'
            'Pretrained and local fits differ in width and training data.\n'
            'The transcoder predicts a different target from the SAEs.\n'
            'Bootstrap intervals resample images in these fixed subsets.'),
        nbf.v4.new_code_cell('pd.DataFrame(summary["transfer"])'),
        nbf.v4.new_markdown_cell(
            '## Stability\n\n'
            'Decoder cosine matching and image-extent overlap are distinct.\n'
            'Archetypal fits share landmarks and mixing initialization.'),
        nbf.v4.new_code_cell('pd.DataFrame(summary["stability"])')]
    figures = [
        ('Training-selected TopK features',
         'topk_k32_s0_gallery_codexgen.png'),
        ('Exploratory foreground-associated pretrained features; selection\n'
         'uses test annotations and is not an unbiased evaluation',
         'pretrained_ra_foreground_gallery_codexgen.png'),
        ('Training-selected transcoder features',
         'prisma_transcoder_gallery_codexgen.png'),
        ('Frozen meet/join queries; patch locations may differ',
         'meet_join_codexgen.png'),
        ('Fixed-contrast gratings and frequency-matched control coordinates',
         'frequency_probe_codexgen.png')]
    for title, filename in figures:
        cells.append(nbf.v4.new_markdown_cell('## ' + title))
        cells.append(nbf.v4.new_code_cell(
            'path = (folder / "figures" /\n'
            f'        "{filename}")\n'
            'display(Image(filename=str(path)))'))
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(display_name='Python 3',
                                       language='python', name='python3')
    NotebookClient(notebook, timeout=180,
                   resources={'metadata': {'path': str(root)}}).execute()
    target = root / 'notebooks/vision/overcomplete_features_codexgen.ipynb'
    nbf.write(notebook, target)
    print(target)


if __name__ == '__main__':
    main()
