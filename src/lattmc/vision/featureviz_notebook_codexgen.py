"""Create and execute the Imagenette/CNN/ViT evidence notebook."""

import os
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from lattmc.vision.paths_codexgen import repository_root


def create():
    root = repository_root()
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(cells=[
        markdown('# Imagenette, Imagewoof, and feature visualization\n\n'
                 'Frozen ResNet34 and DINOv2 ViT-S/14 with independent\n'
                 'Top-32 SAEs. Natural exemplars, optimized stimuli, and\n'
                 'controlled probes test feature hypotheses. These are\n'
                 'not reproductions of Distill circuits.'),
        code('from pathlib import Path\nimport sys\nimport json\n'
             'root = Path.cwd()\n'
             "while not (root / 'src/lattmc/vision').is_dir():\n"
             '    if root == root.parent:\n'
             "        raise RuntimeError('Open within the repository')\n"
             '    root = root.parent\n'
             "sys.path.insert(0, str(root / 'src'))\n"
             "paper = root / 'texs/sparsesurrs/visionlattices'"),
        markdown('## Verify cached evidence\n\n'
                 'Check original image bytes and preprocessing, selected\n'
                 'features, cached codes, reconstruction, and query extents.\n'
                 'The standalone verifier also recomputes full backbone\n'
                 'inference and optimized-stimulus responses.'),
        code('from lattmc.vision.verify_featureviz_codexgen import verify\n'
             'print(json.dumps(verify(inference=False), indent=2))\n'
             "for name in ['resnet34', 'dinov2']:\n"
             "    folder = root / 'vision_tokens' / ('imagenette_' + name)\n"
             "    result = json.loads((folder / 'results'\n"
             "                         / 'experiment_codexgen.json')\n"
             '                        .read_text())\n'
             "    print(name, result['metrics'])"),
        markdown('## Natural photographs and site maps\n\n'
                 'Springer and church coordinates are selected from\n'
                 'Imagenette training labels. Imagewoof is a transfer\n'
                 'collection with no SAE retraining. Actual labels remain\n'
                 'visible, including mismatches.'),
        code('import pymupdf\nfrom IPython.display import Image, display\n'
             'def show(name):\n'
             "    with pymupdf.open(paper / 'figures' / name) as doc:\n"
             '        page = doc[0].get_pixmap(\n'
             '            matrix=pymupdf.Matrix(1.25, 1.25), alpha=False)\n'
             "        display(Image(data=page.tobytes('png')))\n"
             "show('imagenette_resnet34_codexgen.pdf')\n"
             "show('imagenette_dinov2_codexgen.pdf')"),
        markdown('## Input optimization and controlled response tests\n\n'
                 'Optimization uses a smooth pre-Top-k proxy; reported\n'
                 'responses use actual sparse codes. Both random\n'
                 'initializations are shown. Curve, line, and corner\n'
                 'stimuli have matched mean brightness and RMS contrast.\n'
                 'Rotation of photographs also changes crop and sampling.'),
        code("show('optimized_features_codexgen.pdf')\n"
             "show('synthetic_stimuli_codexgen.pdf')\n"
             "show('synthetic_tuning_codexgen.pdf')\n"
             "show('rotation_tuning_codexgen.pdf')"),
        markdown('## Reproduce\n\n'
                 'See vision_tokens/IMAGENETTE.md for complete commands,\n'
                 'upstream provenance, immutable model revision, and\n'
                 'limitations. The minimal environment adds Transformers\n'
                 'for DINOv2. No paid service or generative-image model is\n'
                 'used: stimuli come from differentiating the measured\n'
                 'backbones and sparse encoders.'),
    ])
    notebook.metadata.kernelspec = {
        'display_name': 'Project Python', 'language': 'python',
        'name': 'python3'}
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher',
                                '-f', '{connection_file}']
    client = NotebookClient(notebook, timeout=300, km=manager,
                            resources={'metadata': {'path': str(root)}})
    client.execute()
    nbformat.validate(notebook)
    target = root / 'notebooks/vision/imagenette_features_codexgen.ipynb'
    nbformat.write(notebook, target)
    print('Executed four Imagenette/CNN/ViT notebook cells')


if __name__ == '__main__':
    os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/private/tmp/vision-jupyter')
    create()
