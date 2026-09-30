"""Build an executed, paper-independent patch-context evidence notebook."""

import os
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from lattmc.vision.paths_codexgen import repository_root


def create():
    root = repository_root()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md('# Pretrained vision SAEs: patches, concepts, and witnesses\n\n'
           'Prisma CLIP ViT-B/32 and SAEV DINOv2 ViT-B/14 with registers.\n'
           '200 training images select coordinates and queries; 100 test\n'
           'images measure retrieval. Closure describes the full 300-image\n'
           'context. Coordinate indices are never aligned across models.\n'
           'This notebook needs source and experimental data only.'),
        code('from pathlib import Path\nimport sys\nimport json\n'
             'root = Path.cwd()\n'
             "while not (root / 'src/lattmc/vision').is_dir():\n"
             '    if root == root.parent:\n'
             "        raise RuntimeError('Open within this repository')\n"
             '    root = root.parent\n'
             "sys.path.insert(0, str(root / 'src'))\n"
             "evidence = root / 'vision_tokens/patch_contexts'"),
        md('## Verify saved activations and exact concepts\n\n'
           'The verifier checks full sparse-code caches, source pixels,\n'
           'all 80 vector queries, 20 downset descriptions, and six\n'
           'identical-pixel interventions. Native backbone and full SAE\n'
           'recomputations are available with `--native` in the CLI.\n'
           'The SAEV adapter preserves the old checkpoint encoder formula:\n'
           '`relu((x - b_dec) @ W_enc + b_enc)`. Its published training\n'
           'mean and scalar are retained with symmetric clipping.'),
        code('from lattmc.vision.patch_verify_codexgen import verify\n'
             "for name in ['prisma', 'saev']:\n"
             '    verify(name)\n'
             "    path = evidence / name / 'results'\n"
             "    result = json.loads((path / 'extraction_codexgen.json')\n"
             '                        .read_text())\n'
             "    print(name, result['metrics'])"),
        md('## Meet, join, and common patch witnesses\n\n'
           'Green cells satisfy every query coordinate at one token.\n'
           'Heatmap values are activation / join threshold, clipped to\n'
           'the display range [0, 2]. Cyan boxes mark the strongest common\n'
           'response. These are contextual token responses, not masks.\n'
           'Displayed images are the two highest-ranked test exemplars\n'
           'for each fixed training query; they are illustrative selections.'),
        code('from IPython.display import Image, display\n'
             'def show(name, kind):\n'
             "    folder = evidence / name / 'figures'\n"
             "    display(Image(filename=str(folder /\n"
             "        f'{name}_patch_{kind}_codexgen.png')))\n"
             "for name in ['prisma', 'saev']:\n"
             "    show(name, 'gallery')"),
        md('## Closed downset descriptions\n\n'
           'Shaded rectangles are principal downsets; their union is the\n'
           'exact intersection of two source-image descriptions in two\n'
           'selected coordinates. Multiple maximal generators cannot be\n'
           'replaced by their coordinatewise maximum without changing\n'
           'the description. Exact closure is often too restrictive to\n'
           'retrieve new test photographs; this is recorded, not hidden.'),
        code("show('prisma', 'downsets')\n"
             "for name in ['prisma', 'saev']:\n"
             "    path = evidence / name / 'results/contexts_codexgen.json'\n"
             "    for case in json.loads(path.read_text())['cases']:\n"
             "        print(name, case['class'], 'join pooled / same-site',\n"
             "              case['pooled_test'][3],\n"
             "              case['same_site_test'][3], 'generators',\n"
             "              case['downset_generators'])"),
        md('## Identical pixels, different context\n\n'
           'Each row preserves its marked patch pixels exactly. The\n'
           'joint score is at that designated site and must reach one\n'
           'to satisfy the join. Gray ablations are out of distribution;\n'
           'this is a limited context-dependence test, not evidence that\n'
           'a discovered circuit uses a particular semantic shape.'),
        code("for name in ['prisma', 'saev']:\n"
             "    show(name, 'controls')\n"
             "    path = evidence / name / 'results/controls_codexgen.json'\n"
             '    print(json.dumps(json.loads(path.read_text()), indent=2))'),
        md('## Reproduce\n\n'
           'See `vision_tokens/patch_contexts/README.md` for pinned sources,\n'
           'environment setup, native checks, checkpoint reassembly, and\n'
           'rerun commands. No paper files are required. Reflection results\n'
           'use within-image shuffled positions as a descriptive reference,\n'
           'not a confidence interval over a new image population.'),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
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
    target = root / 'notebooks/vision/patch_contexts_codexgen.ipynb'
    nbformat.write(notebook, target)
    print('Executed five patch-context notebook cells', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/private/tmp/vision-jupyter')
    create()
