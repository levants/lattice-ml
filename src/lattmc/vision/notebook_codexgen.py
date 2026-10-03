"""Build and execute the review notebook with the project Python kernel."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def create_notebook(root: Path) -> Path:
    """Write the offline digit experiment notebook and return its path."""
    root = Path(root)
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(cells=[
        markdown("# Vision lattices: verified digit-image pilot\n\n"
                 "This notebook reviews three executed CNN/SAE runs.\n"
                 "It verifies finite contexts, saved queries, and model\n"
                 "inference. It does not run CLIP, DINO, or a transcoder.\n"
                 "All source files use the `_codexgen` suffix."),
        code("from pathlib import Path\nimport sys\nimport json\n"
             "import numpy as np\nimport torch\n"
             "\n"
             "root = Path.cwd()\n"
             "while not (root / 'pyproject.toml').exists():\n"
             "    if root == root.parent:\n"
             "        raise RuntimeError('Open within the project')\n"
             "    root = root.parent\n"
             "sys.path.insert(0, str(root / 'src'))\n"
             "paper = root / 'texs/sparsesurrs/visionlattices'\n"
             "evidence = root / 'vision_tokens/digits'\n"
             "from lattmc.vision.paths_codexgen import load_digit_cache\n"
             "torch.set_num_threads(2)"),
        markdown("## Exact order-theoretic checks\n\n"
                 "Integer matrices test adjunction, closure, ordinal\n"
                 "scaling, and the spatial counterexample."),
        code("from lattmc.vision.audit_codexgen import audit\n"
             "print(json.dumps(audit(), indent=2))"),
        markdown("## Recompute retrieval and verify provenance\n\n"
                 "Verify hashes, split separation, source labels, and all\n"
                 "150 calibration choices and stored test-score arrays."),
        code("from lattmc.vision.verify_codexgen import verify_results\n"
             "print(json.dumps(verify_results(paper), indent=2))\n"
             "report = json.loads(\n"
             "    (evidence / 'results/results_codexgen.json').read_text())\n"
             "for seed in report['seeds']:\n"
             "    rows = [r for r in report['retrieval']\n"
             "            if r['seed'] == seed]\n"
             "    print('Seed', seed)\n"
             "    for key in ('graded_ap', 'binary_ap',\n"
             "                'sae_cosine_ap', 'dense_cosine_ap'):\n"
             "        print(key, round(np.mean([r[key] for r in rows]), 4))"),
        markdown("## Reload weights and reproduce feature codes\n\n"
                 "This checks cached codes against actual inference from\n"
                 "the saved CNN and SAE weights on every image."),
        code("from lattmc.vision.models_codexgen import DigitCNN, TopKSAE\n"
             "digits = load_digit_cache(evidence, 17)\n"
             "images = torch.tensor(digits['images'][:, None] / 16,\n"
             "                      dtype=torch.float32)\n"
             "for seed in report['seeds']:\n"
             "    weights = torch.load(\n"
             "        evidence / 'checkpoints'\n"
             "        / f'digits_seed_{seed}_codexgen.pt',\n"
             "        map_location='cpu', weights_only=True)\n"
             "    cnn, sae = DigitCNN(), TopKSAE()\n"
             "    cnn.load_state_dict(weights['cnn'])\n"
             "    sae.load_state_dict(weights['sae'])\n"
             "    cnn.eval()\n    sae.eval()\n"
             "    with torch.no_grad():\n"
             "        hidden = cnn.features(images)\n"
             "        sites = hidden.permute(0, 2, 3, 1)\n"
             "        sites = sites.reshape(-1, 16, 32)\n"
             "        codes = sae.encode(sites).numpy()\n"
             "    cache = load_digit_cache(evidence, seed)\n"
             "    np.testing.assert_array_equal(codes, cache['patches'])\n"
             "    print('Exact cached-code reproduction:', seed)"),
        markdown("## A spatial counterexample\n\n"
                 "Two separate sites can jointly satisfy an image-level\n"
                 "query without any one site satisfying it."),
        code("from lattmc.vision.contexts_codexgen import spatial_extents\n"
             "patches = np.array([[[2, 0], [0, 2]]])\n"
             "image, site = spatial_extents(patches, [2, 2])\n"
             "print('Pooled:', image.tolist())\n"
             "print('Same site:', site.tolist())"),
        markdown("## Retraining and foundation-model extensions\n\n"
                 "From the repository root, use the existing environment:\n\n"
                 "```sh\nPYTHONPATH=src uv run --no-sync python -m \\\n"
                 "  lattmc.vision.experiment_codexgen \\\n"
                 "  --output /private/tmp/vision-reproduction\n```\n\n"
                 "The manuscript README gives table and PDF commands.\n"
                 "ViT-Prisma, SAEV, and Overcomplete are researched\n"
                 "extension options; none was used for the pilot.\n"
                 "See LIBRARIES.md for verified source links and the\n"
                 "activation-cache contract. Do not identify dictionary\n"
                 "indices across independently trained models."),
    ])
    notebook.metadata.kernelspec = {
        "display_name": "Project Python", "language": "python",
        "name": "python3",
    }
    path = root / "notebooks/vision/visionlattices_codexgen.ipynb"
    nbformat.write(notebook, path)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher",
                                "-f", "{connection_file}"]
    client = NotebookClient(notebook, timeout=180, km=manager,
                            resources={"metadata": {"path": str(root)}})
    client.execute()
    nbformat.write(notebook, path)
    count = sum(c.cell_type == "code" for c in notebook.cells)
    print(f"Executed {count} cells")
    return path


def create_feature_notebook(root: Path) -> Path:
    """Write the feature-analysis notebook and return its path."""
    root = Path(root)
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(cells=[
        markdown("# Natural-image feature examples\n\n"
                 "Frozen ResNet34 and a Top-16 SAE on a fixed CIFAR-10\n"
                 "sample. These are measured responses, not semantic\n"
                 "detector labels or recovered causal circuits."),
        code("from pathlib import Path\nimport sys\nimport json\n"
             "root = Path.cwd()\n"
             "while not (root / 'pyproject.toml').exists():\n"
             "    if root == root.parent:\n"
             "        raise RuntimeError('Open within the project')\n"
             "    root = root.parent\n"
             "sys.path.insert(0, str(root / 'src'))\n"
             "paper = root / 'texs/sparsesurrs/visionlattices'\n"
             "folder = root / 'vision_tokens/cifar10_resnet34'"),
        markdown("## Recompute feature selection and lattice extents\n\n"
                 "The standalone verification module also reproduces all\n"
                 "1,500 dense and sparse activation arrays from weights.\n"
                 "This notebook checks selection, scores, and queries."),
        code("from lattmc.vision.verify_natural_codexgen import (\n"
             "    verify_natural)\n"
             "print(json.dumps(verify_natural(inference=False), indent=2))\n"
             "summary = json.loads(\n"
             "    (folder / 'results/natural_codexgen.json').read_text())\n"
             "print('Test reconstruction R2:',\n"
             "      round(summary['test_reconstruction_r2'], 3))"),
        markdown("## Highest-activating examples and spatial maps\n\n"
                 "Each row uses the same color scale. Feature association\n"
                 "is selected from training labels; gallery examples are\n"
                 "ranked across all test classes. No mismatch is hidden."),
        code("import pymupdf\nfrom IPython.display import Image, display\n"
             "def show_figure(name):\n"
             "    with pymupdf.open(paper / 'figures' / name) as document:\n"
             "        pix = document[0].get_pixmap(\n"
             "            matrix=pymupdf.Matrix(1.2, 1.2), alpha=False)\n"
             "        display(Image(data=pix.tobytes('png')))\n"
             "show_figure('feature_animals_codexgen.pdf')\n"
             "show_figure('feature_transport_codexgen.pdf')\n"
             "show_figure('feature_tuning_codexgen.pdf')"),
        markdown("## Meet and join examples\n\n"
                 "The query pairs are cat/dog and ship/truck. Extent counts\n"
                 "cover all test images; the pictures show at most three\n"
                 "ranked matches. A join conjoins numeric requirements,\n"
                 "not class labels. Same-site witnesses are separate."),
        code("show_figure('query_cat_dog_codexgen.pdf')\n"
             "show_figure('query_ship_truck_codexgen.pdf')"),
        markdown("## Reproduce the figures\n\n"
                 "From the repository root:\n\n"
                 "```sh\nPYTHONPATH=src uv run --no-sync python -m "
                 "lattmc.vision.examples_codexgen\n```\n\n"
                 "See vision_tokens/README.md for data provenance and\n"
                 "retraining commands. The mirrored repository is\n"
                 "https://github.com/levants/lattice-ml."),
    ])
    notebook.metadata.kernelspec = {
        "display_name": "Project Python", "language": "python",
        "name": "python3",
    }
    path = root / "notebooks/vision/feature_examples_codexgen.ipynb"
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher",
                                "-f", "{connection_file}"]
    client = NotebookClient(notebook, timeout=180, km=manager,
                            resources={"metadata": {"path": str(root)}})
    client.execute()
    nbformat.write(notebook, path)
    print("Executed four natural-image notebook cells")
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/private/tmp/vision-jupyter")
    create_notebook(root)
    create_feature_notebook(root)
