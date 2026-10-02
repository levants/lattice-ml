"""Load frozen vision evidence without exposing semantic labels.

Only split membership is used from dataset metadata. Numerical row IDs
refer to a fixed concatenation of held-out collections; labels are revealed
separately after the first visual reading. No model is fitted or executed.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "vision_tokens/overcomplete"
OUT = DATA / "results/semantic_reading"
FIGURES = OUT / "figures"
DATASETS = ("imagenette", "imagewoof", "pets", "parts", "dtd")
MODELS = ("topk_k32_s0", "prisma_transcoder", "pretrained_ra")
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass
class Evidence:
    """Hold selected codes and native sparse matrices for one dictionary.

    Codes have shape (images, sites, 24); images are uint8 RGB crops.
    Native sparse matrices retain all coordinates for projection controls.
    Mapping pairs identify dataset rows without importing their labels.
    """

    model: str
    features: IntArray
    codes: FloatArray
    training: FloatArray
    native: sparse.csr_matrix
    native_training: sparse.csr_matrix
    images: NDArray[np.uint8]
    training_images: NDArray[np.uint8]
    mapping: list[tuple[str, int]]
    training_rows: IntArray
    provenance: dict[str, str]


def digest(path: Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_evidence(model: str) -> Evidence:
    """Load test crops and codes, retaining training data for query design.

    Parameters
    ----------
    model : str
        Existing cache directory name. Missing artifacts raise OSError.

    Returns
    -------
    Evidence
        Fixed-order held-out records and Imagenette training records.
        No class label, filename, or annotation mask enters selection.
    """
    query_path = DATA / f"results/{model}/queries_codexgen.json"
    features = np.array(json.loads(query_path.read_text())[
        "selected_features"], dtype=np.int64)
    matrices, selected, images, mapping = [], [], [], []
    provenance = {str(query_path.relative_to(ROOT)): digest(query_path)}
    for dataset in DATASETS:
        record_path = DATA / f"dataset/{dataset}_codexgen.json"
        image_path = DATA / f"dataset/{dataset}_codexgen.npz"
        code_path = DATA / f"codes/{model}/{dataset}_codexgen.npz"
        records = json.loads(record_path.read_text())
        ids = np.array([i for i, row in enumerate(records)
                        if row["split"] == "test"], dtype=np.int64)
        with np.load(code_path) as arrays:
            matrix = sparse.csr_matrix(
                (arrays["data"], arrays["indices"], arrays["indptr"]),
                shape=arrays["shape"])
        sites = matrix.shape[0] // len(records)
        assert sites * len(records) == matrix.shape[0]
        rows = (ids[:, None] * sites + np.arange(sites)).ravel()
        matrices.append(matrix[rows])
        codes = matrix[:, features].toarray().astype(np.float64)
        codes = codes.reshape(len(records), sites, len(features))
        selected.append(codes[ids])
        with np.load(image_path) as pixels:
            images.append(pixels["images"][ids])
            if dataset == "imagenette":
                train_ids = np.array([i for i, row in enumerate(records)
                                      if row["split"] == "train"])
                training_images = pixels["images"][train_ids]
        if dataset == "imagenette":
            training = codes[train_ids]
            train_rows = (train_ids[:, None] * sites
                          + np.arange(sites)).ravel()
            native_training = matrix[train_rows]
        mapping.extend((dataset, int(i)) for i in ids)
        for path in (record_path, image_path, code_path):
            provenance[str(path.relative_to(ROOT))] = digest(path)
    return Evidence(
        model, features, np.concatenate(selected), training,
        sparse.vstack(matrices, format="csr"), native_training,
        np.concatenate(images), training_images, mapping, train_ids,
        provenance)


def projected_codes(evidence: Evidence, features: IntArray) -> FloatArray:
    """Return held-out native coordinates as (images, sites, coordinates)."""
    n, sites = evidence.codes.shape[:2]
    return evidence.native[:, features].toarray().astype(
        np.float64).reshape(n, sites, len(features))
