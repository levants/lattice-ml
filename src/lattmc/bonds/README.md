# Bond construction and surrogate comparisons

Run modules from the repository root with its existing uv environment:

```sh
uv run --no-sync python -m src.lattmc.bonds.bond_experiments_codexgen
uv run --no-sync python -m src.lattmc.bonds.summarize_bonds_codexgen
uv run --no-sync python -m src.lattmc.bonds.test_bonds_codexgen
uv run --no-sync python -m src.lattmc.bonds.audit_codexgen
uv run --no-sync python -m src.lattmc.bonds.notebook_codexgen
```

`bond_experiments_codexgen` provides finite saturation, the exact one-pair
construction, sparse context operations, and the cached-corpus experiment.
The reporting module produces the manuscript tables. The test module
checks the constructions independently; the audit checks the paper's
complete input tree. The notebook module regenerates and executes
`notebooks/bonds/bonds_layers_codexgen.ipynb`.

`paths_codexgen` resolves the repository, paper, notebook, and data paths
from its own location. All results go to `notebooks/bonds/data/`, while
LaTeX tables stay in `texs/crossbonds/surrogates/tables/`. Existing raw SAE
and transcoder activation caches remain at their original locations.
See the manuscript README for the data protocol and build instructions.
