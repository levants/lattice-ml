# External SAELens study

The fixed plan is in `external_protocol.md`; explanatory paper sections are
`external-experiments.tex` and `external-findings.tex`. Generated tables are
under `../tables/external-*.tex`, with complete results in the appendix.

Reusable implementation belongs to `src/lattmc/activationstudy` at the
repository root. The synchronized code checkout is
`/Users/ltsinadze/git/lattice-ml`, with the same package and the executed
`notebooks/sae/saelens_external_codexgen.ipynb` companion. No TeX is copied
to that checkout. The package README gives reproduction commands using
the project's existing uv environment.

The canonical completed cache is repository-relative
`data/activation_studies/external_v1`. Paper-relative `external_results/`
and older notebook cache paths are compatibility symlinks to that cache,
not separate mirrors. Raw texts and tokens remain local; derived-data
release archives exclude them.

Four SAEs were evaluated: GPT-2 residual before block 8, GPT-2 MLP output,
and two Gemma-2 residual-after-block-8 SAEs with release L0 identifiers 37
and 301. Data comprise AG News (600/200/400 train/calibration/test) and
DBpedia 14 (1050/350/700). All arrays and checkpoint files have recorded
checksums. All thirteen methods are retained.

Extraction and evaluation ran successfully in the existing uv environment.
The notebook independently executed the cached-score audit. It does not
claim to have rerun GPU inference inside Jupyter. Current imports of
`transformer_lens.HookedTransformer` succeed; GPT-2 keeps that numerical
path for checkpoint compatibility. Gemma uses ordinary Hugging Face hooks
and `SAE.encode`. This run does not establish the cause of an older import
failure under a different dependency lockfile.

See `../PUBLICATION_ASSESSMENT.md` for the scientific assessment and limits.
Release candidates under repository-relative
`artifacts/releases/activation_studies/external_v1` contain a results archive
and a separate derived-activation archive. They exclude third-party model
weights, source text, and token sequences. These files are prepared locally;
they have not been uploaded, tagged, or assigned a DOI.
