# Python tools for lattconference

`lattconferencetools/` is the importable package; `run_codexgen.py` is its
command-line launcher. Run from the paper directory with the repository
environment:

```sh
../../../.venv/bin/python python/run_codexgen.py --help
../../../.venv/bin/python python/run_codexgen.py build_paper_codexgen
```

| Task | Modules in the package |
| --- | --- |
| Article PDFs | `build_paper_codexgen.py` |
| Activation experiments | `activation_experiments_codexgen.py` |
| Generated activation tables | `build_activation_tables_codexgen.py` |
| Cached notebooks | `execute_cached_notebooks_codexgen.py` |
| Checks | `test_activation_experiments_codexgen.py` |
| Review checks | `review_checks_codexgen.py` |
| Resource paths | `paths_codexgen.py` |
| Conference PDF | `build_conference_codexgen.py` |
| Source bundle | `package_arxiv_codexgen.py` |
| Conference experiments | `conference_experiments_codexgen.py` |
| Ablations | `conference_ablation_codexgen.py` |
| Conference results | `summarize_conference_codexgen.py` |
| Artifact checks | `verify_conference_artifacts_codexgen.py` |
| Conference notebook | `execute_conference_notebook_codexgen.py` |
| Conference checks | `test_conference_experiments_codexgen.py` |

Pass a module name without `.py` to the launcher. Tools resolve paper
resources from their location; experiment data and results stay outside
the Python package. Both paper packages can be imported together because
their names differ. Building PDFs does not rerun experiments.
