# Closure-added coordinate experiments

This executed extension isolates exactly the coordinates requested:
`h[i] = FG(u)[i]` if `u[i] == 0` and `FG(u)[i] > 0`, otherwise zero.
It does not include increases on existing coordinates or replace numerical
requirements with support. All thresholds retain their stored amplitudes.

## Results and interpretation

Every meet and join in the preceding contextreading_v1 grid was evaluated:
660 queries across six GPT-2 SAE/transcoder dictionaries, blocks 0, 8, 11.
There are 172 nonempty, nonuniversal original extents: h preserves 81 and
expands 91. Four expansions become universal. The other original extents
are 298 universal and 190 empty. Empty cases use the lattice top convention
and are not evidence learned from selected texts; all their h extents remain
empty. No h in this grid is zero, partly because of common pooled floors.
Counts retain repeated descriptions and are not independent trials.

The mathematical checks establish `G(u) <= G(h)` as set inclusion and
`G(u join h) = G(u)`. Raw `u join h` need not equal `FG(u)`, because h omits
amplitude increases on already positive coordinates. Moreover, h recovers
the original extent exactly if and only if `u <= FG(h)`. This is a
corpus-relative implication, not a causal or semantic equivalence claim.

| Original query | Condition | Added axes | Original extent | h extent |
| --- | --- | ---: | ---: | ---: |
| Research full meet | TC11 | 190 | 99 | 661 |
| Virus/Clippers rank-1 join | TC11 | 1076 | 6 | 6 |
| Biological rank-1 four-way join | TC11 | 1624 | 3 | 3 |
| Cat/dog full meet | TC11 | 255 | 25 | 25 |
| Cat/dog rank-2/3 join | TC11 | 197 | 40 | 389 |
| New/de rank-2/3 join | SAE11 | 21 | 13 | 22 |

TC means GPT-2-small ReLU transcoder, normalized MLP input to MLP output;
SAE means GPT-2-small ReLU SAE, residual-pre. Indices are zero-based.
Dictionary identities, scaling, input-token hashes and source positions
are inherited from the preceding verified extraction records.

## Reading the outputs

For the research h, the eight additional sampled rows include two
biotransformation/evidence passages (43, 65), municipal planning (4728),
entertainment-company appointments (5872), motivational incentives (8161),
wireless regulation (9431), editorial history of TV characters (10778),
and standards of explanation in fictional worlds (17044). The last contrasts
Gandalf and Star Trek, explicitly discussing when physics should constrain
an explanation. These supply concrete links to evidence and explanatory
practice beyond shared keywords, alongside broader institutional reporting.

All eight pass h distributively on fresh replay and fail 2--7 old
requirements. In particular, row 43 fails old coordinates 7172 and 22275;
row 5872 fails 1066, 2463, 7172, 10317, 16100, 18705. Their actual h
inequalities, maxima and token positions are saved. This supports broader
contextual association, not one validated meaning for 190 coordinates.
The entire added extent was not annotated for semantic precision.

For the Cat/dog rank-2/3 h, additional samples concern Batman adaptation,
bicycle-design history, agricultural legislation, literary commentary,
payment-card fraud, television crime, a Python conference, and a childbirth
report spliced with baseball news. Entertainment and animal references
connect some passages but do not explain the whole set. All eight sampled
rows have distributed h witnesses. Broad narrative/expository structure,
splicing and incidental co-occurrence remain plausible alternatives.

The New/de h adds nine rows; the fixed sample reads eight. Urban arts,
employment, sports, acting, crime, politics and biographies occur. All eight
still contain literal New, so deleting the source coordinates does not
establish lexical independence. These new SAE h witnesses were not replayed.

## An important numerical limitation

We replayed 22 rows with both u and h: 44 query/item records. All six
sports-join rows change membership for h only. Although h selects the same
six rows in the original cache, fresh CPU maxima fall below 73--97 of its
1076 cached requirements per row. The largest shortfalls range from
1.28746e-5 to 3.03388e-5. The original two-coordinate u still passes in all
six rows. None is presented as a verified D witness for h after replay.

Closure takes minima over a selected extent, putting many thresholds
exactly on observed values. Consequently small extraction differences can
break membership. This does not contradict the inclusion law on fixed
data. We preserve historical masks, record replay rejection, and do not
silently add a tolerance. A homogeneous full-corpus re-extraction or an
explicitly separate relaxation control would be needed for a robust
cross-extraction claim. All sixteen additional research/Cat-dog rows
retain their h membership under replay.

## Reproduction and organization

Use the project's existing uv environment without upgrades. Sources are
in `src/lattmc/contextstudy`; the executed notebook is
`notebooks/sae/closure_added_codexgen.ipynb`; serialized data and full reading
packets are in `data/activation_studies/closureadded_v1`, outside the paper.
No new package directory contains underscores. Nothing was uploaded.

```sh
export PYTHONPATH=.:src
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/lattice-closure-mpl"
study_root="$PWD"
study_out="$PWD/data/activation_studies/closureadded_v1"
study_protocol="$PWD/experiments/activation_studies/closureadded_v1"
uv run --no-sync python -m lattmc.contextstudy.closureadded_codexgen \
  --root "$study_root" --out "$study_out" --kind tc --block 11 \
  --protocol "$study_protocol/PROTOCOL.md"
```

Run that construction for kinds sae/tc and blocks 0/8/11. It uses only
cached matrices and source traces. `closurewitness_codexgen` with root/out
arguments performs the bounded CPU replay. `closureaudit_codexgen` checks
all definitions/extents through an independent row-oriented dense-block
implementation and verifies every stored token trace, including rejections.
`closuretables_codexgen --out ... --paper ...` regenerates all four tables.
Use a fresh output folder for replication rather than replacing the records.

The mirror receives code, notebook, protocol and manifests, not TeX.
Its local cache symlink follows the existing arrangement; it is not a public
data deposit. The original caches are never overwritten.

## Publication implications

This strengthens the methodology by testing a mathematically precise
projection of closure and exposing both redundant descriptions and broader
extents. It also reveals numerical sensitivity that should remain explicit
in an arXiv version. Independent semantic annotation, new-corpus validation
and latent-mediated causal interventions remain unexecuted. The earlier
venue assessment is unchanged; this extension alone does not make the
long working paper a main-track conference submission.
