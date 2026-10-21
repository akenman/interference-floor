# The Interference Floor

Code, stored reports, and the automated guard suite for the manuscript
**"The Interference Floor: Why a Learned Corrector Degrades the Prior It Was
Meant to Repair"** (TMLR submission).

## What is here

| Path | Contents |
|---|---|
| `paper/` | Manuscript sources (LaTeX + TMLR style), generated tables, figure generators, the twelve preregistrations, the English criteria summary, and the guard scripts. |
| `analysis_output/` | Every stored JSON report the manuscript quotes, plus the checkpoint manifest. |
| `tests/` | The 43-test suite (prior-registry cross-implementation agreement, architecture wiring, checkpoint naming, ladder-extension provenance). |
| `experiments/`, `figures/` | Preregistration inputs and rendered figures. |
| root `*.py` | The two world generators with their validation suites, and the full training / evaluation pipeline. |

## Reproducing the checks

```
python paper/verify_numbers.py     # number provenance guard
python -m pytest tests -q          # 43 passed
python paper/check_latex.py        # LaTeX structure guard -> PASS
python paper/build_pdf.py          # builds paper/latex/main.pdf (needs a TeX distribution)
```

`paper/verify_numbers.py` re-checks every number quoted in the manuscript against
its stored JSON. In the full repository it reports 319/319 PASS; from this archive
it runs 318 of them and automatically skips the single check that recomputes the
training window fraction from the raw trajectory dataset (see *Not included*).

## Not included

The trajectory datasets (`trajectories_v3.npz`, `trajectories_hinge.npz`) and the
bulk of the trained checkpoints are excluded for size. `analysis_output/checkpoints_manifest.json`
lists the checkpoints with 16-character SHA-256 digests; the training pipeline and
the one dataset-dependent provenance check require them.

## License

Released under the MIT License — see `LICENSE`.
