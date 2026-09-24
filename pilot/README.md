# pilot — three new families beside the bank's five

A small, balanced experiment to decide which families (and whether PH) earn a place in the next
bank. Like `cascade/`, it imports `cloudforger` and `cascade/` read-only and writes only under
`pilot/` and `data/pilot/<name>/`.

| family | why | parameters |
|---|---|---|
| `ring` | Neyman–Scott with offspring on a circle of radius ρ, radially smeared by σ. Loops (H₁) are its signature; σ/ρ → 1 fills it into a Thomas blob | κ, μ, ρ, σ |
| `matern1` | Deletes *every* point with a neighbour within R. Pair correlation close to Matérn II's; a second repulsive family | R, λ_p |
| `cell` | Generalised Baddeley–Silverman: a shifted grid of cells of area 1/n̄ holding 0 \| 1 \| k points (probabilities 1/k, 1 − 1/(k−1), 1/(k(k−1))). K(r) = πr² exactly | n̄, k |

The families live in `cloudforger.simulation` (`families.py`, `processes.py`; rules in
`configs/simulation/config.yaml`). They are drawn by the bank's own `draw_theta`/`sample_patterns`
on the `family_pilot` stream set, so no pilot draw can collide with a bank draw, and the bank's
draws are unchanged (its regeneration tests still pass bit-for-bit).

## Design (`config.yaml`)

- 5000 θ per new family × 2 replicates. Every family, bank ones included, uses 3700 / 300 / 1000
  train / val / test θ. The bank families' rows come from inside the bank's own split blocks.
- Features: `classical` (cascade/features.py, 111 columns) and `ph_<tag>` for alpha and DTM(k=10)
  diagrams (`featurize.py`: Betti curves and persistence statistics on the √n axis, 105 columns
  each). Every model is the same gradient-boosted tree, so the feature sets differ only in what they see.
- `cell` is held out of stage 1 and the cascade, because its K function equals Poisson's. Where the
  filter sends it is a result, not a label.

## Running (CPU jobs through SLURM)

```bash
bash pilot/run.sh                                  # generate -> merge -> featurize -> assemble -> analyze
FROM=analyze bash pilot/run.sh                     # rerun later steps only
PILOT_CONFIG=pilot/smoke.yaml bash pilot/run.sh    # 40-theta smoke run (name: smoke)
```

Results: `pilot/results/<name>/{summary.md, report.json, predictions.npz}`. See `analyze.py` for
what each section measures.
