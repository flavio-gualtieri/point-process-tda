# Parameter-estimation feature comparison: matern, 2d

## Summary

- Best normal test loss: `pi_0` with `0.226022`.
- Worst normal test loss: `raw_pc` with `0.764249`.
- Fastest convergence: `betti_1` at epoch `5`.
- Smallest final train/validation gap: `pi_0` with gap `0.0288709`.
- Most stable late validation curve: `betti_1` with late-val std `0.00638209`.
- Best adversarial loss: `pi_0` with `0.444668`.
- Worst adversarial loss: `raw_pc` with `0.997205`.
- Smallest adversarial gap: `betti_1` with adversarial-minus-test `0.092372`.
- Largest adversarial gap: `pi_1` with adversarial-minus-test `0.343184`.

## Ranking by normal test loss

| test_rank | method   | test_loss           | best_val_loss       | best_epoch | final_train_val_gap | convergence_epoch_105 | percent_worse_than_best |
| --------- | -------- | ------------------- | ------------------- | ---------- | ------------------- | --------------------- | ----------------------- |
| 1         | pi_0     | 0.2260218518120902  | 0.17336577867589345 | 226        | 0.02887088125158499 | 97                    | 0.0                     |
| 2         | betti_0  | 0.27223810127803255 | 0.17464065115626265 | 61         | 0.18291097035299403 | 57                    | 20.447690829630744      |
| 3         | pi_1     | 0.5692620050339472  | 0.5135531454551511  | 463        | 0.09970156761430299 | 241                   | 151.86149059039636      |
| 4         | pairwise | 0.6775673769769215  | 0.5085442938455721  | 393        | 0.17708444526885353 | 341                   | 199.7795883648616       |
| 5         | betti_1  | 0.716027713957287   | 0.5552067320521284  | 7          | 0.8167366060005442  | 5                     | 216.79579129923124      |
| 6         | raw_pc   | 0.7642487202371869  | 0.6682021021842957  | 176        | 0.41952787614058534 | 72                    | 238.13045690492226      |

## Ranking by adversarial loss

| adversarial_rank | method   | adversarial_loss    | test_loss           | adversarial_minus_test | adversarial_ratio_to_test | percent_worse_than_best_adversarial |
| ---------------- | -------- | ------------------- | ------------------- | ---------------------- | ------------------------- | ----------------------------------- |
| 1                | pi_0     | 0.44466792543729144 | 0.2260218518120902  | 0.21864607362520125    | 1.9673669686016864        | 0.0                                 |
| 2                | betti_0  | 0.5785042941570282  | 0.27223810127803255 | 0.30626619287899565    | 2.1249938617747364        | 30.098048692880322                  |
| 3                | betti_1  | 0.8083997567494711  | 0.716027713957287   | 0.09237204279218403    | 1.1290062395513567        | 81.79853110711362                   |
| 4                | pairwise | 0.8556839823722839  | 0.6775673769769215  | 0.17811660539536245    | 1.26287659566796          | 92.43213495346998                   |
| 5                | pi_1     | 0.9124463001887003  | 0.5692620050339472  | 0.3431842951547531     | 1.6028582482582652        | 105.19723775700493                  |
| 6                | raw_pc   | 0.9972048997879028  | 0.7642487202371869  | 0.2329561795507159     | 1.3048172321157663        | 124.25833813114367                  |

## How to read the diagnostics

- `test_loss` is the held-out score from the normal train/val/test split.
- `adversarial_loss` is the loss on parameter combinations intentionally excluded from the normal dataset.
- `adversarial_minus_test` measures how much performance degrades on unseen parameter combinations.
- `best_val_loss` checks whether the model selected by validation also generalizes well.
- `final_train_val_gap` is a simple train/validation overfitting diagnostic.
- `convergence_epoch_105` is the first epoch whose validation loss is within 5% of the best validation loss.
- `val_auc` summarizes the whole validation curve; lower means better average validation performance during training.
- `late_val_std` measures how noisy or unstable validation loss was near the end of training.

## Generated figures

- `figs/overlay_train_val_curves.png`
- `figs/validation_curves.png`
- `figs/test_loss_ranking.png`
- `figs/best_val_vs_test.png`
- `figs/train_val_gap.png`
- `figs/convergence_speed.png`
- `figs/adversarial_loss_ranking.png`
- `figs/test_vs_adversarial_bars.png`
- `figs/test_vs_adversarial_scatter.png`
- `figs/adversarial_gap.png`
