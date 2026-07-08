# Parameter-estimation feature comparison: inhom_thomas, 2d

## Summary

- Best normal test loss: `betti_0` with `0.577729`.
- Worst normal test loss: `betti_1` with `0.601186`.
- Fastest convergence: `betti_1` at epoch `3`.
- Smallest final train/validation gap: `pi_0` with gap `0.0405667`.
- Most stable late validation curve: `pairwise` with late-val std `0.00211145`.
- Best adversarial loss: `pairwise` with `0.661431`.
- Worst adversarial loss: `pi_1` with `0.696152`.
- Smallest adversarial gap: `pairwise` with adversarial-minus-test `0.0825839`.
- Largest adversarial gap: `pi_1` with adversarial-minus-test `0.111443`.

## Ranking by normal test loss

| test_rank | method   | test_loss          | best_val_loss      | best_epoch | final_train_val_gap | convergence_epoch_105 | percent_worse_than_best |
| --------- | -------- | ------------------ | ------------------ | ---------- | ------------------- | --------------------- | ----------------------- |
| 1         | betti_0  | 0.5777291351573759 | 0.573090432371412  | 33         | 0.2721852106046031  | 6                     | 0.0                     |
| 2         | pairwise | 0.5788468227766858 | 0.5694936752319336 | 54         | 0.2036416261151348  | 10                    | 0.19346222153144424     |
| 3         | pi_1     | 0.5847086335858728 | 0.5880300440107074 | 196        | 0.07258571137607017 | 39                    | 1.2080918208488192      |
| 4         | pi_0     | 0.5880908420962146 | 0.5770990170751299 | 283        | 0.04056667092899502 | 106                   | 1.793523350006605       |
| 5         | betti_1  | 0.6011856247556855 | 0.5902822310583932 | 8          | 0.7810077538540812  | 3                     | 4.0601188638201435      |

## Ranking by adversarial loss

| adversarial_rank | method   | adversarial_loss   | test_loss          | adversarial_minus_test | adversarial_ratio_to_test | percent_worse_than_best_adversarial |
| ---------------- | -------- | ------------------ | ------------------ | ---------------------- | ------------------------- | ----------------------------------- |
| 1                | pairwise | 0.6614307444671105 | 0.5788468227766858 | 0.08258392169042472    | 1.1426697330638798        | 0.0                                 |
| 2                | betti_0  | 0.6848513858071689 | 0.5777291351573759 | 0.10712225064979297    | 1.1854195056661154        | 3.5409060640093935                  |
| 3                | betti_1  | 0.6856600103707149 | 0.6011856247556855 | 0.0844743856150294     | 1.1405129832393428        | 3.6631599160279444                  |
| 4                | pi_0     | 0.6915234475300229 | 0.5880908420962146 | 0.10343260543380828    | 1.175878619475061         | 4.549637783643848                   |
| 5                | pi_1     | 0.6961516240547443 | 0.5847086335858728 | 0.11144299046887152    | 1.1905957669641705        | 5.24935979738999                    |

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
