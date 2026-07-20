# Parameter-estimation feature comparison: thomas, 2d

## Summary

- Best normal test loss: `betti_0` with `0.16325`.
- Worst normal test loss: `raw_pc` with `0.735503`.
- Fastest convergence: `betti_1` at epoch `6`.
- Smallest final train/validation gap: `pi_0` with gap `-0.0106025`.
- Most stable late validation curve: `betti_0` with late-val std `0.00409541`.
- Best adversarial loss: `pi_0` with `0.149361`.
- Worst adversarial loss: `raw_pc` with `0.679077`.
- Smallest adversarial gap: `raw_pc` with adversarial-minus-test `-0.0564253`.
- Largest adversarial gap: `pi_1` with adversarial-minus-test `-0.00673445`.

## Ranking by normal test loss

| test_rank | method   | test_loss           | best_val_loss       | best_epoch | final_train_val_gap   | convergence_epoch_105 | percent_worse_than_best |
| --------- | -------- | ------------------- | ------------------- | ---------- | --------------------- | --------------------- | ----------------------- |
| 1         | betti_0  | 0.16324958653860194 | 0.14206207115002858 | 79         | 0.04381896151421438   | 29                    | 0.0                     |
| 2         | pi_0     | 0.20107130113468374 | 0.17642039236842946 | 451        | -0.010602488575900931 | 223                   | 23.16803086489808       |
| 3         | pi_1     | 0.23144562318119952 | 0.2222007793444459  | 405        | 0.031033814594309816  | 285                   | 41.77409455580579       |
| 4         | betti_1  | 0.275677792808061   | 0.25170443968106343 | 7          | 0.3208785060043525    | 6                     | 68.8689072072316        |
| 5         | pairwise | 0.46265503411651937 | 0.42671517883577653 | 336        | 0.03843823098184146   | 71                    | 183.40349517952384      |
| 6         | raw_pc   | 0.7355027762792443  | 0.6679429645820331  | 257        | 0.12055530892356014   | 32                    | 350.5388294538361       |

## Ranking by adversarial loss

| adversarial_rank | method   | adversarial_loss    | test_loss           | adversarial_minus_test | adversarial_ratio_to_test | percent_worse_than_best_adversarial |
| ---------------- | -------- | ------------------- | ------------------- | ---------------------- | ------------------------- | ----------------------------------- |
| 1                | pi_0     | 0.1493614683047347  | 0.20107130113468374 | -0.05170983282994904   | 0.7428283771073217        | 0.0                                 |
| 2                | betti_0  | 0.15609544563457506 | 0.16324958653860194 | -0.007154140904026884  | 0.9561766675449728        | 4.508510398479316                   |
| 3                | pi_1     | 0.22471117508520774 | 0.23144562318119952 | -0.006734448095991774  | 0.9709026768213311        | 50.4478883581546                    |
| 4                | betti_1  | 0.26644693547432574 | 0.275677792808061   | -0.009230857333735243  | 0.966515774666833         | 78.39067766172963                   |
| 5                | pairwise | 0.43775200898494193 | 0.46265503411651937 | -0.02490302513157744   | 0.9461736644039074        | 193.08228819217177                  |
| 6                | raw_pc   | 0.679077471068146   | 0.7355027762792443  | -0.05642530521109834   | 0.9232833552355272        | 354.6537194470118                   |

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
