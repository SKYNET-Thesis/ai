VLA FOUR-MODEL COMPARISON
=========================

Models:
    L8 W0.5
    L9 W0.5
    L8 W0.75
    L9 W0.75

Primary factors:
    LFM backbone depth:
        8 vs 9 layers

    Action Expert width:
        0.5 vs 0.75

Held-out comparison:
    111 paired test episodes across all 4 models.

Checkpoint selection:
    Validation MAE only.
    Test set is not used for checkpoint selection.

Visualizations:
    - Four models appear in the same figure.
    - Bar charts contain numeric values.
    - Validation checkpoints contain numeric labels.
    - Distribution charts show mean and median.
    - Latency plots include numeric statistics.
    - Gate plots compare all four models.

Important:
    Lower MAE/MSE/RMSE/nMAE is better.
    Lower latency is better.
    Lower boundary jump is generally better.
