# Stage 1C-A1 FedProx Technical Report v1.0

## 1. Stage objective

This stage evaluates whether the FedProx proximal term improves federated GTSRB training under label-skew non-IID partitions while preserving the same data partition, model, client sampling, training budget, optimizer, learning rate, server update, and initialization.

The paired control is the same FedProx implementation with `mu=0`, which is the strict FedAvg-equivalent control under the selected SGD configuration.

## 2. Experimental design

- Dataset: GTSRB, 43 classes
- Clients: 5
- Partition family: Dirichlet label skew
- Heterogeneity levels: alpha=0.1 and alpha=0.5
- Rounds: 160
- Local epochs: 1
- Batch size: 64
- Client optimizer: SGD
- Client learning rate: 0.1
- Client momentum: 0.0
- Server optimizer: stateless SGDM-compatible update
- Server learning rate: 1.0
- Client weighting: number of examples
- Paired proximal strengths: mu=0 and mu=0.1
- Model selection: global validation accuracy
- Official test set: used only after hyperparameter selection

## 3. Integrity and comparability audit

Both alpha pairs passed:

- 160 contiguous and unique metric rows
- finite core numeric metrics
- zero non-finite update count
- final checkpoints and model files
- test metrics, per-class accuracy, and confusion matrix
- confusion-derived accuracy consistency
- matched paired configuration
- identical Round-0 model weights

## 4. Formal results

| alpha | Metric | mu=0 | mu=0.1 | FedProx change |
|---:|---|---:|---:|---:|
| 0.1 | Best validation accuracy | 89.0216% | 94.6379% | +5.6163 pp |
| 0.1 | Test accuracy | 77.5059% | 84.5210% | +7.0150 pp |
| 0.1 | Macro class accuracy | 73.5995% | 80.7860% | +7.1865 pp |
| 0.1 | Test loss | 0.723341 | 0.533059 | -0.190282 |
| 0.5 | Best validation accuracy | 98.0671% | 99.4914% | +1.4242 pp |
| 0.5 | Test accuracy | 87.3555% | 91.8131% | +4.4576 pp |
| 0.5 | Macro class accuracy | 82.4559% | 87.8022% | +5.3464 pp |
| 0.5 | Test loss | 0.494126 | 0.346118 | -0.148008 |

## 5. Class-level findings

### alpha=0.1

- Improved classes: 33
- Declined classes: 9
- Unchanged classes: 1
- Mean class-accuracy change: +7.1865 percentage points
- Zero-accuracy classes at mu=0: [0]
- Zero-accuracy classes at mu=0.1: []

### alpha=0.5

- Improved classes: 32
- Declined classes: 6
- Unchanged classes: 5
- Mean class-accuracy change: +5.3464 percentage points
- Zero-accuracy classes at mu=0: []
- Zero-accuracy classes at mu=0.1: []

Detailed class and confusion-pair tables are stored in the per-alpha analysis directories.

## 6. Interpretation

FedProx improves the strict paired SGD baseline at both tested heterogeneity levels. The test-accuracy gain is larger at alpha=0.1 than at alpha=0.5, supporting the interpretation that proximal regularization is more valuable when client label distributions are more heterogeneous.

Macro-class accuracy also improves at both alpha values. This indicates that the benefit is not limited to frequent classes and should be interpreted together with the per-class and confusion-pair analyses.

The strict claim is relative to the matched `mu=0` control. Comparisons with historical Stage 1B runs using different optimizer settings are supplementary and must not be presented as the primary causal estimate of the proximal term.

## 7. Limitations

- One partition seed is used in the current formal comparison.
- Five clients and full participation are fixed.
- Only one selected nonzero proximal strength is evaluated formally.
- Results are specific to the selected model, local epoch count, and optimizer configuration.
- Statistical uncertainty across multiple seeds has not yet been estimated.

## 8. Reproducibility artifacts

- `analysis/alpha01/alpha01_classification_summary.json`
- `analysis/alpha01/alpha01_per_class_mu0_vs_mu0p1.csv`
- `analysis/alpha01/alpha01_confusion_pair_comparison.csv`
- `analysis/alpha05/alpha05_classification_summary.json`
- `analysis/alpha05/alpha05_per_class_mu0_vs_mu0p1.csv`
- `analysis/alpha05/alpha05_confusion_pair_comparison.csv`
- `analysis/stage1ca1_cross_alpha_summary.csv`
- `analysis/stage1ca1_cross_alpha_summary.json`
- `analysis/stage1ca1_cross_alpha_summary.md`
- `analysis/alpha05/alpha05_integrity_and_pair_audit.txt`

## 9. Stage conclusion

Stage 1C-A1 is complete. Under matched SGD conditions, FedProx with mu=0.1 produces higher validation accuracy, test accuracy, macro-class accuracy, and lower test loss than the mu=0 paired baseline at both alpha=0.1 and alpha=0.5. The gain is stronger under the more heterogeneous alpha=0.1 partition.

The next algorithm stage is Stage 1C-A2: SCAFFOLD.
