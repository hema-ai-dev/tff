# FedProx paired analysis: alpha=0.1

## Paired global metrics

| Metric | mu=0 | mu=0.1 | Delta |
|---|---:|---:|---:|
| Best validation accuracy | 89.0216% | 94.6379% | +5.6163 pp |
| Test accuracy | 77.5059% | 84.5210% | +7.0150 pp |
| Macro class accuracy | 73.5995% | 80.7860% | +7.1865 pp |
| Test loss | 0.723341 | 0.533059 | -0.190282 |

## Class-level findings

- Improved classes: 33
- Declined classes: 9
- Unchanged classes: 1
- Mean class change: +7.1865 percentage points
- Zero-accuracy classes, mu=0: [0]
- Zero-accuracy classes, mu=0.1: []

## Audit status

- Pair configuration comparable: True
- Initial checkpoint exact: True
- mu=0 confusion accuracy: 0.775059382423
- mu=0.1 confusion accuracy: 0.845209817894

## Output files

- `outputs/stage1c/fedprox_formal/analysis/alpha01/alpha01_per_class_mu0_vs_mu0p1.csv`
- `outputs/stage1c/fedprox_formal/analysis/alpha01/alpha01_confusion_pair_comparison.csv`
- `outputs/stage1c/fedprox_formal/analysis/alpha01/alpha01_classification_summary.json`
