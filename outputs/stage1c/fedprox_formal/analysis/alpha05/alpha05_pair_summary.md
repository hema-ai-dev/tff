# FedProx paired analysis: alpha=0.5

## Paired global metrics

| Metric | mu=0 | mu=0.1 | Delta |
|---|---:|---:|---:|
| Best validation accuracy | 98.0671% | 99.4914% | +1.4242 pp |
| Test accuracy | 87.3555% | 91.8131% | +4.4576 pp |
| Macro class accuracy | 82.4559% | 87.8022% | +5.3464 pp |
| Test loss | 0.494126 | 0.346118 | -0.148008 |

## Class-level findings

- Improved classes: 32
- Declined classes: 6
- Unchanged classes: 5
- Mean class change: +5.3464 percentage points
- Zero-accuracy classes, mu=0: []
- Zero-accuracy classes, mu=0.1: []

## Audit status

- Pair configuration comparable: True
- Initial checkpoint exact: True
- mu=0 confusion accuracy: 0.873555027712
- mu=0.1 confusion accuracy: 0.918131433096

## Output files

- `outputs/stage1c/fedprox_formal/analysis/alpha05/alpha05_per_class_mu0_vs_mu0p1.csv`
- `outputs/stage1c/fedprox_formal/analysis/alpha05/alpha05_confusion_pair_comparison.csv`
- `outputs/stage1c/fedprox_formal/analysis/alpha05/alpha05_classification_summary.json`
