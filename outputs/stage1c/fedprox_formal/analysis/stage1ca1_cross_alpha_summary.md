# Stage 1C-A1 FedProx cross-alpha summary

| alpha | Test acc. mu=0 | Test acc. mu=0.1 | Test delta | Macro mu=0 | Macro mu=0.1 | Macro delta | Improved/Declined/Unchanged |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 77.5059% | 84.5210% | +7.0150 pp | 73.5995% | 80.7860% | +7.1865 pp | 33/9/1 |
| 0.5 | 87.3555% | 91.8131% | +4.4576 pp | 82.4559% | 87.8022% | +5.3464 pp | 32/6/5 |

## Main paired conclusion

Under exactly matched SGD settings, FedProx with mu=0.1 outperforms the mu=0 paired baseline at both heterogeneity levels.
The gain is larger under stronger label skew (alpha=0.1), which is consistent with the intended role of the proximal term.
