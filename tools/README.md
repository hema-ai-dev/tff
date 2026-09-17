# Stage 1C-A1 FedProx finalization package

Files:

- `analyze_fedprox_pair.py`: analyzes one `mu=0` / `mu=0.1` pair.
- `finalize_stage1ca1_fedprox.py`: creates the cross-alpha summary, technical report, manifest, and closure audit.
- `run_stage1ca1_finalize.sh`: runs the complete workflow for alpha=0.1 and alpha=0.5.

Install into the repository:

```bash
cd ~/projects/tff_mnist_fedavg
cp /path/to/analyze_fedprox_pair.py tools/
cp /path/to/finalize_stage1ca1_fedprox.py tools/
cp /path/to/run_stage1ca1_finalize.sh tools/
chmod +x tools/run_stage1ca1_finalize.sh
```

Run:

```bash
bash tools/run_stage1ca1_finalize.sh
```

Expected final success line:

```text
[PASS] Stage 1C-A1 FedProx finalization complete
```
