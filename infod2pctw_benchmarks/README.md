# Benchmark run guide

See the [project overview](../README.md) for the research question, contribution and attribution notes, and the synthetic CPU example.

## Full environment and upstream models

In a fresh checkout, run this from the repository root:

```bash
git submodule update --init
cd infod2pctw_benchmarks
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing requirements include the model dependencies, test tools and Snakemake/Slurm tooling. The benchmark uses Pyro 1.9.1. D²PCCA upstream specifies PyTorch 2.0.1 and Pyro 1.8.4 for its own environment. DPCTW runs without the submodules.

## Run the CLI with your own data

From this directory, point `--data` to a compatible directory containing `arrays.npz` and, for predefined trial splits, `metadata.json`. Use a new output directory for a new run.

```bash
python run_baseline.py --model dpctw \
  --data /path/to/compatible/bundle \
  --config configs/dpctw.yaml --output results/local/dpctw

python evaluate.py --data /path/to/compatible/bundle \
  --results results/local/dpctw --summary results/local/comparison.csv
```

For another baseline, use `d2pcca` or `infodpcca` with its corresponding config. For split-based runs, the evaluator reuses the saved split and normalization, reloads the selected checkpoint, and writes `test_metrics.json`. It can also read compatible exported reconstructions using the legacy artifact path. Use the synthetic test in the root README to try the full pipeline without supplying recordings.

The root YAML files are workflow templates and require compatible bundles and, for reference comparison, matching Info-D²PCTW exports. The Snakefile defaults to the synthetic template. Set the local or HPC paths, reference revision where applicable, and site settings before using it. For the Slurm launchers, set `BENCHMARK_VENV` to your configured environment and pass your allocation with `sbatch --account=YOUR_SLURM_ACCOUNT`.

## Tests

```bash
# Tests that do not need the upstream submodules
python -m pytest -q tests/test_data.py tests/test_result_evaluation.py \
  tests/test_dpctw.py tests/test_dpctw_workflow.py

# Full suite with the pinned upstream models initialized
python -m pytest -q
```
