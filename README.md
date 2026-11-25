# Benchmarking shared and modality-specific dynamics in paired time series

When neural and behavioral signals are recorded during the same trial, how can we separate the dynamics they share from the dynamics unique to each signal, especially when their timing differs? This repository compares three published approaches to that problem. It uses a common data interface and evaluation workflow so their reconstructions and latent representations can be inspected on the same trial partitions.

## My contribution

I built the benchmark infrastructure: the data adapter and masking rules, training and checkpoint interfaces, validation and held-out evaluation, artifact export and provenance, tests, and Snakemake orchestration. I implemented the DPCTW baseline in this repository, including the linear-Gaussian state-space model, EM fitting, Kalman smoothing, and alternating time alignment. The method is based on [Nicolaou et al., *Dynamic Probabilistic CCA for Analysis of Affective Behavior and Fusion of Continuous Annotations*](https://pubmed.ncbi.nlm.nih.gov/26353304/).

The D²PCCA and InfoDPCCA model implementations come from their authors. My wrappers connect them to the benchmark and handle data conversion, training, checkpoint selection and resumption, and standardized outputs. The external source stays at the Git revisions recorded in this repository.

| Component | Source |
| --- | --- |
| D²PCCA model | [Tang et al., ICASSP 2025](https://doi.org/10.1109/ICASSP49660.2025.10887585) and [upstream code at the pinned revision](https://github.com/marcusstang/D2PCCA/tree/f88607ac1c3e2905b91298bc359980a9dac0ce93) |
| InfoDPCCA model | [Tang and Yu, UAI 2025](https://proceedings.mlr.press/v286/tang25b.html) and [upstream code at the pinned revision](https://github.com/marcusstang/InfoDPCCA/tree/02b2e119c9d7e230b2b2cf62534adacaf219698e) |
| Info-D²PCTW reference | The workflow can consume an external result export. The reference model and its results are outside this sample. |

## Data, fitting, and evaluation

1. **Load and prepare trials.** The adapter reads paired arrays `x` and `y` from a NumPy bundle and converts them to trial × time × feature layout. It uses a validity mask and optional JSON metadata with trial identifiers and split assignments. The predefined AgNeuro protocol expects 60 trials (36 train, 12 validation, 12 test) with 32 features in `x` and 16 in `y`.
2. **Fit and select.** Under that protocol, normalization statistics and parameter updates use only training trials. Each baseline selects a checkpoint using its own validation objective. The split manifest, preprocessing statistics, configuration and provenance are saved with the run.
3. **Evaluate and export.** The evaluator reloads the selected checkpoint and applies the saved preprocessing to held-out trials. Exports include shared and view-specific latent arrays, reconstructions, masks, and metrics. DPCTW also exports alignment paths. Its test-time alignment uses both observed views with fixed fitted parameters.

Masked reconstruction error summarizes how well a fitted model reconstructs observed signals on the evaluation trials. Reconstruction alone cannot establish recovery of the true latent factors or prediction of an unobserved view. Validation objectives differ by model and serve checkpoint selection within each method. Bundles without predefined splits use a legacy fit-and-reconstruct path, so their reconstruction scores are in-sample.

## Run a small example on CPU

From the repository root, use Python 3.12. The following installs the dependencies for the DPCTW example and CPU tests. The example uses generated arrays and requires no external models, recordings, or cluster access.

```bash
cd infod2pctw_benchmarks
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.1.3 scipy==1.15.2 torch==2.6.0 'PyYAML>=6,<7' 'pytest>=8,<9'

demo_dir=$(mktemp -d)
python -m pytest -q --basetemp "$demo_dir" \
  tests/test_dpctw_workflow.py::test_predefined_dpctw_end_to_end_and_standalone_reproducibility
find "$demo_dir" -name test_metrics.json -print
```

The test creates an AgNeuro-shaped bundle and fits DPCTW, selects a checkpoint on validation trials, then evaluates the test trials twice. It checks that the metrics and latent arrays agree across evaluations and that checkpoint contents stay intact. Inspect `test_metrics.json`, `trial_splits.json`, and `preprocessing.json` in the generated result directory. These synthetic outputs demonstrate the workflow rather than benchmark performance.

See the [benchmark run guide](infod2pctw_benchmarks/README.md) for CLI examples, submodule setup and the full test command. The broader tests cover input shapes and masks, training-only statistics, alignment constraints and shift recovery, checkpoint resumption, validation selection, and evaluation replay. The CUDA regression requires a CUDA device.

## Where to review the code

| Start here | What to look for |
| --- | --- |
| [Data adapter](infod2pctw_benchmarks/data/agneuro_adapter.py) | Whole-trial split, training-only standardization, and validity masks |
| [DPCTW model](infod2pctw_benchmarks/baselines/dpctw/model.py) and [alignment](infod2pctw_benchmarks/baselines/dpctw/alignment.py) | Alternating fitting and alignment, with fixed model parameters for validation and test |
| [Training entry point](infod2pctw_benchmarks/run_baseline.py), [artifact contract](infod2pctw_benchmarks/benchmark.py), and [evaluator](infod2pctw_benchmarks/evaluate.py) | What goes into a run, what gets saved, and how evaluation is replayed |
| [D²PCCA wrapper](infod2pctw_benchmarks/baselines/d2pcca.py) and [InfoDPCCA wrapper](infod2pctw_benchmarks/baselines/infodpcca.py) | Integration with the pinned upstream models |
| [End-to-end DPCTW test](infod2pctw_benchmarks/tests/test_dpctw_workflow.py) and [Snakefile](Snakefile) | A runnable check of the research protocol and the larger experiment workflow |
