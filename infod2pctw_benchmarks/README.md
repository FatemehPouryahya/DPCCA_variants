# Info-D²PCTW manuscript baselines

This repository runs D²PCCA, InfoDPCCA, and DPCTW against one strict AgNeuro
input/result contract. It does not import the proposed Info-D²PCTW model and it
does not implement plain DPCCA as a separate benchmark.

## Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The checked smoke environment uses Python 3.12.2, PyTorch 2.6.0,
Pyro 1.9.1, NumPy 2.1.3, and SciPy 1.15.2. The author D²PCCA repository pins
PyTorch 2.0.1/Pyro 1.8.4; Pyro 1.9.1 is used here because it supports the
available Python/PyTorch runtime without changing the model API or objective.

## Run

```bash
.venv/bin/python run_baseline.py \
  --model d2pcca \
  --data real_data_debug/2JH3_NoPart_debug_1000 \
  --config configs/d2pcca.yaml
```

Replace the model/config with `infodpcca` or `dpctw`. The checked configs are
deliberately short smoke schedules; increase iteration counts before a
manuscript run.

```bash
.venv/bin/python evaluate.py \
  --data real_data_debug/2JH3_NoPart_debug_1000 \
  --results results/dms_2JH3_NoPart/d2pcca \
            results/dms_2JH3_NoPart/infodpcca \
            results/dms_2JH3_NoPart/dpctw \
  --summary results/dms_2JH3_NoPart/comparison.csv
```

The evaluator also accepts an exported Info-D²PCTW result directory when its
reconstruction keys use `x_reconstruction`/`recon_x`/`x_hat` and
`y_reconstruction`/`recon_y`/`y_hat`.

## Tests

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q -p no:cacheprovider
```

See `BASELINE_AUDIT.md` for source selection, exact adaptations, equations, and
known limitations.
