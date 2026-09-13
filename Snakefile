configfile: "benchmark_config.yaml"

import csv
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import tempfile
import yaml

from snakemake.exceptions import WorkflowError


DATASETS = tuple(config["datasets"])
BASELINES = ("d2pcca", "infodpcca", "dpctw")
ENVIRONMENT = config.get("execution_environment", "local")
if ENVIRONMENT not in ("local", "hpc"):
    raise WorkflowError("execution_environment must be 'local' or 'hpc'")

BENCHMARK_ROOT = Path(config["paths"]["benchmark_root"])
OUTPUT_ROOT = config["paths"]["output_root"].rstrip("/")
PYTHON = config["paths"][f"{ENVIRONMENT}_python"]
RUN_BASELINE = BENCHMARK_ROOT / "run_baseline.py"
EVALUATE = BENCHMARK_ROOT / "evaluate.py"
APPROVED = bool(config.get("training_configs_approved", False))

wildcard_constraints:
    dataset="lfp|fp|miniscope"


def environment_path(dataset, field):
    value = config["datasets"][dataset][field]
    return value[ENVIRONMENT] if isinstance(value, dict) else value


def bundle_path(dataset):
    return environment_path(dataset, "bundle")


def baseline_config(model):
    return config["models"][model]["config"]


def resource(model, key):
    return config["resources"][model][key]


def dpctw_gpu_resource():
    with Path(baseline_config("dpctw")).open() as stream:
        model_config = yaml.safe_load(stream) or {}
    device = str(model_config.get("device", "cpu")).lower()
    if device not in ("cpu", "cuda"):
        raise WorkflowError("DPCTW device must be 'cpu' or 'cuda'")
    return 1 if device == "cuda" else 0


def requested_gpu_count(model):
    return dpctw_gpu_resource() if model == "dpctw" else resource(model, "gpu")


def hpc_resource(key):
    return config["hpc"][key]


def baseline_result_paths(dataset, model):
    root = f"{OUTPUT_ROOT}/{dataset}/{model}"
    return {
        "resolved_config": f"{root}/resolved_config.yaml",
        "provenance": f"{root}/provenance.json",
        "preprocessing": f"{root}/preprocessing.json",
        "metrics": f"{root}/metrics.json",
        "latents": f"{root}/latents.npz",
        "reconstructions": f"{root}/reconstructions.npz",
        "history": f"{root}/history.json",
        "result": f"{root}/result.json",
        "hardware": f"{root}/hardware.json",
    }


def baseline_fit_paths(dataset, model):
    return {
        name: path for name, path in baseline_result_paths(dataset, model).items()
        if name != "metrics"
    }


def all_baseline_results():
    return [
        baseline_result_paths(dataset, model)["result"]
        for dataset in DATASETS
        for model in BASELINES
    ]


def capture_hardware(model):
    record = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "requested_threads": int(resource(model, "threads")),
        "requested_gpu_count": int(requested_gpu_count(model)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if int(requested_gpu_count(model)):
        executable = shutil.which("nvidia-smi")
        if executable:
            query = subprocess.run(
                [executable, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                check=False, capture_output=True, text=True,
            )
            record["nvidia_smi"] = query.stdout.strip() or None
    return record


def execute_baseline(model, dataset, outputs):
    if not APPROVED:
        raise WorkflowError(
            "Baseline execution is locked: approve the proposed manuscript "
            "schedule, update the three model YAMLs, then set "
            "training_configs_approved: true."
        )
    output_dir = Path(str(outputs.result)).parent
    command = [
        PYTHON, str(RUN_BASELINE),
        "--model", model,
        "--data", bundle_path(dataset),
        "--config", baseline_config(model),
        "--output", str(output_dir),
    ]
    subprocess.run(command, check=True)
    Path(str(outputs.hardware)).write_text(
        json.dumps(capture_hardware(model), indent=2, sort_keys=True) + "\n"
    )
    missing = [str(path) for path in outputs if not Path(str(path)).exists()]
    if missing:
        raise WorkflowError(f"{model} did not produce required artifacts: {missing}")


rule all:
    input:
        expand(f"{OUTPUT_ROOT}/{{dataset}}/comparison.csv", dataset=DATASETS),
        expand(f"{OUTPUT_ROOT}/{{dataset}}/infod2pctw/reference.json", dataset=DATASETS),
        f"{OUTPUT_ROOT}/manuscript/model_comparison.csv",
        f"{OUTPUT_ROOT}/manuscript/run_manifest.csv",


rule fit_d2pcca:
    input:
        arrays=lambda wc: f"{bundle_path(wc.dataset)}/arrays.npz",
        metadata=lambda wc: f"{bundle_path(wc.dataset)}/metadata.json",
        model_config=baseline_config("d2pcca"),
    output:
        **baseline_fit_paths("{dataset}", "d2pcca"),
        model=f"{OUTPUT_ROOT}/{{dataset}}/d2pcca/model.pt",
        optimizer=f"{OUTPUT_ROOT}/{{dataset}}/d2pcca/optimizer.pt",
    threads: resource("d2pcca", "threads")
    resources:
        gpu=resource("d2pcca", "gpu"),
        gpu_model=hpc_resource("gpu_model"),
        mem_mb=resource("d2pcca", "mem_mb"),
        runtime=resource("d2pcca", "runtime"),
        slurm_account=hpc_resource("slurm_account"),
    run:
        execute_baseline("d2pcca", wildcards.dataset, output)


rule fit_infodpcca:
    input:
        arrays=lambda wc: f"{bundle_path(wc.dataset)}/arrays.npz",
        metadata=lambda wc: f"{bundle_path(wc.dataset)}/metadata.json",
        model_config=baseline_config("infodpcca"),
    output:
        **baseline_fit_paths("{dataset}", "infodpcca"),
        stage_I_model=f"{OUTPUT_ROOT}/{{dataset}}/infodpcca/stage_I_model.pt",
        stage_I_optimizer=f"{OUTPUT_ROOT}/{{dataset}}/infodpcca/stage_I_optimizer.pt",
        stage_II_model=f"{OUTPUT_ROOT}/{{dataset}}/infodpcca/stage_II_model.pt",
        stage_II_optimizer=f"{OUTPUT_ROOT}/{{dataset}}/infodpcca/stage_II_optimizer.pt",
    threads: resource("infodpcca", "threads")
    resources:
        gpu=resource("infodpcca", "gpu"),
        gpu_model=hpc_resource("gpu_model"),
        mem_mb=resource("infodpcca", "mem_mb"),
        runtime=resource("infodpcca", "runtime"),
        slurm_account=hpc_resource("slurm_account"),
    run:
        execute_baseline("infodpcca", wildcards.dataset, output)


rule fit_dpctw:
    input:
        arrays=lambda wc: f"{bundle_path(wc.dataset)}/arrays.npz",
        metadata=lambda wc: f"{bundle_path(wc.dataset)}/metadata.json",
        model_config=baseline_config("dpctw"),
    output:
        **baseline_fit_paths("{dataset}", "dpctw"),
        model=f"{OUTPUT_ROOT}/{{dataset}}/dpctw/model.npz",
        alignment=f"{OUTPUT_ROOT}/{{dataset}}/dpctw/alignment.npz",
    threads: resource("dpctw", "threads")
    resources:
        gpu=dpctw_gpu_resource(),
        mem_mb=resource("dpctw", "mem_mb"),
        runtime=resource("dpctw", "runtime"),
        slurm_account=hpc_resource("slurm_account"),
    run:
        execute_baseline("dpctw", wildcards.dataset, output)


rule reference_infod2pctw:
    output:
        f"{OUTPUT_ROOT}/{{dataset}}/infod2pctw/reference.json"
    run:
        source = Path(environment_path(wildcards.dataset, "infod2pctw_result"))
        evaluation = Path(environment_path(wildcards.dataset, "infod2pctw_evaluation"))
        evaluation_inputs = (
            "resolved_config.yaml", "provenance.json", "preprocessing.json",
            "latents.npz", "reconstructions.npz", "result.json",
        )
        complete_contract = evaluation_inputs + (
            "metrics.json", "hardware.json",
        )
        record = {
            "dataset": wildcards.dataset,
            "dataset_identity": config["datasets"][wildcards.dataset]["identity"],
            "source_result": str(source),
            "evaluation_export": str(evaluation),
            "scientifically_reusable": True,
            "recorded_git_commit": config["models"]["infod2pctw"]["recorded_git_commit"],
            "evaluation_ready": all(
                (evaluation / name).exists() for name in evaluation_inputs
            ),
            "missing_evaluation_artifacts": [
                name for name in evaluation_inputs if not (evaluation / name).exists()
            ],
            "missing_common_contract_artifacts": [
                name for name in complete_contract if not (evaluation / name).exists()
            ],
        }
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(output[0]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


rule validate_infod2pctw_export:
    input:
        reference=f"{OUTPUT_ROOT}/{{dataset}}/infod2pctw/reference.json"
    output:
        f"{OUTPUT_ROOT}/{{dataset}}/infod2pctw/.evaluation_ready"
    run:
        record = json.loads(Path(input.reference).read_text())
        if not record["evaluation_ready"]:
            raise WorkflowError(
                f"Info-D2PCTW export for {wildcards.dataset} is not ready; missing: "
                + ", ".join(record["missing_evaluation_artifacts"])
            )
        Path(output[0]).write_text(record["evaluation_export"] + "\n")


rule evaluate_dataset:
    input:
        d2pcca=f"{OUTPUT_ROOT}/{{dataset}}/d2pcca/result.json",
        infodpcca=f"{OUTPUT_ROOT}/{{dataset}}/infodpcca/result.json",
        dpctw=f"{OUTPUT_ROOT}/{{dataset}}/dpctw/result.json",
        info_ready=f"{OUTPUT_ROOT}/{{dataset}}/infod2pctw/.evaluation_ready",
    output:
        comparison=f"{OUTPUT_ROOT}/{{dataset}}/comparison.csv",
        d2pcca_metrics=f"{OUTPUT_ROOT}/{{dataset}}/d2pcca/metrics.json",
        infodpcca_metrics=f"{OUTPUT_ROOT}/{{dataset}}/infodpcca/metrics.json",
        dpctw_metrics=f"{OUTPUT_ROOT}/{{dataset}}/dpctw/metrics.json",
        infod2pctw_metrics=f"{OUTPUT_ROOT}/{{dataset}}/infod2pctw/metrics.json",
    run:
        result_dirs = [
            Path(str(input.d2pcca)).parent,
            Path(str(input.infodpcca)).parent,
            Path(str(input.dpctw)).parent,
            Path(environment_path(wildcards.dataset, "infod2pctw_evaluation")),
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv") as temporary:
            subprocess.run(
                [
                    PYTHON, str(EVALUATE), "--data", bundle_path(wildcards.dataset),
                    "--results", *map(str, result_dirs), "--summary", temporary.name,
                ],
                check=True,
            )
        rows = []
        for result_dir in result_dirs:
            metrics = json.loads((result_dir / "metrics.json").read_text())
            rows.append({
                "dataset": wildcards.dataset,
                "model_name": metrics["model_name"],
                "rmse_x": metrics.get("x", {}).get("rmse"),
                "rmse_y": metrics.get("y", {}).get("rmse"),
                "mean_per_feature_normalized_mse_x": metrics.get("x", {}).get("mean_per_feature_normalized_mse"),
                "mean_per_feature_normalized_mse_y": metrics.get("y", {}).get("mean_per_feature_normalized_mse"),
                "training_runtime_seconds": metrics.get("training_runtime_seconds"),
                "inference_runtime_seconds": metrics.get("inference_runtime_seconds"),
                "parameter_count": metrics.get("parameter_count"),
                "result_path": str(result_dir),
            })
        Path(output.infod2pctw_metrics).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            result_dirs[-1] / "metrics.json", Path(output.infod2pctw_metrics)
        )
        Path(output.comparison).parent.mkdir(parents=True, exist_ok=True)
        with Path(output.comparison).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


rule combined_manuscript_table:
    input:
        expand(f"{OUTPUT_ROOT}/{{dataset}}/comparison.csv", dataset=DATASETS)
    output:
        f"{OUTPUT_ROOT}/manuscript/model_comparison.csv"
    run:
        rows = []
        for filename in input:
            with Path(filename).open(newline="") as stream:
                rows.extend(csv.DictReader(stream))
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        with Path(output[0]).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


rule run_manifest:
    input:
        baseline_results=all_baseline_results(),
        baseline_provenance=[
            baseline_result_paths(dataset, model)["provenance"]
            for dataset in DATASETS for model in BASELINES
        ],
        baseline_hardware=[
            baseline_result_paths(dataset, model)["hardware"]
            for dataset in DATASETS for model in BASELINES
        ],
        info_references=expand(
            f"{OUTPUT_ROOT}/{{dataset}}/infod2pctw/reference.json", dataset=DATASETS
        ),
        comparisons=expand(f"{OUTPUT_ROOT}/{{dataset}}/comparison.csv", dataset=DATASETS),
    output:
        f"{OUTPUT_ROOT}/manuscript/run_manifest.csv"
    run:
        rows = []
        for dataset in DATASETS:
            for model in BASELINES:
                paths = baseline_result_paths(dataset, model)
                result = json.loads(Path(paths["result"]).read_text())
                provenance = json.loads(Path(paths["provenance"]).read_text())
                hardware = json.loads(Path(paths["hardware"]).read_text())
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": provenance.get("seed"),
                    "bundle_identity": config["datasets"][dataset]["identity"],
                    "bundle_path": bundle_path(dataset),
                    "model_config": baseline_config(model),
                    "git_commit": provenance.get("benchmark_git_commit"),
                    "third_party_commit": provenance.get("third_party_git_commit"),
                    "hardware": json.dumps(hardware, sort_keys=True),
                    "training_runtime_seconds": result.get("training_runtime_seconds"),
                    "inference_runtime_seconds": result.get("inference_runtime_seconds"),
                    "success": True,
                    "result_path": str(Path(paths["result"]).parent),
                })
            reference_path = Path(f"{OUTPUT_ROOT}/{dataset}/infod2pctw/reference.json")
            reference = json.loads(reference_path.read_text())
            evaluation_dir = Path(reference["evaluation_export"])
            result = json.loads((evaluation_dir / "result.json").read_text())
            provenance = json.loads((evaluation_dir / "provenance.json").read_text())
            hardware_path = evaluation_dir / "hardware.json"
            rows.append({
                "dataset": dataset,
                "model": "infod2pctw",
                "seed": provenance.get("seed", 0),
                "bundle_identity": reference["dataset_identity"],
                "bundle_path": bundle_path(dataset),
                "model_config": config["models"]["infod2pctw"]["config"],
                "git_commit": reference["recorded_git_commit"],
                "third_party_commit": None,
                "hardware": hardware_path.read_text().strip() if hardware_path.exists() else None,
                "training_runtime_seconds": result.get("training_runtime_seconds"),
                "inference_runtime_seconds": result.get("inference_runtime_seconds"),
                "success": True,
                "result_path": reference["source_result"],
            })
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        with Path(output[0]).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
