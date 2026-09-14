#!/usr/bin/env bash
#SBATCH --account=def-rozeske
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:17:00
#SBATCH --nodes=1
#SBATCH --job-name=dpcca_bench_controller
#SBATCH --output=dpcca_bench_controller_%j.log


set -euo pipefail

export XDG_CACHE_HOME="$SCRATCH/.cache"
mkdir -p "$XDG_CACHE_HOME"

# This allocation only runs the lightweight Snakemake controller. Each rule is
# submitted as its own Slurm job with the CPU, memory, runtime, and GPU request
# declared in the Snakefile.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"


module load python/3.11


# source ~/envs/snakemake_cc/bin/activate
source ~/envs/dpcca_baselines/bin/activate


export PYTHONNOUSERSITE=1
cd "$PROJECT_DIR"

# Fail before submission if the controller environment is incomplete. The same
# environment is used by the rule jobs on the compute nodes.
python -c 'import snakemake, snakemake_executor_plugin_slurm, torch, pyro, scipy, sklearn, yaml'


snakemake \
  --snakefile Snakefile \
  --directory "$PROJECT_DIR" \
  --cores "$SLURM_CPUS_PER_TASK" \
  --rerun-incomplete \
  --printshellcmds


# snakemake \
#   --snakefile Snakefile \
#   --directory "$PROJECT_DIR" \
#   --executor slurm \
#   --jobs 1 \
#   --set-resources fit_infodpcca:tasks_per_gpu=0 \
#   --rerun-incomplete \
#   --printshellcmds
