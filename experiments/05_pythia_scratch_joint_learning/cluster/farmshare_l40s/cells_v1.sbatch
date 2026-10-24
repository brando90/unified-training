#!/usr/bin/env bash
#SBATCH -J p05-v1
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH --array=0-8%4
#SBATCH -o logs/%x_%A_%a.log
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
# Submit from $P05_STATE_ROOT with code/experiments/05_pythia_scratch_joint_learning/cluster/submit.sh (it records the job id in jobs.txt), not bare sbatch.
set -euo pipefail
: "${P05_STATE_ROOT:=/home/users/brando9/ut_p05}"
exec bash "$P05_STATE_ROOT/code/experiments/05_pythia_scratch_joint_learning/cluster/jobs.sh" cells v1
