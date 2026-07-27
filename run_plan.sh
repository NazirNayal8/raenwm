#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash run_plan.sh [DATASET] [curve | line] [CKPT_PATH]"
    exit 1
fi

DATASET=$1
SAMPLER=$2
CKPT_PATH=$3
RESULTS_FOLDER="results_plan"

echo "========================================"
echo "Starting Planning (CEM)"
echo "Dataset: ${DATASET}  Sampler: ${SAMPLER}  Checkpoint: ${CKPT_PATH}"
echo "========================================"

# planning_eval.py is now a hydra entrypoint: pass overrides as key=value (no --flags).
CKPT_ARG=""
if [ -n "${CKPT_PATH}" ]; then
    CKPT_ARG="checkpoint.path=${CKPT_PATH}"
fi

python planning_eval.py \
  run.datasets=${DATASET} \
  planning.rollout_stride=1 \
  run.batch_size=24 \
  planning.num_samples=120 \
  planning.topk=3 \
  run.num_workers=12 \
  run.output_dir=${RESULTS_FOLDER} \
  run.run_tag=results_plan \
  planning.opt_steps=1 \
  planning.num_repeat_eval=1 \
  planning.traj_sampler=${SAMPLER} \
  ${CKPT_ARG} \
  run.save_preds=true \
  planning.plot=true

echo "Planning finished successfully!"
