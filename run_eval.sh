#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: bash run_eval.sh [time|rollout] [gt|infer|eval|all] [DATASET] [CKPT_PATH]"
    exit 1
fi

EVAL_MODE=$1
STEP=$2
DATASET=$3
CKPT_PATH=$4

RESULTS_FOLDER="results"
EXP_DIR="${RESULTS_FOLDER}/exp_dir_euler"

echo "========================================"
echo "Starting Evaluation Pipeline"
echo "Mode: ${EVAL_MODE}  Step: ${STEP}  Dataset: ${DATASET}  Checkpoint: ${CKPT_PATH}"
echo "========================================"

# infer.py is now a hydra entrypoint: pass config overrides as key=value (no --flags).
INFER_COMMON="run.datasets=${DATASET} run.batch_size=32 run.num_workers=12 run.input_fps=4 \
checkpoint.path=${CKPT_PATH} sampling.sampling_method=euler sampling.num_steps=50 \
run.output_dir=${RESULTS_FOLDER} run.num_sec_eval=5 \
data.eval_distance.eval_min_dist_cat=-64 data.eval_distance.eval_max_dist_cat=64 data.eval_len_traj_pred=64"

# Define function for Ground-Truth preparation
run_gt() {
    echo "--- Running Ground-Truth Preparation ---"
    if [ "$EVAL_MODE" = "time" ]; then
        python infer.py ${INFER_COMMON} run.eval_type=time run.gt=true
    elif [ "$EVAL_MODE" = "rollout" ]; then
        python infer.py ${INFER_COMMON} run.eval_type=rollout run.gt=true 'run.rollout_fps_values=[4]'
    fi
}

# Define function for Inference
run_infer() {
    echo "--- Running Future Frame Prediction ---"
    if [ "$EVAL_MODE" = "time" ]; then
        python infer.py ${INFER_COMMON} run.eval_type=time run.gt=false
    elif [ "$EVAL_MODE" = "rollout" ]; then
        python infer.py ${INFER_COMMON} run.eval_type=rollout run.gt=false 'run.rollout_fps_values=[4]'
    fi
}

# Define function for Evaluation (evaluate.py remains a plain argparse metrics script)
run_eval() {
    echo "--- Running Metrics Evaluation ---"
    if [ "$EVAL_MODE" = "time" ]; then
        python evaluate.py --datasets ${DATASET} --gt_dir ${RESULTS_FOLDER}/gt --exp_dir ${EXP_DIR} --eval_types time --num_sec_eval 5 --batch_size 16
    elif [ "$EVAL_MODE" = "rollout" ]; then
        python evaluate.py --datasets ${DATASET} --gt_dir ${RESULTS_FOLDER}/gt --exp_dir ${EXP_DIR} --eval_types rollout --rollout_fps_values 4 --input_fps 4 --num_sec_eval 5 --batch_size 16
    fi
}

# Execute based on the STEP argument
case $STEP in
    gt)
        run_gt
        ;;
    infer)
        run_infer
        ;;
    eval)
        run_eval
        ;;
    all)
        run_gt
        run_infer
        run_eval
        ;;
    *)
        echo "Error: Invalid step '${STEP}'. Must be 'gt', 'infer', 'eval', or 'all'."
        exit 1
        ;;
esac

echo "Evaluation pipeline finished successfully!"
