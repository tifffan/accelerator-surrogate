#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=gtr
#SBATCH --output=logs/train_gtr_%j.out
#SBATCH --error=logs/train_gtr_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
# =============================================================================
# SLURM Job Configuration for Graph Transformer (gtr)
# =============================================================================

export SLURM_CPU_BIND="cores"
export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-surrogate

echo "PYTHONPATH is set to: $PYTHONPATH"
cd /sdf/home/t/tiffan/repo/accelerator-surrogate

start_time=$(date +%s)
echo "Start time: $(date)"

BASE_DATA_DIR="/sdf/data/ad/ard/u/tiffan/data/"
BASE_RESULTS_DIR="/sdf/data/ad/ard/u/tiffan/results/"

MODEL="gtr"
DATASET="graph_data_filtered_total_charge_51"
DATA_KEYWORD="knn_k5_weighted"
TASK="predict_n6d"
MODE="train"
NTRAIN=4156
BATCH_SIZE=32
NEPOCHS=2000
HIDDEN_DIM=256
NUM_LAYERS=6

LR=1e-4
LR_SCHEDULER="lin"
LIN_START_EPOCH=100
LIN_END_EPOCH=1000
LIN_FINAL_LR=1e-5
GTR_HEADS=4
GTR_CONCAT=True
GTR_DROPOUT=0.1

# CHECKPOINT=None
CHECKPOINT="/sdf/data/ad/ard/u/tiffan/results/gtr/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b32_lr0.0001_h256_ly6_pr1.00_ep2000_sch_lin_100_1000_1e-05_heads4_concatTrue_dropout0.1/checkpoints/model-249.pth"

RANDOM_SEED=63

python_command="python src/graph_models/train.py \
    --model $MODEL \
    --dataset $DATASET \
    --task $TASK \
    --data_keyword $DATA_KEYWORD \
    --base_data_dir $BASE_DATA_DIR \
    --base_results_dir $BASE_RESULTS_DIR \
    --mode $MODE \
    --ntrain $NTRAIN \
    --batch_size $BATCH_SIZE \
    --nepochs $NEPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --lr $LR \
    --lr_scheduler $LR_SCHEDULER \
    --lin_start_epoch $LIN_START_EPOCH \
    --lin_end_epoch $LIN_END_EPOCH \
    --lin_final_lr $LIN_FINAL_LR \
    --gtr_heads $GTR_HEADS \
    --gtr_concat $GTR_CONCAT \
    --gtr_dropout $GTR_DROPOUT \
    --random_seed $RANDOM_SEED \
    --checkpoint $CHECKPOINT"

echo "Running command: $python_command"
$python_command

end_time=$(date +%s)
echo "End time: $(date)"

duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "Time taken: ${hours}h ${minutes}m ${seconds}s"
