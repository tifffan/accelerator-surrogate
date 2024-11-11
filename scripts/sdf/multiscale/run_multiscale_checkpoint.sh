#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=multiscale
#SBATCH --output=logs/train_multiscale_%j.out
#SBATCH --error=logs/train_multiscale_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=10:00:00

# =============================================================================
# SLURM Job Configuration for Multiscale GNN (multiscale)
# =============================================================================

# Bind CPUs to cores for optimal performance
export SLURM_CPU_BIND="cores"

# # Load necessary modules
# module load conda
# module load cudatoolkit

# # Activate the conda environment
# source activate ignn

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-surrogate

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /sdf/home/t/tiffan/repo/accelerator-surrogate

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# =============================================================================
# Define Variables for Training
# =============================================================================

BASE_DATA_DIR="/sdf/data/ad/ard/u/tiffan/data/"
BASE_RESULTS_DIR="/sdf/data/ad/ard/u/tiffan/results/"

MODEL="multiscale"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
TASK="predict_n6d"             # Replace with your specific task
MODE="train"
NTRAIN=4156
BATCH_SIZE=16
NEPOCHS=3000
HIDDEN_DIM=128
NUM_LAYERS=4                   # Must be even for autoencoders (encoder + decoder)

# Multiscale-specific parameters
MULTISCALE_N_MLP_HIDDEN_LAYERS=2
MULTISCALE_N_MMP_LAYERS=2
MULTISCALE_N_MESSAGE_PASSING_LAYERS=1

# Learning rate scheduler parameters
LR=1e-4
LR_SCHEDULER="lin"
LIN_START_EPOCH=100
LIN_END_EPOCH=1000
LIN_FINAL_LR=1e-5

# Set a random seed for reproducibility
RANDOM_SEED=63

# Checkpoint path
CHECKPOINT="/sdf/data/ad/ard/u/tiffan/results/multiscale/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b16_lr0.0001_h128_ly4_pr1.00_ep2000_sch_lin_100_1000_1e-05_mlph2_mmply2_mply1/checkpoints/model-1999.pth"

# =============================================================================
# Construct the Python Command with All Required Arguments
# =============================================================================

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
    --multiscale_n_mlp_hidden_layers $MULTISCALE_N_MLP_HIDDEN_LAYERS \
    --multiscale_n_mmp_layers $MULTISCALE_N_MMP_LAYERS \
    --multiscale_n_message_passing_layers $MULTISCALE_N_MESSAGE_PASSING_LAYERS
    --lr $LR \
    --lr_scheduler $LR_SCHEDULER \
    --lin_start_epoch $LIN_START_EPOCH \
    --lin_end_epoch $LIN_END_EPOCH \
    --lin_final_lr $LIN_FINAL_LR \
    --random_seed $RANDOM_SEED \
    --checkpoint $CHECKPOINT"

# =============================================================================
# Execute the Training
# =============================================================================

# Print the Python command for verification
echo "Running command: $python_command"

# Execute the Python training script
$python_command

# =============================================================================
# Record and Display the Duration
# =============================================================================

# Record the end time
end_time=$(date +%s)
echo "End time: $(date)"

# Calculate the duration in seconds
duration=$((end_time - start_time))

# Convert duration to hours, minutes, and seconds
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

# Display the total time taken
echo "Time taken: ${hours}h ${minutes}m ${seconds}s"
