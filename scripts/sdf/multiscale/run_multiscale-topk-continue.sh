#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=multiscale-topk
#SBATCH --output=logs/train_multiscale_topk_%j.out
#SBATCH --error=logs/train_multiscale_topk_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=10:00:00
# =============================================================================
# SLURM Job Configuration for TopK Multiscale GNN (multiscale-topk)
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

MODEL="multiscale-topk"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
TASK="predict_n6d"             # Replace with your specific task
MODE="train"
NTRAIN=4156
BATCH_SIZE=32
NEPOCHS=1000
HIDDEN_DIM=128
NUM_LAYERS=4                   # Must be even for autoencoders (encoder + decoder)
POOL_RATIOS="0.8"          # For depth=3 (num_layers=6), pool_ratios=depth-1=2

# Multiscale-specific parameters
MULTISCALE_N_MLP_HIDDEN_LAYERS=0  # Number of hidden layers in MLP node/edge encoder
MULTISCALE_N_MMP_LAYERS=1         # Number of layers in the Multiscale Message Passing (MMP) module
MULTISCALE_N_MESSAGE_PASSING_LAYERS=2  # Number of message passing layers in the Multiscale GNN

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
    --multiscale_n_message_passing_layers $MULTISCALE_N_MESSAGE_PASSING_LAYERS \
    --pool_ratios $POOL_RATIOS"

# =============================================================================
# Execute the Training
# =============================================================================

# Print the Python command for verification
echo "Running command: $python_command"

# Execute the Python training script
$python_command  --checkpoint /sdf/data/ad/ard/u/tiffan/results/multiscale-topk/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b32_lr0.0001_h128_ly4_pr0.80_ep1000_mlph0_mmply1_mply2/checkpoints/model-349.pth

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
