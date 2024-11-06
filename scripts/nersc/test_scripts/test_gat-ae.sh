#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/train_gat_ae_%j.out
#SBATCH --error=logs/train_gat_ae_%j.err

# =============================================================================
# SLURM Job Configuration for Graph Attention AutoEncoder (gat-ae)
# =============================================================================

# Bind CPUs to cores for optimal performance
export SLURM_CPU_BIND="cores"

# Load necessary modules
module load conda
module load cudatoolkit

# Activate the conda environment
source activate ignn

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/global/homes/t/tiffan/repo/accelerator-surrogate

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /global/homes/t/tiffan/repo/accelerator-surrogate

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# =============================================================================
# Define Variables for Training
# =============================================================================

MODEL="gat-ae"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/pscratch/sd/t/tiffan/data/"
BASE_RESULTS_DIR="/pscratch/sd/t/tiffan/results/"
TASK="predict_n6d"             # Replace with your specific task
MODE="train"
NTRAIN=128
BATCH_SIZE=4
NEPOCHS=200
HIDDEN_DIM=32
NUM_LAYERS=6                   # Must be even for autoencoders (encoder + decoder)
POOL_RATIOS="0.7 0.5"          # For depth=3 (num_layers=6), pool_ratios=depth-1=2

# Additional hyperparameters for GAT
GAT_HEADS=4                    # Number of attention heads

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
    --pool_ratios $POOL_RATIOS \
    --gat_heads $GAT_HEADS"

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
