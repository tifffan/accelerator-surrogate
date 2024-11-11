#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=mgn
#SBATCH --output=logs/train_gcn_%j.out
#SBATCH --error=logs/train_gcn_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
# =============================================================================
# SLURM Job Configuration for Mesh Graph Net (mgn)
# =============================================================================

# Bind CPUs to cores for optimal performance
export SLURM_CPU_BIND="cores"

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-surrogate

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /sdf/home/t/tiffan/repo/accelerator-surrogate

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# Define variables for clarity and easy modification
BASE_DATA_DIR="/sdf/data/ad/ard/u/tiffan/data/"
BASE_RESULTS_DIR="/sdf/data/ad/ard/u/tiffan/results/"

MODEL="gcn"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
TASK="predict_n6d"             # Replace with your specific task
MODE="train"
NTRAIN=4156
BATCH_SIZE=32
NEPOCHS=2000
HIDDEN_DIM=256
NUM_LAYERS=6                   # Must be even for autoencoders (encoder + decoder)

# Learning rate scheduler parameters
LR=1e-4
LR_SCHEDULER="lin"
LIN_START_EPOCH=100
LIN_END_EPOCH=1000
LIN_FINAL_LR=1e-5

# Random seed
RANDOM_SEED=63

# Construct the Python command with all required arguments
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
    --random_seed $RANDOM_SEED"

# Print the Python command for verification
echo "Running command: $python_command"

# Execute the Python training script
$python_command

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
