#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=mgn-ae_test
#SBATCH --output=logs/train_mgn_ae_%j.out
#SBATCH --error=logs/train_mgn_ae_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=10:00:00
# =============================================================================
# SLURM Job Configuration for Mesh Graph AutoEncoder (mgn-ae)
# =============================================================================

# Bind CPUs to cores for optimal performance
export SLURM_CPU_BIND="cores"

# Load necessary modules
module load conda
module load cudatoolkit

# Activate the conda environment
source activate ignn

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

MODEL="mgn-ae"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/sdf/data/ad/ard/u/tiffan/data/"
BASE_RESULTS_DIR="/sdf/data/ad/ard/u/tiffan/results/"
TASK="predict_n6d"             # Replace with your specific task
MODE="train"
NTRAIN=128
BATCH_SIZE=4
NEPOCHS=200
HIDDEN_DIM=32
NUM_LAYERS=6                   # Must be even for autoencoders (encoder + decoder)
POOL_RATIOS="0.7 0.5"          # For depth=3 (num_layers=6), pool_ratios=depth-1=2

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
    --pool_ratios $POOL_RATIOS"

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
