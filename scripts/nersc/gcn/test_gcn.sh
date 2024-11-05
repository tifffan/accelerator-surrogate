#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 3:50:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-task=1

# Bind CPUs to cores for optimal performance
export SLURM_CPU_BIND="cores"

# Load necessary modules
module load conda
module load cudatoolkit

# Activate the conda environment
source activate ignn

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/global/homes/t/tiffan/accelerator-surrogate

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /global/homes/t/tiffan/accelerator-surrogate

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# Define variables for clarity and easy modification
MODEL="gcn"
DATASET="graph_data_filtered_total_charge_51"
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/pscratch/sd/t/tiffan/data/"
BASE_RESULTS_DIR="/pscratch/sd/t/tiffan/results/"
TASK="predict_n6d"             # Replace with your task, e.g., "predict_n6d"
MODE="train"
NTRAIN=128
BATCH_SIZE=4
NEPOCHS=2000
HIDDEN_DIM=32
NUM_LAYERS=4
POOL_RATIOS="1.0 1.0"          # Two pooling ratios for 4 layers (num_layers - 2)

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
    --pool_ratios $POOL_RATIOS"

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
