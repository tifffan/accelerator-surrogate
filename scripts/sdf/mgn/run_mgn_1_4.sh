#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=run_mgn_1_4
#SBATCH --output=logs/run_mgn_1_4_%j.out
#SBATCH --error=logs/run_mgn_1_4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=2:30:00

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

MODEL="mgn"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
TASK="predict_n6d"             # Replace with your specific task
MODE="train"
NTRAIN=3324
NVAL=416
NTEST=416
BATCH_SIZE=16
NEPOCHS=1000
HIDDEN_DIM=256
NUM_LAYERS=6                   # Must be even for autoencoders (encoder + decoder)

# Learning rate scheduler parameters
WD=5e-5
LR=1e-3
LR_SCHEDULER="lin"
LIN_START_EPOCH=10
LIN_END_EPOCH=1000
LIN_FINAL_LR=1e-5

# Random seed for reproducibility
RANDOM_SEED=63

# =============================================================================
# New Variables for Testing Preload and Edge Attribute Method
# =============================================================================

# Define the edge attribute computation method (choose from 'v0', 'v0n', 'v1', 'v1n', 'v2', 'v2n', 'v3')
EDGE_ATTR_METHOD="v3"  # Example: using 'v1' is default

# Flag to preload data into memory (set to "--preload_data" to enable, leave empty to disable)
PRELOAD_DATA_FLAG="--preload_data"  # To enable preloading
# PRELOAD_DATA_FLAG=""  # To disable preloading

# =============================================================================
# Construct the Python Command with All Required Arguments
# =============================================================================

python_command="src/graph_models/train.py \
    --model $MODEL \
    --dataset $DATASET \
    --task $TASK \
    --data_keyword $DATA_KEYWORD \
    --base_data_dir $BASE_DATA_DIR \
    --base_results_dir $BASE_RESULTS_DIR \
    --mode $MODE \
    --n_train $NTRAIN \
    --n_val $NVAL \
    --n_val $NTEST \
    --batch_size $BATCH_SIZE \
    --nepochs $NEPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --wd $WD \
    --lr $LR \
    --lr_scheduler $LR_SCHEDULER \
    --lin_start_epoch $((LIN_START_EPOCH * SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --lin_end_epoch $((LIN_END_EPOCH * SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --lin_final_lr $LIN_FINAL_LR \
    --random_seed $RANDOM_SEED \
    --edge_attr_method $EDGE_ATTR_METHOD \
    $PRELOAD_DATA_FLAG"

# =============================================================================
# Execute the Training
# =============================================================================

# Print the command for verification
echo "Running command: $python_command"

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # You can choose any free port
export OMP_NUM_THREADS=4  # Adjust as needed

# Use accelerate launch with srun
srun -l bash -c "
    accelerate launch \
    --num_machines $SLURM_JOB_NUM_NODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --multi_gpu \
    $python_command
"

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
