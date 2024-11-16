#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/train_gcn_seq_%j.out
#SBATCH --error=logs/train_gcn_seq_%j.err

# Bind CPUs to cores for optimal performance
export SLURM_CPU_BIND="cores"

# Load necessary modules
module load conda
module load cudatoolkit
module load pytorch/2.3.1

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

BASE_DATA_DIR="/pscratch/sd/t/tiffan/data/"
BASE_RESULTS_DIR="/global/cfs/cdirs/m669/tiffan/results/"
IDENTICAL_SETTINGS="--identical_settings"

MODEL="gcn"
DATASET="sequence_graph_data_archive_4"
DATA_KEYWORD="knn_k5_weighted"

MODE="train"
INITIAL_STEP=0
FINAL_STEP=10
HORIZON=5
DISCOUNT_FACTOR=0.9

NTRAIN=100
NEPOCHS=1
BATCH_SIZE=64
HIDDEN_DIM=256
NUM_LAYERS=6
POOL_RATIOS=1.0
# VERBOSE="--verbose"

# Learning rate scheduler parameters
LR=1e-4
LR_SCHEDULER="lin"
LIN_START_EPOCH=100
LIN_END_EPOCH=1000
LIN_FINAL_LR=1e-6

# =============================================================================
# Construct the Python Command with All Required Arguments
# =============================================================================

python_command="src/sequence_graph_models/sequence_train_accelerate.py \
    --model $MODEL \
    --dataset $DATASET \
    --data_keyword $DATA_KEYWORD \
    --base_data_dir $BASE_DATA_DIR \
    --base_results_dir $BASE_RESULTS_DIR \
    --initial_step $INITIAL_STEP \
    --final_step $FINAL_STEP \
    --horizon $HORIZON \
    --discount_factor $DISCOUNT_FACTOR \
    $IDENTICAL_SETTINGS \
    --ntrain $NTRAIN \
    --nepochs $NEPOCHS \
    --lr $LR \
    --lr_scheduler $LR_SCHEDULER \
    --lin_start_epoch $LIN_START_EPOCH \
    --lin_end_epoch $LIN_END_EPOCH \
    --lin_final_lr $LIN_FINAL_LR \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --pool_ratios $POOL_RATIOS \
    $VERBOSE"

# =============================================================================
# Execute the Training
# =============================================================================

# Print the command for verification
echo "Running command: $python_command"

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # You can choose any free port
export OMP_NUM_THREADS=8  # Adjust as needed

# Check if sequence_train.py supports accelerate
# If it does, use accelerate launch; otherwise, run the script directly

# For this example, let's assume sequence_train.py supports accelerate
srun -l bash -c "
    accelerate launch \
    --num_machines \$SLURM_JOB_NUM_NODES \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes \$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --multi_gpu \
    $python_command
"

# If sequence_train.py does NOT support accelerate, use the following instead:
# srun -l bash -c "$python_command"

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


# test command:

# python src/sequence_graph_models/sequence_train_accelerate.py   --model gcn   --dataset sequence_graph_data_archive_4   --initial_step 0   --final_step 2   --data_keyword knn_k5_weighted   --base_data_dir /pscratch/sd/t/tiffan/data   --base_results_dir ./results   --ntrain 10   --batch_size 4   --lr 0.001   --hidden_dim 64   --num_layers 3   --discount_factor 0.9   --horizon 2   --nepochs 5
