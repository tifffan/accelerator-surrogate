#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=run_pointnet_training
#SBATCH --output=logs/run_pointnet_%j.out
#SBATCH --error=logs/run_pointnet_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=33:30:00

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

# Data paths
DATA_CATALOG="src/points_models/catalogs/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog_train_sdf.csv"
STATISTICS_FILE="src/points_models/catalogs/global_statistics_filtered_total_charge_51_train.txt"

# Training parameters
BATCH_SIZE=16
N_TRAIN=4156
N_VAL=0
N_TEST=0
RANDOM_SEED=123

NUM_EPOCHS=10
HIDDEN_DIM=64
NUM_LAYERS=3

LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4

# Model and results
MODEL="PointNet1"
BASE_RESULTS_DIR="/sdf/data/ad/ard/u/tiffan/points_results/"
CHECKPOINT=""  # Path to checkpoint if resuming training; leave empty if starting fresh

# Learning rate scheduler parameters
LR_SCHEDULER="lin"  # Options: 'exp', 'lin', or None
# EXP_DECAY_RATE=0.001
# EXP_START_EPOCH=0
LIN_START_EPOCH=10
LIN_END_EPOCH=100
LIN_FINAL_LR=1e-5

# Verbose output
VERBOSE="--verbose"

# WandB settings
# WANDB_PROJECT="points-training"
# WANDB_RUN_NAME="pointnet_run"

# =============================================================================
# Construct the Python Command with All Required Arguments
# =============================================================================

python_command="src/points_models/train.py \
    --data_catalog $DATA_CATALOG \
    --statistics_file $STATISTICS_FILE \
    --batch_size $BATCH_SIZE \
    --n_train $N_TRAIN \
    --n_val $N_VAL \
    --n_test $N_TEST \
    --random_seed $RANDOM_SEED \
    --num_epochs $NUM_EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --model $MODEL \
    --base_results_dir $BASE_RESULTS_DIR \
    $VERBOSE"

# Add checkpoint argument if provided
if [ -n "$CHECKPOINT" ]; then
    python_command="$python_command --checkpoint $CHECKPOINT"
fi

# Add scheduler arguments if scheduler is used
if [ "$LR_SCHEDULER" = "exp" ]; then
    python_command="$python_command \
        --lr_scheduler $LR_SCHEDULER \
        --exp_decay_rate $EXP_DECAY_RATE \
        --exp_start_epoch $EXP_START_EPOCH"
elif [ "$LR_SCHEDULER" = "lin" ]; then
    python_command="$python_command \
        --lr_scheduler $LR_SCHEDULER \
        --lin_start_epoch $LIN_START_EPOCH \
        --lin_end_epoch $LIN_END_EPOCH \
        --lin_final_lr $LIN_FINAL_LR"
fi

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
