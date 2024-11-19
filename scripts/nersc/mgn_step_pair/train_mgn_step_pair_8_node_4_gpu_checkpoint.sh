#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/train_mgn_accelerate_8_4_%j.out
#SBATCH --error=logs/train_mgn_accelerate_8_4_%j.err

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
SETTINGS_FILE="/pscratch/sd/t/tiffan/data/sequence_graph_data_archive_4/settings.pt"
IDENTICAL_SETTINGS="--identical_settings"

MODEL="mgn"
DATASET="sequence_graph_data_archive_4"
DATA_KEYWORD="knn_k5_weighted"
TASK="predict_n6d"
MODE="train"

INITIAL_STEP=20
FINAL_STEP=21

NTRAIN=4156
NEPOCHS=2000
BATCH_SIZE=1
HIDDEN_DIM=256
NUM_LAYERS=6
POOL_RATIOS=1.0
VERBOSE="--verbose"

# Learning rate scheduler parameters
LR=1e-4
LR_SCHEDULER="lin"
LIN_START_EPOCH=3200
LIN_END_EPOCH=32000
LIN_FINAL_LR=1e-5

# LR=1e-4
# LR_SCHEDULER="lin"
# LIN_START_EPOCH=100
# LIN_END_EPOCH=1000
# LIN_FINAL_LR=1e-5

# Random seed
RANDOM_SEED=63

# Checkpoint path
CHECKPOINT="/global/cfs/cdirs/m669/tiffan/results/mgn/sequence_graph_data_archive_4/predict_n6d_init20_final21/knn_k5_weighted_r63_nt4156_b1_lr0.0001_h256_ly6_pr1.00_ep2000_sch_lin_3200_32000_1e-05/checkpoints/model-439.pth"

# =============================================================================
# Construct the Python Command with All Required Arguments
# =============================================================================

python_command="src/graph_models/step_pair_train_accelerate.py \
    --model $MODEL \
    --dataset $DATASET \
    --task $TASK \
    --data_keyword $DATA_KEYWORD \
    --base_data_dir $BASE_DATA_DIR \
    --base_results_dir $BASE_RESULTS_DIR \
    --initial_step $INITIAL_STEP \
    --final_step $FINAL_STEP \
    $IDENTICAL_SETTINGS \
    --settings_file $SETTINGS_FILE \
    --ntrain $NTRAIN \
    --nepochs $NEPOCHS \
    --lr $LR \
    --lr_scheduler $LR_SCHEDULER \
    --lin_start_epoch $LIN_START_EPOCH \
    --lin_end_epoch $LIN_END_EPOCH \
    --lin_final_lr $LIN_FINAL_LR \
    --random_seed $RANDOM_SEED \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --pool_ratios $POOL_RATIOS \
    $VERBOSE \
    --checkpoint $CHECKPOINT"

# =============================================================================
# Execute the Training
# =============================================================================

# Print the command for verification
echo "Running command: $python_command"

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # You can choose any free port
export OMP_NUM_THREADS=32  # Adjust as needed

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
