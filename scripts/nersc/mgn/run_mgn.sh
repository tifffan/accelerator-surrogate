#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/train_mgn_%j.out
#SBATCH --error=logs/train_mgn_%j.err

# =============================================================================
# SLURM Job Configuration for Mesh Graph Net (mgn)
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

python -m pip show accelerate

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# =============================================================================
# Define Variables for Training
# =============================================================================

MODEL="mgn"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/global/cfs/cdirs/m669/tiffan/data/"
BASE_RESULTS_DIR="/global/cfs/cdirs/m669/tiffan/results/"
TASK="predict_n6d"             # Replace with your specific task
MODE="train"
NTRAIN=4156
BATCH_SIZE=32
NEPOCHS=10
HIDDEN_DIM=256
NUM_LAYERS=6                   # Must be even for autoencoders (encoder + decoder)

# =============================================================================
# Construct the Command with All Required Arguments
# =============================================================================

python_command="src/graph_models/train_accelerate.py \
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
    --num_layers $NUM_LAYERS"

# =============================================================================
# Execute the Training with Accelerate
# =============================================================================

# Print the command for verification
echo "Running command: $python_command"

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # You can choose any free port
export OMP_NUM_THREADS=8  # Adjust as needed

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
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

# Display the total time taken
echo "Time taken: ${hours}h ${minutes}m ${seconds}s"