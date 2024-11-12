#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/run_mgn_accelerate_%j.out
#SBATCH --error=logs/run_mgn_accelerate_%j.err

# =============================================================================
# SLURM Job Configuration for Mesh Graph Net (mgn)
# =============================================================================

export SLURM_CPU_BIND="cores"

# Load necessary modules
module load conda
module load cudatoolkit
module load pytorch/2.3.1

# Activate the conda environment
source activate ignn

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/global/homes/t/tiffan/repo/accelerator-surrogate
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /global/homes/t/tiffan/repo/accelerator-surrogate

# Define constants
MODEL="mgn"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/global/cfs/cdirs/m669/tiffan/data/"
BASE_RESULTS_DIR="/global/cfs/cdirs/m669/tiffan/results/"
TASK="predict_n6d"  # Replace with your specific task
MODE="train"
NTRAIN=4156
NEPOCHS=20
HIDDEN_DIM=256
NUM_LAYERS=6  # Must be even for autoencoders (encoder + decoder)

# Batch sizes to iterate over
BATCH_SIZES=(64 128 256)

# Print the number of nodes and GPUs per node
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "Starting training with batch size: $BATCH_SIZE"

    # Record the start time
    start_time=$(date +%s)

    # Define the command with all required arguments
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

    # Set master address and port for distributed training
    export MASTER_ADDR=$(hostname)
    export MASTER_PORT=29500
    export OMP_NUM_THREADS=32

    # Print the command for verification
    echo "Running command: $python_command"

    # Run the training with accelerate launch
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

    # Record the end time
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Calculate time per epoch
    time_per_epoch=$(echo "scale=2; $duration / $NEPOCHS" | bc)

    # Convert duration to hours, minutes, and seconds
    hours=$((duration / 3600))
    minutes=$(((duration % 3600) / 60))
    seconds=$((duration % 60))

    # Display the total time taken and time per epoch
    echo "Batch Size: $BATCH_SIZE"
    echo "Time taken: ${hours}h ${minutes}m ${seconds}s"
    echo "Average time per epoch: ${time_per_epoch}s"
    echo "===================================="
done
