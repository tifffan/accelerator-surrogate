#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/train_mgn_accelerate_%j.out
#SBATCH --error=logs/train_mgn_accelerate_%j.err

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

# Define step pairs
step_pairs=(
    "0 1"
    "0 2"
    "0 5"
    "20 21"
    "20 22"
    "20 25"
)

# Loop through each combination of step pairs
for pair in "${step_pairs[@]}"; do
    INITIAL_STEP=$(echo $pair | awk '{print $1}')
    FINAL_STEP=$(echo $pair | awk '{print $2}')
    
    # Construct the Python command with all required arguments
    python_command="src/graph_models/step_pair_train_accelerate.py \
        --model mgn \
        --dataset sequence_graph_data_archive_4 \
        --task predict_n6d \
        --data_keyword knn_k5_weighted \
        --base_data_dir /pscratch/sd/t/tiffan/data/ \
        --base_results_dir /global/cfs/cdirs/m669/tiffan/results/ \
        --initial_step $INITIAL_STEP \
        --final_step $FINAL_STEP \
        --identical_settings \
        --settings_file /pscratch/sd/t/tiffan/data/sequence_graph_data_archive_4/settings.pt \
        --ntrain 4156 \
        --nepochs 2000 \
        --lr 1e-4 \
        --batch_size 32 \
        --hidden_dim 256 \
        --num_layers 6 \
        --pool_ratios 1.0"

    # Print the command for verification
    echo "Submitting job for initial_step=$INITIAL_STEP and final_step=$FINAL_STEP"
    
    # Submit the job with srun
    srun -l bash -c "
        accelerate launch \
        --num_machines \$SLURM_JOB_NUM_NODES \
        --main_process_ip \$MASTER_ADDR \
        --main_process_port 29500 \
        --machine_rank \$SLURM_PROCID \
        --num_processes \$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
        --multi_gpu \
        $python_command
    " &
done

# Wait for all jobs to finish
wait

# Record and Display the Duration
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
