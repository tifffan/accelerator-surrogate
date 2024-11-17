#!/bin/bash

# Define array of step pairs
declare -a step_pairs=(
    "0 1"
    "0 2"
    "0 5"
    "20 21"
    "20 22"
    "20 25"
)

# Base job name
BASE_JOB_NAME="mgn_train"

# Create logs directory if it doesn't exist
mkdir -p logs

for step_pair in "${step_pairs[@]}"; do
    # Split the pair into initial and final steps
    read -r initial_step final_step <<< "$step_pair"
    
    # Create a unique job name
    job_name="${BASE_JOB_NAME}_${initial_step}_${final_step}"
    
    # Create the submission script
    cat << EOF > "submit_${job_name}.sh"
#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:30:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/train_mgn_accelerate_%j.out
#SBATCH --error=logs/train_mgn_accelerate_%j.err
#SBATCH --job-name=${job_name}

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

# Navigate to the project directory
cd /global/homes/t/tiffan/repo/accelerator-surrogate

# Record the start time
start_time=\$(date +%s)
echo "Start time: \$(date)"

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

NTRAIN=4156
NEPOCHS=2000
BATCH_SIZE=32
HIDDEN_DIM=256
NUM_LAYERS=6
POOL_RATIOS=1.0

# Learning rate scheduler parameters
LR=1e-4
LR_SCHEDULER="lin"
LIN_START_EPOCH=100
LIN_END_EPOCH=1000
LIN_FINAL_LR=1e-5

# Set specific initial and final steps for this job
INITIAL_STEP=${initial_step}
FINAL_STEP=${final_step}

# =============================================================================
# Construct the Python Command
# =============================================================================

python_command="src/graph_models/step_pair_train_accelerate.py \\
    --model \$MODEL \\
    --dataset \$DATASET \\
    --task \$TASK \\
    --data_keyword \$DATA_KEYWORD \\
    --base_data_dir \$BASE_DATA_DIR \\
    --base_results_dir \$BASE_RESULTS_DIR \\
    --initial_step \$INITIAL_STEP \\
    --final_step \$FINAL_STEP \\
    \$IDENTICAL_SETTINGS \\
    --settings_file \$SETTINGS_FILE \\
    --ntrain \$NTRAIN \\
    --nepochs \$NEPOCHS \\
    --lr \$LR \\
    --batch_size \$BATCH_SIZE \\
    --hidden_dim \$HIDDEN_DIM \\
    --num_layers \$NUM_LAYERS \\
    --pool_ratios \$POOL_RATIOS"

# =============================================================================
# Execute the Training
# =============================================================================

# Set master address and port for distributed training
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=32

srun -l bash -c "
    accelerate launch \\
    --num_machines \\\$SLURM_JOB_NUM_NODES \\
    --main_process_ip \\\$MASTER_ADDR \\
    --main_process_port \\\$MASTER_PORT \\
    --machine_rank \\\$SLURM_PROCID \\
    --num_processes \\\$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \\
    --multi_gpu \\
    \$python_command
"

# =============================================================================
# Record and Display the Duration
# =============================================================================

# Record the end time
end_time=\$(date +%s)
echo "End time: \$(date)"

# Calculate and display duration
duration=\$((end_time - start_time))
hours=\$((duration / 3600))
minutes=\$(( (duration % 3600) / 60 ))
seconds=\$((duration % 60))
echo "Time taken: \${hours}h \${minutes}m \${seconds}s"
EOF

    # Make the submission script executable
    chmod +x "submit_${job_name}.sh"
    
    # Submit the job
    echo "Submitting job for initial_step=${initial_step}, final_step=${final_step}"
    sbatch "submit_${job_name}.sh"
    
    # Add a small delay between submissions
    sleep 2
done