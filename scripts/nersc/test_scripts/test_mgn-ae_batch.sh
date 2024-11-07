#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/train_mgn_ae_%j.out
#SBATCH --error=logs/train_mgn_ae_%j.err

# =============================================================================
# SLURM Job Configuration for Mesh Graph AutoEncoder (mgn-ae)
# =============================================================================

export SLURM_CPU_BIND="cores"

# Load necessary modules
module load conda
module load cudatoolkit

# Activate the conda environment
source activate ignn

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/global/homes/t/tiffan/repo/accelerator-surrogate

# Navigate to the project directory
cd /global/homes/t/tiffan/repo/accelerator-surrogate

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# =============================================================================
# Define Variables for Training
# =============================================================================

MODEL="mgn-ae"
DATASET="graph_data_filtered_total_charge_51"
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/pscratch/sd/t/tiffan/data/"
BASE_RESULTS_DIR="/pscratch/sd/t/tiffan/results/"
TASK="predict_n6d"
MODE="train"
NTRAIN=4156
NEPOCHS=1
POOL_RATIO="1.0"
DEFAULT_BATCH_SIZE=32
MAX_BATCH_SIZE=512  # Maximum batch size to consider

# =============================================================================
# Iterate Over Different Configurations and Find Max Batch Size
# =============================================================================

declare -A max_batch_sizes  # To store max batch size for each config

for HIDDEN_DIM in 64 128 256 512; do
  for NUM_LAYERS in 4 6; do
    batch_size=$DEFAULT_BATCH_SIZE
    max_reached=false
    
    while [ "$max_reached" = false ] && [ "$batch_size" -le "$MAX_BATCH_SIZE" ]; do
      RESULTS_DIR="${BASE_RESULTS_DIR}/${MODEL}/${DATASET}/${TASK}/h${HIDDEN_DIM}_l${NUM_LAYERS}_b${batch_size}"

      # Construct the Python command with all required arguments
      python_command="python src/graph_models/train.py \
          --model $MODEL \
          --dataset $DATASET \
          --task $TASK \
          --data_keyword $DATA_KEYWORD \
          --base_data_dir $BASE_DATA_DIR \
          --base_results_dir $RESULTS_DIR \
          --mode $MODE \
          --ntrain $NTRAIN \
          --batch_size $batch_size \
          --nepochs $NEPOCHS \
          --hidden_dim $HIDDEN_DIM \
          --num_layers $NUM_LAYERS \
          --pool_ratios $POOL_RATIO"

      echo "Trying batch size $batch_size for hidden_dim=$HIDDEN_DIM, num_layers=$NUM_LAYERS"
      
      # Run training and check for OOM errors
      if $python_command; then
        max_batch_sizes["${HIDDEN_DIM}_${NUM_LAYERS}"]=$batch_size
        batch_size=$((batch_size * 2))
      else
        echo "Batch size $batch_size too large for hidden_dim=$HIDDEN_DIM, num_layers=$NUM_LAYERS"
        max_reached=true
      fi
    done
  done
done

# =============================================================================
# Record and Display the Results
# =============================================================================

echo "=================== Maximum Batch Sizes ==================="
for config in "${!max_batch_sizes[@]}"; do
  echo "Config (hidden_dim, num_layers): $config, Max Batch Size: ${max_batch_sizes[$config]}"
done

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