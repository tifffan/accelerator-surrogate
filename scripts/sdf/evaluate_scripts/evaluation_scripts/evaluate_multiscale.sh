#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=evaluate_multiscale
#SBATCH --output=logs/evaluate_multiscale_%j.out
#SBATCH --error=logs/evaluate_multiscale_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00:30:00

# =============================================================================
# SLURM Job Configuration for Model Evaluation
# =============================================================================

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
# Define Variables for Evaluation
# =============================================================================

# Specify the checkpoint path
CHECKPOINT="/sdf/data/ad/ard/u/tiffan/results/multiscale/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b16_lr0.001_h128_ly6_pr1.00_ep3000_sch_lin_10_1000_1e-05_mlph2_mmply2_mply2/checkpoints/model-2999.pth"

# Specify the results folder
RESULTS_FOLDER="evaluation_results/multiscale/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b16_lr0.001_h128_ly6_pr1.00_ep3000_sch_lin_10_1000_1e-05_mlph2_mmply2_mply2/model-2999/"

# =============================================================================
# Construct the Python Command with All Required Arguments
# =============================================================================

python_command="python src/graph_models/evaluate_checkpoint.py \
    --checkpoint $CHECKPOINT \
    --results_folder $RESULTS_FOLDER"

# =============================================================================
# Execute the Evaluation
# =============================================================================

# Print the Python command for verification
echo "Running command: $python_command"

# Execute the Python evaluation script
$python_command

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
