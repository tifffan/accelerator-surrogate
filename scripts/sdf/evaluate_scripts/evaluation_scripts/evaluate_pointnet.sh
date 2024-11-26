#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=evaluate_pn0
#SBATCH --output=logs/evaluate_pn0_%j.out
#SBATCH --error=logs/evaluate_pn0_%j.err
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

python src/graph_models/evaluate_checkpoint_3.py  --checkpoint "/sdf/data/ad/ard/u/tiffan/points_results/pn0/hd64_nl4_bs1_lr0.01_wd0.0001_ep2000_r63/checkpoints/model-1449.pth" --results_folder evaluation_results/pn0

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


