# Define variables for clarity and easy modification
MODEL="mgn"
DATASET="graph_data_filtered_total_charge_51"  # Replace with your actual dataset name
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/pscratch/sd/t/tiffan/data/"
BASE_RESULTS_DIR="/pscratch/sd/t/tiffan/results/"
TASK="predict_n6d"             # Replace with your task, e.g., "predict_n6d"
MODE="train"
NTRAIN=128
BATCH_SIZE=4
NEPOCHS=200
HIDDEN_DIM=32
NUM_LAYERS=4
POOL_RATIOS="1.0 1.0"          # Two pooling ratios for 4 layers (num_layers - 2)

python_command="python src/graph_models/train.py \
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
    --num_layers $NUM_LAYERS \
    --pool_ratios $POOL_RATIOS"

# Print the Python command for verification
echo "Running command: $python_command"

# Execute the Python training script
$python_command