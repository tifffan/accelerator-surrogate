python src/graph_models/sequence_train.py \
    --model gcn \
    --dataset sequence_graph_data_archive_4\
    --data_keyword knn_edges_k5_weighted \
    --task predict_n6d \
    --initial_step 0 \
    --final_step 1 \
    --identical_settings \
    --settings_file /sdf/data/ad/ard/u/tiffan/data/sequence_particles_data_archive_4/settings.pt \
    --ntrain 100 \
    --nepochs 10 \
    --lr 0.001 \
    --batch_size 8 \
    --hidden_dim 64 \
    --num_layers 3 \
    --pool_ratios 1.0 \
    --verbose