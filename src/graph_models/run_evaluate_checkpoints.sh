python evaluate_checkpoint_2.py --checkpoint /sdf/data/ad/ard/u/tiffan/results/gcn/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b4_lr0.0001_h256_ly6_pr1.00_ep3000_sch_lin_400_4000_1e-05/checkpoints/model-2999.pth --results_folder evaluation_results/gcn/ --subsample_size 4156

python evaluate_checkpoint_2.py --checkpoint /sdf/data/ad/ard/u/tiffan/results/gtr/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b4_lr0.0001_h256_ly6_pr1.00_ep3000_sch_lin_400_4000_1e-05_heads4_concatTrue_dropout0.1/checkpoints/model-2999.pth --results_folder evaluation_results/gtr/  --subsample_size 4156

python evaluate_checkpoint_2.py --checkpoint /sdf/data/ad/ard/u/tiffan/results/mgn/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b16_lr0.001_h256_ly6_pr1.00_ep3000_sch_lin_40_4000_1e-05/checkpoints/model-2999.pth --results_folder evaluation_results/mgn/  --subsample_size 4156

python evaluate_checkpoint_2.py --checkpoint /sdf/data/ad/ard/u/tiffan/results/multiscale/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b16_lr0.001_h128_ly6_pr1.00_ep3000_sch_lin_10_1000_1e-05_mlph2_mmply2_mply2/checkpoints/model-2999.pth --results_folder evaluation_results/multiscale/  --subsample_size 4156

python evaluate_checkpoint_3.py  --checkpoint "/sdf/data/ad/ard/u/tiffan/points_results/pn0/hd64_nl4_bs1_lr0.01_wd0.0001_ep2000_r63/checkpoints/model-1449.pth" --results_folder evaluation_results/pn0/  --subsample_size 4156



python evaluate_checkpoint_2.py --checkpoint /sdf/data/ad/ard/u/tiffan/results/gcn/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b4_lr0.0001_h256_ly6_pr1.00_ep3000_sch_lin_400_4000_1e-05/checkpoints/model-2999.pth --results_folder evaluation_results/gcn/

python evaluate_checkpoint_2.py --checkpoint /sdf/data/ad/ard/u/tiffan/results/gtr/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b4_lr0.0001_h256_ly6_pr1.00_ep3000_sch_lin_400_4000_1e-05_heads4_concatTrue_dropout0.1/checkpoints/model-2999.pth --results_folder evaluation_results/gtr/

python evaluate_checkpoint_2.py --checkpoint /sdf/data/ad/ard/u/tiffan/results/multiscale/graph_data_filtered_total_charge_51/predict_n6d/knn_k5_weighted_r63_nt4156_b16_lr0.001_h128_ly6_pr1.00_ep3000_sch_lin_10_1000_1e-05_mlph2_mmply2_mply2/checkpoints/model-2999.pth --results_folder evaluation_results/multiscale/

python evaluate_checkpoint_3.py  --checkpoint "/sdf/data/ad/ard/u/tiffan/points_results/pn0/hd64_nl4_bs1_lr0.01_wd0.0001_ep2000_r63/checkpoints/model-1449.pth" --results_folder evaluation_results/pn0/
