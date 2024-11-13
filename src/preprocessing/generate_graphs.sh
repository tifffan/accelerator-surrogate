# environment: 
module load conda
conda activate ignn

cd /global/homes/t/tiffan/slac-point/data/datasets

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt \
# --subsample_size 128  --k 5  --edge_method knn  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_test

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt \
# --subsample_size 128  --k 10  --edge_method knn  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_test

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt \
# --subsample_size 128  --k 20  --edge_method knn  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_test

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt \
# --subsample_size 128  --k 50  --edge_method knn  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_test

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt \
# --subsample_size 128  --distance_threshold 0.5  --edge_method dist  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_test

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt \
# --subsample_size 128  --distance_threshold 1.0  --edge_method dist  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_test

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt \
# --subsample_size 128  --distance_threshold 2.0  --edge_method dist  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_test

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt \
# --subsample_size 128  --edge_method dist  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_test


# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics_filtered_total_charge_51.txt \
# --subsample_size 4156  --edge_method knn --k 5  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_filtered_total_charge_51

# python generate_graphs_from_point_clouds.py  \
# --data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog.csv \
# --statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics_filtered_total_charge_51.txt \
# --subsample_size 4156  --edge_method knn --k 10  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_filtered_total_charge_51

python generate_graphs_from_point_clouds.py  \
--data_catalog /global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog.csv \
--statistics_file /global/homes/t/tiffan/slac-point/data/catalogs/global_statistics_filtered_total_charge_51.txt \
--subsample_size 4156  --edge_method dist --distance_threshold 1.0  --weighted_edge  --output_base_dir /pscratch/sd/t/tiffan/data/graph_data_filtered_total_charge_51