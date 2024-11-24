process_archive_folder
    # python process_archive_folder.py --archive_dir /sdf/data/ad/ard/u/tiffan/Archive_5 --output_file A5_n241_match.csv

plot_histograms_from_catalog
    # python plot_histograms_from_catalog.py --catalog_file Archive_5_n241_match.csv --output_dir ./histograms/Archive5/ --cleaned_catalog_file Archive_5_n241_match_cleaned.csv --default_total_charge 0.5e-9
    
/sdf/data/ad/ard/u/tiffan/Archive_0_n241_match.csv

filter_catalog

extract_sequence_particles

generate_data_catalog

compute_sequence_global_statistics

generate_graphs_from_sequence_particles

todo:

graph dataset with specified step as input and output -> done

sequence graph dataset -> done

sequence or autoregressive dataloader and trainer -> done

todo:

modify step pair dataset to take in settings

also check if setting is normalized or not

sequence data has lots of extra print statement

