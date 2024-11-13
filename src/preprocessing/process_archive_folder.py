# import glob
# import numpy as np
# import pandas as pd
# import argparse
# import os
# from impact import Impact  # Assuming this is the correct import for the Impact class

# # Define the function to calculate various statistics for a given file
# def calc_N(file, settings):
#     I = Impact()
#     no_exceptions = True
#     try:
#         I.load_archive(file)
#     except Exception as e:
#         print(f"Error loading archive for file {file}: {e}")
#         no_exceptions = False
#         return float('NaN'), float('NaN'), float('NaN'), {}, float('NaN'), float('NaN'), float('NaN'), {}, no_exceptions

#     s = {}
#     try:
#         N = len(I['particles']['PR10571']['x']) if 'PR10571' in I['particles'] else 0
#         stdx = np.std(I['particles']['PR10571']['x']) if 'PR10571' in I['particles'] else float('NaN')
#         stdy = np.std(I['particles']['PR10571']['y']) if 'PR10571' in I['particles'] else float('NaN')
#         normemit = I['output']['stats']['norm_emit_x'][-1] if 'norm_emit_x' in I['output']['stats'] else float('NaN')
#         t = I['output']['run_info']['start_time'] if 'start_time' in I['output']['run_info'] else float('NaN')
#     except Exception as e:
#         print(f"Error processing main data for file {file}: {e}")
#         no_exceptions = False
#         N = 0
#         stdx = float('NaN')
#         stdy = float('NaN')
#         normemit = float('NaN')
#         t = float('NaN')

#     init = {}
#     try:
#         init['stdx'] = np.std(I['particles']['initial_particles']['x']) if 'initial_particles' in I['particles'] and 'x' in I['particles']['initial_particles'] else float('NaN')
#         init['stdy'] = np.std(I['particles']['initial_particles']['y']) if 'initial_particles' in I['particles'] and 'y' in I['particles']['initial_particles'] else float('NaN')
#         init['stdt'] = np.std(I['particles']['initial_particles']['t']) if 'initial_particles' in I['particles'] and 't' in I['particles']['initial_particles'] else float('NaN')
#         init['stdpx'] = np.std(I['particles']['initial_particles']['px']) if 'initial_particles' in I['particles'] and 'px' in I['particles']['initial_particles'] else float('NaN')
#         init['stdpy'] = np.std(I['particles']['initial_particles']['py']) if 'initial_particles' in I['particles'] and 'py' in I['particles']['initial_particles'] else float('NaN')
#         init['stdpz'] = np.std(I['particles']['initial_particles']['pz']) if 'initial_particles' in I['particles'] and 'pz' in I['particles']['initial_particles'] else float('NaN')
#     except Exception as e:
#         print(f"Error processing initial particles data for file {file}: {e}")
#         no_exceptions = False
#         init['stdx'] = float('NaN')
#         init['stdy'] = float('NaN')
#         init['stdt'] = float('NaN')
#         init['stdpx'] = float('NaN')
#         init['stdpy'] = float('NaN')
#         init['stdpz'] = float('NaN')

#     try:
#         N241 = len(I['particles']['PR10241']['x']) if 'PR10241' in I['particles'] else 0
#     except Exception as e:
#         print(f"Error processing PR10241 data for file {file}: {e}")
#         no_exceptions = False
#         N241 = 0

#     for setting in settings:
#         try:
#             s[setting] = I[setting]
#         except KeyError:
#             s[setting] = float('NaN')

#     try:
#         s['charge'] = I['particles']['PR10571']['weight'][0] * 2e5 if 'PR10571' in I['particles'] else np.sum(I['initial_particles']['weight'])
#     except Exception as e:
#         print(f"Error calculating charge for file {file}: {e}")
#         no_exceptions = False
#         s['charge'] = float('NaN')

#     return N, stdx, stdy, s, N241, normemit, t, init, no_exceptions

# def process_archive_folder(data_dir, output_file):
#     # Define a list of settings to be extracted from the data
#     settings = [
#         'SOL10111:solenoid_field_scale',
#         'CQ10121:b1_gradient',
#         'SQ10122:b1_gradient',
#         'GUNF:rf_field_scale',
#         'GUNF:theta0_deg',
#         'distgen:total_charge'
#     ]

#     # Use glob to find all files matching the pattern in the specified directory
#     files = glob.glob(data_dir + '/*')  # Make sure to append /* to match files
#     print(f"Number of files found: {len(files)}")

#     # Filter out directories and keep only files
#     files = [f for f in files if os.path.isfile(f)]

#     valid_files_n241_match = []
#     total_files = len(files)
    
#     for idx, file in enumerate(files, start=1):
#         N, stdx, stdy, s, N241, normemit, t, init, no_exceptions = calc_N(file, settings)
#         if no_exceptions and N241 == N:
#             valid_files_n241_match.append(file)
#         print(f"Processed file {idx}/{total_files}")

#     print(f"Number of valid files with N241 == N: {len(valid_files_n241_match)}")

#     # Save the list of valid filenames with N241 == N to a CSV file
#     df_valid_n241_match = pd.DataFrame(valid_files_n241_match, columns=['filename'])
#     df_valid_n241_match.to_csv(output_file, index=False)
#     print(f"Data catalog saved to {output_file}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process archive folder and filter by N241.")
#     parser.add_argument('--archive_dir', type=str, help="Path to the archive folder")
#     parser.add_argument('--output_file', type=str, help="Output file for filtered data catalog")

#     args = parser.parse_args()

#     # Process the archive folder
#     process_archive_folder(args.archive_dir, args.output_file)


import glob
import numpy as np
import pandas as pd
import argparse
import os
from impact import Impact  # Assuming this is the correct import for the Impact class

def calc_N(file):
    I = Impact()
    no_exceptions = True
    try:
        I.load_archive(file)
    except Exception as e:
        print(f"Error loading archive for file {file}: {e}")
        no_exceptions = False
        return 0, 0, no_exceptions

    try:
        N = len(I['particles']['initial_particles']['x']) if 'initial_particles' in I['particles'] else 0
        N241 = len(I['particles']['PR10241']['x']) if 'PR10241' in I['particles'] else 0
    except Exception as e:
        print(f"Error processing particle data for file {file}: {e}")
        no_exceptions = False
        N = 0
        N241 = 0

    return N, N241, no_exceptions

def process_archive_folder(data_dir, output_file):
    files = glob.glob(os.path.join(data_dir, '*'))
    files = [f for f in files if os.path.isfile(f)]

    valid_files_n241_match = []
    total_files = len(files)

    for idx, file in enumerate(files, start=1):
        N, N241, no_exceptions = calc_N(file)
        if no_exceptions and N241 == N:
            valid_files_n241_match.append(file)
        print(f"Processed file {idx}/{total_files}")

    print(f"Number of valid files with N241 == N: {len(valid_files_n241_match)}")

    df_valid_n241_match = pd.DataFrame(valid_files_n241_match, columns=['filename'])
    df_valid_n241_match.to_csv(output_file, index=False)
    print(f"Data catalog saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process archive folder and filter by N241.")
    parser.add_argument('--archive_dir', type=str, help="Path to the archive folder")
    parser.add_argument('--output_file', type=str, help="Output file for filtered data catalog")

    args = parser.parse_args()

    process_archive_folder(args.archive_dir, args.output_file)
