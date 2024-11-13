import pandas as pd
import argparse

# Function to filter the catalog based on provided criteria
def filter_catalog(catalog_file, norm_x_min=None, norm_x_max=None, norm_y_min=None, norm_y_max=None, 
                   norm_z_min=None, norm_z_max=None, total_charge_min=None, total_charge_max=None):
    # Load the catalog CSV file
    df = pd.read_csv(catalog_file)

    # Apply filters based on the arguments provided
    if norm_x_min is not None:
        df = df[df['norm_emit_x'] >= norm_x_min]
    if norm_x_max is not None:
        df = df[df['norm_emit_x'] <= norm_x_max]
    if norm_y_min is not None:
        df = df[df['norm_emit_y'] >= norm_y_min]
    if norm_y_max is not None:
        df = df[df['norm_emit_y'] <= norm_y_max]
    if norm_z_min is not None:
        df = df[df['norm_emit_z'] >= norm_z_min]
    if norm_z_max is not None:
        df = df[df['norm_emit_z'] <= norm_z_max]
    if total_charge_min is not None:
        df = df[df['total_charge'] >= total_charge_min]
    if total_charge_max is not None:
        df = df[df['total_charge'] <= total_charge_max]

    return df

# Main function to handle argument parsing and call the filter function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter data catalog based on norm x, norm y, norm z, and total charge.")
    
    parser.add_argument('catalog_file', type=str, help="Path to the input data catalog (CSV file)")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the filtered data catalog (CSV file)")

    # Optional filter arguments for norm_emit_x, norm_emit_y, norm_emit_z, and total_charge
    parser.add_argument('--norm_x_min', type=float, help="Minimum value for norm_emit_x")
    parser.add_argument('--norm_x_max', type=float, help="Maximum value for norm_emit_x")
    parser.add_argument('--norm_y_min', type=float, help="Minimum value for norm_emit_y")
    parser.add_argument('--norm_y_max', type=float, help="Maximum value for norm_emit_y")
    parser.add_argument('--norm_z_min', type=float, help="Minimum value for norm_emit_z")
    parser.add_argument('--norm_z_max', type=float, help="Maximum value for norm_emit_z")
    parser.add_argument('--total_charge_min', type=float, help="Minimum value for total_charge")
    parser.add_argument('--total_charge_max', type=float, help="Maximum value for total_charge")

    args = parser.parse_args()

    # Call the filter function with the provided arguments
    filtered_df = filter_catalog(args.catalog_file, args.norm_x_min, args.norm_x_max, args.norm_y_min,
                                 args.norm_y_max, args.norm_z_min, args.norm_z_max, 
                                 args.total_charge_min, args.total_charge_max)

    # Save the filtered DataFrame to the specified output file
    filtered_df.to_csv(args.output_file, index=False)

    print(f"Filtered catalog saved to {args.output_file}")
