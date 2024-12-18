import os
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np

# Set global font size for all text elements
plt.rcParams.update({'font.size': 18})  # Increase as needed

def parse_err_files(directory='.'):
    """
    Parses all .err files in the specified directory and extracts initial step,
    final step, and best validation loss from successful runs.

    Args:
        directory (str): The directory to search for .err files. Defaults to current directory.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted data.
    """
    # Initialize a list to store extracted data
    extracted_data = []

    # Define the pattern to extract init and final steps from the filepath
    path_pattern = re.compile(r'init(\d+)_final(\d+)')

    # Define the pattern to extract best validation loss
    loss_pattern = re.compile(r'Best Val Loss:\s*([0-9.eE-]+)')

    # Iterate over all .err files in the directory
    for err_file in glob.glob(os.path.join(directory, '*.err')):
        try:
            with open(err_file, 'r') as file:
                lines = file.readlines()

            # Ensure the file has at least 3 lines to check for success
            if len(lines) < 3:
                print(f"Skipping {err_file}: Not enough lines.")
                continue

            # Extract the last three lines
            last_three = lines[-3:]
            checkpoint_line, training_line, loss_line = last_three

            # Check if the run was successful
            if ("Model checkpoint saved to" in checkpoint_line and
                "Training complete!" in training_line and
                "Best Val Loss" in loss_line):

                # Extract init and final steps from the checkpoint line
                path_match = path_pattern.search(checkpoint_line)
                if not path_match:
                    print(f"Skipping {err_file}: 'initX_finalY' pattern not found.")
                    continue
                init_step = int(path_match.group(1))
                final_step = int(path_match.group(2))

                # Extract best validation loss
                loss_match = loss_pattern.search(loss_line)
                if not loss_match:
                    print(f"Skipping {err_file}: Best Val Loss not found.")
                    continue
                best_val_loss = float(loss_match.group(1))

                # Append the extracted data
                extracted_data.append({
                    'File': os.path.basename(err_file),
                    'Initial Step': init_step,
                    'Final Step': final_step,
                    'Best Val Loss': best_val_loss
                })
            else:
                print(f"Skipping {err_file}: Run was not successful.")

        except Exception as e:
            print(f"Error processing {err_file}: {e}")

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(extracted_data)
    return df

def append_custom_data(df, initial_step, final_step, best_val_loss, file_name='custom_entry'):
    """
    Appends a custom data point to the DataFrame.

    Args:
        df (pd.DataFrame): The existing DataFrame.
        initial_step (int): The initial step value.
        final_step (int): The final step value.
        best_val_loss (float): The best validation loss.
        file_name (str): The name of the file or identifier for the custom entry.
    
    Returns:
        pd.DataFrame: The updated DataFrame with the new entry.
    """
    # Create a dictionary for the new entry
    new_entry = {
        'File': file_name,
        'Initial Step': initial_step,
        'Final Step': final_step,
        'Best Val Loss': best_val_loss
    }

    # Append the new entry to the DataFrame using pd.concat
    new_df = pd.DataFrame([new_entry])
    df = pd.concat([df, new_df], ignore_index=True)
    return df

def visualize_scatter(df, output_path='scatter_plot.png'):
    """
    Visualizes the extracted data on a horizontal scatter plot and saves the figure.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to visualize.
        output_path (str): The file path to save the scatter plot image.
    """
    plt.figure(figsize=(12, 8))  # Increased figure size for better readability
    scatter = plt.scatter(
        df['Initial Step'],
        df['Final Step'],
        c=df['Best Val Loss'],
        cmap='viridis',
        s=200,  # Increased marker size
        edgecolor='k'
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Best Validation Loss', fontsize=18)
    plt.xlabel('Initial Step', fontsize=20)
    plt.ylabel('Final Step', fontsize=20)
    plt.title('Best Validation Loss by Initial and Final Steps', fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {output_path}")

def visualize_curves(df, output_path='curves_plot.png'):
    """
    Visualizes the data by drawing curves connecting initial to final steps.
    The thickness and color of the curves indicate the magnitude of the best validation loss.
    Saves the figure instead of displaying it.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to visualize.
        output_path (str): The file path to save the curves plot image.
    """
    plt.figure(figsize=(12, max(8, len(df) * 0.5)))  # Increased figure size based on data points
    ax = plt.gca()

    # Normalize Best Val Loss for color and line width mapping
    norm = mcolors.Normalize(vmin=df['Best Val Loss'].min(), vmax=df['Best Val Loss'].max())
    cmap = plt.cm.viridis
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Sort the DataFrame to arrange similar val losses together (optional)
    df_sorted = df.sort_values(by='Best Val Loss')

    # Assign y positions to each data point to avoid overlapping
    y_positions = np.arange(1, len(df_sorted) + 1)

    # Iterate over each data point and plot the curve
    for idx, (index, row) in enumerate(df_sorted.iterrows()):
        initial = row['Initial Step']
        final = row['Final Step']
        val_loss = row['Best Val Loss']
        color = scalar_map.to_rgba(val_loss)
        # Map val_loss to line width (e.g., min width 2, max width 6)
        if df['Best Val Loss'].max() != df['Best Val Loss'].min():
            lw = 2 + 4 * (val_loss - df['Best Val Loss'].min()) / (df['Best Val Loss'].max() - df['Best Val Loss'].min())
        else:
            lw = 4  # Default width if all losses are equal

        # Plot a horizontal line connecting initial to final step at the assigned y position
        ax.plot([initial, final], [y_positions[idx], y_positions[idx]], color=color, linewidth=lw, alpha=0.8)

        # Calculate the midpoint for placing the text
        midpoint = (initial + final) / 2
        # Offset the text slightly above the curve
        text_y = y_positions[idx] + 0.1  # Adjust the offset as needed

        # Format the val_loss in scientific notation
        val_loss_sci = f"{val_loss:.2e}"

        # Add the text annotation
        ax.text(midpoint, text_y, val_loss_sci, fontsize=14, ha='center', va='bottom')

    # Create a color bar
    scalar_map.set_array(df['Best Val Loss'])
    cbar = plt.colorbar(scalar_map, ax=ax)
    cbar.set_label('Best Validation Loss', fontsize=18)

    # Set labels and title
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Data Points', fontsize=20)
    plt.title('Validation Loss for Initial-Final Step Pairs', fontsize=24)

    # Hide y-axis ticks as they are arbitrary
    ax.set_yticks([])

    # Set x-axis limits with some padding
    x_min = df['Initial Step'].min() - 10  # Increased padding
    x_max = df['Final Step'].max() + 10
    plt.xlim(x_min, x_max)
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, len(df_sorted) + 1)  # Add 0.5 units of padding above the top row

    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Curves plot saved to {output_path}")

def main():
    """
    Main function to parse .err files, visualize, and save figures.
    """
    # Parse the .err files and extract data
    df = parse_err_files()

    if df.empty:
        print("No successful runs found.")
    else:
        print("Extracted Data:")
        print(df)

    # Append the custom data point (commented out as per user's instruction)
    # initial_step = 5
    # final_step = 76
    # best_val_loss = 3.0853e-04  # Equivalent to 0.00030853
    # df = append_custom_data(df, initial_step, final_step, best_val_loss)

    # print("\nData after appending custom entry:")
    # print(df)

    # Save the updated DataFrame to a CSV file
    output_csv = 'compiled_results_with_custom_entry.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nUpdated dataset saved to {output_csv}")

    # Visualize the extracted data (First Figure: Scatter Plot)
    scatter_output = 'scatter_plot.png'
    visualize_scatter(df, output_path=scatter_output)

    # Visualize the curves (Second Figure: Curves Plot)
    curves_output = 'curves_plot.png'
    visualize_curves(df, output_path=curves_output)

if __name__ == "__main__":
    main()
