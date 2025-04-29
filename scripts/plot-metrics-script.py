#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from matplotlib.ticker import FuncFormatter

# Set style for professional-looking plots
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Create a results directory if it doesn't exist
if not os.path.exists('add_q_plots'):
    os.makedirs('add_q_plots')

# Function to format y-axis to show MB with one decimal place
def mbytes_formatter(x, pos):
    return f'{x:.1f} MB'

# Function to create a single plot
def create_plot(df, x_col, y_col, title, x_label, y_label, color, filename,
                log_scale=False, formatter=None, marker='o'):
    plt.figure(figsize=(12, 8))

    # Create plot with gradient effect
    if log_scale:
        plt.semilogy(df[x_col], df[y_col], marker=marker, linestyle='-',
                   color=color, markersize=8, linewidth=2.5, alpha=0.9)
    else:
        plt.plot(df[x_col], df[y_col], marker=marker, linestyle='-',
                 color=color, markersize=8, linewidth=2.5, alpha=0.9)

    # Fill area under the curve with gradient
    plt.fill_between(df[x_col], df[y_col], color=color, alpha=0.2)

    # Add polynomial trendline
    if len(df) > 3:  # Need at least 4 points for cubic
        try:
            z = np.polyfit(df[x_col], df[y_col], 3)
            p = np.poly1d(z)
            x_trend = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            plt.plot(x_trend, p(x_trend), '--', color='#444444', linewidth=1.5,
                     alpha=0.7, label='Trend')
            plt.legend()
        except:
            pass  # Skip trendline if fitting fails

    # Add grid and customize appearance
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(title, fontweight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Apply custom formatter if provided
    if formatter:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(formatter))

    # Add data labels for start, middle and end points
    indices = [0, len(df)//2, -1]
    for i in indices:
        plt.annotate(f'{df[y_col].iloc[i]:.2f}',
                     xy=(df[x_col].iloc[i], df[y_col].iloc[i]),
                     xytext=(5, 10), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2',
                                    color='#444444'),
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # Customize plot appearance
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to create a combined plot for memory and q-values
def create_combined_memory_q_plot(memory_df, q_df, title, filename):
    # Create a combined convergence plot
    fig, ax1 = plt.figure(figsize=(14, 10)), plt.gca()

    # Plot Q-values on left y-axis
    ax1.set_xlabel('Episodes', fontsize=14)
    ax1.set_ylabel('Average Q-Value', color='#e74c3c', fontsize=14)
    ax1.plot(q_df['Episode'], q_df['AvgQValue'],
             'o-', color='#e74c3c', linewidth=2.5, markersize=8, label='Q-Value')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')

    # Create second y-axis for memory
    ax2 = ax1.twinx()
    ax2.set_ylabel('Memory Usage (MB)', color='#3498db', fontsize=14)
    ax2.plot(memory_df['Episode'], memory_df['MemoryMB'],
             's-', color='#3498db', linewidth=2.5, markersize=8, label='Memory')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2.yaxis.set_major_formatter(FuncFormatter(mbytes_formatter))

    # Title and appearance
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True,
               fancybox=True, shadow=True)

    # Save the combined plot
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to create comparison plots between different programs
def create_comparison_plots(dfs_dict):
    """Create comparison plots between all three programs"""
    if len(dfs_dict) < 2:
        print("Not enough data for comparison plots")
        return

    # Create comparison directory
    if not os.path.exists('add_q_plots/comparison'):
        os.makedirs('add_q_plots/comparison')

    # Define colors and titles for each program
    program_colors = {
        'network': '#e74c3c',
        'grid': '#2ecc71',
        'dice': '#9b59b6'
    }

    program_titles = {
        'network': 'Network Resource Allocation',
        'grid': 'Grid World Navigation',
        'dice': 'Dice Game'
    }

    # Memory usage comparison
    plt.figure(figsize=(14, 10))
    for prog, prog_dfs in dfs_dict.items():
        if 'memory_usage' in prog_dfs:
            plt.plot(prog_dfs['memory_usage']['Episode'],
                     prog_dfs['memory_usage']['MemoryMB'],
                     'o-', color=program_colors[prog], linewidth=2.5,
                     markersize=8, label=program_titles[prog])

    plt.title('Memory Usage Comparison Across Problems', fontsize=16, fontweight='bold')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Memory Usage (MB)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(mbytes_formatter))
    plt.tight_layout()
    plt.savefig('add_q_plots/comparison/memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Q-value convergence comparison (normalized)
    plt.figure(figsize=(14, 10))
    for prog, prog_dfs in dfs_dict.items():
        if 'avg_values' in prog_dfs:
            # Normalize Q-values for fair comparison
            q_values = prog_dfs['avg_values']['AvgQValue']
            normalized_q = (q_values - q_values.min()) / (q_values.max() - q_values.min())
            plt.plot(prog_dfs['avg_values']['Episode'],
                     normalized_q,
                     'o-', color=program_colors[prog], linewidth=2.5,
                     markersize=8, label=program_titles[prog])

    plt.title('Normalized Q-Value Convergence Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Normalized Q-Value (0-1)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('add_q_plots/comparison/convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to process a single experiment type
def process_experiment(file_pattern, experiment_type, experiment_title):
    # Find all CSV files with pattern matching
    experiment_files = glob.glob(file_pattern)

    # Skip if no files found
    if not experiment_files:
        print(f"No files found matching pattern {file_pattern}")
        return None

    # Create output directory for this experiment
    output_dir = f'add_q_plots/{experiment_type}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionary to store dataframes
    dfs = {}

    # Load all available CSVs
    for file in experiment_files:
        # Extract the metric name (e.g., "memory_usage", "avg_values", etc.)
        name = file.replace(f"add_q_{experiment_type}_", "").replace(".csv", "")

        try:
            dfs[name] = pd.read_csv(file)
            print(f"Loaded {experiment_type} file: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not dfs:
        print(f"No valid data loaded for {experiment_type}")
        return None

    # Plot memory usage
    if 'memory_usage' in dfs:
        create_plot(
            dfs['memory_usage'], 'Episode', 'MemoryMB',
            f'{experiment_title}: Memory Usage Over Training',
            'Episodes', 'Memory Usage',
            '#3498db', f'{output_dir}/memory_usage_plot',
            formatter=mbytes_formatter, marker='o'
        )

    # Plot Q-Values (convergence)
    if 'avg_values' in dfs:
        create_plot(
            dfs['avg_values'], 'Episode', 'AvgQValue',
            f'{experiment_title}: Value Convergence',
            'Episodes', 'Average Q-Value',
            '#e74c3c', f'{output_dir}/q_value_convergence',
            marker='o'
        )

    # Plot DAG sizes - handle different column names
    if 'dag_sizes' in dfs:
        y_col = 'AvgNodeCount' if 'AvgNodeCount' in dfs['dag_sizes'].columns else 'NodeCount'
        create_plot(
            dfs['dag_sizes'], 'Episode', y_col,
            f'{experiment_title}: Decision Diagram Size',
            'Episodes', 'Average Node Count',
            '#2ecc71', f'{output_dir}/dag_size_plot',
            marker='o'
        )

    # Plot Bellman errors - handle different column names
    if 'bellman_errors' in dfs:
        y_col = 'AvgError' if 'AvgError' in dfs['bellman_errors'].columns else 'Error'
        create_plot(
            dfs['bellman_errors'], 'Episode', y_col,
            f'{experiment_title}: Bellman Error',
            'Episodes', 'Bellman Error',
            '#9b59b6', f'{output_dir}/bellman_error_plot',
            marker='o'
        )

    # Create combined plot for convergence and memory
    if 'avg_values' in dfs and 'memory_usage' in dfs:
        create_combined_memory_q_plot(
            dfs['memory_usage'],
            dfs['avg_values'],
            f'{experiment_title}: Convergence and Memory Usage',
            f'{output_dir}/convergence_memory_combined'
        )

    # Plot terminal visits if available (specific to experiment types)
    if 'terminal_visits' in dfs or 'goal_visits' in dfs:
        key = 'terminal_visits' if 'terminal_visits' in dfs else 'goal_visits'
        y_col = 'TerminalVisits' if 'TerminalVisits' in dfs[key].columns else 'GoalVisits'

        create_plot(
            dfs[key], 'Episode', y_col,
            f'{experiment_title}: Terminal State Visits',
            'Episodes', 'Terminal State Visits',
            '#f39c12', f'{output_dir}/terminal_visits_plot',
            marker='o'
        )

    print(f"All plots for {experiment_title} created successfully!")
    return dfs

# Function to find and plot all metrics
def plot_all_metrics():
    # Process all three experiment types
    dfs_dict = {}

    # Process network resource allocation
    network_dfs = process_experiment(
        "add_q_network_*.csv",
        "network",
        "Network Resource Allocation"
    )
    if network_dfs:
        dfs_dict['network'] = network_dfs

    # Process grid world
    grid_dfs = process_experiment(
        "add_q_grid_*.csv",
        "grid",
        "Grid World Navigation"
    )
    if grid_dfs:
        dfs_dict['grid'] = grid_dfs

    # Process dice game
    dice_dfs = process_experiment(
        "add_q_dice_*.csv",
        "dice",
        "Dice Game"
    )
    if dice_dfs:
        dfs_dict['dice'] = dice_dfs

    # Create comparison plots if we have data from multiple experiments
    if len(dfs_dict) > 1:
        create_comparison_plots(dfs_dict)

    print("All plots created successfully in the 'add_q_plots' directory!")

# Run the main plotting function
if __name__ == "__main__":
    plot_all_metrics()
