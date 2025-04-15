#!/usr/bin/env python3
"""
Script to plot metrics from ADD-Q implementations.
Usage: python plot_metrics.py file1.csv file2.csv ...
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def identify_metric_type(filename):
    """Identify the type of metric based on filename."""
    base = os.path.basename(filename).lower()
    
    if 'avg_values' in base:
        return 'Average Q-Values'
    elif 'dag_size' in base:
        return 'Decision Diagram Size'
    elif 'bellman_error' in base:
        return 'Bellman Error'
    elif 'memory_usage' in base:
        return 'Memory Usage (MB)'
    elif 'terminal_visit' in base or 'goal_visit' in base:
        return 'Terminal/Goal Visits'
    elif 'episode_time' in base:
        return 'Episode Duration (s)'
    elif 'path_length' in base:
        return 'Path Length'
    elif 'resource' in base:
        return 'Resources Collected'
    else:
        return 'Unknown Metric'

def get_implementation_name(filename):
    """Extract implementation name from filename."""
    base = os.path.basename(filename).lower()
    
    if 'dice' in base:
        return 'Dice Game'
    elif 'grid' in base:
        return 'Grid World'
    elif 'network' in base:
        return 'Network Resource'
    elif 'resource_collector' in base or 'resource_' in base:
        return 'Resource Collector'
    else:
        return 'Unknown Implementation'

def plot_csv(csv_file):
    """Plot data from a CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if it has the expected columns
        if len(df.columns) < 2:
            print(f"Error: CSV file {csv_file} has fewer than 2 columns.")
            return False
        
        # Extract column names
        x_col = df.columns[0]
        y_col = df.columns[1]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(df[x_col], df[y_col], marker='o', markersize=3, linestyle='-', linewidth=1)
        
        # Add labels and title
        metric_type = identify_metric_type(csv_file)
        implementation = get_implementation_name(csv_file)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{metric_type} - {implementation}")
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Improve layout
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.splitext(csv_file)[0] + '.png'
        plt.savefig(output_file, dpi=300)
        print(f"Generated plot: {output_file}")
        
        # Close figure to free memory
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error plotting {csv_file}: {e}")
        return False

def main():
    # Get file list from command line or use wildcard pattern
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
    else:
        # Look for CSV files in current directory
        csv_files = glob('add_q_*.csv')
        
    if not csv_files:
        print("No CSV files specified or found.")
        print("Usage: python plot_metrics.py file1.csv file2.csv ...")
        print("Or run from directory containing ADD-Q CSV files.")
        return
    
    # Count successful plots
    successful_plots = 0
    
    # Process each CSV file
    for csv_file in csv_files:
        if plot_csv(csv_file):
            successful_plots += 1
    
    # Summary
    print(f"\nSummary: Generated {successful_plots} plots from {len(csv_files)} CSV files.")
    
    # Create combined plots for same metric types
    try:
        # Group files by metric type
        metrics = {}
        for csv_file in csv_files:
            metric_type = identify_metric_type(csv_file)
            if metric_type not in metrics:
                metrics[metric_type] = []
            metrics[metric_type].append(csv_file)
        
        # Create combined plots for metrics with multiple files
        for metric_type, files in metrics.items():
            if len(files) > 1:
                plt.figure(figsize=(12, 7))
                
                for csv_file in files:
                    df = pd.read_csv(csv_file)
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    impl = get_implementation_name(csv_file)
                    plt.plot(df[x_col], df[y_col], marker='.', markersize=3, 
                             linestyle='-', linewidth=1, label=impl)
                
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"Comparison of {metric_type} Across Implementations")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                
                # Save combined plot
                safe_metric = metric_type.replace('/', '_').lower().replace(' ', '_')
                output_file = f"combined_{safe_metric}.png"
                plt.savefig(output_file, dpi=300)
                print(f"Generated combined plot: {output_file}")
                plt.close()
        
    except Exception as e:
        print(f"Error creating combined plots: {e}")

if __name__ == '__main__':
    main()
