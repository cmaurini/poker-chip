#!/usr/bin/env python
"""
Post-processing script for poker_chip parametric studies.

This script aggregates results from all simulations and generates summary statistics.
Customize the analysis functions below for your specific needs.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import OmegaConf


def load_simulation_results(results_dir):
    """
    Load all simulation results from the results directory.
    
    Returns a list of dictionaries, one per simulation.
    """
    results_dir = Path(results_dir)
    simulations = []
    
    # Find all subdirectories with parameter files
    for sim_dir in sorted(results_dir.glob("*")):
        if not sim_dir.is_dir() or sim_dir.name.startswith("."):
            continue
        
        param_file = sim_dir / f"{sim_dir.name}-parameters.yml"
        data_file = sim_dir / f"{sim_dir.name}_data.json"
        
        if not param_file.exists():
            continue
        
        # Load parameters
        params = OmegaConf.load(param_file)
        params_dict = OmegaConf.to_container(params, resolve=True)
        
        sim_data = {
            "simulation_id": sim_dir.name,
            "directory": str(sim_dir),
            **{f"param_{k}": v for k, v in params_dict.items() if isinstance(v, (int, float, str, bool))}
        }
        
        # Load data if available
        if data_file.exists():
            try:
                with open(data_file, "r") as f:
                    data = json.load(f)
                    # Extract key metrics from data
                    if isinstance(data, dict):
                        for key in ["time_final", "load_max", "energy_final", "alpha_max"]:
                            if key in data:
                                sim_data[f"result_{key}"] = data[key]
                            elif f"history_{key}" in data and data[f"history_{key}"]:
                                sim_data[f"result_{key}"] = data[f"history_{key}"][-1]
            except Exception as e:
                print(f"Warning: Could not load data from {data_file}: {e}")
        
        simulations.append(sim_data)
    
    return simulations


def create_summary_csv(simulations, output_file):
    """
    Create a CSV summary of all simulations.
    """
    if not simulations:
        print("Warning: No simulations found!")
        return
    
    df = pd.DataFrame(simulations)
    df.to_csv(output_file, index=False)
    print(f"Summary saved to: {output_file}")
    print(f"Total simulations: {len(df)}")
    print("\nColumns:", list(df.columns))


def print_statistics(simulations):
    """
    Print basic statistics about the parameter sweep.
    """
    if not simulations:
        return
    
    df = pd.DataFrame(simulations)
    
    print("\n" + "="*70)
    print("Parametric Study Summary")
    print("="*70)
    print(f"Total simulations: {len(df)}")
    
    # Group by parameter combinations
    param_cols = [col for col in df.columns if col.startswith("param_")]
    if param_cols:
        print(f"\nVaried parameters ({len(param_cols)}):")
        for col in param_cols:
            unique_vals = df[col].unique()
            if len(unique_vals) > 1:
                print(f"  {col}: {len(unique_vals)} values")
                print(f"    Range: {df[col].min()} to {df[col].max()}")
    
    # Summary statistics for results
    result_cols = [col for col in df.columns if col.startswith("result_")]
    if result_cols:
        print(f"\nResult statistics ({len(result_cols)}):")
        for col in result_cols:
            if df[col].dtype in [np.float64, np.int64]:
                print(f"  {col}:")
                print(f"    Mean: {df[col].mean():.6f}")
                print(f"    Std:  {df[col].std():.6f}")
                print(f"    Min:  {df[col].min():.6f}")
                print(f"    Max:  {df[col].max():.6f}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process poker_chip parametric study results"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing simulation results"
    )
    parser.add_argument(
        "--output-summary",
        default="results/sweep_summary.csv",
        help="Output CSV file with summary statistics"
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    simulations = load_simulation_results(args.results_dir)
    
    if simulations:
        print_statistics(simulations)
        
        # Create output directory
        output_path = Path(args.output_summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        create_summary_csv(simulations, args.output_summary)
        
        print("\nPost-processing complete!")
    else:
        print("No simulation results found. Did any simulations complete?")


if __name__ == "__main__":
    main()
