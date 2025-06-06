#!/usr/bin/env python3

"""
Pattern Learning Paper - Plotting Script Runner
Author: Anzal KS (anzal.ks@gmail.com)
Repository: https://github.com/anzalks/

This script reads the configuration from plotting_config.yaml and runs plotting scripts
with the appropriate arguments based on the analysis type (standard or field_normalized).
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path

def load_config(config_path="plotting_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def build_command_args(script_path, args_dict):
    """Build command line arguments from args dictionary."""
    cmd_args = []
    
    # Simple argument mapping - let the config file handle the file mappings
    arg_mapping = {
        'file': '-f', 
        'stats': '-s', 
        'resistance': '-r', 
        'image': '-i', 
        'projimg': '-p',
        'alltrial_path': '-a',  # Figure 3 specific argument
        'cell_stats': '-c', 
        'training': '-t', 
        'firing': '-q',
        'image_i': '-i', 
        'image_p': '-p', 
        'image_m': '-m',
        'sensitisation_data': '-s'  # CHR2 sensitisation data
    }
    
    for key, value in args_dict.items():
        if key in arg_mapping:
            cmd_args.extend([arg_mapping[key], value])
            
    return cmd_args

def run_script(script_path, args_dict, output_dir=None):
    """Run a plotting script with given arguments."""
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return False
    
    # Build command
    cmd = ['python', script_path]
    cmd_args = build_command_args(script_path, args_dict)
    cmd.extend(cmd_args)
    
    # Add output directory if specified
    if output_dir:
        cmd.extend(['-o', output_dir])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Add the plotting_scripts directory to Python path so scripts can find shared_utils
        env = os.environ.copy()
        current_dir = os.getcwd()
        plotting_scripts_dir = os.path.join(current_dir, 'plotting_scripts')
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{plotting_scripts_dir}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = plotting_scripts_dir
            
        result = subprocess.run(cmd, cwd=current_dir, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully completed: {os.path.basename(script_path)}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
        else:
            print(f"✗ Failed: {os.path.basename(script_path)}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception running {script_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run plotting scripts based on plotting_config.yaml')
    parser.add_argument('--figures', nargs='+', help='Specific figures to generate (e.g., figure_1 figure_2)')
    parser.add_argument('--analysis_type', choices=['standard', 'field_normalized'], 
                       default='standard', help='Analysis type to run')
    parser.add_argument('--output_dir', help='Output directory for figures')
    parser.add_argument('--config', default='plotting_config.yaml', help='Path to config file')
    parser.add_argument('--list', action='store_true', help='List available figures')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Set up output directory
    output_dir = args.output_dir
    if not output_dir:
        base_output = config.get('output', {}).get('base_output_dir', 'outputs')
        if args.analysis_type == 'field_normalized':
            output_dir = os.path.join(base_output, 'supplementary_figures')
        else:
            output_dir = os.path.join(base_output, 'main_figures')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get figures configuration
    figures_config = config.get('figures', {})
    
    if args.list:
        print("Available figures:")
        print("\nMain figures:")
        for fig_name in figures_config.get('main_figures', {}):
            print(f"  - {fig_name}")
        print("\nSupplementary figures:")
        for fig_name in figures_config.get('supplementary_figures', {}):
            print(f"  - {fig_name}")
        return 0
    
    # Determine which figures to run
    figures_to_run = []
    
    if args.figures:
        # Run specific figures
        for fig_name in args.figures:
            # Check main figures first
            if fig_name in figures_config.get('main_figures', {}):
                fig_config = figures_config['main_figures'][fig_name]
                figures_to_run.append((fig_name, fig_config, 'main'))
            # Then check supplementary figures
            elif fig_name in figures_config.get('supplementary_figures', {}):
                fig_config = figures_config['supplementary_figures'][fig_name]
                figures_to_run.append((fig_name, fig_config, 'supplementary'))
            else:
                print(f"Figure {fig_name} not found in config")
    else:
        # Run all figures based on analysis type
        if args.analysis_type == 'standard':
            # Run main figures
            for fig_name, fig_config in figures_config.get('main_figures', {}).items():
                figures_to_run.append((fig_name, fig_config, 'main'))
            # Run standard supplementary figures (non-fnorm)
            for fig_name, fig_config in figures_config.get('supplementary_figures', {}).items():
                if 'fnorm' not in fig_name and 'field_norm' not in fig_name:
                    figures_to_run.append((fig_name, fig_config, 'supplementary'))
        else:
            # Run field normalized figures
            for fig_name, fig_config in figures_config.get('supplementary_figures', {}).items():
                if 'fnorm' in fig_name or 'field_norm' in fig_name:
                    figures_to_run.append((fig_name, fig_config, 'supplementary'))
    
    if not figures_to_run:
        print("No figures to run")
        return 1
    
    print(f"Running {len(figures_to_run)} figures with {args.analysis_type} analysis...")
    
    success_count = 0
    total_count = len(figures_to_run)
    
    for fig_name, fig_config, fig_type in figures_to_run:
        print(f"\n--- Running {fig_name} ---")
        
        script_path = fig_config.get('script')
        if not script_path:
            print(f"No script specified for {fig_name}")
            continue
            
        # Get arguments for the analysis type
        args_config = fig_config.get('args', {})
        if args.analysis_type == 'field_normalized':
            script_args = args_config.get('field_normalized', args_config.get('standard', {}))
        else:
            script_args = args_config.get('standard', {})
            
        if not script_args:
            print(f"No arguments found for {fig_name} with {args.analysis_type} analysis")
            continue
        
        # Run the script
        if run_script(script_path, script_args, output_dir):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully completed: {success_count}/{total_count} figures")
    print(f"Output directory: {output_dir}")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main()) 