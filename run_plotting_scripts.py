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
    
    # Standardized argument mapping - consistent across all scripts
    arg_mapping = {
        'file': '-f',                # --pikl-path (main pickle file)
        'stats': '-s',               # --sortedcell-path (stats/classification data)
        'resistance': '-r',          # --inR-path (resistance data)
        'image': '-i',               # --illustration-path (main illustration)
        'projimg': '-p',             # --projection-image / --inRillustration-path
        'alltrial_path': '-t',       # --alltrial-path (Figure 3 format)
        'all_trials': '-t',          # --alltrials-path (all other figures) - same as alltrial_path
        'cell_stats': '-c',          # --cellstat-path (cell statistics)
        'training': '-n',            # --training-path (training data, now uses -n)
        'firing': '-q',              # --firingproperties-path (firing properties)
        'image_i': '-i',             # Figure 2 multi-image support (main image)
        'image_p': '-p',             # Figure 2 projection image
        'image_m': '-m',             # Figure 2 additional image
        'sensitisation_data': '-s',  # CHR2 sensitisation data (reuses stats flag)
        # RMP distribution script specific mappings
        'cell_stats': '-c',          # RMP script cell stats path (duplicate mapping for clarity)
        'classified_cells_path': '-s' # RMP script classified cells path
    }
    
    # Special handling for RMP distribution script parameters
    rmp_mapping = {}
    if 'figure_generation_script_rmp_distribution.py' in str(script_path):
        rmp_mapping = {
            'cell_stats': '-c',          # --cell-stats-path
            'stats': '-s',               # --classified-cells-path  
            'file': '-f',                # --epsp-data-path
            'firing': '-p',              # --firing-properties-path
            'sensitisation_data': '-x'   # --sensitisation-data-path
        }
    
    for key, value in args_dict.items():
        # Use RMP-specific mapping if available, otherwise use standard mapping
        if rmp_mapping and key in rmp_mapping:
            cmd_args.extend([rmp_mapping[key], value])
        elif key in arg_mapping:
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
    parser.add_argument('--all_fig', action='store_true', help='Generate all figures (main + supplementary)')
    parser.add_argument('--main_fig', action='store_true', help='Generate only main figures')
    parser.add_argument('--supplementary_fig', action='store_true', help='Generate only supplementary figures')
    parser.add_argument('--alpha_labels_off', action='store_true', 
                       help='Disable alphabetical subplot labels (A, B, C, etc.) for all figures')
    parser.add_argument('--alpha_labels_on', action='store_true', 
                       help='Enable alphabetical subplot labels (A, B, C, etc.) for all figures (default)')
    parser.add_argument('--no_labels', action='store_true', 
                       help='Alias for --alpha_labels_off (disable all subplot labels)')
    
    # Figure format control
    parser.add_argument('--format', choices=['png', 'pdf', 'svg', 'eps'], 
                       default='png', help='Output format for all figures (default: png)')
    parser.add_argument('--multi_format', nargs='+', choices=['png', 'pdf', 'svg', 'eps'],
                       help='Save figures in multiple formats (e.g., --multi_format png pdf svg)')
    parser.add_argument('--dpi', type=int, default=300, 
                       help='DPI for figure output (default: 300)')
    parser.add_argument('--high_quality', action='store_true',
                       help='Use high quality settings (DPI=600, optimized for publication)')
    parser.add_argument('--transparent', action='store_true',
                       help='Save figures with transparent background')
    
    args = parser.parse_args()
    
    # Handle subplot labels control
    if args.alpha_labels_off or args.no_labels:
        # Set environment variable that scripts can check
        os.environ['SUBPLOT_LABELS_ENABLED'] = 'False'
        print("🏷️  Alphabetical subplot labels DISABLED for all figures")
    elif args.alpha_labels_on:
        os.environ['SUBPLOT_LABELS_ENABLED'] = 'True'
        print("🏷️  Alphabetical subplot labels ENABLED for all figures")
    else:
        # Default behavior - labels enabled
        os.environ['SUBPLOT_LABELS_ENABLED'] = 'True'
    
    # Handle figure format control
    if args.multi_format:
        # Multiple formats requested
        os.environ['FIGURE_FORMATS'] = ','.join(args.multi_format)
        print(f"📊 Figure formats: {', '.join(args.multi_format).upper()}")
    else:
        # Single format
        os.environ['FIGURE_FORMAT'] = args.format
        print(f"📊 Figure format: {args.format.upper()}")
    
    # Handle quality settings
    if args.high_quality:
        os.environ['FIGURE_DPI'] = '600'
        print("🎯 High quality mode: DPI=600")
    else:
        os.environ['FIGURE_DPI'] = str(args.dpi)
        print(f"🎯 Figure DPI: {args.dpi}")
    
    if args.transparent:
        os.environ['FIGURE_TRANSPARENT'] = 'True'
        print("🔍 Transparent background enabled")
    else:
        os.environ['FIGURE_TRANSPARENT'] = 'False'
    
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
    elif args.all_fig:
        # Run all figures (main + supplementary)
        print("Running all figures (main + supplementary)...")
        # Add main figures
        for fig_name, fig_config in figures_config.get('main_figures', {}).items():
            figures_to_run.append((fig_name, fig_config, 'main'))
        # Add all supplementary figures
        for fig_name, fig_config in figures_config.get('supplementary_figures', {}).items():
            figures_to_run.append((fig_name, fig_config, 'supplementary'))
    elif args.main_fig:
        # Run only main figures
        print("Running main figures only...")
        for fig_name, fig_config in figures_config.get('main_figures', {}).items():
            figures_to_run.append((fig_name, fig_config, 'main'))
    elif args.supplementary_fig:
        # Run only supplementary figures
        print("Running supplementary figures only...")
        for fig_name, fig_config in figures_config.get('supplementary_figures', {}).items():
            figures_to_run.append((fig_name, fig_config, 'supplementary'))
    else:
        # Run all figures based on analysis type (legacy behavior)
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
    
    # Update the analysis message to reflect auto-detection for new options
    if args.all_fig or args.main_fig or args.supplementary_fig:
        print(f"Running {len(figures_to_run)} figures with auto-detected analysis types...")
    else:
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
        
        # Auto-detect analysis type based on what's available in the config
        # For the new options (all_fig, main_fig, supplementary_fig), run whatever is available
        if args.all_fig or args.main_fig or args.supplementary_fig:
            # Try field_normalized first (for fnorm figures), then standard
            if 'field_normalized' in args_config:
                script_args = args_config['field_normalized']
                current_analysis_type = 'field_normalized'
            elif 'standard' in args_config:
                script_args = args_config['standard']
                current_analysis_type = 'standard'
            else:
                script_args = {}
                current_analysis_type = 'unknown'
        else:
            # Use the user-specified analysis type (legacy behavior)
            if args.analysis_type == 'field_normalized':
                script_args = args_config.get('field_normalized', args_config.get('standard', {}))
                current_analysis_type = 'field_normalized'
            else:
                script_args = args_config.get('standard', {})
                current_analysis_type = 'standard'
            
        if not script_args:
            print(f"No arguments found for {fig_name} with {current_analysis_type} analysis")
            continue
        
        # Determine correct output directory based on figure type
        if args.output_dir:
            # User specified output directory, use that
            figure_output_dir = output_dir
        else:
            # Use appropriate directory based on figure type
            base_output = config.get('output', {}).get('base_output_dir', 'outputs')
            if fig_type == 'supplementary':
                figure_output_dir = os.path.join(base_output, 'supplementary_figures')
            else:
                figure_output_dir = os.path.join(base_output, 'main_figures')
        
        os.makedirs(figure_output_dir, exist_ok=True)
        
        # Run the script
        if run_script(script_path, script_args, figure_output_dir):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully completed: {success_count}/{total_count} figures")
    print(f"Output directory: {output_dir}")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main()) 