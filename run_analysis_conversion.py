#!/usr/bin/env python3

"""
Pattern Learning Paper - Analysis and Conversion Script Runner
Author: Anzal KS (anzal.ks@gmail.com)
Repository: https://github.com/anzalks/

This script reads the configuration from analysis_config.yaml and runs analysis
and conversion scripts with the appropriate arguments.
"""

import os
import sys
import yaml
import argparse
import subprocess
import time
from pathlib import Path

def load_config(config_path="analysis_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def build_command_args(script_config):
    """Build command line arguments from script configuration."""
    cmd_args = []
    
    if 'arguments' in script_config:
        for arg_name, arg_config in script_config['arguments'].items():
            if isinstance(arg_config, dict):
                flag = arg_config.get('flag', f'--{arg_name}')
                default_value = arg_config.get('default')
                arg_type = arg_config.get('type', 'value')
                
                if arg_type == 'flag' and default_value:
                    cmd_args.append(flag)
                elif arg_type != 'flag' and default_value is not None:
                    cmd_args.extend([flag, str(default_value)])
    
    return cmd_args

def run_script(script_name, script_config, custom_args=None):
    """Run a single script with its configuration."""
    script_path = script_config['script']
    description = script_config.get('description', 'No description available')
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    # Check if script exists
    if not Path(script_path).exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    # Build command
    cmd = ['python', script_path]
    
    # Add arguments from config
    config_args = build_command_args(script_config)
    cmd.extend(config_args)
    
    # Add custom arguments if provided
    if custom_args:
        cmd.extend(custom_args)
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run the script
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"\n✅ SUCCESS: {script_name} completed in {(end_time - start_time)/60:.1f} minutes")
        
        # Print last few lines of output
        if result.stdout:
            output_lines = result.stdout.strip().split('\n')
            print(f"Last few lines of output:")
            for line in output_lines[-5:]:
                print(f"  {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"\n❌ FAILED: {script_name} failed after {(end_time - start_time)/60:.1f} minutes")
        print(f"Return code: {e.returncode}")
        
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        
        return False

def run_workflow(workflow_name, config):
    """Run a complete workflow."""
    workflows = config.get('workflows', {})
    
    if workflow_name not in workflows:
        print(f"ERROR: Workflow '{workflow_name}' not found in configuration")
        return False
    
    workflow = workflows[workflow_name]
    steps = workflow.get('steps', [])
    description = workflow.get('description', 'No description available')
    
    print(f"\n{'='*80}")
    print(f"STARTING WORKFLOW: {workflow_name}")
    print(f"Description: {description}")
    print(f"Steps: {', '.join(steps)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    success_count = 0
    
    for step in steps:
        # Check in conversion scripts first
        if step in config.get('conversion_scripts', {}):
            script_config = config['conversion_scripts'][step]
            success = run_script(step, script_config)
        # Then check in analysis scripts
        elif step in config.get('analysis_scripts', {}):
            script_config = config['analysis_scripts'][step]
            success = run_script(step, script_config)
        else:
            print(f"ERROR: Step '{step}' not found in configuration")
            success = False
        
        if success:
            success_count += 1
        else:
            print(f"WORKFLOW FAILED at step: {step}")
            break
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    
    if success_count == len(steps):
        print(f"\n🎉 WORKFLOW COMPLETED SUCCESSFULLY: {workflow_name}")
        print(f"Total time: {total_time:.1f} minutes")
        print(f"All {len(steps)} steps completed successfully")
    else:
        print(f"\n💥 WORKFLOW FAILED: {workflow_name}")
        print(f"Total time: {total_time:.1f} minutes")
        print(f"Completed {success_count}/{len(steps)} steps")
    
    return success_count == len(steps)

def list_available_scripts(config):
    """List all available scripts and workflows."""
    print("\n📋 AVAILABLE SCRIPTS AND WORKFLOWS")
    print("="*50)
    
    print("\n🔄 CONVERSION SCRIPTS:")
    for name, script_config in config.get('conversion_scripts', {}).items():
        description = script_config.get('description', 'No description')
        print(f"  • {name}: {description}")
    
    print("\n📊 ANALYSIS SCRIPTS:")
    for name, script_config in config.get('analysis_scripts', {}).items():
        description = script_config.get('description', 'No description')
        print(f"  • {name}: {description}")
    
    print("\n🔗 WORKFLOWS:")
    for name, workflow in config.get('workflows', {}).items():
        description = workflow.get('description', 'No description')
        steps = workflow.get('steps', [])
        print(f"  • {name}: {description}")
        print(f"    Steps: {' → '.join(steps)}")

def main():
    parser = argparse.ArgumentParser(
        description="Run analysis and conversion scripts for pattern learning paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available scripts and workflows
  python run_analysis_conversion.py --list
  
  # Run a single conversion script
  python run_analysis_conversion.py --script abf_to_hdf5
  
  # Run a single analysis script
  python run_analysis_conversion.py --script extract_features
  
  # Run a complete workflow
  python run_analysis_conversion.py --workflow full_conversion
  
  # Run the complete pipeline
  python run_analysis_conversion.py --workflow complete_pipeline
  
  # Run with custom arguments
  python run_analysis_conversion.py --script extract_features --args "--use-full-ram"
        """
    )
    
    parser.add_argument('--config', '-c', default='analysis_config.yaml',
                        help='Path to configuration file (default: analysis_config.yaml)')
    
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available scripts and workflows')
    
    parser.add_argument('--script', '-s', type=str,
                        help='Run a specific script by name')
    
    parser.add_argument('--workflow', '-w', type=str,
                        help='Run a specific workflow by name')
    
    parser.add_argument('--args', '-a', type=str,
                        help='Additional arguments to pass to the script (space-separated)')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in configuration file: {e}")
        return 1
    
    # Handle list command
    if args.list:
        list_available_scripts(config)
        return 0
    
    # Handle script command
    if args.script:
        custom_args = args.args.split() if args.args else None
        
        # Check in conversion scripts
        if args.script in config.get('conversion_scripts', {}):
            script_config = config['conversion_scripts'][args.script]
            success = run_script(args.script, script_config, custom_args)
        # Check in analysis scripts
        elif args.script in config.get('analysis_scripts', {}):
            script_config = config['analysis_scripts'][args.script]
            success = run_script(args.script, script_config, custom_args)
        else:
            print(f"ERROR: Script '{args.script}' not found in configuration")
            print("Use --list to see available scripts")
            return 1
        
        return 0 if success else 1
    
    # Handle workflow command
    if args.workflow:
        success = run_workflow(args.workflow, config)
        return 0 if success else 1
    
    # If no specific command, show help
    parser.print_help()
    return 0

if __name__ == '__main__':
    sys.exit(main()) 