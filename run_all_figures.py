#!/usr/bin/env python3
"""
Master Pipeline Script for Pattern Learning Paper Figure Generation
Executes all figures with flexible options and parallel processing

Author: Anzal KS (anzal.ks@gmail.com)
Repository: https://github.com/anzalks/
Copyright 2024-, Anzal KS
"""

import argparse
import sys
import os
import subprocess
import concurrent.futures
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import signal

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from shared_utilities import PatternLearningUtils, setup_logging
except ImportError as e:
    print(f"Error: Could not import shared_utilities from src directory.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    print(f"Expected path: {os.path.join(os.path.dirname(__file__), 'src', 'shared_utilities.py')}")
    print(f"Import error: {e}")
    
    # Try alternative path
    try:
        sys.path.insert(0, 'src')
        from shared_utilities import PatternLearningUtils, setup_logging
        print("Successfully imported from 'src' directory")
    except ImportError:
        print("Failed to import from both paths. Please ensure src/shared_utilities.py exists.")
        sys.exit(1)

class FigurePipeline:
    """Main pipeline controller for figure generation"""
    
    def __init__(self, config_path: str = "config.yaml", data_dir: Optional[str] = None):
        """Initialize the pipeline"""
        try:
            self.utils = PatternLearningUtils(config_path)
            self.config = self.utils.config_manager.config
            self.logger = self.utils.logger
            
            # Override data directory if provided
            if data_dir:
                self.config['data_paths']['base_data_dir'] = data_dir
                self.logger.info(f"Using custom data directory: {data_dir}")
            
            self.total_figures = 0
            self.completed_figures = 0
            self.failed_figures = []
            self.execution_stats = {}
            
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            sys.exit(1)
    
    def validate_environment(self) -> bool:
        """Validate environment and dependencies"""
        self.logger.info("Validating environment...")
        
        # Check Python executable
        python_exec = self.config['environment']['python_executable']
        try:
            result = subprocess.run([python_exec, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.logger.error(f"Python executable not working: {python_exec}")
                return False
            self.logger.info(f"Python version: {result.stdout.strip()}")
        except Exception as e:
            self.logger.error(f"Error checking Python: {e}")
            return False
        
        # Check required packages (basic check)
        required_packages = self.config['environment']['required_packages']
        missing_packages = []
        
        for package in required_packages:
            try:
                result = subprocess.run([python_exec, '-c', f'import {package}'], 
                                      capture_output=True, timeout=5)
                if result.returncode != 0:
                    missing_packages.append(package)
            except Exception:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing required packages: {missing_packages}")
            return False
        
        self.logger.info("Environment validation passed")
        return True
    
    def get_figure_list(self, mode: str, specific_figures: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
        """Get list of figures to process based on mode"""
        figures_to_process = []
        
        figures_config = self.config['figures']
        
        if mode == "all":
            # All main and supplementary figures
            for fig_name, fig_config in figures_config['main_figures'].items():
                for analysis_type in fig_config['analysis_types']:
                    figures_to_process.append((fig_name, "main_figures", analysis_type))
            
            for fig_name, fig_config in figures_config['supplementary_figures'].items():
                for analysis_type in fig_config['analysis_types']:
                    figures_to_process.append((fig_name, "supplementary_figures", analysis_type))
        
        elif mode == "main_only":
            for fig_name, fig_config in figures_config['main_figures'].items():
                for analysis_type in fig_config['analysis_types']:
                    figures_to_process.append((fig_name, "main_figures", analysis_type))
        
        elif mode == "supp_only":
            for fig_name, fig_config in figures_config['supplementary_figures'].items():
                for analysis_type in fig_config['analysis_types']:
                    figures_to_process.append((fig_name, "supplementary_figures", analysis_type))
        
        elif mode == "individual" and specific_figures:
            for fig_name in specific_figures:
                # Find figure in config
                if fig_name in figures_config['main_figures']:
                    fig_config = figures_config['main_figures'][fig_name]
                    for analysis_type in fig_config['analysis_types']:
                        figures_to_process.append((fig_name, "main_figures", analysis_type))
                elif fig_name in figures_config['supplementary_figures']:
                    fig_config = figures_config['supplementary_figures'][fig_name]
                    for analysis_type in fig_config['analysis_types']:
                        figures_to_process.append((fig_name, "supplementary_figures", analysis_type))
                else:
                    self.logger.warning(f"Figure {fig_name} not found in configuration")
        
        elif mode == "standard_only":
            for category in ['main_figures', 'supplementary_figures']:
                for fig_name, fig_config in figures_config[category].items():
                    if 'standard' in fig_config['analysis_types']:
                        figures_to_process.append((fig_name, category, 'standard'))
        
        elif mode == "fnorm_only":
            for category in ['main_figures', 'supplementary_figures']:
                for fig_name, fig_config in figures_config[category].items():
                    if 'field_normalized' in fig_config['analysis_types']:
                        figures_to_process.append((fig_name, category, 'field_normalized'))
        
        self.total_figures = len(figures_to_process)
        self.logger.info(f"Total figures to process: {self.total_figures}")
        
        return figures_to_process
    
    def build_command(self, figure_name: str, figure_type: str, analysis_type: str) -> List[str]:
        """Build command to execute figure script"""
        try:
            # Get figure configuration
            if figure_type == "main_figures":
                fig_config = self.config['figures']['main_figures'][figure_name]
            else:
                fig_config = self.config['figures']['supplementary_figures'][figure_name]
            
            # Base script path
            script_path = fig_config['script']
            
            # Modify script path for field normalized analysis
            if analysis_type == 'field_normalized' and not script_path.endswith('_fnorm.py'):
                # Try to find the _fnorm version
                base_script = script_path.replace('.py', '')
                fnorm_script = f"{base_script}_fnorm.py"
                if os.path.exists(fnorm_script):
                    script_path = fnorm_script
                else:
                    self.logger.warning(f"Field normalized script not found for {figure_name}, using standard script with --analysis-type field_normalized")
            
            # Build command using the new standardized approach
            python_exec = self.config['environment']['python_executable']
            command = [python_exec, script_path]
            
            # Add standardized arguments that all scripts now accept
            command.extend(['--data-dir', self.config['data_paths']['base_data_dir']])
            command.extend(['--analysis-type', analysis_type])
            
            self.logger.debug(f"Built command for {figure_name}: {' '.join(command)}")
            return command
            
        except Exception as e:
            self.logger.error(f"Error building command for {figure_name}: {e}")
            raise
    
    def execute_figure(self, figure_name: str, figure_type: str, analysis_type: str) -> Dict[str, Any]:
        """Execute a single figure generation"""
        start_time = time.time()
        
        self.logger.info(f"Starting {figure_name} ({analysis_type})")
        
        try:
            # Validate dependencies first
            if figure_type == "main_figures":
                fig_config = self.config['figures']['main_figures'][figure_name]
            else:
                fig_config = self.config['figures']['supplementary_figures'][figure_name]
            
            dependencies = fig_config['data_dependencies']
            if not self.utils.data_manager.validate_dependencies(dependencies):
                raise ValueError(f"Missing dependencies for {figure_name}")
            
            # Build and execute command
            command = self.build_command(figure_name, figure_type, analysis_type)
            if not command:
                raise ValueError("Could not build execution command")
            
            self.logger.debug(f"Executing: {' '.join(command)}")
            
            # Execute with timeout
            timeout = self.config['pipeline']['timeout_per_figure']
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                self.completed_figures += 1
                self.logger.info(f"✓ {figure_name} ({analysis_type}) completed in {execution_time:.1f}s")
                
                return {
                    'figure': figure_name,
                    'type': figure_type,
                    'analysis': analysis_type,
                    'status': 'success',
                    'execution_time': execution_time,
                    'command': ' '.join(command)
                }
            else:
                self.failed_figures.append(f"{figure_name}_{analysis_type}")
                self.logger.error(f"✗ {figure_name} ({analysis_type}) failed")
                self.logger.error(f"Error output: {result.stderr}")
                
                return {
                    'figure': figure_name,
                    'type': figure_type,
                    'analysis': analysis_type,
                    'status': 'failed',
                    'execution_time': execution_time,
                    'error': result.stderr,
                    'command': ' '.join(command)
                }
        
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            self.failed_figures.append(f"{figure_name}_{analysis_type}")
            self.logger.error(f"✗ {figure_name} ({analysis_type}) timed out after {timeout}s")
            
            return {
                'figure': figure_name,
                'type': figure_type,
                'analysis': analysis_type,
                'status': 'timeout',
                'execution_time': execution_time,
                'error': f'Timeout after {timeout}s'
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_figures.append(f"{figure_name}_{analysis_type}")
            self.logger.error(f"✗ {figure_name} ({analysis_type}) error: {e}")
            
            return {
                'figure': figure_name,
                'type': figure_type,
                'analysis': analysis_type,
                'status': 'error',
                'execution_time': execution_time,
                'error': str(e)
            }
    
    def run_parallel(self, figures_list: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Run figures in parallel"""
        max_workers = self.config['pipeline']['max_workers']
        results = []
        
        self.logger.info(f"Starting parallel execution with {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_figure = {
                executor.submit(self.execute_figure, fig_name, fig_type, analysis_type): 
                (fig_name, fig_type, analysis_type)
                for fig_name, fig_type, analysis_type in figures_list
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_figure):
                fig_name, fig_type, analysis_type = future_to_figure[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress update
                    progress = (len(results) / self.total_figures) * 100
                    self.logger.info(f"Progress: {len(results)}/{self.total_figures} ({progress:.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Exception in {fig_name}: {e}")
                    results.append({
                        'figure': fig_name,
                        'type': fig_type,
                        'analysis': analysis_type,
                        'status': 'exception',
                        'error': str(e)
                    })
        
        return results
    
    def run_sequential(self, figures_list: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Run figures sequentially"""
        results = []
        
        self.logger.info("Starting sequential execution")
        
        for i, (fig_name, fig_type, analysis_type) in enumerate(figures_list, 1):
            self.logger.info(f"Processing {i}/{self.total_figures}: {fig_name} ({analysis_type})")
            
            result = self.execute_figure(fig_name, fig_type, analysis_type)
            results.append(result)
            
            # Stop on first error if configured
            if (result['status'] != 'success' and 
                not self.config['pipeline']['continue_on_error']):
                self.logger.error("Stopping execution due to error")
                break
        
        return results
    
    def save_execution_report(self, results: List[Dict[str, Any]]):
        """Save execution report"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"outputs/logs/execution_report_{timestamp}.json"
        
        # Ensure logs directory exists
        os.makedirs("outputs/logs", exist_ok=True)
        
        report = {
            'timestamp': timestamp,
            'total_figures': self.total_figures,
            'completed_figures': self.completed_figures,
            'failed_figures': len(self.failed_figures),
            'success_rate': (self.completed_figures / self.total_figures * 100) if self.total_figures > 0 else 0,
            'results': results
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Execution report saved: {report_file}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
    
    def run(self, mode: str, parallel: bool = True, specific_figures: Optional[List[str]] = None) -> bool:
        """Main execution method"""
        start_time = time.time()
        
        self.logger.info(f"Starting Pattern Learning Figure Pipeline")
        self.logger.info(f"Mode: {mode}, Parallel: {parallel}")
        
        # Validate environment
        if not self.validate_environment():
            self.logger.error("Environment validation failed")
            return False
        
        # Get figures to process
        figures_list = self.get_figure_list(mode, specific_figures)
        if not figures_list:
            self.logger.error("No figures to process")
            return False
        
        # Execute figures
        if parallel and len(figures_list) > 1:
            results = self.run_parallel(figures_list)
        else:
            results = self.run_sequential(figures_list)
        
        # Calculate execution statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r['status'] == 'success')
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total execution time: {total_time:.1f}s")
        self.logger.info(f"Figures processed: {len(results)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {len(results) - successful}")
        
        if self.failed_figures:
            self.logger.error(f"Failed figures: {', '.join(self.failed_figures)}")
        
        # Save execution report
        self.save_execution_report(results)
        
        return len(self.failed_figures) == 0

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\nInterrupt received, stopping pipeline...")
    sys.exit(1)

def main():
    """Main entry point"""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Pattern Learning Paper - Figure Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures
  python run_all_figures.py --mode all
  
  # Generate only main figures with custom data directory
  python run_all_figures.py --mode main_only --data-dir /path/to/data
  
  # Generate specific figures sequentially
  python run_all_figures.py --mode individual --figures figure_1 figure_2 --sequential
  
  # Generate only field normalized analysis
  python run_all_figures.py --mode fnorm_only --parallel
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["all", "main_only", "supp_only", "individual", "standard_only", "fnorm_only"],
        default="all",
        help="Execution mode (default: all)"
    )
    
    parser.add_argument(
        "--figures", "-f",
        nargs="+",
        help="Specific figures to generate (use with --mode individual)"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        help="Custom data directory (overrides config)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        default=True,
        help="Run in parallel mode (default)"
    )
    
    parser.add_argument(
        "--sequential", "-s",
        action="store_true",
        help="Run in sequential mode"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "individual" and not args.figures:
        parser.error("--figures required when using --mode individual")
    
    if args.sequential:
        args.parallel = False
    
    # Initialize pipeline
    try:
        pipeline = FigurePipeline(args.config, args.data_dir)
        
        # Run pipeline
        success = pipeline.run(
            mode=args.mode,
            parallel=args.parallel,
            specific_figures=args.figures
        )
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 