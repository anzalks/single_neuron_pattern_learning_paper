#!/usr/bin/env python3
"""
Simplified CLI for Pattern Learning Figure Generation
Provides easy access to common pipeline operations

Author: Anzal KS (anzal.ks@gmail.com)
Repository: https://github.com/anzalks/
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """Simplified CLI for figure generation"""
    
    parser = argparse.ArgumentParser(
        description="Pattern Learning Paper - Simplified Figure Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Common Usage Examples:

1. Generate all figures with your data directory:
   python generate_figures.py --data-dir /path/to/your/data

2. Generate only main figures:
   python generate_figures.py --main-only --data-dir /path/to/data

3. Generate specific figures:
   python generate_figures.py --figures 1 2 3 --data-dir /path/to/data

4. Generate field-normalized analysis only:
   python generate_figures.py --field-norm --data-dir /path/to/data

5. Test with a single figure:
   python generate_figures.py --test --data-dir /path/to/data

Data Directory Structure Expected:
/your/data/directory/
├── analysis_scripts/pickle_files_from_analysis/
│   ├── pd_all_cells_mean.pickle
│   ├── all_cells_classified_dict.pickle
│   ├── all_cells_fnorm_classifeied_dict.pickle
│   └── ... (other pickle files)
├── data/pickle_files/
│   └── cell_stats.h5
└── data/illustations/
    ├── figure_2_1.jpg
    └── ... (other images)
        """
    )
    
    # Data source
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Path to your data directory (required)"
    )
    
    # Execution modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    
    mode_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate all figures (default)"
    )
    
    mode_group.add_argument(
        "--main-only", "-m",
        action="store_true",
        help="Generate only main figures (1-8)"
    )
    
    mode_group.add_argument(
        "--supp-only", "-s",
        action="store_true",
        help="Generate only supplementary figures"
    )
    
    mode_group.add_argument(
        "--field-norm", "-fn",
        action="store_true",
        help="Generate only field-normalized analysis"
    )
    
    mode_group.add_argument(
        "--standard-only", "-st",
        action="store_true",
        help="Generate only standard analysis (no field-norm)"
    )
    
    mode_group.add_argument(
        "--figures", "-f",
        nargs="+",
        type=int,
        choices=range(1, 9),
        metavar="N",
        help="Generate specific figures (1-8)"
    )
    
    mode_group.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test mode: generate only Figure 1 (fastest)"
    )
    
    # Execution options
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run figures sequentially (default: parallel)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Check for key data files
    required_paths = [
        data_dir / "analysis_scripts" / "pickle_files_from_analysis",
        data_dir / "data" / "pickle_files",
        data_dir / "data" / "illustations"
    ]
    
    missing_paths = [p for p in required_paths if not p.exists()]
    if missing_paths:
        print("Warning: Some expected data directories are missing:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nThis may cause some figures to fail. Continue anyway? (y/N): ", end="")
        if input().lower() != 'y':
            sys.exit(1)
    
    # Determine execution mode
    if args.main_only:
        mode = "main_only"
    elif args.supp_only:
        mode = "supp_only"
    elif args.field_norm:
        mode = "fnorm_only"
    elif args.standard_only:
        mode = "standard_only"
    elif args.figures:
        mode = "individual"
        # Convert figure numbers to figure names
        figure_names = [f"figure_{i}" for i in args.figures]
    elif args.test:
        mode = "individual"
        figure_names = ["figure_1"]
    else:
        mode = "all"
    
    # Build command for run_all_figures.py
    cmd = [sys.executable, "run_all_figures.py"]
    cmd.extend(["--mode", mode])
    cmd.extend(["--data-dir", str(data_dir)])
    
    if mode == "individual":
        cmd.extend(["--figures"] + figure_names)
    
    if args.sequential:
        cmd.append("--sequential")
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Show command
    print("=" * 60)
    print("PATTERN LEARNING FIGURE GENERATION")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Mode: {mode}")
    if mode == "individual":
        print(f"Figures: {', '.join(figure_names)}")
    print(f"Execution: {'Sequential' if args.sequential else 'Parallel'}")
    print()
    
    if args.dry_run:
        print("Dry run - would execute:")
        print(" ".join(cmd))
        return
    
    print("Starting figure generation...")
    print("Command:", " ".join(cmd))
    print()
    
    # Execute the pipeline
    try:
        import subprocess
        result = subprocess.run(cmd, cwd=os.getcwd())
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 