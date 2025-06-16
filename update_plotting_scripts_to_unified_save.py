#!/usr/bin/env python3

"""
Script to update plotting scripts to use the unified saving system from bpf module.
Author: Anzal KS (anzal.ks@gmail.com)

This script demonstrates how to migrate from:
    plt.savefig(outpath, bbox_inches='tight')
    
To the new unified system:
    bpf.save_figure_smart(fig, output_dir, filename)
"""

import os
import re
import glob
from pathlib import Path

def update_script_save_calls(script_path):
    """
    Update a single script to use the unified saving system.
    
    This function demonstrates the migration pattern but doesn't automatically
    modify files to avoid breaking existing functionality.
    """
    print(f"\n📄 Analyzing: {script_path}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find all savefig calls
    savefig_pattern = r'plt\.savefig\s*\(\s*([^,)]+)(?:,\s*([^)]*))?\s*\)'
    matches = re.findall(savefig_pattern, content)
    
    if matches:
        print(f"   Found {len(matches)} savefig calls:")
        for i, (path_arg, other_args) in enumerate(matches, 1):
            print(f"   {i}. plt.savefig({path_arg.strip()}, {other_args.strip()})")
        
        print(f"\n   💡 Migration suggestions:")
        print(f"   1. Add import: from shared_utils import baisic_plot_fuctnions_and_features as bpf")
        print(f"   2. Replace plt.savefig calls with:")
        print(f"      # Old: plt.savefig(outpath, bbox_inches='tight')")
        print(f"      # New: bpf.save_figure_smart(fig, output_dir, filename_without_extension)")
        print(f"   3. Extract directory and filename from existing outpath variables")
    else:
        print("   ✅ No plt.savefig calls found")

def demonstrate_migration_examples():
    """Show concrete examples of how to migrate different savefig patterns."""
    
    print("\n" + "="*80)
    print("🔄 MIGRATION EXAMPLES")
    print("="*80)
    
    examples = [
        {
            "old": '''outpath = f"{outdir}/figure_1.png"
plt.savefig(outpath, bbox_inches='tight')''',
            "new": '''# Using unified saving system
bpf.save_figure_smart(fig, outdir, "figure_1")'''
        },
        {
            "old": '''plt.savefig(f"{output_dir}/supplementary_figure_2.png", 
          bbox_inches='tight', dpi=300)''',
            "new": '''# Using unified saving system (DPI controlled globally)
bpf.save_figure_smart(fig, output_dir, "supplementary_figure_2")'''
        },
        {
            "old": '''filename = "complex_figure_name"
outpath = f"{outdir}/{filename}.png"
plt.savefig(outpath, bbox_inches='tight', facecolor='white')''',
            "new": '''# Using unified saving system
filename = "complex_figure_name"
bpf.save_figure_smart(fig, outdir, filename)'''
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}:")
        print("   OLD CODE:")
        for line in example["old"].split('\n'):
            print(f"     {line}")
        print("   NEW CODE:")
        for line in example["new"].split('\n'):
            print(f"     {line}")

def show_benefits():
    """Show the benefits of the unified saving system."""
    
    print("\n" + "="*80)
    print("✨ BENEFITS OF UNIFIED SAVING SYSTEM")
    print("="*80)
    
    benefits = [
        "🎯 Consistent quality settings across all figures",
        "📊 Global format control (PNG, PDF, SVG, EPS)",
        "🔄 Multiple format support with single command",
        "⚙️  Environment variable control from main script",
        "📐 Standardized DPI and transparency settings",
        "🛡️  Backward compatibility with existing scripts",
        "🎨 Publication-ready quality presets",
        "📁 Automatic directory creation"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")

def show_usage_examples():
    """Show how to use the new system."""
    
    print("\n" + "="*80)
    print("🚀 USAGE EXAMPLES")
    print("="*80)
    
    print("\n1️⃣  Basic usage in plotting scripts:")
    print("   from shared_utils import baisic_plot_fuctnions_and_features as bpf")
    print("   ")
    print("   # Create your figure")
    print("   fig, ax = plt.subplots()")
    print("   # ... plotting code ...")
    print("   ")
    print("   # Save with unified system")
    print("   bpf.save_figure_smart(fig, output_dir, 'my_figure')")
    
    print("\n2️⃣  Command line format control:")
    print("   # Save as PNG (default)")
    print("   python run_plotting_scripts.py --all_fig")
    print("   ")
    print("   # Save as PDF")
    print("   python run_plotting_scripts.py --all_fig --format pdf")
    print("   ")
    print("   # Save in multiple formats")
    print("   python run_plotting_scripts.py --all_fig --multi_format png pdf svg")
    print("   ")
    print("   # High quality for publication")
    print("   python run_plotting_scripts.py --all_fig --high_quality --format pdf")
    print("   ")
    print("   # Transparent background")
    print("   python run_plotting_scripts.py --all_fig --transparent --format png")

def main():
    """Main function to analyze plotting scripts."""
    
    print("🔧 PLOTTING SCRIPTS UNIFIED SAVING SYSTEM MIGRATION TOOL")
    print("="*80)
    
    # Find all plotting scripts
    script_patterns = [
        "plotting_scripts/main_figures/*.py",
        "plotting_scripts/supplementary_figures/*.py"
    ]
    
    all_scripts = []
    for pattern in script_patterns:
        all_scripts.extend(glob.glob(pattern))
    
    print(f"📊 Found {len(all_scripts)} plotting scripts to analyze")
    
    # Analyze each script
    total_savefig_calls = 0
    for script_path in sorted(all_scripts):
        if os.path.basename(script_path).startswith('figure_generation_script'):
            update_script_save_calls(script_path)
    
    # Show examples and benefits
    demonstrate_migration_examples()
    show_benefits()
    show_usage_examples()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("📝 Next steps:")
    print("   1. Review the migration examples above")
    print("   2. Update your plotting scripts to use bpf.save_figure_smart()")
    print("   3. Test with different format flags")
    print("   4. Enjoy consistent, high-quality figure output! 🎉")

if __name__ == "__main__":
    main() 