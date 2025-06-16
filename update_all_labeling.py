#!/usr/bin/env python3
"""
Universal Subplot Labeling Update Script
Author: Anzal KS (anzal.ks@gmail.com)

This script automatically updates all figure generation scripts to use
the universal labeling method while preserving exact text positions
and all other visual features.

Updates:
1. Direct text() calls -> bpf.add_subplot_label()
2. label_axis() calls -> bpf.add_subplot_labels_from_list()
3. Array-based panel labels -> bpf.add_subplot_label()
4. Sequential capital letters -> bpf.add_subplot_label()
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


class LabelingUpdater:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.plotting_scripts_dir = self.base_dir / "plotting_scripts"
        self.backup_dir = self.base_dir / "backup_original_scripts"
        
        # Patterns for different labeling methods
        self.patterns = {
            # Pattern 1: Direct text calls with single letters
            'direct_text': re.compile(
                r"(\w+)\.text\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*['\"]([A-M])['\"]"
                r"\s*,\s*transform\s*=\s*\1\.transAxes\s*,"
                r"\s*fontsize\s*=\s*(\d+)"
                r"(?:\s*,\s*fontweight\s*=\s*['\"](\w+)['\"])?"
                r"(?:\s*,\s*ha\s*=\s*['\"](\w+)['\"])?"
                r"(?:\s*,\s*va\s*=\s*['\"](\w+)['\"])?\s*\)"
            ),
            
            # Pattern 2: label_axis calls
            'label_axis': re.compile(
                r"label_axis\(\s*([^,]+)\s*,\s*['\"]([A-M])['\"]"
                r"(?:\s*,\s*xpos\s*=\s*(-?[\d.]+))?"
                r"(?:\s*,\s*ypos\s*=\s*(-?[\d.]+))?\s*\)"
            ),
            
            # Pattern 3: Panel labels in loops
            'panel_labels': re.compile(
                r"panel_labels\s*=\s*\[(.*?)\].*?"
                r"(\w+)\.text\(\s*\w+\s*,\s*\w+\s*,\s*panel_labels\[i\]",
                re.DOTALL
            )
        }
    
    def create_backup(self):
        """Create backup of original scripts"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        # Copy the entire plotting_scripts directory
        shutil.copytree(self.plotting_scripts_dir, self.backup_dir / "plotting_scripts")
        print(f"✓ Created backup at: {self.backup_dir}")
    
    def get_all_python_files(self) -> List[Path]:
        """Get all Python files in plotting_scripts directory"""
        python_files = []
        for root, dirs, files in os.walk(self.plotting_scripts_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def update_direct_text_calls(self, content: str) -> str:
        """Update direct text() calls to use bpf.add_subplot_label()"""
        def replace_direct_text(match):
            axis_name = match.group(1)
            xpos = match.group(2)
            ypos = match.group(3)
            label = match.group(4)
            fontsize = match.group(5)
            fontweight = match.group(6) or 'bold'
            ha = match.group(7) or 'center'
            va = match.group(8) or 'center'
            
            return (f"bpf.add_subplot_label({axis_name}, '{label}', "
                   f"xpos={xpos}, ypos={ypos}, fontsize={fontsize}, "
                   f"fontweight='{fontweight}', ha='{ha}', va='{va}')")
        
        return self.patterns['direct_text'].sub(replace_direct_text, content)
    
    def update_label_axis_calls(self, content: str) -> str:
        """Update label_axis() calls to use bpf.add_subplot_labels_from_list()"""
        def replace_label_axis(match):
            axes_list = match.group(1).strip()
            letter = match.group(2)
            xpos = match.group(3) or '-0.1'
            ypos = match.group(4) or '1.1'
            
            # Generate variable names
            var_name = f"{letter.lower()}_axes"
            labels_var = f"{letter.lower()}_labels"
            
            return (f"{var_name} = {axes_list}\n    "
                   f"{labels_var} = bpf.generate_letter_roman_labels('{letter}', len({var_name}))\n    "
                   f"bpf.add_subplot_labels_from_list({var_name}, {labels_var}, \n"
                   f"                                base_params={{'xpos': {xpos}, 'ypos': {ypos}, "
                   f"'fontsize': 16, 'fontweight': 'bold'}})")
        
        return self.patterns['label_axis'].sub(replace_label_axis, content)
    
    def update_panel_labels(self, content: str) -> str:
        """Update panel_labels array-based approach"""
        # This is more complex and may need manual review
        # For now, we'll identify and flag these for manual update
        matches = self.patterns['panel_labels'].finditer(content)
        for match in matches:
            print(f"⚠️  Found panel_labels pattern - may need manual review")
        return content
    
    def add_import_if_needed(self, content: str) -> str:
        """Add bpf import if not present"""
        if 'from shared_utils import baisic_plot_fuctnions_and_features as bpf' in content:
            return content
        
        # Find existing imports and add after them
        import_pattern = re.compile(r'(from shared_utils import.*?)\n', re.MULTILINE)
        if import_pattern.search(content):
            return content  # Already has shared_utils import
        
        # Add import after other imports
        lines = content.split('\n')
        import_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_index = i + 1
        
        lines.insert(import_index, 'from shared_utils import baisic_plot_fuctnions_and_features as bpf')
        return '\n'.join(lines)
    
    def update_file(self, file_path: Path) -> Dict[str, int]:
        """Update a single file and return statistics"""
        print(f"Updating: {file_path.relative_to(self.base_dir)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        updated_content = original_content
        stats = {'direct_text': 0, 'label_axis': 0, 'panel_labels': 0}
        
        # Count and update direct text calls
        direct_matches = len(self.patterns['direct_text'].findall(updated_content))
        updated_content = self.update_direct_text_calls(updated_content)
        stats['direct_text'] = direct_matches
        
        # Count and update label_axis calls
        axis_matches = len(self.patterns['label_axis'].findall(updated_content))
        updated_content = self.update_label_axis_calls(updated_content)
        stats['label_axis'] = axis_matches
        
        # Handle panel_labels (flagged for manual review)
        panel_matches = len(self.patterns['panel_labels'].findall(updated_content))
        updated_content = self.update_panel_labels(updated_content)
        stats['panel_labels'] = panel_matches
        
        # Add import if needed
        updated_content = self.add_import_if_needed(updated_content)
        
        # Write updated content
        if updated_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"  ✓ Updated: {sum(stats.values())} labeling calls")
        else:
            print(f"  - No changes needed")
        
        return stats
    
    def update_all_scripts(self) -> Dict[str, Dict[str, int]]:
        """Update all plotting scripts"""
        python_files = self.get_all_python_files()
        all_stats = {}
        
        print(f"Found {len(python_files)} Python files to process")
        print("=" * 60)
        
        for file_path in python_files:
            try:
                stats = self.update_file(file_path)
                all_stats[str(file_path.relative_to(self.base_dir))] = stats
            except Exception as e:
                print(f"  ✗ Error updating {file_path}: {e}")
                all_stats[str(file_path.relative_to(self.base_dir))] = {'error': str(e)}
        
        return all_stats
    
    def print_summary(self, all_stats: Dict[str, Dict[str, int]]):
        """Print update summary"""
        print("\n" + "=" * 60)
        print("UPDATE SUMMARY")
        print("=" * 60)
        
        total_direct = sum(stats.get('direct_text', 0) for stats in all_stats.values())
        total_axis = sum(stats.get('label_axis', 0) for stats in all_stats.values())
        total_panel = sum(stats.get('panel_labels', 0) for stats in all_stats.values())
        total_files = len([f for f, s in all_stats.items() if sum(s.values() if isinstance(s, dict) and 'error' not in s else [0]) > 0])
        
        print(f"Files processed: {len(all_stats)}")
        print(f"Files updated: {total_files}")
        print(f"Direct text() calls updated: {total_direct}")
        print(f"label_axis() calls updated: {total_axis}")
        print(f"Panel labels flagged: {total_panel}")
        print(f"Total labeling calls updated: {total_direct + total_axis}")
        
        # Show files with errors
        error_files = [f for f, s in all_stats.items() if isinstance(s, dict) and 'error' in s]
        if error_files:
            print(f"\n⚠️  Files with errors: {len(error_files)}")
            for file in error_files:
                print(f"  - {file}")
        
        print(f"\n✓ Labeling standardization complete!")
        print(f"✓ Original scripts backed up to: {self.backup_dir}")
        print("\nNOTE: Please review and test the updated scripts before running them.")


def main():
    """Main execution function"""
    print("Universal Subplot Labeling Update Script")
    print("Author: Anzal KS (anzal.ks@gmail.com)")
    print("=" * 60)
    
    updater = LabelingUpdater()
    
    # Create backup
    updater.create_backup()
    
    # Update all scripts
    all_stats = updater.update_all_scripts()
    
    # Print summary
    updater.print_summary(all_stats)


if __name__ == "__main__":
    main() 