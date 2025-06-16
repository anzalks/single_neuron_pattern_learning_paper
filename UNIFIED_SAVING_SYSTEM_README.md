# Unified Figure Saving System

**Author:** Anzal KS (anzal.ks@gmail.com)  
**Repository:** https://github.com/anzalks/  
**Date:** June 2024

## 🎯 Overview

The Unified Figure Saving System provides consistent, high-quality figure output across all plotting scripts in the pattern learning paper project. It centralizes format control, quality settings, and output management through the `baisic_plot_fuctnions_and_features.py` (bpf) module.

## ✨ Key Features

- **🎨 Multiple Format Support**: PNG, PDF, SVG, EPS
- **📊 Global Format Control**: Command-line flags control all figures
- **🎯 Quality Presets**: Standard (300 DPI) and High-Quality (600 DPI) modes
- **🔄 Multi-Format Output**: Save single figure in multiple formats simultaneously
- **⚙️ Environment Variable Control**: Scriptable format settings
- **🛡️ Backward Compatibility**: Works with existing plotting scripts
- **📁 Automatic Directory Creation**: No manual directory management needed

## 🚀 Quick Start

### Basic Usage in Plotting Scripts

```python
from shared_utils import baisic_plot_fuctnions_and_features as bpf

# Create your figure
fig, ax = plt.subplots()
# ... plotting code ...

# Save with unified system (replaces plt.savefig)
bpf.save_figure_smart(fig, output_dir, 'figure_name')
```

### Command Line Format Control

```bash
# Default PNG format
python run_plotting_scripts.py --all_fig

# Save as PDF
python run_plotting_scripts.py --all_fig --format pdf

# Save in multiple formats
python run_plotting_scripts.py --all_fig --multi_format png pdf svg

# High quality for publication (600 DPI)
python run_plotting_scripts.py --all_fig --high_quality --format pdf

# Transparent background
python run_plotting_scripts.py --all_fig --transparent --format png

# Custom DPI
python run_plotting_scripts.py --all_fig --dpi 450 --format png
```

## 📋 Available Functions

### Core Functions

#### `save_figure_smart(fig, output_dir, filename)`
**Recommended function for all new code**
- Automatically handles global format settings
- Supports both single and multiple format output
- Uses environment variables set by main script

```python
# Simple usage - respects global settings
bpf.save_figure_smart(fig, "outputs/figures", "my_figure")

# Returns list of saved file paths
saved_files = bpf.save_figure_smart(fig, output_dir, filename)
```

#### `save_figure(fig, output_dir, filename, format_override=None, quality_override=None)`
**Advanced function with manual control**
- Override global settings for specific figures
- Custom quality parameters

```python
# Override format for this figure only
bpf.save_figure(fig, output_dir, "special_figure", format_override="svg")

# Custom quality settings
custom_quality = {'dpi': 450, 'transparent': True}
bpf.save_figure(fig, output_dir, "figure", quality_override=custom_quality)
```

#### `save_figure_multiple_formats(fig, output_dir, filename, formats=['png', 'pdf'])`
**Multi-format saving**
- Save single figure in multiple formats
- Useful for publication workflows

```python
# Save in publication formats
formats = ['png', 'pdf', 'svg']
bpf.save_figure_multiple_formats(fig, output_dir, "publication_figure", formats)
```

### Utility Functions

#### `set_figure_format(format_type)`
```python
bpf.set_figure_format('pdf')  # Set global format
```

#### `get_figure_format()`
```python
current_format = bpf.get_figure_format()  # Get current format
```

#### `get_figure_quality_settings(format_type=None)`
```python
settings = bpf.get_figure_quality_settings('png')  # Get quality settings
```

#### `update_figure_quality_settings(format_type, **kwargs)`
```python
bpf.update_figure_quality_settings('png', dpi=450, transparent=True)
```

## 🔧 Command Line Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--format` | Single output format | `--format pdf` |
| `--multi_format` | Multiple output formats | `--multi_format png pdf svg` |
| `--dpi` | Custom DPI setting | `--dpi 450` |
| `--high_quality` | High quality mode (600 DPI) | `--high_quality` |
| `--transparent` | Transparent background | `--transparent` |

## 📊 Quality Settings

### Standard Quality (Default)
- **DPI**: 300
- **Background**: White
- **Transparency**: False
- **Bbox**: Tight with 0.1 padding

### High Quality Mode
- **DPI**: 600
- **Background**: White
- **Transparency**: False
- **Bbox**: Tight with 0.1 padding

### Custom Quality Override
```python
custom_settings = {
    'dpi': 450,
    'transparent': True,
    'facecolor': 'none',
    'bbox_inches': 'tight',
    'pad_inches': 0.05
}
bpf.save_figure(fig, output_dir, filename, quality_override=custom_settings)
```

## 🔄 Migration Guide

### From Old System
```python
# OLD CODE
outpath = f"{outdir}/figure_1.png"
plt.savefig(outpath, bbox_inches='tight')
```

### To New System
```python
# NEW CODE
bpf.save_figure_smart(fig, outdir, "figure_1")
```

### Migration Benefits
- ✅ Format controlled by command line
- ✅ Consistent quality across all figures
- ✅ Multiple format support
- ✅ No hardcoded file extensions
- ✅ Automatic directory creation

## 🌍 Environment Variables

The system uses these environment variables (set automatically by `run_plotting_scripts.py`):

| Variable | Description | Example |
|----------|-------------|---------|
| `FIGURE_FORMAT` | Single format | `png`, `pdf`, `svg`, `eps` |
| `FIGURE_FORMATS` | Multiple formats | `png,pdf,svg` |
| `FIGURE_DPI` | DPI setting | `300`, `600` |
| `FIGURE_TRANSPARENT` | Transparency | `True`, `False` |

## 📁 File Organization

The system maintains the existing directory structure:
```
outputs/
├── main_figures/
│   ├── Figure_1/
│   │   ├── figure_1.png
│   │   ├── figure_1.pdf
│   │   └── figure_1.svg
│   └── Figure_2/
│       └── figure_2.png
└── supplementary_figures/
    ├── supplimentary_figure_1/
    │   └── supplimentary_figure_1.png
    └── Figure_7_fnorm/
        └── figure_7_fnorm.png
```

## 🧪 Testing Examples

### Test Single Format
```bash
python run_plotting_scripts.py --figures figure_1 --format pdf
```

### Test Multiple Formats
```bash
python run_plotting_scripts.py --figures figure_1 --multi_format png pdf svg
```

### Test High Quality
```bash
python run_plotting_scripts.py --figures figure_1 --high_quality --format png
```

### Test All Figures with Custom Settings
```bash
python run_plotting_scripts.py --all_fig --format pdf --dpi 450 --transparent
```

## 🔍 Troubleshooting

### Common Issues

1. **"Module not found" error**
   - Ensure scripts import: `from shared_utils import baisic_plot_fuctnions_and_features as bpf`

2. **Format not changing**
   - Check if script uses `bpf.save_figure_smart()` instead of `plt.savefig()`

3. **Directory not created**
   - The system automatically creates directories; check permissions

4. **Quality settings not applied**
   - Verify environment variables are set by running through `run_plotting_scripts.py`

### Debug Information
```python
# Check current settings
print(f"Format: {bpf.get_figure_format()}")
print(f"DPI: {bpf.FIGURE_DPI}")
print(f"Transparent: {bpf.FIGURE_TRANSPARENT}")
print(f"Multi-formats: {bpf.FIGURE_FORMATS}")
```

## 📈 Performance Impact

| Format | File Size (typical) | Generation Time | Use Case |
|--------|-------------------|-----------------|----------|
| PNG | 1-4 MB | Fast | Web, presentations |
| PDF | 1-2 MB | Medium | Publications, print |
| SVG | 2-5 MB | Medium | Vector editing, web |
| EPS | 2-4 MB | Slow | Legacy publications |

## 🎯 Best Practices

1. **Use `save_figure_smart()`** for all new plotting scripts
2. **Test multiple formats** before final publication
3. **Use high-quality mode** for final publication figures
4. **Keep filenames consistent** across all scripts
5. **Document custom quality overrides** in script comments

## 🔮 Future Enhancements

- [ ] Automatic format optimization based on content
- [ ] Batch format conversion utilities
- [ ] Integration with version control for figure tracking
- [ ] Custom quality profiles for different journals
- [ ] Automatic figure compression for web deployment

## 📞 Support

For issues or questions:
- **Author**: Anzal KS
- **Email**: anzal.ks@gmail.com
- **Repository**: https://github.com/anzalks/

---

*This unified saving system ensures consistent, high-quality figure output across the entire pattern learning paper project while maintaining flexibility for different publication requirements.* 