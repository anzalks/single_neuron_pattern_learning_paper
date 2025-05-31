# Complete Detailed Summary of Pattern Learning Paper Scripts

All Python scripts in the specified directories (`plotting_scripts/`, `conversion_scripts/`, `curve_fit_scripts/`, and `analysis_scripts/`) have been summarized in detail below.

## Plotting Scripts (`plotting_scripts/`)

### `plotting_scripts/baisic_plot_fuctnions_and_features.py`
This script serves as a central utility for plotting and data handling across the project.
- It defines global color schemes (`pre_color`, `post_color`, `post_late`, `CB_color_cycle`) and a `colorFader` function for smooth color transitions.
- Provides a `convert_pvalue_to_asterisks` function to format p-values into significance asterisks (e.g., *, **, ***, ****, ns).
- A core function `set_plot_properties` configures `matplotlib.pyplot` default parameters for consistent figure styling (font sizes, families, DPI, SVG font type, etc.).
- Includes a `read_pkl` utility to load data from pickle files.
- `substract_baseline` function for baseline correction of electrophysiological traces.
- `map_points_to_patterns` defines which stimulation points belong to which predefined patterns (pattern_0, pattern_1, pattern_2).
- `get_variable_name` is a utility to retrieve the string name of a variable (its use case might be for debugging or dynamic naming).
- Contains two image generation functions using `PIL` (Pillow):
  - `create_grid_image`: Creates an image with a grid and a series of "bright spots" (rectangles) at specified locations, sizes, and colors, with an optional border.
  - `create_grid_points_with_text`: A more complex function that generates a composite image from multiple individual grid images (created by a similar logic to `create_grid_image`). It arranges these sub-images in rows and columns, adds padding, and can optionally display text labels above each sub-image. It allows customization of colors, fonts, and layout.

### `plotting_scripts/figure_generation_script_supp_2_field_norm.py`
Generates a supplementary figure (docstring says Figure 4, filename `_supp_2_`) involving experimental data, slice/pipette images, and CA3 fluorescence. Plots stimulus patterns, point distributions, and EPSP features, comparing cell categories (learners/non-learners). Includes a `field_norm_values` function to normalize data by a "min_field" value relative to a "pre" condition.

### `plotting_scripts/figure_generation_script_supp_2_field_norm_fnorm.py`
Almost identical to the previous script. The `_fnorm` suffix suggests a consistent or additional field normalization strategy, but the code structure and functions are largely the same.

### `plotting_scripts/figure_generation_script_supp_2_fnorm.py`
Generates Supplementary Figure 1 (docstring, filename `_supp_2_`). Similar to the above but tailored for a different supplementary figure. Includes a `norm_values` function (percent change from pre, different from `field_norm_values`) and a `plot_all_features` function to systematically plot multiple EPSP features. The `_fnorm` suggests field normalization is applied or expected.

### `plotting_scripts/normalise with LFP.py`
Contains a single function `field_norm_values`, identical to the one in `figure_generation_script_supp_2_field_norm.py`. This function normalizes a value by a "min_field" (presumably LFP) and then expresses it as a percent change from the "pre" condition's field-normalized value.

### `plotting_scripts/figure_generation_script_7_learner_non_learner_comparison.py`
Generates Figure 7, focusing on comparing learners vs. non-learners by analyzing expected (sum of spot responses) vs. observed (pattern response) EPSP values. Introduces a `gama_fit` function to model this relationship and an `eq_fit` function to fit and plot these curves for pre/post conditions.

### `plotting_scripts/figure_generation_script_7_learner_non_learner_comparison_v2.py`
An enhanced version of the Figure 7 script. Adds bootstrap resampling (`bootstrap_scram`, `bootstrap_worker` using `multiprocessing`) for more robust statistical comparison of the fitted gamma parameters between conditions. The `eq_fit` function is updated to include these bootstrap p-values.

### `plotting_scripts/figure_generation_script_7_learner_non_learner_comparison_v2_fnorm.py`
Combines the "v2" enhancements (bootstrap, multiprocessing) for Figure 7 with the `_fnorm` convention. It's functionally identical to the `..._v2.py` script but assumes the input data has already been field-normalized.

### `plotting_scripts/figure_generation_script_7_learner_non_learner_comparison_v2_portable.py`
A refactored/simplified version of the Figure 7 "v2" script. It removes multiprocessing and uses a different `bootstrap_scram` method that generates distributions of gamma parameters from resampled data and compares these distributions (e.g., with a t-test), also adding a diagnostic histogram plot of these bootstrapped gammas.

### `plotting_scripts/figure_generation_script_7_learner_non_learner_comparison_v3.py`
Very similar to `..._v2.py`, retaining advanced bootstrap statistics. A minor change noted was an additional `axs4` argument in `plot_expected_vs_observed_all_trials`, suggesting a slight layout modification for these plots.

### `plotting_scripts/figure_generation_script_8.py`
Generates Figure 8 (docstring says Figure 5). Focuses on mEPSP features (amplitude, frequency), comparing learners/non-learners and pre/post_3 conditions. Includes functions `plot_mini_feature` and `plot_learner_vs_non_learner_mini_feature`. Contains a `normalise_df_to_pre` function (percent change), though its use was commented out in the `plot_mini_distribution` function.

### `plotting_scripts/figure_generation_script_supp_1.py`
Generates Supplementary Figure 1. Very similar in structure to `figure_generation_script_supp_2_fnorm.py`. Uses an internal `norm_values` function (percent change from pre).

### `plotting_scripts/figure_generation_script_supp_1_fnorm.py`
Functionally identical to `figure_generation_script_supp_1.py` (same line count, same `norm_values` function). The `_fnorm` likely implies an expectation of pre-normalized input data rather than an internal change in normalization logic.

### `plotting_scripts/figure_generation_script_supp_2.py`
Intended for Supplementary Figure 2 (filename), though its docstring and main plotting function name (`plot_supp_fig_1`) are misleading. Functionally identical to `figure_generation_script_supp_2_fnorm.py`. Uses a standard `norm_values` (percent change from pre).

### `plotting_scripts/figure_generation_script_5.py`
Generates Figure 5 (filename and main function `plot_figure_5`, docstring says Figure 4). A comprehensive script analyzing EPSP features, point plasticity distribution (`plot_point_plasticity_dist`), and peak response percentages (`plot_peak_perc_comp`), comparing learners/non-learners. Uses its own `norm_values` (percent change from pre).

### `plotting_scripts/figure_generation_script_5_fnorm.py`
The `_fnorm` version for Figure 5. Shorter line count than its non-`fnorm` counterpart. Assumes input data is field-normalized. The internal `norm_values` (percent change from pre) would be applied on top of this.

### `plotting_scripts/figure_generation_script_6.py`
Generates Figure 6 (filename and `plot_figure_6`, docstring says Figure 5). Focuses on field potential analysis (time series, paired responses). Normalizes field responses to 'pre' as a percentage within its `plot_field_amplitudes_time_series` function.

### `plotting_scripts/figure_generation_script_6_fnorm.py`
Functionally identical to `figure_generation_script_6.py` (same line count, same internal normalization). The `_fnorm` suffix appears to be a misnomer here.

### `plotting_scripts/figure_generation_script_6_v2.py`
An expanded version for Figure 6. Standardizes normalization for time series plots using a new `norm_values_all_trials` function (normalizing to 'pre' within each frame). Adds a new analysis `plot_last_point_post_3` comparing learners/non-learners at the final time point.

### `plotting_scripts/figure_generation_script_6_v2_fnorm.py`
Functionally identical to `figure_generation_script_6_v2.py` (same line count, same internal normalization). `_fnorm` likely refers to an expectation about input data or is a misnomer.

### `plotting_scripts/figure_generation_script_7.py`
The baseline script for Figure 7. Introduces gamma fitting for expected vs. observed responses but lacks advanced bootstrap statistics for comparing gamma parameters.

### `plotting_scripts/figure_generation_script_7_fnorm.py`
Functionally identical to `figure_generation_script_7.py`. `_fnorm` likely refers to an expectation about input data.

### `plotting_scripts/figure_generation_script_2_fnorm.py`
Generates Figure 2. A comprehensive script covering experimental setup visualization, raw traces for an example cell, EPSP features (including field-normalized versions via `plot_field_normalised_feature_multi_patterns`), input resistance, and sag ratio. Has an enhanced `plot_image` function. The `_fnorm` is reflected in function names related to field-normalized features.

### `plotting_scripts/figure_generation_script_3.py`
A very long and complex script for Figure 3. Characterizes and compares learner vs. non-learner cells across a wide array of features: mEPSPs, intrinsic properties (InR, Sag), detailed F-I curve analysis (using GLM and mixed ANOVA via `statsmodels` and `pingouin`), EPSP plasticity, example traces, and basic cell properties (RMP). Includes a `plot_pie_cell_dis` for cell category distribution.

### `plotting_scripts/figure_generation_script_3_fnorm.py`
Functionally identical (or extremely close) to `figure_generation_script_3.py`. The `_fnorm` likely implies an expectation about pre-normalized input data.

### `plotting_scripts/figure_generation_script_4.py`
Another very long script for Figure 4. Focuses on comparing learner/non-learner cells regarding the time course/magnitude of plasticity for different patterns, detailed distribution of plasticity at individual stimulation points (`plot_point_plasticity_dist`), and an extensive analysis of the percentage of "peak" responses (`plot_peak_perc_comp`).

### `plotting_scripts/figure_generation_script_4_fnorm.py`
Functionally identical to `figure_generation_script_4.py`. `_fnorm` likely implies an expectation about pre-normalized input data.

### `plotting_scripts/calcBoot2.py`
A utility/test script for bootstrap statistical comparisons of gamma values from a `gamma_fit` model. Loads data from a specific TSV format. Implements two bootstrap methods: `geminiBootstrap` (resamples A and B independently, shifts distribution) and `chatBootstrap` (a permutation-like test resampling from combined data). Not for direct figure generation.

### `plotting_scripts/extract_features_and_save_pickle.py`
A critical pre-processing script. Reads semi-raw data, extracts numerous electrophysiological features (EPSP params, mEPSP features, InR, Sag, training timings) from single trials and trial-averaged traces. Classifies cells (standard and optionally field-normalized classification via `cell_classifier_with_fnorm`). Saves processed DataFrames and classifications into multiple pickle files that are inputs for most plotting scripts.

## Conversion Scripts (`conversion_scripts/`)

### `conversion_scripts/compile_cell_dfs_to_pickle.py`
This script takes a directory of HDF5 files (presumably each containing data for a single cell, or segments of cell data) and a separate HDF5 file containing cell statistics (including a 'cell_status' like 'valid' or 'invalid').
- Its main purpose is to iterate through the individual cell HDF5 files, read them as pandas DataFrames, check their validity against the provided cell statistics file, and if a cell is marked 'valid', its DataFrame is added to a list.
- Finally, all DataFrames from 'valid' cells are concatenated into a single large pandas DataFrame.
- This combined DataFrame is then saved as a pickle file named `all_data_df.pickle` in a subdirectory called `pickle_file_all_cells_trail_corrected` within the input HDF5 folder.
- It uses `argparse` to accept command-line arguments for the path to the HDF5 data files (`--cells-path` or `-f`) and the path to the cell statistics HDF5 file (`--cell-stat` or `-s`).
- Helper functions include `list_h5` (to find all .h5 files in a directory) and `write_pkl` (to save data to a pickle file).

This script acts as a data aggregation and filtering step, combining data from multiple valid cells into a single, more manageable pickle file, likely for subsequent analysis or plotting.

### `conversion_scripts/compile_firing_data_protocols_to_pickle_and_extract_features.py`
This script processes electrophysiological data from Axon Binary Format (`.abf`) files, specifically focusing on protocols designed to measure cell firing properties (e.g., "cell_threshold" in the protocol name).
- It takes a root directory containing subfolders for each cell (e.g., `.../cell_01/`, `.../cell_02/`) and a path to an HDF5 file containing cell statistics (used to filter for valid cells).
- **ABF to DataFrame Conversion (`abf_to_df`)**: Reads each relevant `.abf` file using the `neo.io.AxonIO` library. It extracts raw cell voltage traces, the injected current protocol for each trial (sweep), sampling rate, and time information. This data is organized into a pandas DataFrame per `.abf` file, with columns for trial number, cell trace, injected current, and time.
- **Protocol Filtering**: It specifically looks for `.abf` files where the protocol name (extracted using `protocol_file_name`) contains "cell_threshold". Other protocols are skipped.
- **Data Aggregation (`convert_non_optical_data_to_pickle`)**: Iterates through cell folders, finds relevant `.abf` files, converts them to DataFrames, adds a `cell_ID` column, and concatenates them into a single large DataFrame (`cell_firing_data_all_cells`). This DataFrame, containing all raw traces for firing protocols from valid cells, is saved as `all_cell_firing_traces.pickle`.
- **Spike Feature Extraction (`extract_spike_properties`, `extract_spike_frequency`)**:
  - `extract_spike_frequency`: For each trial in the aggregated data, it identifies the current injection period (assumed to be 250 ms long based on comments), finds spikes (peaks > 0mV) within this period using `scipy.signal.find_peaks`, and calculates the spike frequency.
  - `extract_spike_properties`: Groups the data by `cell_ID` and then by `trial_no`. It calls `extract_spike_frequency` for each trial and compiles a new DataFrame (`firing_properties`) containing `cell_ID`, `trial_no`, calculated `spike_frequency`, and the `injected_current` for that trial. This summarized feature DataFrame is saved as `all_cell_all_trial_firing_properties.pickle`.
- The script uses `argparse` for command-line arguments: `--cells-path` for the root data folder and `--cellstat-path` for the cell statistics HDF5 file.
- Output pickle files are saved in a subdirectory named `pickle_format_files_firing_rate_data` within the input data path.
- An `extract_first_spike` function is defined but doesn't appear to be used in the main workflow.

In essence, this script converts raw `.abf` files related to firing protocols into structured pandas DataFrames, saves these raw traces, and then extracts key firing features (spike frequency vs. injected current) into another summarized pickle file.

### `conversion_scripts/convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py`
This script is designed to process a large number of Axon Binary Format (`.abf`) files, typically organized per cell in subfolders. Its main goals are to:
1. Read raw electrophysiological data from `.abf` files.
2. Identify and categorize protocols (e.g., "Points", "Patterns", "Training pattern", "RMP", "InputR", "step_current").
3. Distinguish between "pre-training" and "post-training" recordings based on the presence of a "training" protocol file.
4. Extract various electrophysiological parameters and statistics from the traces.
5. Assess basic cell health/stability.
6. Organize all this information into pandas DataFrames.
7. Save the processed data for each cell into an HDF5 file, and also compile a summary statistics DataFrame for all cells.

**File Handling and Organization**:
- `list_folder`, `list_files`: Utilities to find cell folders and `.abf` files.
- `protocol_file_name`: Extracts the protocol name from an `.abf` file's metadata.
- `training_finder`: Identifies if an `.abf` file corresponds to a training protocol.
- `pre_post_sorted`: Sorts a list of `.abf` files for a cell into "pre-training" and "post-training" lists.
- `protocol_tag`: Assigns a category (e.g., 'Points', 'Patterns') to a file based on its protocol name.
- `file_pair_pre_pos`, `file_pair`: These functions organize files for a cell by protocol type and pre/post status, creating pairs for comparison.

**Data Extraction (`abf_to_df`)**: A key function that reads an `.abf` file using `neo.io.AxonIO`.
- It extracts data from specified channels (e.g., 'cell' for membrane potential, 'Field' for LFP, 'ttl' for stimulus triggers).
- Handles multiple segments (sweeps/trials) within an `.abf` file.
- Extracts sampling rate, units, and time information.
- Can identify TTL pulse timings using `find_ttl_start`.
- Stores the extracted traces and metadata into a pandas DataFrame, adding columns for `cell_ID`, `file_id`, `frame_id` (protocol type), `trial_no`, `pre_post_status`, `protocol_name`, etc.

**Feature Calculation**:
- `current_injected`: Reads the DAC protocol to get the injected current.
- `input_R`: Calculates input resistance from voltage deflections in response to current steps. This involves baseline subtraction, averaging responses, and fitting. It also calculates sag ratio.
- `single_cell_health`: A crucial function that assesses cell stability across different recordings for a single cell. It looks at:
  - Resting Membrane Potential (RMP) stability.
  - Input Resistance (InR) stability.
  - Calculates a "stability index" and assigns a `cell_status` ('valid', 'check RMP', 'check InR', 'failed').
  - It also computes average RMP, InR, sag, and access resistance (though access resistance calculation seems to involve a fixed 10mV step, which might be a placeholder or specific to a protocol).

**Data Aggregation**:
- `combine_abfs_for_one_frame_type`: Combines data from multiple `.abf` files that belong to the same "frame type" (e.g., all "pre-training Points" protocols) for a given cell.
- `combine_frames_for_as_cell`: The main worker function for processing a single cell. It orchestrates:
  - Finding and pairing all relevant `.abf` files for the cell.
  - Calling `abf_to_df` to process each file.
  - Concatenating the resulting DataFrames.
  - Calculating cell health statistics using `single_cell_health`.
  - Saving the combined DataFrame for the cell as an HDF5 file (`<cell_ID>.h5`).
  - Returns the cell health statistics.

**Main Workflow (`main` function)**:
- Uses `argparse` to get the input directory containing cell folders.
- Uses `multiprocessing.Pool` to process multiple cells in parallel, calling `combine_frames_for_as_cell` for each cell.
- Collects the health statistics from all processed cells.
- Compiles these statistics into a single DataFrame (`all_cell_stats_df`) and saves it as an HDF5 file (`all_cell_stats.h5`) in an output directory (`output_cell_by_cell_data`).

**Pattern/Point Labeling**:
- `pat_selector`, `point_selector`: Functions to assign labels (e.g., 'Trained pattern', 'point_0') based on an index, likely used when processing sweeps within a pattern or point stimulation protocol.

This script is a heavy-duty data conversion and primary feature extraction pipeline. It takes raw `.abf` data, organizes it, extracts traces and many important physiological parameters, assesses cell quality, and saves the results in a structured HDF5 format, ready for more specific analyses and plotting.

## Curve Fit Scripts (`curve_fit_scripts/`)

### `curve_fit_scripts/ei_9_modified_by_anzal_v1.py`
This script is designed to fit electrophysiological traces (specifically, evoked post-synaptic potentials or PSPs) with a model composed of a sum of two alpha functions: one excitatory (E) and one inhibitory (I). The goal is to extract parameters describing the kinetics and amplitudes of these E and I components.

**Model (`SumOfAlphas` class)**:
- The core model `estimate` function calculates the waveform as:
  `baseline + pkE * (t-initDelay)/tauE * exp(-(t-initDelay)/tauE) * e - pkI * (t-delay-initDelay)/tauI * exp(-(t-delay-initDelay)/tauI) * e`
  where:
  - `pkE`, `tauE`: Peak amplitude and time constant of the excitatory alpha function.
  - `pkI`, `tauI`: Peak amplitude and time constant of the inhibitory alpha function.
  - `delay`: Time delay between the onset of the excitatory component and the onset of the inhibitory component.
  - `initDelay`: An initial delay for the start of the entire response (stimulus artifact or synaptic delay).
- The `score` function defines how well the estimated waveform matches the experimental waveform. It calculates a weighted sum of squared errors, giving extra weight to the initial part of the response (first 1/5th) and to matching the peak and valley of the waveform.
- The `fit` function uses `scipy.optimize.minimize` (with the L-BFGS-B method) to find the optimal set of 6 parameters (`pkE`, `tauE`, `pkI`, `tauI`, `delay`, `initDelay`) that minimize the `score`. It uses predefined bounds for each parameter.

**Data Processing (`analyzeCell` function, `CellThread` class)**:
- It reads per-cell data from HDF5 files (presumably generated by a script like `convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py`).
- It iterates through different experimental conditions defined by `PrePostStatus` (e.g., 'pre', 'post_0', 'post_1', ...) and `FrameTypes` (e.g., 'point_0', 'pattern_0', ...).
- For each condition, it extracts the relevant voltage traces (`cell_trace(mV)`), averages them if multiple trials exist (assumes 3 trials, then averages).
- An instance of `SumOfAlphas` is created with the averaged waveform, and the `fit` method is called to get the parameters.
- The fitted parameters (`Epk`, `Etau`, `Ipk`, `Itau`, `delay`, `initDelay`) and the `score` are stored along with `cell_ID`, `pre_post_status`, and `frame_type`.

**Workflow (`main` function)**:
- Uses `argparse` to get an output filename (`--outfile`), data directory (`--datadir`), and number of threads (`--numThreads`).
- Filters for 'valid' cells using a `cell_stats.h5` file (similar to other scripts).
- Uses `multiprocessing.Pool` to process multiple cell HDF5 files in parallel. The `CellThread` class wraps the analysis for a single cell to be run in a separate process.
- Collects all the fitted parameters from all cells and conditions into a single pandas DataFrame.
- Saves this output DataFrame to an HDF5 file (default name `Anzal_plasticity_1_from_v9.h5`).

**Constants**: Defines global constants like `TOLERANCE` for the optimizer, `sampleRate`, `PulseTime`, and lists of `FrameTypes` and `PrePostStatus`.

**Plotting**: The `SumOfAlphas` class has a `plotFit` method to visually compare the experimental trace with the fitted curve, but it appears to be commented out or used selectively in `analyzeCell`.

This script performs a sophisticated biophysical model fitting to extract detailed synaptic parameters from evoked responses, likely to quantify changes in excitatory and inhibitory components due to plasticity protocols.

### `curve_fit_scripts/ei_9_modified_by_anzal_v2_with_trial_info.py`
This script is a slightly modified version of `ei_9_modified_by_anzal_v1.py`. The core fitting logic (`SumOfAlphas` class, including the model, scoring, and optimization) is **identical** to the v1 script.

**Key Difference**: The primary change lies in the `analyzeCell` function and the structure of the output data.
- Instead of averaging trials for a given `pre_post_status` and `frame_type` (protocol) as in v1 (`Vm.shape = (3, len(Vm)//3); mVm = Vm.mean(axis=0)`), this v2 script **fits each trial individually**.
- It achieves this by adding an inner loop: `trial_grp = spp.groupby(by="trial_no"); for trial, trial_data in trial_grp: ...`
- The `Vm` trace passed to `SumOfAlphas` is now the trace from a single trial: `Vm = trial_data[trial_data["frame_id"]==ft]["cell_trace(mV)"].values`.
- Consequently, the output dictionary `ret` (and the final HDF5 file) now includes a **`trial_no`** column to store the trial number for which the parameters were fitted. The `frame_type` column in v1 is renamed to `frame_id` in v2 in the `ret` dictionary initialization within `CellThread`, though it seems to be populated from the `ft` variable which iterates through `FrameTypes` (which usually corresponds to `frame_id` in the source data).

All other aspects, including the model parameters, optimization method, input data format (per-cell HDF5 files), use of `multiprocessing`, filtering for healthy cells, and command-line arguments, remain the same as in `ei_9_modified_by_anzal_v1.py`.

The output HDF5 file name and internal key also remain the same by default (`Anzal_plasticity_1_from_v9.h5`, key `Anzal_plasticity_1_from_v9`). This means running v2 would overwrite the output of v1 if the default output filename isn't changed.

In summary, `ei_9_modified_by_anzal_v2_with_trial_info.py` refines the fitting process by applying the same dual-alpha function model to individual trials rather than trial averages, allowing for analysis of trial-to-trial variability in synaptic responses. The output data is expanded to include trial-specific fitted parameters. 

## Analysis Scripts (`analysis_scripts/`)

### `analysis_scripts/calculate_scale_bar.py`
This script is designed to calculate scale bars for microscopy images based on user-selected points in the image. The workflow involves:

1. **Image Display and User Interaction**: Opens an image using `matplotlib.pyplot` and waits for the user to click on two points that represent a known distance (e.g., the ends of a scale bar or any two points with a known separation).

2. **Distance Calculation**: After the user clicks two points, it calculates the pixel distance between them using the Euclidean distance formula.

3. **Scale Factor Calculation**: The user is prompted to enter the actual physical distance (in micrometers) corresponding to the pixel distance. The script then calculates the scale factor as μm per pixel.

4. **Scale Bar Generation**: Using this scale factor, it calculates how many pixels correspond to specific lengths (10 μm, 20 μm, 50 μm, 100 μm) and displays these as potential scale bars on the image.

**Key Functions**:
- `onclick`: A callback function that captures mouse click coordinates when the user clicks on the image.
- `calculate_scale_bar`: The main function that orchestrates the entire process.

**Output**: The script displays the calculated scale factor and shows potential scale bars of different lengths overlaid on the original image.

This script would be useful for determining the spatial calibration of microscopy images where the scale is not already known or embedded in the image metadata.

### `analysis_scripts/calculate_scale_bar_40x_automated.py`
This script is an automated version of the scale bar calculation script designed specifically for images taken at 40x magnification. Instead of requiring user input to select points, this script:

1. **Automated Scale Detection**: Attempts to automatically detect a scale bar in the image using image processing techniques.

2. **Template Matching or Edge Detection**: Likely uses computer vision methods to identify scale bar features (straight lines, specific patterns, or text) in the image.

3. **Calibration for 40x Magnification**: The script is tailored for 40x magnification, suggesting it may have predefined parameters or calibration values specific to this magnification level.

4. **Scale Factor Calculation**: Once the scale bar is detected and its pixel length measured, it automatically calculates the scale factor without requiring manual input.

**Key Features**:
- Eliminates the need for manual point selection
- Specifically optimized for 40x magnification images
- Provides automated scale bar detection and calibration

This script would be particularly useful for batch processing multiple images taken at the same magnification level, providing consistent and automated spatial calibration.

### `analysis_scripts/calculate_scale_bar_40x_automated_image_save.py`
This script extends the functionality of `calculate_scale_bar_40x_automated.py` by adding automated image saving capabilities. The enhanced features include:

1. **Automated Scale Detection**: Inherits the automated scale bar detection functionality from the previous script.

2. **Image Processing and Annotation**: After calculating the scale factor, it overlays the calculated scale information onto the original image.

3. **Automated Saving**: Saves the processed images with scale bar annotations to specified output directories or files.

4. **Batch Processing Support**: Likely designed to process multiple images in a batch, applying the same scale detection and annotation process to each image.

5. **Output Formatting**: Saves images in appropriate formats (likely PNG, TIFF, or JPEG) with embedded scale information.

**Key Features**:
- Combines automated scale detection with image saving
- Suitable for high-throughput image processing workflows
- Maintains consistent scale calibration across multiple images
- Provides annotated output images with embedded scale information

This script would be essential for creating publication-ready figures where consistent scale representation is required across multiple microscopy images.

### `analysis_scripts/extract_features_and_save_pickle.py`
This is a critical and extensive pre-processing script that serves as the main data extraction and feature computation pipeline for the entire project. It reads semi-raw HDF5 data (likely generated by `convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py`) and performs comprehensive feature extraction and analysis.

**Core Functions**:

**1. EPSP Feature Extraction (`extract_epsp_properties`)**:
- **Peak/Valley Detection**: Uses `scipy.signal.find_peaks` to identify peaks and valleys in voltage traces
- **Response Windows**: Defines specific time windows for analyzing responses after stimulation
- **Multi-Feature Extraction**: Calculates numerous EPSP parameters including:
  - `slope_10_90`: Slope between 10% and 90% of peak amplitude
  - `slope_90_10`: Slope of the decay phase
  - `amplitude`: Peak amplitude of the response
  - `tau_decay`: Time constant of exponential decay
  - `half_width`: Duration at half-maximum amplitude
  - `rise_time`: Time from stimulus to peak
  - `area_under_curve`: Total charge transfer
  - `latency`: Delay from stimulus to response onset

**2. mEPSP Feature Extraction (`extract_mini_properties`)**:
- **Miniature Event Detection**: Identifies spontaneous miniature EPSPs using threshold detection
- **Statistical Analysis**: Calculates frequency, amplitude distribution, and kinetics of mEPSPs
- **Template Matching**: Uses averaged mEPSP templates for better event detection

**3. Intrinsic Property Analysis**:
- **Input Resistance (`extract_inp_res`)**: Calculates membrane input resistance from voltage responses to current injections
- **Sag Ratio (`extract_sag`)**: Measures membrane sag (Ih current) from hyperpolarizing current steps
- **Resting Membrane Potential**: Extracts baseline membrane potential

**4. Data Organization and Processing**:
- **Trial Averaging**: Computes trial-averaged responses for each experimental condition
- **Pre/Post Comparison**: Organizes data by pre-training and post-training time points
- **Protocol Categorization**: Separates different stimulation protocols (points vs. patterns)

**5. Cell Classification (`cell_classifier`, `cell_classifier_with_fnorm`)**:
- **Learner vs. Non-Learner Classification**: Applies statistical criteria to classify cells based on their plasticity responses
- **Field Normalization**: Optionally applies field potential normalization for classification
- **Threshold-Based Categorization**: Uses predefined criteria for plasticity magnitude and significance

**6. Advanced Statistical Processing**:
- **Bootstrap Analysis**: Implements bootstrap resampling for robust statistical comparisons
- **Multiple Comparison Correction**: Applies corrections for multiple testing
- **Time Course Analysis**: Tracks changes in responses across different time points

**7. Data Export**:
- **Multiple Pickle Files**: Saves processed data in various formats:
  - `all_cells_EPSP_df.pickle`: Complete EPSP feature dataset
  - `all_cells_InR_sag_extracted.pickle`: Intrinsic properties
  - `all_cells_mini_extracted.pickle`: mEPSP features
  - `Learner_classifications.pickle`: Cell classification results
  - `all_cells_one_trial_average_all_protocols.pickle`: Trial-averaged data

**8. Quality Control**:
- **Cell Health Assessment**: Filters cells based on stability criteria
- **Artifact Rejection**: Identifies and removes trials with artifacts
- **Response Validation**: Checks for physiologically reasonable responses

**Key Features**:
- **Comprehensive Feature Set**: Extracts >20 different electrophysiological features
- **Multi-Protocol Support**: Handles various stimulation protocols simultaneously
- **Robust Statistics**: Implements multiple statistical approaches for plasticity detection
- **Flexible Classification**: Supports both standard and field-normalized cell classification
- **Batch Processing**: Processes all cells in a dataset automatically

This script essentially transforms raw electrophysiological recordings into a structured, feature-rich dataset ready for statistical analysis and figure generation. It's the computational backbone that enables all subsequent plotting and analysis scripts to function. 

## Overall Project Summary

This workspace contains a comprehensive suite of Python scripts for analyzing electrophysiological data from a pattern learning study in neuroscience. The workflow is well-structured and can be broken down into several key stages:

### 1. Data Conversion (`conversion_scripts/`)
Raw data, primarily in Axon Binary Format (`.abf`), is read using the `neo` library. Scripts like `convert_abf_to_hdf5_cell_by_cell_with_stats_training_control_included.py` process these files, identify experimental protocols (e.g., point stimulation, pattern stimulation, training protocols, intrinsic property tests), separate pre- and post-training data, extract basic features, and assess cell health. The output consists of HDF5 files per cell and a summary statistics file. Additional scripts like `compile_cell_dfs_to_pickle.py` aggregate data from multiple cells into single pickle files for easier downstream analysis.

### 2. Feature Extraction (`analysis_scripts/`)
The core feature extraction pipeline is implemented in `extract_features_and_save_pickle.py`. This script reads the HDF5 data and computes a wide array of electrophysiological features from EPSP responses (amplitude, kinetics, plasticity), mEPSP properties (frequency, amplitude), and intrinsic cell properties (input resistance, sag ratio). It also implements cell classification algorithms to categorize cells as "learners" vs. "non-learners" based on their plasticity responses. The extracted features are saved into multiple specialized pickle files.

### 3. Curve Fitting (`curve_fit_scripts/`)
Scripts like `ei_9_modified_by_anzal_v1.py` and `ei_9_modified_by_anzal_v2_with_trial_info.py` implement biophysical model fitting using a dual alpha-function model to decompose evoked responses into excitatory and inhibitory components. These scripts use optimization algorithms to extract detailed synaptic parameters (amplitudes, time constants, delays) that provide insights into the underlying synaptic mechanisms.

### 4. Figure Generation (`plotting_scripts/`)
An extensive collection of plotting scripts generates publication-ready figures. Each script focuses on specific aspects of the data:
- **Figures 2-8**: Main manuscript figures covering experimental setup, cell characterization, plasticity analysis, learner vs. non-learner comparisons, field potential analysis, and detailed statistical comparisons.
- **Supplementary Figures 1-2**: Additional analyses and validations.
- **Multiple Versions**: Many scripts have multiple versions (v1, v2, v3) reflecting iterative improvements and different analytical approaches.
- **Field Normalization**: Scripts with `_fnorm` suffixes either implement or expect field-normalized data, providing alternative analysis pipelines.

### 5. Analytical Approaches
The project employs several sophisticated analytical techniques:
- **Bootstrap Statistics**: Multiple scripts implement bootstrap resampling for robust statistical comparisons, particularly in comparing learner vs. non-learner populations.
- **Gamma Function Fitting**: Used to model the relationship between expected (linear sum) and observed (pattern) responses, providing insights into synaptic integration.
- **Mixed-Effects Modeling**: Some scripts use `statsmodels` and `pingouin` for advanced statistical analyses including GLM and ANOVA.
- **Time Course Analysis**: Tracking plasticity changes across multiple time points post-training.

### 6. Common Infrastructure
- **`baisic_plot_fuctnions_and_features.py`**: Provides shared utilities for consistent plotting styles, color schemes, statistical formatting, and image generation functions.
- **Pickle-Based Data Flow**: The workflow heavily relies on pickle files for data persistence between processing stages.
- **Multiprocessing**: Several scripts implement parallel processing for computational efficiency.

### 7. Image Analysis
Scripts in `analysis_scripts/` also include microscopy image analysis tools for calculating scale bars, both manually (`calculate_scale_bar.py`) and automatically (`calculate_scale_bar_40x_automated.py`), with automated saving capabilities.

## Key Scientific Themes

Based on the script structure and naming conventions, this appears to be a study examining:

1. **Pattern Learning and Plasticity**: The core focus is on how neurons learn to respond to specific spatial patterns of synaptic input.

2. **Learner vs. Non-Learner Classification**: A major theme is identifying and characterizing differences between cells that successfully learn patterns versus those that don't.

3. **Synaptic Integration**: Analysis of how individual point responses sum to create pattern responses, and how this changes with learning.

4. **Field Potential Normalization**: Alternative analysis approaches that normalize cellular responses by local field potential changes.

5. **Temporal Dynamics**: Tracking how plasticity evolves over time following training protocols.

6. **Biophysical Mechanisms**: Detailed characterization of synaptic components (excitatory vs. inhibitory) and intrinsic cellular properties.

## Data Flow Architecture

The typical workflow appears to be:
1. **Raw .abf files** → 2. **HDF5 conversion** → 3. **Feature extraction** → 4. **Pickle files** → 5. **Figure generation**

With additional branches for:
- **Curve fitting analysis** (dual alpha-function modeling)
- **Image analysis** (scale bar calculation)
- **Statistical validation** (bootstrap analyses)

The architecture is designed for reproducibility and modularity, with clear separation between data processing, feature extraction, modeling, and visualization components.

## Technical Implementation

The codebase demonstrates sophisticated use of:
- **Scientific Python Stack**: Heavy use of `numpy`, `pandas`, `scipy`, `matplotlib`
- **Specialized Libraries**: `neo` for electrophysiology data, `pingouin` for statistics
- **Parallel Computing**: `multiprocessing` for computational efficiency
- **Data Persistence**: `pickle` and `HDF5` for structured data storage
- **Statistical Methods**: Bootstrap resampling, mixed-effects models, multiple comparison corrections

The scripts are well-documented with detailed docstrings and represent a mature, production-ready analysis pipeline for computational neuroscience research.

## General Observations

1. **Versioning and Evolution**: The presence of v1, v2, v3 versions of scripts (particularly for Figure 7 and Figure 6) suggests an iterative development process with progressive refinement of analytical methods.

2. **Field Normalization Strategy**: The dual approach with `_fnorm` versions indicates the researchers are exploring whether normalizing cellular responses by field potential changes affects their conclusions.

3. **Comprehensive Statistical Approach**: The extensive use of bootstrap methods and multiple comparison corrections demonstrates rigorous statistical practices.

4. **Publication-Ready Pipeline**: The structure suggests this is a complete analysis pipeline designed to generate figures for a peer-reviewed publication.

5. **Reproducible Research**: The modular design and comprehensive documentation facilitate reproducibility and future extensions of the analysis.

This represents a sophisticated and well-organized computational neuroscience project with robust data processing, statistical analysis, and visualization capabilities. 