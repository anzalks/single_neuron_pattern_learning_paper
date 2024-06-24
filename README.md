# single_neuron_pattern_learning_paper
All scripts that's associated with the pattern learning paper

"plotting scripts parsers from bash/zsh on vim"

"figure 1"
:!python % 
-f /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/baseline_traces_all_cells.pickle 
-i /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/with\ fluorescence\ and\ pipette.bmp 
-p /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/Scre
enshot\ 2023-06-28\ at\ 22.21.46.png

"figure 2"
:!python % 
-f /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/pd_all_cells_mean.pickle 
-s /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/all_cells_classified_dict.pickle 
-r /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/all_cells_inR.pickle 
-i /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/illustations/figure_2_1.jpg 
-p /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/illustations/figure_2_2.jpg


"figure 3"
:!python % -f /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/pd_all_cells_mean.pickle -s /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/all_cells_classified_dict.pickle -i /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/illustations/Figure_3_1.jpg -c /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/pickle_files/cell_stats.h5 -a /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/pd_all_cells_all_trials.pickle -t /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/pd_training_data_all_cells_all_trials.pickle -q /Users/anzalks/Documents/Expt_data/CA1_recordings/x_spread_recording_02Hz_30mins/cells_cumulated/cells_min_30mins_long/pickle_format_files_firing_rate_data/all_cell_all_trial_firing_properties.pickle 

"figure 4"
:!python % -f /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/pd_all_cells_mean.pickle -s /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/all_cells_classified_dict.pickle -i /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/illustations/Figure_3_1.jpg -c /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/pickle_files/cell_stats.h5

"figure 5"
:!python % -f /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/pd_all_cells_mean.pickle -s /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/analysis_scripts/pickle_files_from_analysis/all_cells_classified_dict.pickle -i /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/illustations/Figure_3_1.jpg -c /Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/pickle_files/cell_stats.h5
