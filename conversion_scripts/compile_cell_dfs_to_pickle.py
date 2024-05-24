__author__           = "Anzal KS"
__copyright__        = "Copyright 2023-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import time
import pickle
from tqdm import tqdm


class Args: pass 
args_ = Args()

def list_h5(p):
    f_list = []
    f_list=list(p.glob('**/*h5'))
    f_list.sort()
    return f_list

def write_pkl(file_to_write,file_name):
    with open(f'{file_name}.pickle', 'wb') as handle:
        pickle.dump(file_to_write, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f" wrote file to pickle")
    
def combine_multicell_to_pkl(cell_stats, h5_data_cells,outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = f"{outdir}/all_data_df"
    cells_combined = []
    for c, cell in enumerate(h5_data_cells):
        #print(f"read {c}")
        cell = pd.read_hdf(cell)
        print(f"cell: {cell.head()}")
        cell_ID = cell['cell_ID'][0]
        cell_stat =cell_stats[cell_stats.index==cell_ID]
        #print(f' cell stat head= {cell_stats.head()}')
        cell_validity = cell_stat['cell_stats'][0]['cell_status']
        if cell_validity!='valid':
            print(f"cell_ID: {cell_ID} is invalid")
            print(f'cell status : {cell["frame_status"].unique()}')
            continue
        else:
            print(f"cell_ID: {cell_ID} is valid")
            print(f'cell status : {cell["frame_status"].unique()}')
            cells_combined.append(cell)
    
    all_cells = pd.concat(cells_combined).reset_index(drop=True)
    print("converting to pickle...")
    write_pkl(all_cells,outfile)

def main():
    # Argument parser.
    description = '''Script to analyse cell health and plot relevent values'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cells-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'folder path to cell data in h5 format'
                       )
    parser.add_argument('--cell-stat', '-s'
                        , required = False,default ='./', type=str
                        , help = 'path of dataframe with cell stats in h5 format'
                       )
    
    args = parser.parse_args()
    #print(args.cells_path)
    h5_folder_path = Path(args.cells_path)
    h5_files = list_h5(h5_folder_path)
    cell_stats =pd.read_hdf(args.cell_stat)
#    print(cell_stats)
    outdir = h5_folder_path/'pickle_file_all_cells_trail_corrected'
    outdir.mkdir(exist_ok=True, parents=True)
    combine_multicell_to_pkl(cell_stats,h5_files,outdir)


if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {np.around(((tf-ts)/60),1)} (mins)')
