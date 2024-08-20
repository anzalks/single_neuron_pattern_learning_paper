###################################################################
##
##  ei6.py adds another parameter to fit, the initial delay for the
##  start of the stimulus
##  It also refines the scoring function to include peak and valley fits.
##
###################################################################
import pandas
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import time
import os
import scipy.optimize as sci
from multiprocessing import Pool

datadir ="/Users/anzalks/Documents/pattern_learning_paper/data/CA1_3_pattern_training_experiment_all_cells/hdf5"
cell_stats =pandas.read_hdf("/Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/pickle_files/cell_stats.h5")

outdir="/Users/anzalks/Documents/pattern_learning_paper/data/CA1_3_pattern_training_experiment_all_cells/hdf5/cell_stats_plots/curve_fit_plots_v9"

TOLERANCE = 0.0001
sampleRate = 20000.0
FrameTypes = ['point_0', 'point_1', 'point_2', 'point_3', 'point_4', 
        'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10', 
        'point_11', 'pattern_0', 'pattern_1', 'pattern_2']
PrePostStatus = ['pre', 'post_0', 'post_1', 'post_2', 'post_3', 'post_4', 'post_5']
PulseTime = 0.005

def healthy_cell_list(cell_stats):
    h_cells = []
    for a in cell_stats.iterrows():
        if a[1]['cell_stats']['cell_status']=='valid':
            h_cells.append(a[0])
    return h_cells

class CellThread():
    def __init__( self, fname, t0 ):
        self.fname = fname
        self.t0 = t0
        self.ret = { "cell_ID": [], "pre_post_status":[], "frame_id":[],
                    "trial_no":[],"Epk": [], "Etau": [], "Ipk":[], 
                    "Itau":[], "delay":[],"initDelay":[], "score":[] }

    def analyze( self ):
        dat = pandas.read_hdf( self.fname )
        cellId = analyzeCell( dat, self.ret )
        print( "\nCompleted for cell {} in {:.3f} sec, numPlots = {}".format( cellId, time.time() - self.t0, len( self.ret["cell_ID"] ) ) )
        return self.ret


class SumOfAlphas():
    def __init__( self, waveform ):
        self.dt = 1/1e4 if len( waveform ) == 50000 else 1/2e4
        self.waveform = waveform[:int( round( 0.5/self.dt) ) ]
        self.baseline = np.mean( waveform[:50] )
        self.sampleT = np.arange( 0.0, 0.5, self.dt )

    def estimate( self, params ):
        [pkE, tauE, pkI, tauI, delay, initDelay] = params
        st = (self.sampleT - initDelay) / tauE
        st[st<0] = 0.0
        a1 = pkE * st * np.exp( -st) * np.e
        st = (self.sampleT - delay - initDelay) / tauI
        st[st<0] = 0.0
        a2 = pkI * st * np.exp( -st ) * np.e
        return self.baseline + a1 - a2

    def score( self, params ):
        pkSamples = int( 20e-3 * 20000 )    # 20 ms window * sample ratei used 15ms here
        valSamples = int( 200e-3 * 20000 )    # 100 ms window * sample rate
        # Params have 5 entries: pkE, tauE, pkI, tauI, delay
        # We need to weight the first 50 ms extra.
        d2Wt = 2.0
        pkValWt = 10.0 #wt of 10 used here
        est = self.estimate( params )
        delta = est - self.waveform
        d2 = delta[:len(delta)//5]
        pk = max( est[:pkSamples] ) - max( self.waveform[:pkSamples] )
        val = min( est[:valSamples] ) - min( self.waveform[:valSamples] )
        pkValFit = pk*pk + val*val
        return pkValFit * pkValWt + np.dot( delta, delta ) / len( delta ) + d2Wt * np.dot( d2, d2 ) / len(d2)

    def fit( self ):
        # Params have 6 entries: pkE, tauE, pkI, tauI, delay, initDelay
        initGuess = [max( self.waveform ), 0.005, min( self.waveform ), 0.02, 0.005, 0.002 ]
        self.t0 = time.time()
        result = sci.minimize( self.score, initGuess, method ="L-BFGS-B",
                              bounds = [ (0.01, 20), (0.001, 0.04),
                                        (0.01, 20), (0.006, 0.15),
                                        (0.001, 0.1), (0.0005,0.2)
                                       ], tol = TOLERANCE 
                             )
        self.ans = result.x
        #print( "kpk = {:.1f}, runtime={:.3f}, t1=({:.3f}, {:.3f}), t2=({:.3f},{:.3f}), delay={:.3f}, ratio={:.3f}, score = {:.2f}, initScore = {:.2f}".format(self.kpk, time.time() - t0, ans[0], ans[1], ans[2], ans[3], ans[4], ans[5], self.score( ans ), self.score( initGuess ) ) )
        self.optScore = self.score( self.ans )
        return self.ans

    def printFit( self ):
        print( "{:8s}{:8s}{:8s}{:8s}{:8s}{:8s}{:8s}{:8s}".format( "time", "Epk", "Etau", "Ipk", "Itau", "delay", "initDelay", "score" ) )
        print( "{:4.2f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.2f}".format(time.time() - self.t0, self.ans[0], self.ans[1], self.ans[2], 
            self.ans[3], self.ans[4], self.ans[5], self.optScore ) )


    def plotFit( self ):
        plt.plot( self.sampleT, self.waveform, "b", label= "expt" )
        plt.plot( self.sampleT, self.estimate( self.ans), "r:", label= "est" )
        plt.legend()
        plt.xlabel( "Time (s)" )
        plt.ylabel( "Vm (mV)" )
        plt.title( "Plot of fit" )
        plt.show(block=False)
        plt.pause(2)
        plt.close()


########################################################################

def analyzeCell( dat, ret ):
    cellId = dat['cell_ID'].unique()[0]
    print()
    print( cellId, end = "", flush=True)
    for pp in PrePostStatus:
        spp = dat.loc[dat["pre_post_status"] == pp]
        # It is more efficient to get a subset of the big frame and then 
        # search only within it.
        for ft in FrameTypes:
            Vm = spp.loc[spp["frame_id"] == ft]["cell_trace(mV)"].values
            if len( Vm ) == 0:
                continue
            trial_grp = spp.groupby(by="trial_no")
            for trial, trial_data in trial_grp:
                Vm = trial_data[trial_data["frame_id"]==ft]["cell_trace(mV)"].values
                #Vm.shape = (3, len(Vm)//3)
                #print(Vm)
                #mVm = Vm.mean( axis = 0 )
                soa = SumOfAlphas( Vm )
                params = soa.fit()
                ret["cell_ID"].append( cellId )
                ret["pre_post_status"].append( pp )
                ret["frame_id"].append( ft )
                ret["trial_no"].append(trial)
                ret["Epk"].append( params[0] )
                ret["Etau"].append( params[1] )
                ret["Ipk"].append( params[2] )
                ret["Itau"].append( params[3] )
                ret["delay"].append( params[4] )
                ret["initDelay"].append( params[5] )
                ret["score"].append( soa.optScore )
                print( ".", end = "", flush=True)
                #soa.printFit()
                #if ft in ['pattern_0', 'pattern_1', 'pattern_2']:
                    #   soa.plotFit()
        print( "@", end = "", flush=True)
    return cellId

def main():
    parser = argparse.ArgumentParser( description = "Program to analyze Anzal's current-clamp data for long-term plasticity" )
    parser.add_argument( "-o", "--outfile", type = str, help = "Optional: specify cell file to analyze.", default ="Anzal_plasticity_1_from_v9.h5" )
    parser.add_argument( "-d", "--datadir", type = str, help = "Optional: specify directory with .h5 files to analyze.", default = datadir )
    parser.add_argument( "-n", "--numThreads", type = int, help = "Optional: Number of threads to use.", default = 6 )
    args = parser.parse_args()
    t0 = time.time()
    h_cells = healthy_cell_list(cell_stats)
    pool = Pool( processes = args.numThreads )

    ret = { "cell_ID": [], "pre_post_status":[], "frame_id":[],
           "trial_no":[],"Epk": [], "Etau": [], "Ipk":[], "Itau":[], "delay":[],
            "initDelay":[], "score":[] }
    ctList = []
    threadRet = []

    for cellfile in os.listdir( args.datadir ):
        if cellfile.endswith( ".h5" ):
            cname =cellfile.split("_with")[0]
            if cname in h_cells:
                ct = CellThread( "{}/{}".format( datadir, cellfile ), t0 )
                ctList.append( ct )
                threadRet.append( pool.apply_async( ct.analyze ) )

    results = [ rr.get() for rr in threadRet ]

    for ii in results:
        for key in ret:
            ret[key].extend( ii[key] )

    '''
    for cc in ctList:
        print( "len = ", len( cc.ret["cell_ID"] ) )
        for rr in ret:
            ret[rr].extend( cc.ret[rr] )
        #ret.update( cc.ret )
    '''

    # tauE ~  5 ms, tauI ~~ 30ms, 100ms., 20ms,50ms. 
    # It is toward the faster end for the 1sq stim.
    outFrame = pandas.DataFrame(ret)
    print( outFrame )
    outFrame.to_hdf( args.outfile, "Anzal_plasticity_1_from_v9", "w" )
    print( "Completed in {:.3f} seconds".format( time.time() - t0 ) )

if __name__ == "__main__":
    main()

