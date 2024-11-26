import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

def load_tsv_data(filename):
    """
    Loads data from a TSV file with the specified format into a pandas DataFrame.

    Args:
        filename (str): The path to the TSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded data.
    """

    data = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            celltype = parts[0]
            index = int(parts[1])
            label = parts[2]
            numSamples = (len( parts ) - 3)//2
            x = [float(xx) for xx in parts[3:3+numSamples]]
            y = [float(yy) for yy in parts[3+numSamples:3+numSamples*2]]
            print( "LENS = ", len(x), len(y) )
            data.append([celltype, index, label, x, y])
            [fitgamma], cov = scipy.optimize.curve_fit(gamma_fit, 
                    x, y, p0 = [2.0], bounds=(0, 60))
            print( "{}  {}  {}  {:.3f}".format( celltype, index, label, fitgamma ) )

    df = pd.DataFrame(data, columns=['celltype', 'Index', 'Label', 'X_Values', 'Y_Values'])
    return data

def geminiBootstrap(A, B, nIter = 1000):
    # Calculate the observed difference of means
    observed_diff = fitGamma(A) - fitGamma(B)
    #print( "GAMMAS = ", fitGamma(A), fitGamma(B) )

    # Generate bootstrap samples and calculate the test statistic
    bootstrap_diffs = []
    idx = np.arange(len(A))
    for _ in range(nIter):
        temp = np.random.choice(idx, size=len(idx), replace=True)
        bootstrap_A = A[temp]
        temp = np.random.choice(idx, size=len(idx), replace=True)
        bootstrap_B = B[temp]
        '''
        bootstrap_A = np.random.choice(A, size=len(A), replace=True)
        bootstrap_B = np.random.choice(B, size=len(A), replace=True)
        '''
        bootstrap_diff = fitGamma(bootstrap_A) - fitGamma(bootstrap_B)
        bootstrap_diffs.append(bootstrap_diff)
    
    # Shift the distribution
    bootstrap_diffs = np.array(bootstrap_diffs) - observed_diff
    
    # Calculate the p-value (two-tailed test)
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value

def chatBootstrap(A, B, nIter=1000):
    # Combine datasets
    combined = np.concatenate([A, B])

    # Calculate observed test statistic (e.g., difference in means)
    observed_stat = fitGamma(A) - fitGamma(B)

    # Bootstrap resampling
    bootstrap_stats = []
    idx = np.arange(len(A))
    for _ in range(nIter):
        # Resample with replacement 
        temp = np.random.choice(idx, size=len(A), replace=True)
        A_boot = combined[temp]
        temp = np.random.choice(idx, size=len(A), replace=True)
        B_boot = combined[temp]
        '''
        A_boot = np.random.choice(combined, size=len(A), replace=True)
        B_boot = np.random.choice(combined, size=len(B), replace=True)
        '''

        # Compute test statistic for resampled data
        bootstrap_stats.append(fitGamma(A_boot) - fitGamma(B_boot))

    # Convert to numpy array
    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))

    return observed_stat, p_value

def fit_plots( x, y ):
    # Perform curve fitting only for the 'gam' parameter
    params, cov = scipy.optimize.curve_fit(gamma_fit, x, y, p0 = [2.0], bounds=(0, 60))

def gamma_fit(expt, gamma):
    """
    Gamma function fit model with fixed parameters:
    beta = 1, alpha = 0.
    """
    beta = 1  # Fixed value
    alpha = 0  # Fixed value
    return expt - (((beta * expt) / (gamma + expt)) * expt) - alpha

def fitGamma( data ):
    data = data.transpose()
    [ret], cov = scipy.optimize.curve_fit(gamma_fit, 
        data[0], data[1], p0 = [2.0], bounds=(0, 60))
    return ret

def main():
    filename = 'patdat.txt'  # Replace with your actual filename
    data = load_tsv_data(filename)
    for idx in range(0,12,2):
        pre = data[0+idx]
        post = data[1+idx]
        A = np.array( [pre[3], pre[4]]).transpose()
        B = np.array( [post[3], post[4]]).transpose()
        delta, p = geminiBootstrap( A, B )
        print( "GEMINI: {}  {}  DELTA={:.3f}, P={:.3f}".format(pre[0],pre[1], delta, p ) )
        delta, p = chatBootstrap( A, B )
        print( "CHAT: {}  {}  DELTA={:.3f}, P={:.3f}".format(pre[0],pre[1], delta, p ) )
    #delta, p = chatBootstrap( A, B )
    #print( delta, p )

if __name__ == "__main__":
    main()
