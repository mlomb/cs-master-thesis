#
# Based on https://github.com/official-stockfish/nnue-pytorch/blob/master/perf_sigmoid_fitter.py
#

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
from tqdm import tqdm
import sys
import seaborn as sns

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True

def sigmoid(x, a, ib):
    y = 1 / (1 + np.exp(-ib*(x-a)))
    return (y)

def fit_data(x, y, sigma):
    # 1/361 is the initial guess. It's good enough to find the solution
    p0 = [0, 1/361]
    popt, pcov = curve_fit(sigmoid, x, y, p0, sigma, method='dogbox')
    return popt[0], popt[1]

def do_plot(data, plot_filename):
    plt.figure(figsize=(6,2.5))

    x = list(data.keys())
    y = [data[k][1] for k in x]
    x, y = zip(*list(sorted(zip(x, y), key=lambda x:x[0])))
    
    # plot of the perf% by eval and the fitted sigmoid
    x = list(data.keys())
    y = [data[k][0] / data[k][1] for k in x]
    # sigma is uncertainties, we con't care how correct it is.
    # The inverted counts are good enough.
    sigma = [1 / data[k][1] for k in x]
    a, ib = fit_data(x, y, sigma)
    print('inv b: ', ib, 'a: ', a)
    print('b: ', 1/ib)
    
    sns.scatterplot(x=x, y=y, label='Winrate', alpha=0.8, color="black") # perf


    # plot fit
    # plot fit
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = [sigmoid(xx, a, ib) for xx in x_fit]
    plt.plot(x_fit, y_fit, label='$\\sigma((e-{:.2f})/{:.2f})$'.format(a, 1.0/ib), color="lime")

    plt.legend(loc="upper left")
    plt.ylabel('Winrate')
    plt.xlabel('Evaluation')


    plt.tight_layout()
    plt.savefig(plot_filename)
    print('plot saved at {}'.format(plot_filename))



def gather_statistics(filename, count, bucket_size):
    '''
    It goes through all batches and collects evals and the outcomes.
    The evals are bucketed by bucket_size. Perf% is computed based on the
    evals and corresponding game outcomes.
    The result is a dictionary of the form { eval : (perf%, count) }
    '''
    data = dict()

    with open(filename) as file:
        i = 0
        for line in tqdm(file, total=count):
            score, result = line.strip().split(",")

            # https://github.com/official-stockfish/nnue-pytorch/blob/14124a0c9c6d70b25f46e5bbe443c1c97fd55fee/training_data_loader.cpp#L436C22-L436C47
            outcome = (float(result) + 1) / 2
            bucket = round(float(score) / bucket_size) * bucket_size

            bucket_id = int(bucket)
            pp = float(outcome) # perf

            if bucket_id in data:
                t = data[bucket_id]
                data[bucket_id] = (t[0] + pp, t[1] + 1)
            else:
                data[bucket_id] = (pp, 1)

            i += 1
            if i >= count:
                break

    return data

def main():
    filename = sys.argv[1]
    count = 1000000 if len(sys.argv) < 3 else int(sys.argv[2])
    bucket_size = 16 if len(sys.argv) < 4 else int(sys.argv[3])
    data = gather_statistics(filename, count, bucket_size)
    do_plot(data, "sigmoid_fit.pdf")

if __name__ == '__main__':
    main()
