import neurolib.utils.functions as func
import neurolib.utils.stimulus as stim
import numpy as np
import matplotlib.pyplot as plt
import scipy
from neurolib.models.hopf import HopfModel
from neurolib.models.wc import WCModel
from neurolib.utils.loadData import Dataset
from tqdm.notebook import tqdm_notebook
from numba import jit
from ddc_functions import dCov, partial_correlation
import time
from sklearn.preprocessing import MinMaxScaler


def generate_connectivity_and_weights_backup(n, p, maxweight=100, h = [-1,1], seed = 42, simmetric = False, threshold = 0.1):
    """
    Generates a connectivity matrix and a weight matrix with random values.
    The connectivity matrix is a numpy array with values ranging from 0 to 1.
    The weight matrix is a numpy array of the same shape with values ranging from 0 to 100.
    If a connection is non-zero, the weight should also be non-zero.
    Both matrices have the same height and width.
    
    Args:
        n: number of nodes
        p: probability of connection between nodes

    Returns:
        connectivity: a numpy array of shape (n, n)
        weights: a numpy array of the same shape with values of 0 or between 0 and 100
    """
    rng = np.random.default_rng(seed)
    connectivity = rng.uniform(low=h[0], high=h[1], size=(n, n))
    for i in range(len(connectivity)):
        for j in range(len(connectivity)):
            if rng.random() > p:
                connectivity[i, j] = 0
            
    connectivity[np.abs(connectivity) < threshold] = 0
            
    #connectivity[connectivity < 1-p] = 0
    #connectivity = rng.choice([-1, 1], size=(n, n), p=[1-p, p])
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if connectivity[i, j] != 0:
                weights[i, j] = rng.uniform(0, maxweight)
            if connectivity[i, j] != 0 and connectivity[j, i] != 0 and not simmetric:
                coin = rng.choice(np.array([0,1]))
                if coin == 0:
                    connectivity[i, j] = 0
                else:
                    connectivity[j, i] = 0


    return connectivity, weights

def generate_connectivity_and_weights(n, p, maxweight=100, h = [-1,1], seed = 42, threshold = 0, bidirectional = True):
    """
    Generates a connectivity matrix and a weight matrix with random values.
    The connectivity matrix is a numpy array with values ranging from 0 to 1.
    The weight matrix is a numpy array of the same shape with values ranging from 0 to 100.
    If a connection is non-zero, the weight should also be non-zero.
    Both matrices have the same height and width.
    
    Args:
        n: number of nodes
        p: probability of connection between nodes

    Returns:
        connectivity: a numpy array of shape (n, n)
        weights: a numpy array of the same shape with values of 0 or between 0 and 100
    """

    rng = np.random.default_rng(seed)
    count = n*(n-1)*p
    connectivity = np.zeros((n, n))
    while count > 0:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if bidirectional:
            if i != j and connectivity[i, j] == 0:
                l = rng.uniform(h[0], h[1])
                if l > threshold:
                    connectivity[i, j] = l
                    count -= 1
        else:
            if i != j and connectivity[i, j] == 0 and connectivity[j, i] == 0:
                l = rng.uniform(h[0], h[1])
                if l > threshold:
                    connectivity[i, j] = l
                    count -= 1

    #connectivity[connectivity < 1-p] = 0
    #connectivity = rng.choice([-1, 1], size=(n, n), p=[1-p, p])
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if connectivity[i, j] != 0:
                weights[i, j] = rng.uniform(0, maxweight)
    


    return connectivity, weights

def generate_xs_ys(n, seed = 42):
    """Generate x and y values for the model to run on the same x and ys for the same seed

    Args:
        n (int): amount of x and ys that should be generated
        seed (int, optional): Seed for the random values. Defaults to 42.

    Returns:
        np.array, np.array: Two arrays. First being the xs and second being the ys
    """
    rng = np.random.default_rng(seed)
    xs = np.array([[rng.uniform(-1,1)]for i in range(n)])
    ys = np.array([[rng.uniform(-1,1)]for i in range(n)])
    return xs, ys

def run_model(conn, dconn, seed, model_type = "hopf"):
    """Runs the neurolib model with the given parameters"""
    xs, ys = generate_xs_ys(len(conn))
    if model_type == "hopf":
        model = HopfModel(Cmat = conn, Dmat = dconn, seed = seed)
        model.params['w'] = 0.5
        model.params['signalV'] = 0.0
        model.params['duration'] = 20 * 1000
        model.params['sigma_ou'] = 0.1
        model.params['xs_init'] = xs
        model.params['ys_init'] = ys
        model.params['x_ou'] = np.zeros(len(conn))
        model.params['y_ou'] = np.zeros(len(conn))
        model.params['K_gl'] = 0.6
        model.run(chunkwise=True)
        return model.x.T
    
    elif model_type == "wc":
        model = WCModel(Cmat = conn, Dmat = dconn, seed = seed)
        model.params['exc_ext'] = 0.65
        model.params['signalV'] = 0
        model.params['duration'] = 20 * 1000 
        model.params['sigma_ou'] = 0.1
        model.params['K_gl'] = 3.15
        model.run(chunkwise=True)
        return model.exc.T  
    
    
    
    

def m_conn(conn):
    m_conn = np.zeros_like(conn)
    for i in range(len(m_conn)):
        for j in range(len(m_conn)):
            m_conn[i, j] += conn[i, j] + conn[j, i]
    return m_conn

    

def lag_corr(time_series_a, time_series_b, fft = True):
    if fft:
        # Compute the Fourier transforms of the time series
        fft_a = np.fft.fft(time_series_a)
        fft_b = np.fft.fft(time_series_b)

        # Compute the cross-correlation using the Fourier transforms
        cross_corr = np.fft.ifft(fft_a * np.conj(fft_b))

        # Find the lag corresponding to the maximum correlation
        max_corr_lag = np.argmax(np.abs(cross_corr)) - len(time_series_a) + 1

    else:
        # Compute the cross-correlation between the time series
        cross_corr = np.correlate(time_series_a, time_series_b, mode='full')

        # Find the lag corresponding to the maximum correlation
        max_corr_lag = np.argmax(cross_corr) - (len(time_series_a) - 1)

    return max_corr_lag


def lag_corr_new(time_series_a, time_series_b, fft = True):
    if fft:
        # Compute the Fourier transforms of the time series
        fft_a = np.fft.fft(time_series_a)
        fft_b = np.fft.fft(time_series_b)

        # Compute the cross-correlation using the Fourier transforms
        cross_corr = np.fft.ifft(fft_a * np.conj(fft_b))

        # Find the lag corresponding to the maximum correlation
        max_corr_lag = np.argmax(np.abs(cross_corr)) - len(time_series_a) + 1
    else:
        # Compute the cross-correlation between the time series
        cross_corr = np.correlate(time_series_a, time_series_b, mode='full')

        # Find the lag corresponding to the maximum correlation
        max_corr_lag = -(np.argmax(cross_corr) - (len(time_series_a) - 1))

    return max_corr_lag


def lag_corr_matrix_backup(data_model, mean_factor = 1, std_factor = 0):
    n_corr = np.corrcoef(data_model.T)
    for i in range(len(n_corr)):
        n_corr[i,i] = 0
    n_corr[n_corr < np.mean(n_corr) * mean_factor + np.std(n_corr) * std_factor] = 0
    l_conn = np.zeros_like(n_corr)
    f_conn = np.zeros_like(n_corr)
    for i in range(len(n_corr)):
        for j in range(len(n_corr)):
            if f_conn[i, j] == 0:
                l1 = lag_corr(data_model.T[i], data_model.T[j])
                l2 = lag_corr(data_model.T[j], data_model.T[i])
                if l1 == l2:
                    l_conn[i,j] = n_corr[i, j]
                    l_conn[j,i] = n_corr[i, j]
                elif l1 < l2:
                    l_conn[i,j] = n_corr[i, j]
                else:
                    l_conn[j,i] = n_corr[i, j]
                
                if l1 == 0:
                    l1 = 0.01
                if l2 == 0:
                    l2 = 0.01
                f_conn[i, j] = l1
                f_conn[j, i] = l2
    return l_conn, f_conn

def lag_corr_matrix(data_model, mean_factor=1, std_factor=0, use_threshold = False, l1_threshold = 0, l2_threshold = 0):
    n_corr = np.corrcoef(data_model.T)
    np.fill_diagonal(n_corr, 0)
    if use_threshold:
        threshold = np.mean(n_corr) * mean_factor + np.std(n_corr) * std_factor
        n_corr[n_corr < threshold] = 0

    l_conn = np.zeros_like(n_corr)
    f_conn = np.zeros_like(n_corr)

    for i, j in zip(*np.nonzero(n_corr)):
        if f_conn[i, j] == 0:
            l1 = lag_corr(data_model.T[i], data_model.T[j])
            l2 = lag_corr(data_model.T[j], data_model.T[i])

            
            if l1 == l2:
               l_conn[i, j] = n_corr[i, j]
               l_conn[j, i] = n_corr[i, j]
            elif l1 < l2:
               l_conn[i, j] = n_corr[i, j]
            else:
               l_conn[j, i] = n_corr[i, j]

            if l1 == 0:
                l1 = 0.01
            if l2 == 0:
                l2 = 0.01
            f_conn[i, j] = l1
            f_conn[j, i] = l2

    return l_conn, f_conn, n_corr



def lag_corr_matrix_new(data_model, mean_factor=1, std_factor=0, use_threshold = False, fft = True, dtw = False, offset = 0):
    n_corr = np.corrcoef(data_model.T)
    np.fill_diagonal(n_corr, 0)
    if use_threshold:
        threshold = np.mean(n_corr) * mean_factor + np.std(n_corr) * std_factor
        n_corr[n_corr < threshold] = 0

    l_conn = np.zeros_like(n_corr)
    f_conn = np.zeros_like(n_corr)

    for i, j in zip(*np.nonzero(n_corr)):
        if f_conn[i, j] == 0:
            l1 = lag_corr_new(data_model.T[i], data_model.T[j], fft)
            l2 = lag_corr_new(data_model.T[j], data_model.T[i], fft)
            #print(i, j)
            #print(l1, l2)


            if np.abs(l1-l2) <= offset:
               l_conn[i, j] = n_corr[i, j]
               l_conn[j, i] = n_corr[i, j]
            elif l1 < l2:
               l_conn[i, j] = n_corr[i, j]
            else:
               l_conn[j, i] = n_corr[i, j]

            if l1 == 0:
                l1 = 0.01
            if l2 == 0:
                l2 = 0.01
            f_conn[i, j] = l1
            f_conn[j, i] = l2

    return l_conn, f_conn, n_corr

def old_merge_matrix(m1, m2, threshold1 = 0.5, threshold2 = 0.25):
    x1 = np.zeros_like(m1)
    x2 = np.zeros_like(m2)

    for i in range(len(m1)):
        for j in range(len(m1)):
            if m1[i,j] >= threshold2 and m2[i,j] != 0:
                x1[i,j] = m1[i,j]

            elif m2[i,j] <= threshold1:
                x1[i,j] = 0
            else:
                x1[i,j] = m1[i,j]
    return x1

def merge_matrix(m1, m2, th1 = 0, th2 = 0):
    x1 = np.zeros_like(m1)
    for i in range(len(m1)):
        for j in range(len(m1)):
            if m1[i, j] <= th1 and m2[i, j] <= th1:
                x1[i, j] = m1[i, j]
            elif m1[i, j] >= th2 and m2[i, j] >= th2:
                x1[i, j] = m1[i, j]
            else:
                x1[i, j] = 0
    return x1

def c_matrix(matrix1, matrix2):
    bm1 = matrix1 != 0
    bm2 = matrix2 != 0
    return np.logical_xor(bm1, bm2)


def calc_matrix(data_model, conn, th1 = 0, th2 = 0, pcorr = True, threshold = 0.1, l1_threshold = 0, l2_treshold = 0):
    red_conn = conn
    for i in range(len(red_conn)):
        red_conn[i,i] = 0
    if pcorr:
        dcov = partial_correlation(data_model, 0.05)
    else:
        dcov, _, _ = dCov(data_model, 0.1, 0.1, 0.1, False, False, True)   
    for i in range(len(dcov)):
        dcov[i,i] = 0
    #dcov[np.abs(dcov) < threshold] = 0
    l_conn, _, n_corr = lag_corr_matrix(data_model, 1, 0, False, l1_threshold, l2_treshold)
    c_m = merge_matrix(l_conn, dcov, th1, th2)
    c_m[np.abs(c_m) < threshold] = 0
    return red_conn, c_m, dcov, l_conn, n_corr

def normalize_matrix(matrix1):
    matrix1_max = np.max(np.abs(matrix1))
    return matrix1 * (1/matrix1_max)

def calc_percentage(red_conn, c_m, zscore = False):
    if zscore:
        red_conn = normalize_matrix(red_conn)
        c_m = normalize_matrix(c_m)

    dist = np.abs(red_conn - c_m)
    mask = red_conn != 0
    dist[mask] = dist[mask] / np.abs(red_conn[mask])
    dist_absolute = np.ones_like(np.abs(dist)) - np.abs(dist)
    dist_absolute[dist_absolute < 0] = 0
    return np.mean(dist_absolute), dist, dist_absolute

def pearson_corr(m1, m2, zscore = False):
    if zscore:
        m1 = normalize_matrix(m1)
        m2 = normalize_matrix(m2)
    return scipy.stats.pearsonr(m1.flatten(), m2.flatten())[0]


def get_comparison(data_model, conn, th1 = 0, th2 = 0, pcorr = True, threshold = 0, zscore = False, l1_threshold = 0, l2_threshold = 0):
    red_conn, c_m, dcov, l_conn, n_corr = calc_matrix(data_model, conn, th1, th2, pcorr, threshold, l1_threshold, l2_threshold)
    #dist = np.abs(red_conn - c_m)
    #mask = dist != 0
    #dist[mask] = dist[mask] / red_conn[mask]
    #dist = np.ones_like(dist) - dist
    #comparison = c_matrix(red_conn, c_m)
    #performance = np.count_nonzero(comparison)/(len(red_conn)*len(red_conn))

    performance, comparison, dist_abs = calc_percentage(red_conn, c_m, zscore)
    return performance, comparison, red_conn, c_m, dcov, l_conn, dist_abs, n_corr

def calc_lagged(data_model, l1_threshold = 0, l2_treshold = 0, th1 = 0, th2 = 0, threshold = 0.1, pcorr = True, ddc = [0.1,0.1,0.1]):
    if pcorr:
        dcov = partial_correlation(data_model, 0.05)
    else:
        dcov, _, _ = dCov(data_model, ddc[0], ddc[1], ddc[2], False, False, True)   

    for i in range(len(dcov)):
        dcov[i,i] = 0
        
    #dcov[np.abs(dcov) < threshold] = 0
    l_conn, _, n_corr = lag_corr_matrix(data_model, 1, 0, False, l1_threshold, l2_treshold)
    c_m = merge_matrix(l_conn, dcov, th1, th2)
    c_m[np.abs(c_m) < threshold] = 0
    return c_m, dcov, l_conn, n_corr

def run_with_seed(seed,num_neurons = 10, p = 0.1, maxweight = 100, th1 = 0, th2 = 0, pcorr = True, hilo = [0,1], simmetry = False, threshold = 0, zscore = False, l1_threshold = 0, l2_threshold = 0):
    conn, dconn = generate_connectivity_and_weights(num_neurons,p,maxweight, hilo, seed, simmetry)
    data_model = run_model(conn, dconn, seed)
    performance, comparison, red_conn, c_m, dcov, l_conn, dist_abs, n_corr = get_comparison(data_model, conn, th1, th2, pcorr, threshold, zscore, l1_threshold, l2_threshold)
    del data_model
    return performance, comparison, red_conn, c_m, dconn, dcov, l_conn, dist_abs, n_corr



def run_correlation(seed, num_neurons = 10, p = 0.1, maxweight = 100, th1 = 0, th2 = 0, pcorr = True, hilo = [0,1], simmetry = False, distance = 0, sim_amounts = 1):
    conn, dconn = generate_connectivity_and_weights(num_neurons,p,maxweight, hilo, seed, simmetry)
    data_model = run_model(conn, dconn, seed)
    performance, comparison, red_conn, c_m, dcov, l_conn, dist_abs, n_corr = get_comparison(data_model, conn, th1, th2, pcorr)
    sim_models = []
    for i in range(sim_amounts):
        s = seed + i
        if distance == 0:
            s_model = run_model(c_m, dconn, s)
        else:
            s_model = run_model(c_m, np.zeros_like(dconn), s)
        sim_models.append(s_model)
        
    correlations = [[] for _ in range(len(sim_models[0].T))]
    for sim_model in sim_models:
        for i in range(len(sim_model.T)):
            correlations[i].append(np.corrcoef(sim_model.T[i], scipy.stats.zscore(data_model.T[i], ddof=1))[0,1])
    return performance, comparison, red_conn, c_m, dconn, correlations, sim_models, data_model, dcov, l_conn

def generate_dataset(n_datasets, num_neurons, p = 0.1, maxweight = 100, hilo = [-1,1], simmetry = False):
    dataset = []
    for i in tqdm_notebook(range(n_datasets)):
        seed = np.random.randint(0, 100000)
        conn, dconn = generate_connectivity_and_weights(num_neurons,p,maxweight, hilo, seed, simmetry)
        data_model = run_model(conn, dconn, seed)
        dataset.append(np.array([conn, dconn, data_model]))
    return dataset

def run_with_data(seed, num_neurons = 10, p = 0.1, maxweight = 100, th1 = 0, th2 = 0, pcorr = True, hilo = [0,1], simmetry = False, threshold = 0, zscore = False, l1_threshold = 0, l2_threshold = 0):
    conn, dconn = generate_connectivity_and_weights(num_neurons,p,maxweight, hilo, seed, simmetry)
    data_model = run_model(conn, dconn, seed)
    performance, comparison, red_conn, c_m, dcov, l_conn, dist_abs, n_corr = get_comparison(data_model, conn, th1, th2, pcorr, threshold, zscore, l1_threshold, l2_threshold)
    return c_m, data_model, conn, red_conn

def generate_iter_dataset(n_datasets, num_neurons, seed = 42, p = 0.1, maxweight = 100, hilo = [-1,1], simmetry = False):
    dataset = []
    rng = np.random.default_rng(seed)
    conn, dconn = generate_connectivity_and_weights(num_neurons,p,maxweight, hilo, rng.integers(0,100000), simmetry)
    for _ in tqdm_notebook(range(n_datasets)):
        s = rng.integers(0, 100000)
        data_model = run_model(conn, dconn, s)
        dataset.append(data_model)
    return dataset, conn, dconn

def generate_corresponding_dataset(nums, num_neurons, seed = 42, p = 0.1, maxweight = 100, hilo = [-1,1], simmetry = False):
    data_models, conns, dconns = [], [], []
    rng = np.random.default_rng(seed)
    for i in tqdm_notebook(range(nums)):
        conn, dconn = generate_connectivity_and_weights(num_neurons, p , maxweight, hilo, rng.integers(0,100000), simmetry)
        s = rng.integers(0, 100000)
        data_model = run_model(conn, dconn, s)
        data_models.append(data_model[1000:])
        conns.append(conn)
        dconns.append(dconn)
    return data_models, conns, dconns

def calc_cmx(data, cut = 15, threshold = 0.2, use_threshold = True, use_abs = True):
    if use_abs:
        l_t = np.abs(lag_corr_matrix_new(scipy.stats.zscore(data[::cut], axis=0),fft = False)[0])
        l_tf = np.abs(lag_corr_matrix_new(scipy.stats.zscore(data[::cut], axis=0),fft = True)[0])
    else:
        l_t = lag_corr_matrix_new(scipy.stats.zscore(data[::cut], axis=0),fft = False)[0]
        l_tf = lag_corr_matrix_new(scipy.stats.zscore(data[::cut], axis=0),fft = True)[0]

    c_mx = (l_t + l_tf)/2
    if use_threshold:
        c_mx[c_mx < threshold] = 0
        c_mx[c_mx > 0] = 1
    return c_mx