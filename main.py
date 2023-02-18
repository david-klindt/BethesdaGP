import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from scipy.stats import norm


def load_data(file):
    WS = pd.read_excel(file)
    WS_np = np.array(WS)
    num_subject, num_measurement = 0, 0
    for line in WS_np:
        if str(line[1]).startswith('measurement'):
            num_measurement += 1
        else:
            try:
                ind = int(line[1])
                if num_subject < ind:
                    num_subject = ind
            except:
                pass
    print("Number of subjects:", num_subject,
          "Number of measurements:", num_measurement)
    x_values = WS_np[1, 2:].astype(int)
    data = np.zeros((num_subject, num_measurement, x_values.size))
    ind_m, ind_s = -1, -1
    for line in WS_np:
        if str(line[1]).startswith('measurement'):
            ind_m += 1
            ind_s = 0
        else:
            data[ind_s, ind_m] = line[2:].astype(float).copy()
            ind_s += 1
    # remove only nan concentrations
    remove_last = np.all(np.isnan(data), axis=(0, 1)).sum()
    # print(remove_last)
    if remove_last > 0:
        x_values = x_values[:-remove_last]
        data = data[:, :, :-remove_last]
    return data


def fit_model(data, level=50, n_restarts_optimizer=100, tol=1e-6):
    # define model
    kernel = 1.0 * RBF() + WhiteKernel(
        noise_level=2.0, noise_level_bounds=(1.0, 20.0)
    )
    # x coordinates
    X = np.concatenate([np.log2(x_values) for _ in range(num_measurement)])[:, None]
    X_test = []
    models, log_likelihoods = [], []
    mean_predictions = []
    std_predictions = []
    level_prob = []
    for i in range(num_subject):
        print('Fitting model for subject', i + 1)
        t0 = time.time()
        Y = np.concatenate([d for d in data[i]])
        # filter out nans
        good_ind = np.logical_not(np.isnan(Y))
        x = X[good_ind]
        y = Y[good_ind]
        x_test = np.linspace(x.min(), x.max(), 10000)
        X_test.append(x_test)
        # fit model
        models.append(
            GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=42,
            )
        )
        models[-1].fit(x, y)
        log_likelihoods.append(models[-1].log_marginal_likelihood())
        print('   Fitting took %.2fs' % (time.time() - t0))
        print('   Fitted model:', models[-1].kernel_)
        print('   Quality of fit (log-likelihood, higher is better)=%.2f' % log_likelihoods[-1])
        # predict
        mean_prediction, std_prediction = models[-1].predict(
            x_test[:, None], return_std=True)
        mean_predictions.append(mean_prediction)
        std_predictions.append(std_prediction)
        # Bayesian inference
        x_prob = norm.pdf(
            level, loc=mean_predictions[i], scale=std_predictions[i]
        ) + 1e-9
        x_prob /= np.sum(x_prob)
        level_prob.append(x_prob)
    return mean_predictions, std_predictions

