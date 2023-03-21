import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import sys
import warnings
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from scipy.stats import norm


class GP_Model:
    def __init__(
            self,
            level=50,  # Restaktivitaet level of interest
            n_restarts_optimizer=100,  # number of restarts for GP fitting
            tol=1e-6,  # minimal value of pdf, numerical stability + uniform prior
            seed=42,  # random seed
            noise_level=2.0,  # initial scale for noise kernel
            noise_lower_bound=1.0,  # lower scale for noise kernel
            noise_upper_bound=20.0,  # upper scale for noise kernel
            num_test=10000,  # number of test points for inference
            dilution_lower_bound=None,  # lower bound for GP inference (in log2)
    ):
        # parameters
        self.level = level
        self.n_restarts_optimizer = n_restarts_optimizer
        self.tol = tol
        self.seed = seed
        self.noise_level = noise_level
        self.noise_level_bounds = (noise_lower_bound, noise_upper_bound)
        self.num_test = num_test
        self.dilution_lower_bound = dilution_lower_bound
        # fitting
        self.X_test = []
        self.level_prob = []
        self.models = []
        self.log_likelihoods = []
        self.mean_predictions  = []
        self.std_predictions = []

    def load_data(self, file):
        WS = pd.read_excel(file)
        WS_np = np.array(WS)
        self.num_subject, self.num_measurement = 0, 0
        for line in WS_np:
            if str(line[1]).startswith('measurement'):
                self.num_measurement += 1
            else:
                try:
                    ind = int(line[1])
                    if self.num_subject < ind:
                        self.num_subject = ind
                except:
                    pass
        print("Number of subjects:", self.num_subject,
              "Number of measurements:", self.num_measurement)
        self.x_values = WS_np[1, 2:].astype(int)
        self.data = np.zeros((
            self.num_subject, self.num_measurement, self.x_values.size))
        ind_m, ind_s = -1, -1
        for line in WS_np:
            if str(line[1]).startswith('measurement'):
                ind_m += 1
                ind_s = 0
            else:
                self.data[ind_s, ind_m] = line[2:].astype(float).copy()
                ind_s += 1
        # remove only nan concentrations
        remove_last = np.all(np.isnan(self.data), axis=(0, 1)).sum()
        if remove_last > 0:
            self.x_values = self.x_values[:-remove_last]
            self.data = self.data[:, :, :-remove_last]

    def fit_model(self):
        # define model
        kernel = 1.0 * RBF() + WhiteKernel(
            noise_level=self.noise_level,
            noise_level_bounds=self.noise_level_bounds
        )
        # x coordinates
        X = np.concatenate([np.log2(self.x_values) for _ in range(
            self.num_measurement)])[:, None]
        for i in range(self.num_subject):
            print('Fitting model for subject', i + 1)
            t0 = time.time()
            Y = np.concatenate([d for d in self.data[i]])
            # filter out nans
            good_ind = np.logical_not(np.isnan(Y))
            x = X[good_ind]
            y = Y[good_ind]
            self.X_test.append(
                np.linspace(
                x.min() if self.dilution_lower_bound is None else self.dilution_lower_bound,
                x.max(), self.num_test)
            )
            # fit model
            self.models.append(
                GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=self.n_restarts_optimizer,
                    random_state=self.seed,
                )
            )
            self.models[-1].fit(x, y)
            self.log_likelihoods.append(
                self.models[-1].log_marginal_likelihood())
            print('   Fitting took %.2fs' % (time.time() - t0))
            print('   Fitted model:', self.models[-1].kernel_)
            print('   Quality of fit (log-likelihood, higher is better)=%.2f' % self.log_likelihoods[-1])
            # predict
            mean_prediction, std_prediction = self.models[-1].predict(
                self.X_test[-1][:, None], return_std=True)
            self.mean_predictions.append(mean_prediction)
            self.std_predictions.append(std_prediction)
            # Bayesian inference
            x_prob = norm.pdf(
                self.level, loc=self.mean_predictions[i],
                scale=self.std_predictions[i]
            ) + self.tol
            x_prob /= np.sum(x_prob)
            self.level_prob.append(x_prob)


### Plotting ###
def plot_skeleton(
        model,  # GP_Model
        i,  # index of subject in data
        plot_all=False,  # if False, shows only first measurement
        log_scale=True  # shows log inverse concentration as x ticks
):
    handle = plt.hlines(50, model.X_test[i][0], model.X_test[i][-1],
                        linestyle='--', color='grey')
    handles = [handle]
    labels= [f"Saturation = 50%"]
    longest = 0
    if plot_all:
        num_plot = model.num_measurement
    else:
        num_plot = 1
    for j in range(num_plot):
        handle = plt.scatter(np.log2(model.x_values), model.data[i, j])
        handles.append(handle)
        labels.append("measurement %s" % (j + 1))
        if np.sum(np.logical_not(np.isnan(model.data[i, j]))) > longest:
            longest = np.sum(np.logical_not(np.isnan(model.data[i, j])))
    # determine x ticks
    if log_scale:
        xticks = np.log2(model.x_values)[:longest].astype(int)
        plt.xlabel("Log inverse concentration")
    else:
        xticks = np.array([r'$2^{%s}$' % x for x in np.log2(model.x_values).astype(int)])
        plt.xlabel("Inverse concentration")
    plt.xticks(np.log2(model.x_values)[:longest], xticks[:longest])
    plt.ylabel("Saturation [%]")
    plt.grid()
    plt.xlim(-.5, longest - .5)
    plt.ylim(-10, 1.1 * np.nanmax(model.data[i]))
    return longest, handles, labels


def get_confidence_interval(confidence_interval):
    if confidence_interval == 0.9:
        ci = 1.65
    elif confidence_interval == 0.95:
        ci = 1.96
    elif confidence_interval == 0.99:
        ci = 2.58
    else:
        raise ValueError("confidence_interval not defined.")
    return ci


def make_single_plot(
        model,  # GP_Model
        i,  # index of subject in data
        confidence_interval=0.95,  # confidence_interval
        plot_all=False,  # if False, shows only first measurement
        title="Gaussian Process",  # default title
        log_scale=True  # shows log inverse concentration as x ticks
):
    longest, handles, labels = plot_skeleton(
        model, i, plot_all=plot_all, log_scale=log_scale)
    handle = plt.plot(model.X_test[i], model.mean_predictions[i])[0]
    handles.append(handle)
    labels.append("mean prediction")
    ci = get_confidence_interval(confidence_interval)
    handle = plt.fill_between(
        model.X_test[i],
        model.mean_predictions[i] - ci * model.std_predictions[i],
        model.mean_predictions[i] + ci * model.std_predictions[i],
        alpha=0.5, color='lightblue',
    )
    handles.append(handle)
    labels.append("%s confidence interval" % confidence_interval)
    plt.title(title)
    ax1 = plt.gca()
    ax2 = plt.gca().twinx()
    _ = ax2.plot(model.X_test[i], model.level_prob[i], 'red')
    ax2.set_ylabel(f'P(Saturation={model.level}%)', color='red')
    return handles, labels, ax1


def plot_result(
        model,  # GP_Model
        save_dir,  # Folder for saving (will be created)
        plot_size=1.0,  # Relative plot size
        confidence_interval=0.95,  # confidence_intervals
        dpi=300,  #  figure resolution
        log_scale=True  # shows log inverse concentration as x ticks
):
    os.makedirs(save_dir, exist_ok=True)
    # do twice for separate and big fig
    for ind in range(2):
        if ind == 0:
            plt.figure(figsize=(plot_size * 12, plot_size * 4 * model.num_subject))
        for i in range(model.num_subject):
            if ind == 0:
                plt.subplot(model.num_subject, 2, i * 2 + 1)
            else:
                plt.figure(figsize=(plot_size * 12, plot_size * 4 * 1))
                plt.subplot(1, 2, 1)
            handles, labels, ax1 = make_single_plot(
                model, i, confidence_interval=confidence_interval,
                title="Subject %s" % (i + 1), plot_all=True,
                log_scale=log_scale
            )
            if ind == 0:
                plt.subplot(model.num_subject, 2, i * 2 + 2)
            else:
                plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.legend(handles, labels, loc='upper left')
            mean = np.sum(model.X_test[i] * model.level_prob[i])
            max = model.X_test[i][np.argmax(model.level_prob[i])]
            q5 = model.X_test[i][np.where(np.cumsum(
                model.level_prob[i]) >= 1 - confidence_interval)[0][0]]
            q95 = model.X_test[i][np.where(np.cumsum(
                    model.level_prob[i]) >= confidence_interval)[0][0]]
            if not log_scale:
                mean = 2 ** mean
                max = 2 ** max
                q5 = 2 ** q5
                q95 = 2 ** q95
            plt.text(
                0.05, 0,
                'GP fitting results:\n' +
                f'Mean = {mean:.2f}' +
                f'\nMaximum = {max:.2f}' +
                f'\n5%' + ' = %.2f' % q5 +
                f'\n95%' + ' = %.2f' % q95 +
                f'\nKernel 1: {model.models[i].kernel_.k1}' +
                f'\nKernel 2: {model.models[i].kernel_.k2}' +
                f'\nLogLikelihood = {model.models[i].log_marginal_likelihood():.2f}'
            )
            plt.tight_layout()
            if ind == 1:
                save_file = os.path.join(
                    save_dir, 'subject_%s.png' % (i + 1)
                )
                print('saving figure in:', save_file)
                plt.savefig(save_file, dpi=dpi)
                plt.show()
        if ind == 0:
            save_file = os.path.join(
                save_dir, 'all_subjects.png'
            )
            print('saving figure in:', save_file)
            plt.savefig(save_file, dpi=dpi)
            plt.close()


def plot_fig1(
        model,  # GP_Model
        save_dir,  # Folder for saving (will be created)
        plot_size=1.0,  # Relative plot size
        confidence_interval=0.95,   # confidence_intervals
        point_size=300,  # size of points in sactter plot
        dpi=300,  #  figure resolution
        gp_output='mean',  # what estimate to return for GP
        log_scale=True  # shows log inverse concentration as x ticks
):
    os.makedirs(save_dir, exist_ok=True)
    methods = {
        'closest_to_50': closest_to_50,
        'first_over_25': first_over_25,
        'mean_between_25_75': mean_between_25_75,
        'interpolation': interpolation
    }
    df = pd.DataFrame(
        index=['subject %s' % (i + 1) for i in range(model.num_subject)],
        columns=list(methods.keys()) + ['GP']
    )
    ax2_limits = np.max(model.level_prob)
    ax2_limits = (- 0.05 * ax2_limits, 1.05 * ax2_limits)
    plt.figure(figsize=(plot_size * 12 * 2, plot_size * 4 * model.num_subject))
    for i in range(model.num_subject):
        title = "Subject %s" % (i + 1)
        for k, m in enumerate(methods):
            plt.subplot(model.num_subject, 6, 1 + k + i * 6)
            longest, handles, labels = plot_skeleton(
                model, i, plot_all=False, log_scale=log_scale)
            try:
                if m in ['closest_to_50', 'first_over_25',
                         'mean_between_25_75', 'interpolation']:
                    estimate, ind = methods[m](
                        np.log2(model.x_values)[:longest], model.data[i, 0, :longest]
                    )
                    df[m]['subject %s' % (i + 1)] = 2. ** estimate
                    plt.scatter(
                        np.log2(model.x_values)[:longest][ind],
                        model.data[i, 0][:longest][ind],
                        s=point_size,
                        facecolors='none', edgecolors='r'
                    )
                    if m in ['first_over_25', 'mean_between_25_75']:
                        plt.hlines(25, model.X_test[i][0], model.X_test[i][-1],
                                            linestyle='--', color='green')
                    if m == 'mean_between_25_75':
                        plt.hlines(75, model.X_test[i][0], model.X_test[i][-1],
                                            linestyle='--', color='green')
                    if m == 'interpolation':
                        plt.plot(
                            np.log2(model.x_values)[:longest][ind],
                            model.data[i, 0, :longest][ind],
                            color='black', alpha=.5
                        )
                plt.vlines(estimate, -10, 1.1 * np.nanmax(model.data[i]))
                if not log_scale:
                    estimate = 2 ** estimate
                plt.text(0, 0, 'estimate: %.4f' % estimate)
                plt.title(title + ' - ' + m)
            except Exception as e:
                plt.title(e)
        plt.subplot(model.num_subject, 6, 5 + i * 6)
        handles, labels, ax1 = make_single_plot(
            model, i, confidence_interval=confidence_interval,
            plot_all=False, log_scale=log_scale,
            title=title + ' - ' + 'Gaussian Process')
        plt.yticks([])
        plt.ylim(*ax2_limits)
        # take mean estimate
        if gp_output == 'mean':
            estimate = np.sum(model.X_test[i] * model.level_prob[i])
        elif gp_output == 'max':
            estimate = model.X_test[i][np.argmax(model.level_prob[i])]
        else:
            raise ValueError("gp_output not defined, must be one of: {'mean', 'max'}")
        df['GP']['subject %s' % (i + 1)] = 2. ** estimate
        plt.vlines(estimate, -10, 1.1 * np.nanmax(model.data[i]))
        if not log_scale:
            estimate = 2 ** estimate
        ax1.text(0, 0, 'estimate: %.4f' % estimate)
    plt.tight_layout()
    save_file = os.path.join(save_dir, 'fig1.png')
    print('saving figure and output table in:', save_file)
    plt.savefig(save_file, dpi=dpi)
    plt.show()
    df.to_excel(os.path.join(save_dir, 'output.xlsx'))


### Other Methods ###
def closest_to_50(x, y):
    "Returns x value that was closest to y=50"
    assert x.shape == y.shape
    y_ = y.copy()
    y_[np.isnan(y_)] = np.inf
    ind = np.argmin(abs(y_ - 50))
    return x[ind], ind

def first_over_25(x, y):
    "Returns first x value that was over y=25"
    assert x.shape == y.shape
    for ind, value in enumerate(y):
        if value > 25:
            return x[ind], ind
    else:
        raise ValueError("No measurement > 25!")

def mean_between_25_75(x, y):
    "Returns mean of all x values in y=[25, 75]"
    assert x.shape == y.shape
    indices = np.logical_and(
        25 < y, y < 75
    )
    #print('used for mean:', y[indices])
    if np.sum(indices) > 0:
        return np.mean(x[indices]), indices
    else:
        raise ValueError("No measurement between 25 and 75!")

def interpolation(x, y):
  "Returns interpolated x for closest y_0<50 and y_1>50."
  assert x.shape == y.shape
  # checks
  if np.min(y) > 50:
    raise ValueError("All measurements above 50!")
  if np.max(y) < 50:
    raise ValueError("All measurements under 50!")
  # search from last value
  ind_1 = len(y) - 1
  ind_0 = len(y) - 2
  while True:
      if y[ind_0] <= 50 and 50 <= y[ind_1]:
          break
      else:
          ind_0 -= 1
          ind_1 -= 1
      if ind_0 < 0:
          raise ValueError("Interpolation conditions not met!")
  # interpolate
  x_1 = x[ind_0]
  x_2 = x[ind_1]
  y_1 = y[ind_0]
  y_2 = y[ind_1]
  intercept = y_1 - (y_2 - y_1) / (x_2 - x_1) * x_1
  y_c = 50
  x_c = (y_c - intercept) * (x_2 - x_1) / (y_2 - y_1)
  return x_c, np.array((ind_0, ind_1))

