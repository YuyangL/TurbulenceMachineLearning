import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from FieldData import FieldData
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor
from Preprocess.Feature import getInvariantFeatureSet, getSupplementaryInvariantFeatures
from Utility import interpolateGridData
from Preprocess.FeatureExtraction import splitTrainTestDataList
from Preprocess.GridSearchSetup import setupDecisionTreeGridSearchCV, setupRandomForestGridSearch, setupAdaBoostGridSearchCV,  performEstimatorGridSearch
import time as t
# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

from scipy.interpolate import griddata
from numba import njit, prange
from Utilities import timer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import RegressorChain
import matplotlib.pyplot as plt
from PlottingTool import BaseFigure, Plot2D_Image, Plot2D_MultiAxes
from scipy import ndimage
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import copy
from joblib import load, dump
from sklearn.model_selection import validation_curve
import os

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both RANS and LES
rans_case_name = 'RANS_Re10595'  # str
les_case_name = 'LES_Breuer/Re_10595'  # str
# LES data name to read
les_data_name = 'Hill_Re_10595_Breuer.csv'  # str
# Absolute directory of this flow case
casedir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'


"""
Machine Learning Settings
"""
# Feature set number
fs = 'grad(TKE)_grad(p)+'  # '1', '12', '123'
# Name of the ML estimator
estimator_name = 'tbgb'  # "tbdt", "tbrf", "tbab", "tbrc", 'tbnn''
# Whether to presort X for every feature before finding the best split at each node
presort = True  # bool
# Maximum number of features to consider for best split
max_features = (1/3., 2/3., 1.)  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples at leaf
min_samples_leaf = 2  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples to perform a split
min_samples_split = (0.0005, 0.001, 0.002)  # list/tuple(int / float 0-1) or int / float 0-1
# L2 regularization fraction to penalize large optimal 10 g found through LS fit of min_g(bij - Tij*g)
alpha_g_fit = 0#(0., 0.001)  # list/tuple(float) or float
# L2 regularization coefficient to penalize large optimal 10 g during best split finder
alpha_g_split = (0., 0.00001, 0.001)  # list/tuple(float) or float
# Best split finding scheme to speed up the process of locating the best split in samples of each node
split_finder = "brent"  # "brute", "brent", "1000", "auto"
# Cap of optimal g magnitude after LS fit
g_cap = None  # int or float or None
# Realizability iterator to shift predicted bij to be realizable
realize_iter = 0  # int
# For specific estimators only
if estimator_name in ('TBDT', 'tbdt'):
    tbkey = 'tree__tb'
    max_depth = None
    bij_novelty = None
if estimator_name in ("TBRF", "tbrf"):
    n_estimators = 8  # int
    oob_score = True  # bool
    median_predict = True  # bool
    # What to do with bij novelties
    bij_novelty = None  # 'excl', 'reset', None
    tbkey = 'rf__tb'
    max_depth = None
elif estimator_name in ("TBAB", "tbab"):
    n_estimators = 8  # int
    learning_rate = (0.1, 0.2, 0.4)  # int
    # Override global setting
    min_samples_split = 0.002
    loss = "square"  # 'linear'/'square'/'exponential'
    # What to do with bij novelties
    bij_novelty = None  # 'reset', None
    tbkey = 'ab__tb'
elif estimator_name in ('TBGB', 'tbgb'):
    n_estimators = 8
    learning_rate = (0.1, 0.2, 0.4)
    subsample = 0.8
    # Override global setting
    min_samples_split = 0.002
    # FIXME: this functionality doesn't work
    n_iter_no_change = None
    tol = 1e-8
    bij_novelty = None  # 'reset', None
    loss = 'ls'
    tbkey = 'gb__tb'

# Seed value for reproducibility
seed = 123
# For debugging, verbose on bij reconstruction from Tij and g;
# and/or verbose on "brent"/"brute"/"1000"/"auto" split finding scheme
tb_verbose, split_verbose = False, False  # bool; bool
# Fraction of data for testing
test_fraction = 0.2  # float [0-1]
# Whether verbose on GSCV. 0 means off, larger means more verbose
gs_verbose = 2  # int
# Number of n-fold cross validation
cv = 4  # int or None


"""
Process User Inputs, No Need to Change
"""
# Average fields of interest for reading and processing
fields = ('U', 'k', 'p', 'omega',
          'grad_U', 'grad_k', 'grad_p')
# Ensemble name of fields useful for Machine Learning
ml_field_ensemble_name = 'ML_Fields_' + rans_case_name
# Initialize case object
case = FieldData(casename=rans_case_name, casedir=casedir, times=time, fields=fields, save=False)
if estimator_name == "tbdt":
    estimator_name = "TBDT"
    # Dictionary of hyper-parameters
    paramdict = dict(min_samples_split=min_samples_split,
                     max_features=max_features,
                     alpha_g_split=alpha_g_split)
elif estimator_name == "tbrf":
    estimator_name = "TBRF"
    paramdict = dict(min_samples_split=min_samples_split,
                     max_features=max_features,
                     alpha_g_split=alpha_g_split)
elif estimator_name == 'tbab':
    estimator_name = 'TBAB'
    paramdict = dict(base_estimator__max_depth=(5, 10),
                     base_estimator__max_features=max_features,
                     base_estimator__alpha_g_split=alpha_g_split,
                     learning_rate=learning_rate)
elif estimator_name == 'tbgb':
    estimator_name = 'TBGB'
    paramdict = dict(max_depth=(5, 10),
                     max_features=max_features,
                     alpha_g_split=alpha_g_split,
                     learning_rate=learning_rate)

figdir = case.result_paths[time] + '/ValidationCurve/'
os.makedirs(figdir, exist_ok=True)


"""
Read Data
"""
list_data_train = case.readPickleData(time, 'list_47data_train_seed' + str(seed))
list_data_test = case.readPickleData(time, 'list_47data_test_seed' + str(seed))
cc_train, cc_test = list_data_train[0], list_data_test[0]
ccx_train, ccy_train, ccz_train = cc_train[:, 0], cc_train[:, 1], cc_train[:, 2]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_train, y_train, tb_train = list_data_train[1:4]
x_test, y_test, tb_test = list_data_test[1:4]


"""
Initialize Regressors
"""
if estimator_name == 'TBDT':
    regressor = DecisionTreeRegressor(presort=presort, max_depth=max_depth, tb_verbose=tb_verbose,
                                      min_samples_leaf=min_samples_leaf,
                                      split_finder=split_finder, split_verbose=split_verbose,
                                      alpha_g_fit=alpha_g_fit,
                                      g_cap=g_cap,
                                      realize_iter=realize_iter,
                                      random_state=seed,
                                      max_features=1.,
                                      min_samples_split=0.002,
                                      alpha_g_split=0.)
elif estimator_name == 'TBRF':
    regressor = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, verbose=gs_verbose,
                                      oob_score=True,
                                      max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      realize_iter=realize_iter,
                                      split_verbose=split_verbose,
                                      split_finder=split_finder,
                                      tb_verbose=tb_verbose,
                                      median_predict=True,
                                      bij_novelty=bij_novelty,
                                      g_cap=g_cap,
                                      random_state=seed,
                                      alpha_g_split=0.,
                                      min_samples_split=0.002,
                                      max_features=1.,
                                      # [DEPRECATED]
                                      alpha_g_fit=alpha_g_fit)
elif estimator_name == 'TBAB' or estimator_name == 'TBGB':
    base = DecisionTreeRegressor(presort=presort, tb_verbose=tb_verbose,
                                 min_samples_leaf=min_samples_leaf,
                                 min_samples_split=0.002,
                                 split_finder=split_finder, split_verbose=split_verbose,
                                 g_cap=g_cap,
                                 realize_iter=realize_iter,
                                 max_depth=5,
                                 max_features=1.,
                                 alpha_g_split=0.,
                                 # [DEPRECATED]
                                 alpha_g_fit=alpha_g_fit)
    if estimator_name == 'TBAB':
        regressor = AdaBoostRegressor(base_estimator=base,
                                      n_estimators=n_estimators,
                                      loss=loss,
                                      random_state=seed,
                                      bij_novelty=bij_novelty,
                                      learning_rate=0.1)
    else:
        regressor = GradientBoostingRegressor(subsample=subsample,
                                      n_estimators=n_estimators,
                                      loss=loss,
                                      random_state=seed,
                                      bij_novelty=bij_novelty,
                                      learning_rate=0.1,
                                              criterion='mse',
                                              min_samples_leaf=0.001,
                                              min_samples_split=0.002,
                                              max_depth=5,
                                              max_features=1.,
                                              init='zero',
                                              verbose=2,
                                              split_finder=split_finder, split_verbose=split_verbose,
                                              g_cap=g_cap,
                                              realize_iter=realize_iter,
                                              alpha_g_split=0.,
                                              presort=presort)

"""
Validation Curve
"""
# train_scores_all, test_scores_all = (np.empty((1, cv)),)*2
train_score_all, test_score_all = [], []
train_score_mean, test_score_mean = [], []
train_score_std, test_score_std = [], []
labels0, param_ranges = [], []
i = 0
for param_name, param_range in paramdict.items():
    train_score, test_score = validation_curve(regressor, x_train, y_train, param_name=param_name, param_range=param_range,
                              cv=cv, n_jobs=-1, verbose=gs_verbose, tb=tb_train, bij_novelty=bij_novelty)
    # train_scores_all = np.concatenate((train_scores_all, train_scores))
    # test_scores_all = np.concatenate((test_scores_all, test_scores))
    train_score_all.append(train_score)
    test_score_all.append(test_score)
    # Raw hyper-parameter names used by Scikit-learn
    labels0.append(param_name)
    train_score_mean.append(np.maximum(np.mean(train_score, axis=1), -10))
    train_score_std.append(np.std(train_score, axis=1))
    test_score_mean.append(np.maximum(np.mean(test_score, axis=1), -10))
    test_score_std.append(np.std(test_score, axis=1))
    param_ranges.append(param_range)
    i += 1

# Actual x axes labels for plotting, to be formatted
labels = ['1', '2', '3', '4']
xscales = ['linear', 'linear', 'linear', 'linear', 'linear']
# Reformat hyper-parameter names for plotting as Latex doesn't recognize '_'
for i in range(len(labels0)):
    if 'max_feat' in labels0[i]:
        labels[i] = 'Max features \%'
    elif 'max_depth' in labels0[i]:
        labels[i] = 'Max depths'
    elif 'samples_split' in labels0[i]:
        labels[i] = 'Min split samples \%'
    elif 'alpha_g' in labels0[i]:
        labels[i] = r'$\alpha_{g}$'
        xscales[i] = 'log'
    elif 'learning' in labels0[i]:
        labels[i] = 'Learning rate'

if 'TBAB' or 'TBGB' in estimator_name:
    ymin, ymax = 1e9, -1e9
    for i in range(len(test_score_mean)):
        ymin0 = min(np.minimum(np.array(test_score_mean[i]).ravel(), np.array(train_score_mean[i]).ravel()))
        ymax0 = max(np.maximum(np.array(train_score_mean[i]).ravel(), np.array(test_score_mean[i]).ravel()))
        if ymin0 < ymin: ymin = ymin0
        if ymax0 > ymax: ymax = ymax0
else:
    ymin = min(np.minimum(np.array(test_score_mean).ravel(), np.array(train_score_mean).ravel()))
    ymax = max(np.maximum(np.array(train_score_mean).ravel(), np.array(test_score_mean).ravel()))

ymin *= 0.9 if ymin >= 0. else 1.1
ymax *= 1.1 if ymax > 0 else 0.9
ylim = (ymin, ymax)
# Prepare other hyper-parametrs than the 1st one
list_x2, list_y2 = [], []
for i in range(1, len(labels0)):
    # For alhpa_g_split, instead of plotting for 0, plot 1e-7 instead for log to work
    list_x2.append(np.maximum(param_ranges[i], 1e-7))
    list_x2.append(np.maximum(param_ranges[i], 1e-7))
    list_y2.append(train_score_mean[i])
    list_y2.append(test_score_mean[i])

# If 3 or less hyper-parameters to plot, plot them in 1 figure
if len(labels0) <= 3:
    myplot = Plot2D_MultiAxes((param_ranges[0],)*2, (train_score_mean[0], test_score_mean[0]), list_x2,
                              list_y2,
                            ax2label=labels[1:], save=True, show=False, figdir=figdir, name=estimator_name,
                              figwidth='half', figheight_multiplier=1.25,
                              ylim=ylim,
                              ylabel='$R^2$ score', xlabel=labels[0])
    myplot.initializeFigure()

    linelabel = ('Train', 'CV')
    ls = ('-', '--')
    myplot.markercolors = None
    linelabel2 = ('Train', 'CV')*5
    yscale = 'symlog' if max(np.abs(ylim)) - min(np.abs(ylim)) > 10 else 'linear'
    myplot.linelabel2 = np.arange(1, myplot.narr2 + 1) if linelabel2 is None else linelabel2

    for i in range(2):
        myplot.axes.plot(myplot.list_x[i], myplot.list_y[i], ls=ls[i], color=myplot.colors[0],
                       alpha=myplot.alpha,
                       marker=myplot.markers[0])

    # Dummy plots purely for the legend
    myplot.axes.plot(myplot.list_x[0], 1e9*np.ones_like(myplot.list_x[0]), ls=ls[0], label=linelabel[0], color=myplot.gray)
    myplot.axes.plot(myplot.list_x[0], 1e9*np.ones_like(myplot.list_x[0]), ls=ls[1], label=linelabel[1], color=myplot.gray)
    myplot.axes.set_yscale(yscale)
    myplot.axes.tick_params(axis='x', colors=myplot.colors[0])
    myplot.axes.xaxis.label.set_color(myplot.colors[0])
    myplot.axes2, myplot.plots2 = ([None]*myplot.narr2,)*2
    i = 0
    for i0 in range(0, myplot.narr2, 2):
        # xscale2 = 'symlog' if max(myplot.list_x2[i0])/np.maximum(min(np.abs(myplot.list_x2[i0])), 1e-10) > 100 else 'linear'
        color = myplot.colors[i + myplot.narr] if myplot.plot_type2[i] != 'shade' else (150/255., 150/255., 150/255.)
        # If new axes are x
        if myplot.ax2loc == 'x':
            myplot.axes2[i] = myplot.axes.twiny()
            # Move twinned axis ticks and label from top to bottom
            myplot.axes2[i].xaxis.set_ticks_position("bottom")
            myplot.axes2[i].xaxis.set_label_position("bottom")
            # Offset the twin axis below the host
            myplot.axes2[i].spines['bottom'].set_position(('axes', -.375*(i + 1)))
        # Else if new axes are y
        else:
            myplot.axes2[i] = myplot.axes.twinx()
            myplot.axes2[i].spines['right'].set_position(('axes', 1 + 0.1*i))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom/right spine
        myplot.axes2[i].set_frame_on(True)
        myplot.axes2[i].patch.set_visible(False)
        for sp in myplot.axes2[i].spines.values():
            sp.set_visible(False)

        if myplot.ax2loc == 'x':
            myplot.axes2[i].spines["bottom"].set_visible(True)
        else:
            myplot.axes2[i].spines["right"].set_visible(True)

        if myplot.ax2loc == 'x':

            myplot.axes2[i].set_xlabel(myplot.ax2label[i])

            myplot.axes2[i].tick_params(axis='x', colors=myplot.colors[i + 1])
            myplot.axes2[i].xaxis.label.set_color(myplot.colors[i + 1])
        else:

            myplot.axes2[i].set_ylabel(myplot.ax2label[i])
            myplot.axes2[i].tick_params(axis='y', colors=color)
            myplot.axes2[i].yaxis.label.set_color(color)

        myplot.axes2[i].set_xscale(xscales[i + 1])#, myplot.axes2[i].set_yscale(yscale)
        if myplot.x2lim is not None: myplot.axes2[i].set_xlim(myplot.x2lim)
        if myplot.y2lim is not None: myplot.axes2[i].set_ylim(myplot.y2lim)

        # Plot
        # Set the 2nd axis plots layer in the back. The higher zorder, the more front the plot is
        myplot.axes.set_zorder(myplot.axes2[i].get_zorder() + 1)
        if myplot.plot_type2[i] == 'shade':
            myplot.plots2[i] = myplot.axes2[i].fill_between(myplot.list_x2[i], 0, myplot.list_y2[i], alpha=1,
                                                        facecolor=(150/255., 150/255., 150/255.),
                                                        interpolate=False)
        elif myplot.plot_type2[i] == 'line':
            myplot.axes2[i].plot(myplot.list_x2[i0], myplot.list_y2[i0],
                                                 ls=ls[0],
                                                 color=myplot.colors[i + 1], alpha=myplot.alpha,
                                                 marker=myplot.markers[i + 1])
            myplot.axes2[i].plot(myplot.list_x2[i0 + 1], myplot.list_y2[i0 + 1],
                                                     ls=ls[1],
                                                     color=myplot.colors[i + 1], alpha=myplot.alpha,
                                                     marker=myplot.markers[i + 1])

        i += 1

    myplot.finalizeFigure(tight_layout=True, xyscale=(xscales[0], yscale))
# Otherwise, if 4 hyper-parameters or more, divide it to two plots
else:
    # 1st plot, plotting first 2 hyper-parameters
    myplot = Plot2D_MultiAxes((param_ranges[0],)*2, (train_score_mean[0], test_score_mean[0]), list_x2[0:2],
                              list_y2[0:2],
                              ax2label=labels[1], save=True, show=False, figdir=figdir, name=estimator_name,
                              figwidth='half', figheight_multiplier=1.2,
                              ylim=ylim,
                              ylabel='$R^2$ score', xlabel=labels[0])
    myplot.initializeFigure()

    linelabel = ('Train', 'CV')
    ls = ('-', '--')
    myplot.markercolors = None
    linelabel2 = ('Train', 'CV')*5
    yscale = 'symlog' if max(np.abs(ylim)) - min(np.abs(ylim)) > 10 else 'linear'
    myplot.linelabel2 = np.arange(1, myplot.narr2 + 1) if linelabel2 is None else linelabel2

    for i in range(2):
        myplot.axes.plot(myplot.list_x[i], myplot.list_y[i], ls=ls[i], color=myplot.colors[0],
                         alpha=myplot.alpha,
                         marker=myplot.markers[0])

    # Dummy plots purely for the legend
    myplot.axes.plot(myplot.list_x[0], 1e9*np.ones_like(myplot.list_x[0]), ls=ls[0], label=linelabel[0],
                     color=myplot.gray)
    myplot.axes.plot(myplot.list_x[0], 1e9*np.ones_like(myplot.list_x[0]), ls=ls[1], label=linelabel[1],
                     color=myplot.gray)
    myplot.axes.set_yscale(yscale)
    myplot.axes.tick_params(axis='x', colors=myplot.colors[0])
    myplot.axes.xaxis.label.set_color(myplot.colors[0])
    myplot.axes2, myplot.plots2 = ([None]*myplot.narr2,)*2
    i = 0
    for i0 in range(0, myplot.narr2, 2):
        # xscale2 = 'symlog' if max(myplot.list_x2[i0])/np.maximum(min(np.abs(myplot.list_x2[i0])), 1e-10) > 100 else 'linear'
        color = myplot.colors[i + myplot.narr] if myplot.plot_type2[i] != 'shade' else (150/255., 150/255., 150/255.)
        # If new axes are x
        if myplot.ax2loc == 'x':
            myplot.axes2[i] = myplot.axes.twiny()
            # Move twinned axis ticks and label from top to bottom
            myplot.axes2[i].xaxis.set_ticks_position("bottom")
            myplot.axes2[i].xaxis.set_label_position("bottom")
            # Offset the twin axis below the host
            myplot.axes2[i].spines['bottom'].set_position(('axes', -0.25*(i + 1)))
        # Else if new axes are y
        else:
            myplot.axes2[i] = myplot.axes.twinx()
            myplot.axes2[i].spines['right'].set_position(('axes', 1 + 0.1*i))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom/right spine
        myplot.axes2[i].set_frame_on(True)
        myplot.axes2[i].patch.set_visible(False)
        for sp in myplot.axes2[i].spines.values():
            sp.set_visible(False)

        if myplot.ax2loc == 'x':
            myplot.axes2[i].spines["bottom"].set_visible(True)
        else:
            myplot.axes2[i].spines["right"].set_visible(True)

        if myplot.ax2loc == 'x':

            myplot.axes2[i].set_xlabel(myplot.ax2label[i])

            myplot.axes2[i].tick_params(axis='x', colors=myplot.colors[i + 1])
            myplot.axes2[i].xaxis.label.set_color(myplot.colors[i + 1])
        else:

            myplot.axes2[i].set_ylabel(myplot.ax2label[i])
            myplot.axes2[i].tick_params(axis='y', colors=color)
            myplot.axes2[i].yaxis.label.set_color(color)

        myplot.axes2[i].set_xscale(xscales[i + 1])  # , myplot.axes2[i].set_yscale(yscale)
        if myplot.x2lim is not None: myplot.axes2[i].set_xlim(myplot.x2lim)
        if myplot.y2lim is not None: myplot.axes2[i].set_ylim(myplot.y2lim)

        # Plot
        # Set the 2nd axis plots layer in the back. The higher zorder, the more front the plot is
        myplot.axes.set_zorder(myplot.axes2[i].get_zorder() + 1)
        if myplot.plot_type2[i] == 'shade':
            myplot.plots2[i] = myplot.axes2[i].fill_between(myplot.list_x2[i], 0, myplot.list_y2[i], alpha=1,
                                                            facecolor=(150/255., 150/255., 150/255.),
                                                            interpolate=False)
        elif myplot.plot_type2[i] == 'line':
            myplot.axes2[i].plot(myplot.list_x2[i0], myplot.list_y2[i0],
                                 ls=ls[0],
                                 color=myplot.colors[i + 1], alpha=myplot.alpha,
                                 marker=myplot.markers[i + 1])
            myplot.axes2[i].plot(myplot.list_x2[i0 + 1], myplot.list_y2[i0 + 1],
                                 ls=ls[1],
                                 color=myplot.colors[i + 1], alpha=myplot.alpha,
                                 marker=myplot.markers[i + 1])

        i += 1

    myplot.finalizeFigure(tight_layout=True, xyscale=(xscales[0], yscale))


    # 2nd plot
    if len(labels0[2:]) > 2:
        figheight_multiplier = 1.25
        axis_offset = 0.3
        tight_layout = False
    else:
        figheight_multiplier = 1.2
        axis_offset = 0.25
        tight_layout = True

    list_x1, list_y1 = list_x2[2:4], list_y2[2:4]
    list_x2, list_y2 = list_x2[4:], list_y2[4:]

    myplot = Plot2D_MultiAxes(list_x1, list_y1, list_x2,
                              list_y2,
                              ax2label=labels[3:], save=True, show=False, figdir=figdir, name=estimator_name + '2',
                              figwidth='half', figheight_multiplier=figheight_multiplier,
                              ylim=ylim,
                              ylabel='$R^2$ score', xlabel=labels[2])
    myplot.initializeFigure()

    linelabel = ('Train', 'CV')
    ls = ('-', '--')
    myplot.markercolors = None
    linelabel2 = ('Train', 'CV')*5
    yscale = 'symlog' if max(np.abs(ylim)) - min(np.abs(ylim)) > 10 else 'linear'
    myplot.linelabel2 = np.arange(1, myplot.narr2 + 1) if linelabel2 is None else linelabel2

    for i in range(2):
        myplot.axes.plot(myplot.list_x[i], myplot.list_y[i], ls=ls[i], color=myplot.colors[0],
                         alpha=myplot.alpha,
                         marker=myplot.markers[0])

    # Dummy plots purely for the legend
    myplot.axes.plot(myplot.list_x[0], 1e9*np.ones_like(myplot.list_x[0]), ls=ls[0], label=linelabel[0],
                     color=myplot.gray)
    myplot.axes.plot(myplot.list_x[0], 1e9*np.ones_like(myplot.list_x[0]), ls=ls[1], label=linelabel[1],
                     color=myplot.gray)
    myplot.axes.set_yscale(yscale)
    myplot.axes.tick_params(axis='x', colors=myplot.colors[0])
    myplot.axes.xaxis.label.set_color(myplot.colors[0])
    myplot.axes2, myplot.plots2 = ([None]*myplot.narr2,)*2
    i = 0
    for i0 in range(0, myplot.narr2, 2):
        # xscale2 = 'symlog' if max(myplot.list_x2[i0])/np.maximum(min(np.abs(myplot.list_x2[i0])), 1e-10) > 100 else 'linear'
        color = myplot.colors[i + myplot.narr] if myplot.plot_type2[i] != 'shade' else (160/255., 160/255., 160/255.)
        # If new axes are x
        if myplot.ax2loc == 'x':
            myplot.axes2[i] = myplot.axes.twiny()
            # Move twinned axis ticks and label from top to bottom
            myplot.axes2[i].xaxis.set_ticks_position("bottom")
            myplot.axes2[i].xaxis.set_label_position("bottom")
            # Offset the twin axis below the host
            myplot.axes2[i].spines['bottom'].set_position(('axes', -axis_offset*(i + 1)))
        # Else if new axes are y
        else:
            myplot.axes2[i] = myplot.axes.twinx()
            myplot.axes2[i].spines['right'].set_position(('axes', 1 + 0.1*i))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom/right spine
        myplot.axes2[i].set_frame_on(True)
        myplot.axes2[i].patch.set_visible(False)
        for sp in myplot.axes2[i].spines.values():
            sp.set_visible(False)

        if myplot.ax2loc == 'x':
            myplot.axes2[i].spines["bottom"].set_visible(True)
        else:
            myplot.axes2[i].spines["right"].set_visible(True)

        if myplot.ax2loc == 'x':

            myplot.axes2[i].set_xlabel(myplot.ax2label[i])

            myplot.axes2[i].tick_params(axis='x', colors=myplot.colors[i + 1])
            myplot.axes2[i].xaxis.label.set_color(myplot.colors[i + 1])
        else:

            myplot.axes2[i].set_ylabel(myplot.ax2label[i])
            myplot.axes2[i].tick_params(axis='y', colors=color)
            myplot.axes2[i].yaxis.label.set_color(color)

        myplot.axes2[i].set_xscale(xscales[i + 3])  # , myplot.axes2[i].set_yscale(yscale)
        if myplot.x2lim is not None: myplot.axes2[i].set_xlim(myplot.x2lim)
        if myplot.y2lim is not None: myplot.axes2[i].set_ylim(myplot.y2lim)

        # Plot
        # Set the 2nd axis plots layer in the back. The higher zorder, the more front the plot is
        myplot.axes.set_zorder(myplot.axes2[i].get_zorder() + 1)
        if myplot.plot_type2[i] == 'shade':
            myplot.plots2[i] = myplot.axes2[i].fill_between(myplot.list_x2[i], 0, myplot.list_y2[i], alpha=1,
                                                            facecolor=(160/255., 160/255., 160/255.),
                                                            interpolate=False)
        elif myplot.plot_type2[i] == 'line':
            myplot.axes2[i].plot(myplot.list_x2[i0], myplot.list_y2[i0],
                                 ls=ls[0],
                                 color=myplot.colors[i + 1], alpha=myplot.alpha,
                                 marker=myplot.markers[i + 1])
            myplot.axes2[i].plot(myplot.list_x2[i0 + 1], myplot.list_y2[i0 + 1],
                                 ls=ls[1],
                                 color=myplot.colors[i + 1], alpha=myplot.alpha,
                                 marker=myplot.markers[i + 1])

        i += 1

    myplot.finalizeFigure(tight_layout=tight_layout, xyscale=(xscales[2], yscale))


