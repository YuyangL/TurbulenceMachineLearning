import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from FieldData import FieldData
from Preprocess.GridSearchSetup import setupDecisionTreeGridSearchCV
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData

# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from warnings import warn
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import RegressorChain
from joblib import load, dump
import time as t
from Utility import interpolateGridData
from scipy import ndimage
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PlottingTool import BaseFigure, Plot2D_Image
from copy import copy


"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case
casename = 'ALM_N_H_OneTurb'  # str
# Absolute directory of this flow case
casedir = '/media/yluan'  # str
# Which time to extract input and output for ML
time = 'last'  # str/float/int or 'last'
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# What keyword does the gradient fields contain
gradFieldKW = 'grad'  # str
# Slice names for prediction visualization
sliceNames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
              'twoDupstreamTurbine', 'rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine', 'sevenDdownstreamTurbine')
# Whether process field data, invariants, features from scratch,
# or use raw field pickle data and process invariants and features
# or use raw field and invariants pickle data and process features
process_raw_field, process_invariants = False, False  # bool
# Flow field counter-clockwise rotation in x-y plane
# so that tensor fields are parallel/perpendicular to flow direction
fieldRot = np.pi/6  # float [rad]
# When fieldRot is not 0, whether to infer spatial correlation fields and rotate them
# Otherwise the name of the fields needs to be specified
# Strain rate / rotation rate tensor fields not supported
if fieldRot != 0.:
    spatialCorrelationFields = ('infer',)  # list/tuple(str) or list/tuple('infer')
    spatial_corr_slices = ('infer',)

# Whether confine and visualize the domain of interest, useful if mesh is too large
confineBox, plotConfinedBox = True, True  # bool; bool
# Only when confineBox is True:
if confineBox:
    # Subscript of the confined field file name
    confinedFieldNameSub = 'Confined'
    # Whether auto generate confine box for each case,
    # the confinement is bordered by
    # 'first': 1st refinement zone
    # 'second': 2nd refined zone
    # 'third': left half of turbine in 2nd refinement zone
    boxAutoDim = 'second'  # 'first', 'second', 'third', None
    # Only when boxAutoDim is False:
    if boxAutoDim is None:
        # Confine box counter-clockwise rotation in x-y plane
        boxRot = np.pi/6  # float [rad]
        # Confine box origin, width, length, height
        boxOrig = (0, 0, 0)  # (x, y, z)
        boxL, boxW, boxH = 0, 0, 0  # float

# Absolute cap value for Sij and Rij; TB coefficients' basis;  and TB coefficients
cap_sijrij, cap_tij = 1e9, 1e9  # float/int
# Enforce 0 trace when computing Tij?
tij_0trace = False

# Save anything when possible
save_fields, resultFolder = True, 'Result'  # bool; str


"""
Machine Learning Settings
"""
# Feature set number
fs = 'grad(TKE)_grad(p)'  # '1', '12', '123'
# Name of the ML estimator
estimator_name = 'tbdt'  # "tbdt", "tbrf", "tbab", "tbrc"
scaler = None  # "robust", "standard" or None
# Whether to presort X for every feature before finding the best split at each node
presort = True  # bool
# Maximum number of features to consider for best split
max_features = 1.#(0.8, 1.)  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples at leaf
min_samples_leaf = 4#(4, 16)  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples to perform a split
min_samples_split = 16#(8, 64)  # list/tuple(int / float 0-1) or int / float 0-1
# Max depth of the tree to prevent overfitting
max_depth = None  # int
# L2 regularization fraction to penalize large optimal 10 g found through LS fit of min_g(bij - Tij*g)
alpha_g_fit = 0#(0., 0.001)  # list/tuple(float) or float
# L2 regularization coefficient to penalize large optimal 10 g during best split finder
alpha_g_split = 0#(0., 0.001)  # list/tuple(float) or float
# Best split finding scheme to speed up the process of locating the best split in samples of each node
split_finder = "brent"  # "brute", "brent", "1000", "auto"
# Cap of optimal g magnitude after LS fit
g_cap = None  # int or float or None
# Realizability iterator to shift predicted bij to be realizable
realize_iter = 0  # int
# For TBRF only
if estimator_name in ("TBRF", "tbrf"):
    n_estimators = 8  # int
    oob_score = True  # bool
    median_predict = True  # bool
    # What to do with bij novelties
    bij_novelty = 'excl'

if estimator_name in ("TBAB", "tbab"):
    n_estimators = 96  # int
    learning_rate = 0.1  # int
    loss = "linear"  # 'linear'/'square'/'exponential'
    # What to do with bij novelties
    bij_novelty = 'lim'

if estimator_name in ('TBRC', 'tbrc'):
    # b11, b22 have rank 10, b12 has rank 9, b33 has rank 5, b13 and b23 are 0 and have rank 10
    order = [0, 3, 1, 5, 4, 2]  # [4, 2, 5, 3, 1, 0]

# Seed value for reproducibility
seed = 123
# For debugging, verbose on bij reconstruction from Tij and g;
# and/or verbose on "brent"/"brute"/"1000"/"auto" split finding scheme
tb_verbose, split_verbose = False, False  # bool; bool
# Fraction of data for testing
test_fraction = 0.2  # float [0-1]
# Whether verbose on GSCV. 0 means off, larger means more verbose
gscv_verbose = 2  # int
# Number of n-fold cross validation
cv = 5  # int or None


"""
Visualization Settings
"""
uniform_mesh_size = 1e6
bijlims = (-0.5, 2/3.)
savefig = True
show = False
ext, dpi = 'png', 1000



"""
Process User Inputs, No Need to Change
"""
# Average fields of interest for reading and processing
if fs == 'grad(TKE)_grad(p)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_p_rghAvg', 'grad_kResolved', 'grad_kSGSmean', 'UAvg')
elif fs == 'grad(TKE)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_kResolved', 'grad_kSGSmean')
elif fs == 'grad(p)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_p_rghAvg', 'UAvg')
else:
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg')


# Ensemble name of fields useful for Machine Learning
mlFieldEnsembleName = 'ML_Fields_' + casename
# Automatically select time if time is set to 'latest'
if time == 'last':
    if casename == 'ALM_N_H_ParTurb':
        time = '22000.0918025'
    elif casename == 'ALM_N_H_OneTurb':
        time = '24995.0788025'

else:
    time = str(time)

# Automatically define the confined domain region
if confineBox and boxAutoDim is not None:
    boxRot = np.pi/6
    if casename == 'ALM_N_H_ParTurb':
        # 1st refinement zone as confinement box
        if boxAutoDim == 'first':
            boxOrig = (1074.225, 599.464, 0)
            boxL, boxW, boxH = 1134, 1134, 405
            confinedFieldNameSub += '1'
        # 2nd refinement zone as confinement box
        elif boxAutoDim == 'second':
            boxOrig = (1120.344, 771.583, 0)
            boxL, boxW, boxH = 882, 378, 216
            confinedFieldNameSub += '2'
        elif boxAutoDim == 'third':
            boxOrig = (1120.344, 771.583, 0)
            boxL, boxW, boxH = 882, 378/2., 216
            confinedFieldNameSub += '2'

    elif casename == 'ALM_N_H_OneTurb':
        if boxAutoDim == 'first':
            boxOrig = (948.225, 817.702, 0)
            boxL, boxW, boxH = 1134, 630, 405
            confinedFieldNameSub += '1'
        elif boxAutoDim == 'second':
            boxOrig = (994.344, 989.583, 0)
            boxL, boxW, boxH = 882, 378, 216
            confinedFieldNameSub += '2'
        elif boxAutoDim == 'third':
            boxOrig = (994.344, 989.583, 0)
            boxL, boxW, boxH = 882, 378/2., 216
            confinedFieldNameSub += '3'

if not confineBox:
    confinedFieldNameSub = ''

# Subscript for the slice names
sliceNamesSub = 'Slice'

# Ensemble file name, containing fields related to ML
mlFieldEnsembleNameFull = mlFieldEnsembleName + '_' + confinedFieldNameSub
# Initialize case object
case = FieldData(caseName=casename, caseDir=casedir, times=time, fields=fields, save=save_fields, resultFolder=resultFolder)
if estimator_name == "tbdt":
    estimator_name = "TBDT"
elif estimator_name == "tbrf":
    estimator_name = "TBRF"
elif estimator_name == 'tbab':
    estimator_name = 'TBAB'
elif estimator_name in ('TBRC', 'tbrc'):
    estimator_name = 'TBRC'
    # At least 10 samples at leaf so that LS fit g is not under-determined
    min_samples_leaf = max(min_samples_leaf, 10)
    # Realizability iterator of bij prediction is turned off
    realize_iter = 0


"""
Read Train & Test Data
"""
# First GSCV data
list_data_gs = case.readPickleData(time, 'list_data_GS_' + confinedFieldNameSub)
cc_gs = list_data_gs[0]
x_gs = list_data_gs[1]
y_gs = list_data_gs[2]
tb_gs = list_data_gs[3]
del list_data_gs

# Second train data
list_data_train = case.readPickleData(time, 'list_data_train_' + confinedFieldNameSub)
cc_train = list_data_train[0]
x_train = list_data_train[1]
y_train = list_data_train[2]
tb_train = list_data_train[3]
del list_data_train

# Then test data which are various slices
# TODO: make it a loop
slice_type = sliceNames[0]
list_data_test = case.readPickleData(time, 'list_data_test_' + slice_type)
cc_test = list_data_test[0]
ccx_test, ccy_test = cc_test[:, 0], cc_test[:, 1]
x_test = list_data_test[1]
y_test = list_data_test[2]
tb_test = list_data_test[3]
del list_data_test


"""
Machine Learning
"""
base_estimator = DecisionTreeRegressor(presort=presort, max_depth=max_depth, tb_verbose=tb_verbose,
                                       min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split,
                                       split_finder=split_finder, split_verbose=split_verbose,
                                       max_features=max_features,
                                       alpha_g_fit=alpha_g_fit,
                                       alpha_g_split=alpha_g_split,
                                       g_cap=g_cap,
                                       realize_iter=realize_iter)

if estimator_name == 'TBDT' and cv is not None:
    regressor, tune_params = setupDecisionTreeGridSearchCV(max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                           alpha_g_fit=alpha_g_fit, alpha_g_split=alpha_g_split,
                                                           presort=presort, split_finder=split_finder,
                                                           tb_verbose=tb_verbose, split_verbose=split_verbose, scaler=scaler, rand_state=seed, gscv_verbose=gscv_verbose,
                                                           cv=cv, max_depth=max_depth,
                                                           g_cap=g_cap,
                                                           realize_iter=realize_iter)
elif estimator_name == 'TBRF':
    regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf, max_features=max_features,
                                      oob_score=oob_score, n_jobs=-1,
                                      random_state=seed, verbose=gscv_verbose,
                                      tb_verbose=tb_verbose, split_finder=split_finder,
                                      split_verbose=split_verbose, alpha_g_fit=alpha_g_fit,
                                      alpha_g_split=alpha_g_split, g_cap=g_cap,
                                      realize_iter=realize_iter,
                                      median_predict=median_predict,
                                      bij_novelty=bij_novelty)
elif estimator_name == 'TBAB':
    regressor = AdaBoostRegressor(base_estimator=base_estimator,
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  loss=loss,
                                  random_state=seed,
                                  bij_novelty=bij_novelty)
elif estimator_name == 'TBRC':
    regressor = RegressorChain(base_estimator=base_estimator,
                               order=order,
                               cv=cv,
                               random_state=seed)

else:
    regressor = base_estimator

t0 = t.time()
regressor.fit(x_train, y_train, tb=tb_train)
t1 = t.time()
print('\nFinished DecisionTreeRegressor in {:.4f} s'.format(t1 - t0))

# Save estimator
dump(regressor, case.resultPaths[time] + estimator_name + '.joblib')


score_test = regressor.score(x_test, y_test, tb=tb_test)
score_train = regressor.score(x_train, y_train, tb=tb_train)

t0 = t.time()
y_pred_test = regressor.predict(x_test, tb=tb_test)
y_pred_train = regressor.predict(x_train, tb=tb_train)
t1 = t.time()
print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))


"""
Postprocess Machine Learning Predictions
"""
t0 = t.time()
_, eigval_test, _ = processReynoldsStress(y_test, make_anisotropic=False, realization_iter=0)
_, eigval_pred_test, _ = processReynoldsStress(y_pred_test, make_anisotropic=False, realization_iter=realize_iter)
t1 = t.time()
print('\nFinished processing Reynolds stress in {:.4f} s'.format(t1 - t0))

t0 = t.time()
xy_bary_test, rgb_bary_test = getBarycentricMapData(eigval_test)
xy_bary_pred_test, rgb_bary_pred_test = getBarycentricMapData(eigval_pred_test)
t1 = t.time()
print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))

t0 = t.time()
ccx_test_mesh, ccy_test_mesh, _, rgb_bary_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, rgb_bary_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for barycentric map in {:.4f} s'.format(t1 - t0))

t0 = t.time()
_, _, _, y_test_mesh = interpolateGridData(ccx_test, ccy_test, y_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, y_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for bij in {:.4f} s'.format(t1 - t0))


"""
Plotting
"""
rgb_bary_test_mesh = ndimage.rotate(rgb_bary_test_mesh, 90)
rgb_bary_pred_test_mesh = ndimage.rotate(rgb_bary_pred_test_mesh, 90)
xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
figname = 'barycentric_periodichill_test_seed' + str(seed)
bary_map = BaseFigure((None,), (None,), name=figname, xlabel=xlabel,
                      ylabel=ylabel, save=savefig, show=show,
                      figdir=case.resultPaths[time],
                      figheight_multiplier=0.7)
extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
if savefig:
    plt.savefig(case.resultPaths[time] + figname + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_pred_test_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
if savefig:
    plt.savefig(case.resultPaths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()


fignames_predtest = ('b11_periodichill_pred_test_seed' + str(seed),
                     'b12_periodichill_pred_test_seed' + str(seed),
                     'b13_periodichill_pred_test_seed' + str(seed),
                     'b22_periodichill_pred_test_seed' + str(seed),
                     'b23_periodichill_pred_test_seed' + str(seed),
                     'b33_periodichill_pred_test_seed' + str(seed))
fignames_test = ('b11_periodichill_test_seed' + str(seed),
                 'b12_periodichill_test_seed' + str(seed),
                 'b13_periodichill_test_seed' + str(seed),
                 'b22_periodichill_test_seed' + str(seed),
                 'b23_periodichill_test_seed' + str(seed),
                 'b33_periodichill_test_seed' + str(seed))

zlabels = ('$b_{11}$ [-]', '$b_{12}$ [-]', '$b_{13}$ [-]', '$b_{22}$ [-]', '$b_{23}$ [-]', '$b_{33}$ [-]')
zlabels_pred = ('$\hat{b}_{11}$ [-]', '$\hat{b}_{12}$ [-]', '$\hat{b}_{13}$ [-]', '$\hat{b}_{22}$ [-]', '$\hat{b}_{23}$ [-]', '$\hat{b}_{33}$ [-]')
figheight_multiplier = 1.1
# Go through every bij component and plot
for i in range(len(zlabels)):
    bij_predtest_plot = Plot2D_Image(val=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xlabel=xlabel,
                                     ylabel=ylabel, val_label=zlabels_pred[i],
                                     save=savefig, show=show,
                                     figdir=case.resultPaths[time],
                                     figwidth='1/3',
                                     val_lim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test,
                                     figheight_multiplier=figheight_multiplier)
    bij_predtest_plot.initializeFigure()
    bij_predtest_plot.plotFigure()
    bij_predtest_plot.finalizeFigure()

    bij_test_plot = Plot2D_Image(val=y_test_mesh[:, :, i], name=fignames_test[i], xlabel=xlabel,
                                 ylabel=ylabel, val_label=zlabels[i],
                                 save=savefig, show=show,
                                 figdir=case.resultPaths[time],
                                 figwidth='1/3',
                                 val_lim=bijlims,
                                 rotate_img=True,
                                 extent=extent_test,
                                 figheight_multiplier=figheight_multiplier)
    bij_test_plot.initializeFigure()
    bij_test_plot.plotFigure()
    bij_test_plot.finalizeFigure()




