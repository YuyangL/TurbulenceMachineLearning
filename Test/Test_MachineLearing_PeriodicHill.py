import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from PostProcess_FieldData import FieldData
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData
from Preprocess.Feature import getInvariantFeatureSet
from Utility import interpolateGridData
from Preprocess.FeatureExtraction import splitTrainTestDataList
from Preprocess.GridSearchSetup import setupDecisionTreeGridSearchCV, setupRandomForestGridSearch, setupAdaBoostGridSearch,  performEstimatorGridSearch
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import RegressorChain
import matplotlib.pyplot as plt
from PlottingTool import BaseFigure, Plot2D_Image
from scipy import ndimage
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import copy
from joblib import load, dump
from tbnn import NetworkStructure, TBNN


"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both RANS and LES
rans_case_name = 'RANS_Re10595'  # str
les_case_name = 'LES_Breuer/Re_10595'  # str
# LES data name to read
les_data_name = 'Hill_Re_10595_Breuer.csv'  # str
# Absolute directory of this flow case
caseDir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# Eddy-viscosity coefficient to convert omega to epsilon via
# epsilon = Cmu*k*omega
cmu = 0.09  # float
# Whether process field data, invariants, features from scratch,
# or use raw field pickle data and process invariants and features
# or use raw field and invariants pickle data and process features
process_raw_field, process_invariants = False, False  # bool
# The following is only relevant if processInvariants is True
if process_invariants:
    # Absolute cap value for Sij and Rij; tensor basis Tij
    cap_sij_rij, cap_tij = 1e9, 1e9  # float/int


"""
Machine Learning Settings
"""
# Whether to calculate features or directly read from pickle data
calculate_features = False  # bool
# Whether to split train and test data or directly read from pickle data
split_train_test_data = False  # bool
# Feature set number
fs = 'grad(TKE)_grad(p)'  # '1', '12', '123'
# Whether to train the model or directly load it from saved joblib file;
# and whether to save estimator after training
train_model, save_estimator = True, True  # bool
# Name of the ML estimator
estimator_name = 'tbnn'  # "tbdt", "tbrf", "tbab", "tbrc", 'tbnn''
# Whether to presort X for every feature before finding the best split at each node
presort = True  # bool
# Maximum number of features to consider for best split
max_features = 0.8 # (0.8, 1.)  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples at leaf
min_samples_leaf = 4  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples to perform a split
min_samples_split = 16 #(8, 64)  # list/tuple(int / float 0-1) or int / float 0-1
# Max depth of the tree to prevent overfitting
max_depth = 10#(3, 5)  # int
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
# For specific estimators only
if estimator_name in ("TBRF", "tbrf"):
    n_estimators = 8  # int
    oob_score = True  # bool
    median_predict = True  # bool
    # What to do with bij novelties
    bij_novelty = 'excl'
elif estimator_name in ("TBAB", "tbab"):
    n_estimators = 5  # int
    learning_rate = (0.1, 0.2)  # int
    loss = "square"  # 'linear'/'square'/'exponential'
    # What to do with bij novelties
    bij_novelty = 'lim'
elif estimator_name in ('TBRC', 'tbrc'):
    # b11, b22 have rank 10, b12 has rank 9, b33 has rank 5, b13 and b23 are 0 and have rank 10
    order = [0, 3, 1, 5, 4, 2]  # [4, 2, 5, 3, 1, 0]
elif estimator_name in ('TBGB', 'tbgb'):
    n_estimators = 10
    learning_rate = 1
    subsample = 1
    n_iter_no_change = None
    tol = 1e-4
    bij_novelty = 'excl'
    loss = 'ls'
elif estimator_name in ('TBNN', 'tbnn'):
    num_layers = 16  # Number of hidden layers in the TBNN
    num_nodes = 32  # Number of nodes per hidden layer
    max_epochs = 2000  # Max number of epochs during training
    min_epochs = 1000  # Min number of training epochs required
    interval = 100  # Frequency at which convergence is checked
    average_interval = 4  # Number of intervals averaged over for early stopping criteria

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
cv = None # int or None


"""
Plot Settings
"""
# Whether to plot velocity magnitude to verify readFieldData()
plot_u = False  # bool
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
# Number of contour levels in bij's plot
contour_lvl = 50  # int
# Limit for bij plot
bijlims = (-1/2., 2/3.)  # (float, float)
# Save anything when possible
save_fields = True  # bool
# Save figures and show figures
save_fig, show = True, False  # bool; bool
if save_fig:
    # Figure extension and DPI
    ext, dpi = 'png', 1000  # str; int


"""
Process User Inputs, No Need to Change
"""
# Average fields of interest for reading and processing
fields = ('U', 'k', 'p', 'omega',
          'grad_U', 'grad_k', 'grad_p')
# if fs == "grad(TKE)_grad(p)":
#     fields = ('U', 'k', 'p', 'omega',
#               'grad_U', 'grad_k', 'grad_p')
# elif fs == "grad(TKE)":
#     fields = ('k', 'omega',
#               'grad_U', 'grad_k')
# elif fs == "grad(p)":
#     fields = ('U', 'k', 'p', 'omega',
#               'grad_U', 'grad_p')
# else:
#     fields = ('k', 'omega',
#               'grad_U')

# Ensemble name of fields useful for Machine Learning
ml_field_ensemble_name = 'ML_Fields_' + rans_case_name
# Initialize case object
case = FieldData(caseName=rans_case_name, caseDir=caseDir, times=time, fields=fields, save=save_fields)
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
elif estimator_name == 'tbgb':
    estimator_name = 'TBGB'
elif estimator_name == 'tbnn':
    estimator_name = 'TBNN'


"""
Read and Process Raw Field Data
"""
if process_raw_field:
    # Read raw field data specified in fields
    field_data = case.readFieldData()
    # Assign fields to their corresponding variable
    grad_u, u = field_data['grad_U'], field_data['U']
    grad_k, k = field_data['grad_k'], field_data['k']
    grad_p, p = field_data['grad_p'], field_data['p']
    omega = field_data['omega']
    # Get turbulent energy dissipation rate
    epsilon = cmu*k*omega
    # Convert 1D array to 2D so that I can hstack them to 1 array ensemble, n_points x 1
    k, epsilon = k.reshape((-1, 1)), epsilon.reshape((-1, 1))
    # Assemble all useful fields for Machine Learning
    ml_field_ensemble = np.hstack((grad_k, k, epsilon, grad_u, u, grad_p))
    print('\nField variables identified')
    # Read cell center coordinates of the whole domain, nCell x 0
    ccx, ccy, ccz, cc = case.readCellCenterCoordinates()
    # Save all whole fields and cell centers
    case.savePickleData(time, ml_field_ensemble, fileNames=ml_field_ensemble_name)
    case.savePickleData(time, cc, fileNames='cc')
# Else if directly read pickle data
else:
    # Load rotated and/or confined field data useful for Machine Learning
    ml_field_ensemble = case.readPickleData(time, ml_field_ensemble_name)
    grad_k, k = ml_field_ensemble[:, :3], ml_field_ensemble[:, 3]
    epsilon = ml_field_ensemble[:, 4]
    grad_u, u = ml_field_ensemble[:, 5:14], ml_field_ensemble[:, 14:17]
    grad_p = ml_field_ensemble[:, 17:20]
    # Load confined cell centers too
    cc = case.readPickleData(time, 'cc')

if plot_u:
    umag = np.sqrt(u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2)
    ccx_mesh_u, ccy_mesh_u, _, umag_mesh = interpolateGridData(cc[:, 0], cc[:, 1], umag, mesh_target=uniform_mesh_size, fill_val=0)
    umag_mesh = ndimage.rotate(umag_mesh, 90)
    plt.figure('U magnitude', constrained_layout=True)
    plt.imshow(umag_mesh, origin='upper', aspect='equal', cmap='inferno')


"""
Calculate Field Invariants
"""
if process_invariants:
    # Step 1: strain rate and rotation rate tensor Sij and Rij
    sij, rij = case.getStrainAndRotationRateTensorField(grad_u, tke=k, eps=epsilon, cap=cap_sij_rij)
    # Step 2: 10 invariant bases TB
    tb = case.getInvariantBasesField(sij, rij, quadratic_only=False, is_scale=True, zero_trace=False)
    # # Since TB is n_points x 10 x 3 x 3, reshape it to n_points x 10 x 9
    # tb = tb.reshape((tb.shape[0], tb.shape[1], 9))

    # Step 3: anisotropy tensor bij from LES of Breuer csv data
    # 0: x; 1: y; 6: u'u'; 7: v'v'; 8: w'w'; 9: u'v'
    les_data = np.genfromtxt(caseDir + '/' + les_case_name + '/' + les_data_name,
                             delimiter=',', skip_header=1, usecols=(0, 1, 6, 7, 8, 9))
    # Assign each column to corresponding field variable
    # LES cell centers with z being 0
    cc_les = np.zeros((len(les_data), 3))
    cc_les[:, :2] = les_data[:, :2]
    # LES Reynolds stress has 0 u'w' and v'w' 
    uu_prime2_les = np.zeros((len(cc_les), 6))
    # u'u', u'v'
    uu_prime2_les[:, 0], uu_prime2_les[:, 1] = les_data[:, 2], les_data[:, 5]
    # v'v'
    uu_prime2_les[:, 3] = les_data[:, 3]
    # w'w'
    uu_prime2_les[:, 5] = les_data[:, 4]
    # Get LES anisotropy tensor field bij
    bij_les_all = case.getAnisotropyTensorField(uu_prime2_les, use_oldshape=False)
    # Interpolate LES bij to the same grid of RANS
    bij_les = np.empty((len(cc), 6))
    # Go through every bij component and interpolate
    print("\nInterpolating LES bij to RANS grid...")
    for i in range(6):
        bij_les[:, i] = griddata(cc_les[:, :2], bij_les_all[:, i], cc[:, :2], method=interp_method)

    # # Expand bij from 6 components to its full 9 components form
    # bij_les = np.zeros((len(cc), 9))
    # # b11, b12, b13
    # bij_les[:, :3] = bij6_les[:, :3]
    # # b21, b22, b23
    # bij_les[:, 3], bij_les[:, 4:6] = bij6_les[:, 1], bij6_les[:, 3:5]
    # # b31, b32, b33
    # bij_les[:, 6], bij_les[:, 7:] = bij6_les[:, 2], bij6_les[:, 4:]
    # If save_fields, save the processed RANS invariants and LES bij (interpolated to same grid of RANS)
    if save_fields:
        case.savePickleData(time, sij, fileNames = ('Sij'))
        case.savePickleData(time, rij, fileNames = ('Rij'))
        case.savePickleData(time, tb, fileNames = ('Tij'))
        case.savePickleData(time, bij_les, fileNames = ('bij_LES'))

# Else if read RANS invariants and LES bij data from pickle
else:
    invariants = case.readPickleData(time, fileNames=('Sij',
                                                        'Rij',
                                                        'Tij',
                                                        'bij_LES'))
    sij = invariants['Sij']
    rij = invariants['Rij']
    tb = invariants['Tij']
    bij_les = invariants['bij_LES']
    cc = case.readPickleData(time, fileNames='cc')


"""
Calculate Feature Sets
"""
if calculate_features:
    if fs == 'grad(TKE)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k, k=k, eps=epsilon)
    elif fs == 'grad(p)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_p=grad_p, u=u, grad_u=grad_u)
    elif fs == 'grad(TKE)_grad(p)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k=grad_k, grad_p=grad_p, k=k, eps=epsilon, u=u, grad_u=grad_u)

    # If only feature set 1 used for ML input, then do train test data split here
    if save_fields:
        case.savePickleData(time, fs_data, fileNames = ('FS_' + fs))

# Else, directly read feature set data
else:
    fs_data = case.readPickleData(time, fileNames = ('FS_' + fs))


"""
Machine Learning Train, Test Data Preparation
"""
@timer
@njit(parallel=True)
def transposeTensorBasis(tb):
    tb_transpose = np.empty((len(tb), tb.shape[2], tb.shape[1]))
    for i in prange(len(tb)):
        tb_transpose[i] = tb[i].T

    return tb_transpose

# If split train and test data instead of directly read from pickle data
if split_train_test_data:
    # # Transpose Tij so that it's n_samples x n_outputs x n_bases
    # tb = transposeTensorBasis(tb)

    # X is RANS invariant features
    x = fs_data
    # y is LES bij interpolated to RANS grid
    y = bij_les
    # Train-test data split, incl. cell centers and Tij
    list_data_train, list_data_test = splitTrainTestDataList([cc, x, y, tb], test_fraction=test_fraction, seed=seed)
    if save_fields:
        # Extra tuple treatment to list_data_t* that's already a tuple since savePickleData thinks tuple means multiple files
        case.savePickleData(time, (list_data_train,), 'list_data_train_seed' + str(seed))
        case.savePickleData(time, (list_data_test,), 'list_data_test_seed' + str(seed))

# Else if directly read train and test data from pickle data
else:
    list_data_train = case.readPickleData(time, 'list_data_train_seed' + str(seed))
    list_data_test = case.readPickleData(time, 'list_data_test_seed' + str(seed))

cc_train, cc_test = list_data_train[0], list_data_test[0]
ccx_train, ccy_train, ccz_train = cc_train[:, 0], cc_train[:, 1], cc_train[:, 2]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_train, y_train, tb_train = list_data_train[1:4]
x_test, y_test, tb_test = list_data_test[1:4]
# # Contract symmetric tensors to 6 components in case of 9 components
# y_train, y_test = contractSymmetricTensor(y_train), contractSymmetricTensor(y_test)
# tb_train, tb_test = transposeTensorBasis(tb_train), transposeTensorBasis(tb_test)
# tb_train, tb_test = contractSymmetricTensor(tb_train), contractSymmetricTensor(tb_test)
# tb_train, tb_test = transposeTensorBasis(tb_train), transposeTensorBasis(tb_test)


"""
Machine Learning Training
"""
if train_model:
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
        regressor, tuneparams = setupDecisionTreeGridSearchCV(gs_max_features=max_features, gs_min_samples_split=min_samples_split,
                                                               gs_alpha_g_split=alpha_g_split,
                                                               min_samples_leaf=min_samples_leaf,
                                                               alpha_g_fit=alpha_g_fit,
                                                  presort=presort, split_finder=split_finder,
                                                  tb_verbose=tb_verbose, split_verbose=split_verbose, rand_state=seed, gscv_verbose=gs_verbose,
                                                               cv=cv, max_depth=max_depth,
                                                               g_cap=g_cap,
                                                               realize_iter=realize_iter)
    elif estimator_name == 'TBRF':
        regressor, tuneparams = setupRandomForestGridSearch(n_estimators=n_estimators, max_depth=max_depth, gs_min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf, gs_max_features=max_features,
                                          oob_score=oob_score, n_jobs=-1,
                                          rand_state=seed, verbose=gs_verbose,
                                          tb_verbose=tb_verbose, split_finder=split_finder,
                                          split_verbose=split_verbose, alpha_g_fit=alpha_g_fit,
                                          gs_alpha_g_split=alpha_g_split, g_cap=g_cap,
                                          realize_iter=realize_iter,
                                          median_predict=median_predict,
                                          bij_novelty=bij_novelty)
    elif estimator_name == 'TBAB':
        regressor, tuneparams = setupAdaBoostGridSearch(gs_max_features=max_features, gs_max_depth=max_depth, gs_alpha_g_split=alpha_g_split,                             gs_learning_rate=learning_rate,
                                                        cv=cv,
                                                        gscv_verbose=gs_verbose,
                                      n_estimators=n_estimators,
                                      loss=loss,
                                      rand_state=seed,
                                      bij_novelty=bij_novelty)
    elif estimator_name == 'TBRC':
        regressor = RegressorChain(base_estimator=base_estimator,
                                   order=order,
                                   cv=cv,
                                   random_state=seed)
    elif estimator_name == 'TBGB':
        regressor = GradientBoostingRegressor(learning_rate=learning_rate,
                                              n_estimators=n_estimators,
                                              subsample=subsample,
                                              criterion='mse',
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              max_depth=max_depth,
                                              random_state=seed,
                                              max_features=max_features,
                                              verbose=gs_verbose,
                                              presort=presort,
                                              n_iter_no_change=n_iter_no_change,
                                              tol=tol,
                                              init='zero',
                                              alpha_g_split=alpha_g_split,
                                              alpha_g_fit=alpha_g_fit,
                                              realize_iter=realize_iter,
                                              tb_verbose=tb_verbose,
                                              split_verbose=split_verbose,
                                              split_finder=split_finder,
                                              g_cap=g_cap,
                                              bij_novelty=bij_novelty,
                                              loss=loss)
    elif estimator_name == 'TBNN':
        # Swap n_bases and n_outputs axes as TBNN requires Tij of shape (n_samples, n_bases, n_outputs)
        tb_train = np.swapaxes(tb_train, 1, 2)
        tb_test = np.swapaxes(tb_test, 1, 2)
        # Define network structure
        structure = NetworkStructure()
        structure.set_num_layers(num_layers)
        structure.set_num_nodes(num_nodes)
        # Initialize and fit TBNN
        regressor = TBNN(structure)
        regressor.fit(x_train, tb_train, y_train, max_epochs=max_epochs, min_epochs=min_epochs, interval=interval,
                 average_interval=average_interval)
    else:
        regressor = base_estimator

    t0 = t.time()
    if estimator_name in ('TBDT', 'TBAB', 'TBRC', 'TBGB'):
        regressor.fit(x_train, y_train, tb=tb_train)
    else:
        performEstimatorGridSearch(regressor, tuneparams, x_train, y_train, tb_train, x_test, y_test, tb_test, verbose=gs_verbose, refit=True)

    t1 = t.time()
    print('\nFinished DecisionTreeRegressor in {:.4f} s'.format(t1 - t0))
    if save_estimator:
        dump(regressor, case.resultPaths[time] + estimator_name + '.joblib')

else:
    regressor = load(case.resultPaths[time] + estimator_name + '.joblib')

if estimator_name != 'TBNN':
    score_test = regressor.score(x_test, y_test, tb=tb_test)
    score_train = regressor.score(x_train, y_train, tb=tb_train)

# plt.figure(num="DBRT", figsize=(16, 10))
# try:
#     plot = plot_tree(regressor.best_estimator_, fontsize=6, max_depth=5, filled=True, rounded=True, proportion=True, impurity=False)
# except AttributeError:
#     plot = plot_tree(regressor, fontsize=6, max_depth=5, filled=True, rounded=True, proportion=True, impurity=False)

t0 = t.time()
y_pred_test = regressor.predict(x_test, tb=tb_test)
y_pred_train = regressor.predict(x_train, tb=tb_train)
t1 = t.time()
print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))

if estimator_name == 'TBNN':
    score_train = regressor.rmse_score(y_train, y_pred_train)
    score_test = regressor.rmse_score(y_test, y_pred_test)

# print('\n\nLoading regressor... \n')
# reg2 = load(case.resultPaths[time] + estimator_name + '.joblib')
# score_test2 = reg2.score(x_test, y_test, tb=tb_test)
# score_train2 = reg2.score(x_train, y_train, tb=tb_train)
#
# t0 = t.time()
# y_pred_test2 = reg2.predict(x_test, tb=tb_test)
# y_pred_train2 = reg2.predict(x_train, tb=tb_train)
# t1 = t.time()
# print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))


"""
Postprocess Machine Learning Predictions
"""
t0 = t.time()
_, eigval_test, _ = processReynoldsStress(y_test, make_anisotropic=False, realization_iter=0)
_, eigval_train, _ = processReynoldsStress(y_train, make_anisotropic=False, realization_iter=0)
y_pred_test3, eigval_pred_test, _ = processReynoldsStress(y_pred_test, make_anisotropic=False, realization_iter=realize_iter)
y_pred_train3, eigval_pred_train, _ = processReynoldsStress(y_pred_train, make_anisotropic=False, realization_iter=realize_iter)
t1 = t.time()
print('\nFinished processing Reynolds stress in {:.4f} s'.format(t1 - t0))

t0 = t.time()
xy_bary_test, rgb_bary_test = getBarycentricMapData(eigval_test)
xy_bary_train, rgb_bary_train = getBarycentricMapData(eigval_train)
xy_bary_pred_test, rgb_bary_pred_test = getBarycentricMapData(eigval_pred_test)
xy_bary_pred_train, rgb_bary_pred_train = getBarycentricMapData(eigval_pred_train)

# Manually limit RGB values
rgb_bary_pred_test[rgb_bary_pred_test > 1.] = 1.
rgb_bary_pred_train[rgb_bary_pred_train > 1,] = 1.
t1 = t.time()
print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))

t0 = t.time()
ccx_test_mesh, ccy_test_mesh, _, rgb_bary_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
ccx_train_mesh, ccy_train_mesh, _, rgb_bary_train_mesh = interpolateGridData(ccx_train, ccy_train, rgb_bary_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, rgb_bary_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, rgb_bary_pred_train_mesh = interpolateGridData(ccx_train, ccy_train, rgb_bary_pred_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for barycentric map in {:.4f} s'.format(t1 - t0))

t0 = t.time()
_, _, _, y_test_mesh = interpolateGridData(ccx_test, ccy_test, y_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_train_mesh = interpolateGridData(ccx_train, ccy_train, y_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, y_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_pred_train_mesh = interpolateGridData(ccx_train, ccy_train, y_pred_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for bij in {:.4f} s'.format(t1 - t0))


"""
Plotting
"""
rgb_bary_test_mesh = ndimage.rotate(rgb_bary_test_mesh, 90)
rgb_bary_train_mesh = ndimage.rotate(rgb_bary_train_mesh, 90)
rgb_bary_pred_test_mesh = ndimage.rotate(rgb_bary_pred_test_mesh, 90)
rgb_bary_pred_train_mesh = ndimage.rotate(rgb_bary_pred_train_mesh, 90)
xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
geometry = np.genfromtxt(caseDir + '/' + rans_case_name + '/'  + "geometry.csv", delimiter=",")[:, :2]
figname = 'barycentric_periodichill_test_seed' + str(seed)
bary_map = BaseFigure((None,), (None,), name=figname, xlabel=xlabel,
                      ylabel=ylabel, save=save_fig, show=show,
                      figdir=case.resultPaths[time],
                      figheight_multiplier=0.7)
path = Path(geometry)
patch = PathPatch(path, linewidth=0., facecolor=bary_map.gray)
# patch is considered "a single artist" so have to make copy to use more than once
patches = []
for _ in range(28):
    patches.append(copy(patch))

patches = iter(patches)
extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
extent_train = (ccx_train.min(), ccx_train.max(), ccy_train.min(), ccy_train.max())
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.resultPaths[time] + figname + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_pred_test_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.resultPaths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_train_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_train_mesh, origin='upper', aspect='equal', extent=extent_train)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.resultPaths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_pred_train_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_train_mesh, origin='upper', aspect='equal', extent=extent_train)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
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
fignames_predtrain = ('b11_periodichill_pred_train_seed' + str(seed),
                     'b12_periodichill_pred_train_seed' + str(seed),
                     'b13_periodichill_pred_train_seed' + str(seed),
                     'b22_periodichill_pred_train_seed' + str(seed),
                     'b23_periodichill_pred_train_seed' + str(seed),
                     'b33_periodichill_pred_train_seed' + str(seed))
fignames_train = ('b11_periodichill_train_seed' + str(seed),
                 'b12_periodichill_train_seed' + str(seed),
                 'b13_periodichill_train_seed' + str(seed),
                 'b22_periodichill_train_seed' + str(seed),
                 'b23_periodichill_train_seed' + str(seed),
                 'b33_periodichill_train_seed' + str(seed))
zlabels = ('$b_{11}$ [-]', '$b_{12}$ [-]', '$b_{13}$ [-]', '$b_{22}$ [-]', '$b_{23}$ [-]', '$b_{33}$ [-]')
zlabels_pred = ('$\hat{b}_{11}$ [-]', '$\hat{b}_{12}$ [-]', '$\hat{b}_{13}$ [-]', '$\hat{b}_{22}$ [-]', '$\hat{b}_{23}$ [-]', '$\hat{b}_{33}$ [-]')
figheight_multiplier = 1.1
# Go through every bij component and plot
for i in range(len(zlabels)):
    bij_predtest_plot = Plot2D_Image(val=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xlabel=xlabel,
                                     ylabel=ylabel, val_label=zlabels_pred[i],
                                     save=save_fig, show=show,
                                     figdir=case.resultPaths[time],
                                     figwidth='1/3',
                                     val_lim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test,
                                     figheight_multiplier=figheight_multiplier)
    bij_predtest_plot.initializeFigure()
    bij_predtest_plot.plotFigure()
    bij_predtest_plot.axes.add_patch(next(patches))
    bij_predtest_plot.finalizeFigure()

    bij_test_plot = Plot2D_Image(val=y_test_mesh[:, :, i], name=fignames_test[i], xlabel=xlabel,
                                     ylabel=ylabel, val_label=zlabels[i],
                                     save=save_fig, show=show,
                                     figdir=case.resultPaths[time],
                                     figwidth='1/3',
                                     val_lim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test,
                                    figheight_multiplier=figheight_multiplier)
    bij_test_plot.initializeFigure()
    bij_test_plot.plotFigure()
    bij_test_plot.axes.add_patch(next(patches))
    bij_test_plot.finalizeFigure()

    bij_predtrain_plot = Plot2D_Image(val=y_pred_train_mesh[:, :, i], name=fignames_predtrain[i], xlabel=xlabel,
                                     ylabel=ylabel, val_label=zlabels_pred[i],
                                     save=save_fig, show=show,
                                     figdir=case.resultPaths[time],
                                     figwidth='1/3',
                                     val_lim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test,
                                      figheight_multiplier=figheight_multiplier)
    bij_predtrain_plot.initializeFigure()
    bij_predtrain_plot.plotFigure()
    bij_predtrain_plot.axes.add_patch(next(patches))
    bij_predtrain_plot.finalizeFigure()

    bij_train_plot = Plot2D_Image(val=y_train_mesh[:, :, i], name=fignames_train[i], xlabel=xlabel,
                                     ylabel=ylabel, val_label=zlabels[i],
                                     save=save_fig, show=show,
                                     figdir=case.resultPaths[time],
                                     figwidth='1/3',
                                     val_lim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test,
                                    figheight_multiplier=figheight_multiplier)
    bij_train_plot.initializeFigure()
    bij_train_plot.plotFigure()
    bij_train_plot.axes.add_patch(next(patches))
    bij_train_plot.finalizeFigure()


    # bij_predtest_plot = Plot2D(ccx_test_mesh, ccy_test_mesh, z2D=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xLabel=xlabel,
    #                       yLabel=ylabel, zLabel=zlabels[i],
    #                       save=save_fig, show=show,
    #                       figDir=case.resultPaths[time],
    #                            figWidth='1/3')
    # bij_predtest_plot.initializeFigure()
    # bij_predtest_plot.plotFigure(contourLvl=contour_lvl, zlims=bijlims)
    # bij_predtest_plot.axes.add_patch(next(patches))
    # bij_predtest_plot.finalizeFigure()
    #
    # bij_test_plot = Plot2D(ccx_test_mesh, ccy_test_mesh, z2D=y_test_mesh[:, :, i], name=fignames_test[i], xLabel=xlabel,
    #                            yLabel=ylabel, zLabel=zlabels[i],
    #                            save=save_fig, show=show,
    #                            figDir=case.resultPaths[time],
    #                            figWidth='1/3')
    # bij_test_plot.initializeFigure()
    # bij_test_plot.plotFigure(contourLvl=contour_lvl, zlims=bijlims)
    # bij_test_plot.axes.add_patch(next(patches))
    # bij_test_plot.finalizeFigure()
    #
    # bij_predtrain_plot = Plot2D(ccx_train_mesh, ccy_train_mesh, z2D=y_pred_train_mesh[:, :, i], name=fignames_predtrain[i],
    #                            xLabel=xlabel,
    #                            yLabel=ylabel, zLabel=zlabels[i],
    #                            save=save_fig, show=show,
    #                            figDir=case.resultPaths[time],
    #                            figWidth='1/3')
    # bij_predtrain_plot.initializeFigure()
    # bij_predtrain_plot.plotFigure(contourLvl=contour_lvl, zlims=bijlims)
    # bij_predtrain_plot.axes.add_patch(next(patches))
    # bij_predtrain_plot.finalizeFigure()
    #
    # bij_train_plot = Plot2D(ccx_train_mesh, ccy_train_mesh, z2D=y_train_mesh[:, :, i], name=fignames_train[i], xLabel=xlabel,
    #                        yLabel=ylabel, zLabel=zlabels[i],
    #                        save=save_fig, show=show,
    #                        figDir=case.resultPaths[time],
    #                        figWidth='1/3')
    # bij_train_plot.initializeFigure()
    # bij_train_plot.plotFigure(contourLvl=contour_lvl, zlims=bijlims)
    # bij_train_plot.axes.add_patch(next(patches))
    # bij_train_plot.finalizeFigure()






