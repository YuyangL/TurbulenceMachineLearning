import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from FieldData import FieldData
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor
from Preprocess.Feature import getInvariantFeatureSet, getSupplementaryInvariantFeatures
from Utility import interpolateGridData
from Preprocess.FeatureExtraction import splitTrainTestDataList
from Preprocess.GridSearchSetup import setupDecisionTreeGridSearchCV, setupRandomForestGridSearch, setupAdaBoostGridSearchCV, setupGradientBoostGridSearchCV, performEstimatorGridSearch, performEstimatorGridSearchCV
import time as t
# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

from scipy.interpolate import griddata
from numba import njit, prange
from Utility import timer
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
from sklearn.feature_selection import VarianceThreshold
from scipy.interpolate import interp1d
from tempfile import mkdtemp
from numba import njit, prange

"""
User Inputs, Anything Can Be Changed Here
"""
# Reynolds number of the flow case
re = 5600
# Name of the flow case in both RANS and LES
rans_case_name = 'RANS_Re' + str(re)  # str
les_case_name = 'LES_Breuer/Re_' + str(re)  # str
# LES data name to read
les_data_name = 'Hill_Re_' + str(re) + '_Breuer.csv'  # str
# Absolute directory of this flow case
casedir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# Eddy-viscosity coefficient to convert omega to epsilon via
# epsilon = Cmu*k*omega
cmu = 0.09  # float
# Kinematic viscosity for Re = 10595
nu = 9.438414346389807e-05  # float
# Whether process field data, invariants, features from scratch,
# or use raw field pickle data and process invariants and features
# or use raw field and invariants pickle data and process features
process_raw_field, process_invariants = True, True  # bool
# The following is only relevant if processInvariants is True
if process_invariants:
    # Absolute cap value for Sij and Rij; tensor basis Tij
    cap_sij_rij, cap_tij = 1e9, 1e9  # float/int

cap_lam = 1e9


"""
Machine Learning Settings
"""
# Whether to calculate features or directly read from pickle data
calculate_features = False  # bool
# Whether to split train and test data or directly read from pickle data
split_train_test_data = False  # bool
# Feature set number
fs = 'grad(TKE)_grad(p)+'  # '1', '12', '123'
scaler = None  # 'maxabs', 'minmax', None
# Whether remove low variance features
var_threshold = -1  # float
rf_selector_n_estimators = 0
rf_selector_threshold = '0.25*median'  # 'median', 'mean', None
# Whether to train the model or directly load it from saved joblib file;
# and whether to save estimator after training
train_model, save_estimator = False, False  # bool
# Name of the ML estimator
estimator_name = 'tbdt'  # "tbdt", "tbrf", "tbab", "tbrc", 'tbnn''
# Whether to presort X for every feature before finding the best split at each node
presort = True  # bool
# Maximum number of features to consider for best split
max_features = 1.#(2/3., 1.)  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples at leaf
min_samples_leaf = 4  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples to perform a split
min_samples_split = 16 #(8, 64)  # list/tuple(int / float 0-1) or int / float 0-1
# Max depth of the tree to prevent overfitting
max_depth = None#(3, 5)  # int
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
if estimator_name in ('TBDT', 'tbdt'):
    tbkey = 'tree__tb'
if estimator_name in ("TBRF", "tbrf"):
    n_estimators = 8  # int
    oob_score = True  # bool
    median_predict = True  # bool
    # What to do with bij novelties
    bij_novelty = None  # 'excl', 'reset', None
    return_all_predictions = False
    tbkey = 'rf__tb'
elif estimator_name in ("TBAB", "tbab"):
    n_estimators = 16  # int
    learning_rate = 0.1#(0.1, 0.2)  # int
    loss = "square"  # 'linear'/'square'/'exponential'
    # What to do with bij novelties
    bij_novelty = None  # 'reset', None
    tbkey = 'ab__tb'
elif estimator_name in ('TBRC', 'tbrc'):
    # b11, b22 have rank 10, b12 has rank 9, b33 has rank 5, b13 and b23 are 0 and have rank 10
    order = [0, 3, 1, 5, 4, 2]  # [4, 2, 5, 3, 1, 0]
elif estimator_name in ('TBGB', 'tbgb'):
    n_estimators = 16
    learning_rate = 0.1
    subsample = 0.8
    n_iter_no_change = 3
    tol = 1e-8
    bij_novelty = None  # 'reset', None
    loss = 'huber'
    tbkey = 'gb__tb'
elif estimator_name in ('TBNN', 'tbnn'):
    num_layers = 8  # Number of hidden layers in the TBNN
    num_nodes = 30  # Number of nodes per hidden layer
    max_epochs = 1000  # Max number of epochs during training
    min_epochs = 100  # Min number of training epochs required
    interval = 10  # Frequency at which convergence is checked
    average_interval = 5  # Number of intervals averaged over for early stopping criteria
    learning_rate_decay = 0.99
    init_learning_rate = 1e-2

# Seed value for reproducibility
seed = 123
# For debugging, verbose on bij reconstruction from Tij and g;
# and/or verbose on "brent"/"brute"/"1000"/"auto" split finding scheme
tb_verbose, split_verbose = False, False  # bool; bool
# Fraction of data for testing
test_fraction = 1.  # float [0-1]
# Whether verbose on GSCV. 0 means off, larger means more verbose
gs_verbose = 2  # int
# Number of n-fold cross validation
cv = None  # int or None


"""
Plot Settings
"""
# Whether to plot velocity magnitude to verify readFieldData()
plot_u = False  # bool
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
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
case = FieldData(casename=rans_case_name, casedir=casedir, times=time, fields=fields, save=save_fields)
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
    case.savePickleData(time, ml_field_ensemble, filenames=ml_field_ensemble_name)
    case.savePickleData(time, cc, filenames='CC')
# Else if directly read pickle data
else:
    # Load rotated and/or confined field data useful for Machine Learning
    ml_field_ensemble = case.readPickleData(time, ml_field_ensemble_name)
    grad_k, k = ml_field_ensemble[:, :3], ml_field_ensemble[:, 3]
    epsilon = ml_field_ensemble[:, 4]
    grad_u, u = ml_field_ensemble[:, 5:14], ml_field_ensemble[:, 14:17]
    grad_p = ml_field_ensemble[:, 17:20]
    # Load confined cell centers too
    cc = case.readPickleData(time, 'CC')

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
    les_data = np.genfromtxt(casedir + '/' + les_case_name + '/' + les_data_name,
                             delimiter=',', skip_header=1, usecols=(0, 1, 6, 7, 8, 9))
    # Mean velocity data from the same csv file, order is <U>, <V>, <W>
    mean_les_data = np.genfromtxt(casedir + '/' + les_case_name + '/' + les_data_name,
                                  delimiter=',', skip_header=1, usecols=(2, 3, 4))

    # Assign each column to corresponding field variable
    # LES cell centers with z being 0
    cc_les = np.zeros((len(les_data), 3))
    cc_les[:, :2] = les_data[:, :2]
    for i in range(1, len(cc_les)):
        if cc_les[i, 0] < cc_les[i - 1, 0]:
            nx = i
            break

    ccx_les_mesh = cc_les[:, 0].reshape((-1, nx))
    ccy_les_mesh = cc_les[:, 1].reshape((-1, nx))
    umean_les_mesh = mean_les_data[:, 0].reshape((-1, nx))
    vmean_les_mesh = mean_les_data[:, 1].reshape((-1, nx))
    wmean_les_mesh = mean_les_data[:, 2].reshape((-1, nx))
    umean_les_mesh = np.dstack((umean_les_mesh, vmean_les_mesh, wmean_les_mesh))
    ccy_sortidx = np.argsort(ccy_les_mesh[:, 0])
    ccx_les_mesh = ccx_les_mesh[ccy_sortidx]
    ccy_les_mesh = ccy_les_mesh[ccy_sortidx]
    umean_les_mesh = umean_les_mesh[ccy_sortidx]
    dudx_mesh = np.ones_like(umean_les_mesh)*9999
    # Go through each U component and get x derivative
    for i in range(3):
        # Upwind diff for first cell
        dudx_mesh[:, 0, i] = (umean_les_mesh[:, 1, i] - umean_les_mesh[:, 0, i])/(ccx_les_mesh[:, 1] - ccx_les_mesh[:, 0])
        # Central diff for inner cells
        dudx_mesh[:, 1:-1, i] = (umean_les_mesh[:, 2:, i] - umean_les_mesh[:, :-2, i])/(ccx_les_mesh[:, 2:] - ccx_les_mesh[:, :-2])
        # Downwind diff for last cell
        dudx_mesh[:, -1, i] = (umean_les_mesh[:, -1, i] - umean_les_mesh[:, -2, i])/(ccx_les_mesh[:, -1] - ccx_les_mesh[:, -2])

    wronggrad_x = np.where(dudx_mesh == 9999)

    dudy_mesh = np.ones_like(umean_les_mesh)*9999
    # Go through each U component and get y derivative
    for i in range(3):
        dudy_mesh[0, :, i] = (umean_les_mesh[1, :, i] - umean_les_mesh[0, :, i])/(
                    ccy_les_mesh[1, :] - ccy_les_mesh[0, :])
        dudy_mesh[1:-1, :, i] = (umean_les_mesh[2:, :, i] - umean_les_mesh[:-2, :, i])/(
                    ccy_les_mesh[2:, :] - ccy_les_mesh[:-2, :])
        dudy_mesh[-1, :, i] = (umean_les_mesh[-1, :, i] - umean_les_mesh[-2, :, i])/(
                    ccy_les_mesh[-1, :] - ccy_les_mesh[-2, :])

    wronggrad_y = np.where(dudy_mesh == 9999)

    # Go through every dx component and interpolate to RANS grid (not mesh)
    print("\nInterpolating LES velocity gradients dudx, dudy to RANS grid...")
    dudx = np.empty((len(cc), 3))
    dudy = np.empty((len(cc), 3))
    # Since 2D flow case, d*/dz is always 0
    dudz = np.zeros((len(cc), 3))
    cc_les_sorted = np.array([ccx_les_mesh.ravel(), ccy_les_mesh.ravel()]).T
    for i in range(3):
        dudx[:, i] = griddata(cc_les_sorted, dudx_mesh[..., i].ravel(), cc[:, :2], method=interp_method)
        dudy[:, i] = griddata(cc_les_sorted, dudy_mesh[..., i].ravel(), cc[:, :2], method=interp_method)

    @njit(parallel=True)
    def regroupGradU(dudx, dudy, dudz):
        grad_u_les = np.empty((len(cc), 9))
        for i in prange(len(dudx)):
            # Enforcing zero trace since div(U) = 0
            # dw/dz + du/dx + dv/dy = 0.
            dudz[i, 2] = 0. - dudx[i, 0] - dudy[i, 1]
            # du/dx, du/dy, du/dz
            grad_u_les[i, 0], grad_u_les[i, 1], grad_u_les[i, 2] = dudx[i, 0], dudy[i, 0], dudz[i, 0]
            # dv/dx, dv/dy, dv/dz
            grad_u_les[i, 3], grad_u_les[i, 4], grad_u_les[i, 5] = dudx[i, 1], dudy[i, 1], dudz[i, 1]
            # dw/dx, dw/dy, dw/dz
            grad_u_les[i, 6], grad_u_les[i, 7], grad_u_les[i, 8] = dudx[i, 2], dudy[i, 2], dudz[i, 2]

        return grad_u_les

    # Finally, LES grad(U) of shape (n_samples, 9), where n_samples is RANS samples
    grad_u_les = regroupGradU(dudx, dudy, dudz)

    # LES Reynolds stress has 0 u'w' and v'w'
    uu_prime2_les_all = np.zeros((len(cc_les), 6))
    # u'u', u'v'
    uu_prime2_les_all[:, 0], uu_prime2_les_all[:, 1] = les_data[:, 2], les_data[:, 5]
    # v'v'
    uu_prime2_les_all[:, 3] = les_data[:, 3]
    # w'w'
    uu_prime2_les_all[:, 5] = les_data[:, 4]
    # Get LES anisotropy tensor field bij
    bij_les_all = case.getAnisotropyTensorField(uu_prime2_les_all, use_oldshape=False)
    # Sort according to y from low to high after mesh grid treatment
    uu_prime2_les_all_mesh = uu_prime2_les_all.reshape((-1, nx, 6))[ccy_sortidx]
    # Calculate turbulent shear stress gradient -div(ui'uj')
    @njit(parallel=True, fastmath=True)
    def calcSymmTensorDivergence2D(tensor_mesh, ccx_mesh, ccy_mesh):
        drdx = np.empty_like(tensor_mesh)
        drdy = np.empty_like(tensor_mesh)
        for i in prange(tensor_mesh.shape[2]):
            # First dRij/dx
            # Upwind diff for first cell
            drdx[:, 0, i] = (tensor_mesh[:, 1, i] - tensor_mesh[:, 0, i])/(
                        ccx_mesh[:, 1] - ccx_mesh[:, 0])
            # Central diff for inner cells
            drdx[:, 1:-1, i] = (tensor_mesh[:, 2:, i] - tensor_mesh[:, :-2, i])/(
                        ccx_mesh[:, 2:] - ccx_mesh[:, :-2])
            # Downwind diff for last cell
            drdx[:, -1, i] = (tensor_mesh[:, -1, i] - tensor_mesh[:, -2, i])/(
                        ccx_mesh[:, -1] - ccx_mesh[:, -2])

            # Repeat for dRij/dy
            drdy[0, :, i] = (tensor_mesh[1, :, i] - tensor_mesh[0, :, i])/(
                    ccy_mesh[1, :] - ccy_mesh[0, :])
            drdy[1:-1, :, i] = (tensor_mesh[2:, :, i] - tensor_mesh[:-2, :, i])/(
                    ccy_mesh[2:, :] - ccy_mesh[:-2, :])
            drdy[-1, :, i] = (tensor_mesh[-1, :, i] - tensor_mesh[-2, :, i])/(
                    ccy_mesh[-1, :] - ccy_mesh[-2, :])

        # Now perform the divergence
        tensor_div_mesh = np.empty((tensor_mesh.shape[0], tensor_mesh.shape[1], 3))
        # 1st row: dR11/dx + dR12/dy + dR13/dz
        # Assuming no component vary with z
        tensor_div_mesh[..., 0] = drdx[..., 0] + drdy[..., 1] + 0.
        # 2nd row: dR12/dx + dR22/dy + dR23/dz
        tensor_div_mesh[..., 1] = drdx[..., 1] + drdy[..., 3] + 0.
        # 3rd row: dR13/dx + dR23/dy + dR33/dz
        tensor_div_mesh[..., 2] = drdx[..., 2] + drdy[..., 4] + 0.
        return tensor_div_mesh

    div_uuprime2_les_mesh = -calcSymmTensorDivergence2D(uu_prime2_les_all_mesh, ccx_les_mesh, ccy_les_mesh)
    # Go through all 3 components of the divergence and interpolate it to RANS grid
    div_uuprime2_les = np.empty((len(cc), 3))
    for i in range(3):
        div_uuprime2_les[:, i] = griddata(cc_les_sorted, div_uuprime2_les_mesh[..., i].ravel(), cc[:, :2], method=interp_method)

    # Interpolate LES bij to the same grid of RANS
    uu_prime2_les = np.empty((len(cc), 6))
    bij_les = np.empty((len(cc), 6))
    # Go through every bij component and interpolate
    print("\nInterpolating LES ui'uj' and bij to RANS grid...")
    for i in range(6):
        uu_prime2_les[:, i] = griddata(cc_les[:, :2], uu_prime2_les_all[:, i], cc[:, :2], method=interp_method)
        bij_les[:, i] = griddata(cc_les[:, :2], bij_les_all[:, i], cc[:, :2], method=interp_method)

    # LES TKE in RANS grid
    k_les = .5*(uu_prime2_les[:, 0] + uu_prime2_les[:, 3] + uu_prime2_les[:, 5])

    # uup2 = np.empty((len(cc), 6))
    # for i in range(6):
    #     uup2[:, i] = 2/3.*k_les + 2.*k_les*bij_les[:, i] if i in (0, 3, 5) else 2.*k_les*bij_les[:, i]
    #
    # uup2 = expandSymmetricTensor(uup2).reshape((-1, 3, 3))
    # uup3 = expandSymmetricTensor(uu_prime2_les).reshape((-1, 3, 3))
    # diff = uup3 - uup2



    # Calculate LES TKE production G (density normalized) based on RANS grid
    # Actual G is ui'uj':u_i,j
    # G with Boussinesq apprx is 2nut*Sij:Sij
    # G with predicted bij is -2k*bij:u_i,j since 2nut*Sij ~ -2k*bij
    g_tke_les = np.empty(len(cc))
    for i in range(len(cc)):
        # A 3 x 3 Reynolds stress at current sample
        uu_prime2_les_i = expandSymmetricTensor(uu_prime2_les[i]).reshape((3, 3))
        grad_u_les_i = grad_u_les[i].reshape((3, 3))
        # Double inner dot for Rij:u_i,j
        g_tke_les[i] = np.tensordot(-uu_prime2_les_i, grad_u_les_i)

        # bij_les_i = expandSymmetricTensor(bij_les[i]).reshape((3, 3))
        # g2[i] = -2.*k_les[i]*np.tensordot(bij_les_i, grad_u_les_i)


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
        case.savePickleData(time, sij, filenames=('Sij'))
        case.savePickleData(time, rij, filenames=('Rij'))
        case.savePickleData(time, tb, filenames=('Tij'))
        case.savePickleData(time, bij_les, filenames=('bij_LES'))
        case.savePickleData(time, uu_prime2_les, filenames=('uuPrime2_LES'))
        # Save TKE production G related fields too
        case.savePickleData(time, g_tke_les, filenames=('G_LES'))
        case.savePickleData(time, grad_u_les, filenames=('grad(U)_LES'))
        case.savePickleData(time, k_les, filenames=('k_LES'))
        # Also save turbulent shear stress gradient
        case.savePickleData(time, div_uuprime2_les, 'div(uuPrime2)_LES')


# Else if read RANS invariants and LES bij data from pickle
else:
    invariants = case.readPickleData(time, filenames=('Sij',
                                                        'Rij',
                                                        'Tij',
                                                        'bij_LES'))
    sij = invariants['Sij']
    rij = invariants['Rij']
    tb = invariants['Tij']
    bij_les = invariants['bij_LES']
    cc = case.readPickleData(time, filenames='CC')


"""
Calculate Feature Sets
"""
if calculate_features:
    if fs == 'grad(TKE)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k, k=k, eps=epsilon)
    elif fs == 'grad(p)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_p=grad_p, u=u, grad_u=grad_u)
    elif 'grad(TKE)_grad(p)' in fs:
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k=grad_k, grad_p=grad_p, k=k, eps=epsilon, u=u, grad_u=grad_u)
        if '+' in fs:
            nu *= np.ones_like(k)
            geometry = np.genfromtxt(casedir + '/' + rans_case_name + '/' + "geometry.csv", delimiter=",")[:1000, :2]
            # fperiodic = interp1d(geometry[:, 0], geometry[:, 1], kind='cubic')
            # d0 = cc[:, 1] - fperiodic(cc[:, 0])
            d1 = 3.036 - cc[:, 1]
            # d = np.minimum(d0, d1)
            d = d1
            fs_data2, labels2 = getSupplementaryInvariantFeatures(k, d, epsilon, nu, sij)
            fs_data = np.hstack((fs_data, fs_data2))

    # If only feature set 1 used for ML input, then do train test data split here
    if save_fields:
        case.savePickleData(time, fs_data, filenames=('FS_' + fs))

# Else, directly read feature set data
else:
    fs_data = case.readPickleData(time, filenames=('FS_' + fs))


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
    if estimator_name == 'TBNN':
        sij = expandSymmetricTensor(sij).reshape((-1, 3, 3))
        rij = rij.reshape((-1, 3, 3))
        x, x_mu, x_std = case.calcScalarBasis(sij, rij, is_train=True, cap=cap_lam, is_scale=True)
    else:
        # X is RANS invariant features
        x = fs_data

    # y is LES bij interpolated to RANS grid
    y = bij_les
    # Train-test data split, incl. cell centers and Tij
    list_data_train, list_data_test = splitTrainTestDataList([cc, x, y, tb], test_fraction=test_fraction, seed=seed)
    if save_fields:
        if estimator_name == 'TBNN':
            # Extra tuple treatment to list_data_t* that's already a tuple since savePickleData thinks tuple means multiple files
            case.savePickleData(time, (list_data_train,), 'list_data_train_seed' + str(seed) + '_TBNN')
            case.savePickleData(time, (list_data_test,), 'list_data_test_seed' + str(seed) + '_TBNN')
        else:
            # Extra tuple treatment to list_data_t* that's already a tuple since savePickleData thinks tuple means multiple files
            case.savePickleData(time, (list_data_train,), 'list_data_train_seed' + str(seed))
            case.savePickleData(time, (list_data_test,), 'list_data_test_seed' + str(seed))

# Else if directly read train and test data from pickle data
else:
    if estimator_name == 'TBNN':
        list_data_train = case.readPickleData(time, 'list_data_train_seed' + str(seed) + '_TBNN')
        list_data_test = case.readPickleData(time, 'list_data_test_seed' + str(seed) + '_TBNN')
    else:
        list_data_train = case.readPickleData(time, 'list_data_train_seed' + str(seed))
        list_data_test = case.readPickleData(time, 'list_data_test_seed' + str(seed))

cc_train, cc_test = list_data_train[0], list_data_test[0]
ccx_train, ccy_train, ccz_train = cc_train[:, 0], cc_train[:, 1], cc_train[:, 2]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_train, y_train, tb_train = list_data_train[1:4]
x_test, y_test, tb_test = list_data_test[1:4]
if estimator_name == 'TBNN':
    # Swap n_bases and n_outputs axes as TBNN requires Tij of shape (n_samples, n_bases, n_outputs)
    tb_train = np.swapaxes(tb_train, 1, 2)
    tb_test = np.swapaxes(tb_test, 1, 2)


# """
# Remove Low Variance Features
# """
# if feat_sel:
#     selector = VarianceThreshold(threshold=var_threshold)
#     x_train = selector.fit_transform(x_train)
#     x_test = selector.transform(x_test)


"""
Machine Learning Training
"""
if train_model:
    # cachedir = mkdtemp()
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
        regressor_gs, regressor, tuneparams, tbkey = setupDecisionTreeGridSearchCV(gs_max_features=max_features, gs_min_samples_split=min_samples_split,
                                                               gs_alpha_g_split=alpha_g_split,
                                                               min_samples_leaf=min_samples_leaf,
                                                               alpha_g_fit=alpha_g_fit,
                                                  presort=presort, split_finder=split_finder,
                                                  tb_verbose=tb_verbose, split_verbose=split_verbose, rand_state=seed, gs_verbose=gs_verbose,
                                                               cv=cv, max_depth=max_depth,
                                                               g_cap=g_cap,
                                                               realize_iter=realize_iter,
                                                               var_threshold=var_threshold,
                                                                               scaler=scaler,
                                                                                rf_selector_n_estimators=rf_selector_n_estimators,
                                                                                rf_selector_threshold=rf_selector_threshold,
                                                                               pipeline_cachedir=None)
    elif estimator_name == 'TBRF':
        regressor_gs, regressor, tuneparams, tbkey = setupRandomForestGridSearch(n_estimators=n_estimators, max_depth=max_depth, gs_min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf, gs_max_features=max_features,
                                          oob_score=oob_score, n_jobs=-1,
                                          rand_state=seed, gs_verbose=gs_verbose,
                                          tb_verbose=tb_verbose, split_finder=split_finder,
                                          split_verbose=split_verbose, alpha_g_fit=alpha_g_fit,
                                          gs_alpha_g_split=alpha_g_split, g_cap=g_cap,
                                          realize_iter=realize_iter,
                                          median_predict=median_predict,
                                          bij_novelty=None,
                                                                   var_threshold=var_threshold,
                                                                   scaler=scaler,
                                                                   rf_selector_n_estimators=rf_selector_n_estimators,
                                                                   rf_selector_threshold=rf_selector_threshold)
    elif estimator_name == 'TBAB':
        regressor_gs, regressor, tuneparams, tbkey = setupAdaBoostGridSearchCV(gs_max_features=max_features, gs_max_depth=max_depth, gs_alpha_g_split=alpha_g_split,                             gs_learning_rate=learning_rate,
                                                        cv=cv,
                                                        gs_verbose=gs_verbose,
                                      n_estimators=n_estimators,
                                      loss=loss,
                                      rand_state=seed,
                                      bij_novelty=bij_novelty,
                                                                        scaler=scaler,
                                                                        var_threshold=var_threshold,
                                                                               rf_selector_n_estimators=rf_selector_n_estimators,
                                                                               rf_selector_threshold=rf_selector_threshold,
                                                                               split_finder=split_finder, split_verbose=split_verbose)
    elif estimator_name == 'TBRC':
        regressor = RegressorChain(base_estimator=base_estimator,
                                   order=order,
                                   cv=cv,
                                   random_state=seed)
    elif estimator_name == 'TBGB':
        regressor_gs, regressor, tuneparams, tbkey = setupGradientBoostGridSearchCV(gs_learning_rate=learning_rate,
                                              n_estimators=n_estimators,
                                              subsample=subsample,
                                              criterion='mse',
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              gs_max_depth=max_depth,
                                              random_state=seed,
                                              gs_max_features=max_features,
                                              gs_verbose=gs_verbose,
                                              presort=presort,
                                              n_iter_no_change=n_iter_no_change,
                                              tol=tol,
                                              init='zero',
                                              gs_alpha_g_split=alpha_g_split,
                                              alpha_g_fit=alpha_g_fit,
                                              realize_iter=realize_iter,
                                              tb_verbose=tb_verbose,
                                              split_verbose=split_verbose,
                                              split_finder=split_finder,
                                              g_cap=g_cap,
                                              bij_novelty=bij_novelty,
                                              loss=loss,
                                                                                    scaler=scaler,
                                                                                    var_threshold=var_threshold,
                                                                                    rf_selector_n_estimators=rf_selector_n_estimators,
                                                                                    rf_selector_threshold=rf_selector_threshold)
    elif estimator_name == 'TBNN':
        # Define network structure
        structure = NetworkStructure()
        structure.set_num_layers(num_layers)
        structure.set_num_nodes(num_nodes)
        # Initialize and fit TBNN
        regressor = TBNN(structure, print_freq=interval,learning_rate_decay=learning_rate_decay)
        # Optimal according to Ling et al. (2016)
        regressor.set_min_learning_rate(2.5e-7)
        regressor.set_train_fraction(1. - test_fraction)
        regressor.fit(x_train, tb_train, y_train, max_epochs=max_epochs, min_epochs=min_epochs, interval=interval,
                 average_interval=average_interval, init_learning_rate=init_learning_rate)
    else:
        regressor = base_estimator
        tbkey = 'tb'

    t0 = t.time()
    fit_param, test_param = {}, {}
    fit_param[tbkey] = tb_train
    # test_param[tbkey] = tb_test
    if estimator_name in ('TBDT', 'TBAB', 'TBRC', 'TBGB'):
        if cv is None:
            regressor.fit(x_train, y_train, **fit_param)
        else:
            _, _ = performEstimatorGridSearchCV(regressor_gs, regressor, x_train, y_train,
                                                       tb_kw=tbkey, tb_gs=tb_train, savedir=case.result_paths[time], final_name=estimator_name)

    else:
        _, _ = performEstimatorGridSearch(regressor_gs, regressor,
                                   tuneparams, x_train, y_train,
                                   tbkey, tb_train, refit=True,
                                          savedir=case.result_paths[time], final_name=estimator_name)

    t1 = t.time()
    print('\nFinished {0} in {1:.4f} s'.format(estimator_name, t1 - t0))
    if save_estimator:
        dump(regressor, case.result_paths[time] + estimator_name + '.joblib')

else:
    regressor = load(case.result_paths[time] + estimator_name + '.joblib')

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
# reg2 = load(case.result_paths[time] + estimator_name + '.joblib')
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
rgb_bary_pred_train[rgb_bary_pred_train > 1.] = 1.
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
Plot Barycentric Map
"""
rgb_bary_test_mesh = ndimage.rotate(rgb_bary_test_mesh, 90)
rgb_bary_train_mesh = ndimage.rotate(rgb_bary_train_mesh, 90)
rgb_bary_pred_test_mesh = ndimage.rotate(rgb_bary_pred_test_mesh, 90)
rgb_bary_pred_train_mesh = ndimage.rotate(rgb_bary_pred_train_mesh, 90)
xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
geometry = np.genfromtxt(casedir + '/' + rans_case_name + '/'  + "geometry.csv", delimiter=",")[:, :2]
figname = 'barycentric_periodichill_test_seed' + str(seed)
bary_map = BaseFigure((None,), (None,), name=figname, xlabel=xlabel,
                      ylabel=ylabel, save=save_fig, show=show,
                      figdir=case.result_paths[time],
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
    plt.savefig(case.result_paths[time] + figname + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_pred_test_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.result_paths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_train_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_train_mesh, origin='upper', aspect='equal', extent=extent_train)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.result_paths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_pred_train_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_train_mesh, origin='upper', aspect='equal', extent=extent_train)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.result_paths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()


"""
Plot bij
"""
# fignames_predtest = ('b11_periodichill_pred_test_seed' + str(seed),
#             'b12_periodichill_pred_test_seed' + str(seed),
#             'b13_periodichill_pred_test_seed' + str(seed),
#             'b22_periodichill_pred_test_seed' + str(seed),
#             'b23_periodichill_pred_test_seed' + str(seed),
#             'b33_periodichill_pred_test_seed' + str(seed))
# fignames_test = ('b11_periodichill_test_seed' + str(seed),
#                      'b12_periodichill_test_seed' + str(seed),
#                      'b13_periodichill_test_seed' + str(seed),
#                      'b22_periodichill_test_seed' + str(seed),
#                      'b23_periodichill_test_seed' + str(seed),
#                      'b33_periodichill_test_seed' + str(seed))
# fignames_predtrain = ('b11_periodichill_pred_train_seed' + str(seed),
#                      'b12_periodichill_pred_train_seed' + str(seed),
#                      'b13_periodichill_pred_train_seed' + str(seed),
#                      'b22_periodichill_pred_train_seed' + str(seed),
#                      'b23_periodichill_pred_train_seed' + str(seed),
#                      'b33_periodichill_pred_train_seed' + str(seed))
# fignames_train = ('b11_periodichill_train_seed' + str(seed),
#                  'b12_periodichill_train_seed' + str(seed),
#                  'b13_periodichill_train_seed' + str(seed),
#                  'b22_periodichill_train_seed' + str(seed),
#                  'b23_periodichill_train_seed' + str(seed),
#                  'b33_periodichill_train_seed' + str(seed))
# zlabels = ('$b_{11}$ [-]', '$b_{12}$ [-]', '$b_{13}$ [-]', '$b_{22}$ [-]', '$b_{23}$ [-]', '$b_{33}$ [-]')
# zlabels_pred = ('$\hat{b}_{11}$ [-]', '$\hat{b}_{12}$ [-]', '$\hat{b}_{13}$ [-]', '$\hat{b}_{22}$ [-]', '$\hat{b}_{23}$ [-]', '$\hat{b}_{33}$ [-]')
# figheight_multiplier = 1.1
# # Go through every bij component and plot
# for i in range(len(zlabels)):
#     bij_predtest_plot = Plot2D_Image(val=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xlabel=xlabel,
#                                      ylabel=ylabel, val_label=zlabels_pred[i],
#                                      save=save_fig, show=show,
#                                      figdir=case.result_paths[time],
#                                      figwidth='1/3',
#                                      val_lim=bijlims,
#                                      rotate_img=True,
#                                      extent=extent_test,
#                                      figheight_multiplier=figheight_multiplier)
#     bij_predtest_plot.initializeFigure()
#     bij_predtest_plot.plotFigure()
#     bij_predtest_plot.axes.add_patch(next(patches))
#     bij_predtest_plot.finalizeFigure()
# 
#     bij_test_plot = Plot2D_Image(val=y_test_mesh[:, :, i], name=fignames_test[i], xlabel=xlabel,
#                                      ylabel=ylabel, val_label=zlabels[i],
#                                      save=save_fig, show=show,
#                                      figdir=case.result_paths[time],
#                                      figwidth='1/3',
#                                      val_lim=bijlims,
#                                      rotate_img=True,
#                                      extent=extent_test,
#                                     figheight_multiplier=figheight_multiplier)
#     bij_test_plot.initializeFigure()
#     bij_test_plot.plotFigure()
#     bij_test_plot.axes.add_patch(next(patches))
#     bij_test_plot.finalizeFigure()
# 
#     bij_predtrain_plot = Plot2D_Image(val=y_pred_train_mesh[:, :, i], name=fignames_predtrain[i], xlabel=xlabel,
#                                      ylabel=ylabel, val_label=zlabels_pred[i],
#                                      save=save_fig, show=show,
#                                      figdir=case.result_paths[time],
#                                      figwidth='1/3',
#                                      val_lim=bijlims,
#                                      rotate_img=True,
#                                      extent=extent_test,
#                                       figheight_multiplier=figheight_multiplier)
#     bij_predtrain_plot.initializeFigure()
#     bij_predtrain_plot.plotFigure()
#     bij_predtrain_plot.axes.add_patch(next(patches))
#     bij_predtrain_plot.finalizeFigure()
# 
#     bij_train_plot = Plot2D_Image(val=y_train_mesh[:, :, i], name=fignames_train[i], xlabel=xlabel,
#                                      ylabel=ylabel, val_label=zlabels[i],
#                                      save=save_fig, show=show,
#                                      figdir=case.result_paths[time],
#                                      figwidth='1/3',
#                                      val_lim=bijlims,
#                                      rotate_img=True,
#                                      extent=extent_test,
#                                     figheight_multiplier=figheight_multiplier)
#     bij_train_plot.initializeFigure()
#     bij_train_plot.plotFigure()
#     bij_train_plot.axes.add_patch(next(patches))
#     bij_train_plot.finalizeFigure()





    # bij_predtest_plot = Plot2D(ccx_test_mesh, ccy_test_mesh, z2D=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xLabel=xlabel,
    #                       yLabel=ylabel, zLabel=zlabels[i],
    #                       save=save_fig, show=show,
    #                       figDir=case.result_paths[time],
    #                            figWidth='1/3')
    # bij_predtest_plot.initializeFigure()
    # bij_predtest_plot.plotFigure(contourLvl=contour_lvl, zlims=bijlims)
    # bij_predtest_plot.axes.add_patch(next(patches))
    # bij_predtest_plot.finalizeFigure()
    #
    # bij_test_plot = Plot2D(ccx_test_mesh, ccy_test_mesh, z2D=y_test_mesh[:, :, i], name=fignames_test[i], xLabel=xlabel,
    #                            yLabel=ylabel, zLabel=zlabels[i],
    #                            save=save_fig, show=show,
    #                            figDir=case.result_paths[time],
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
    #                            figDir=case.result_paths[time],
    #                            figWidth='1/3')
    # bij_predtrain_plot.initializeFigure()
    # bij_predtrain_plot.plotFigure(contourLvl=contour_lvl, zlims=bijlims)
    # bij_predtrain_plot.axes.add_patch(next(patches))
    # bij_predtrain_plot.finalizeFigure()
    #
    # bij_train_plot = Plot2D(ccx_train_mesh, ccy_train_mesh, z2D=y_train_mesh[:, :, i], name=fignames_train[i], xLabel=xlabel,
    #                        yLabel=ylabel, zLabel=zlabels[i],
    #                        save=save_fig, show=show,
    #                        figDir=case.result_paths[time],
    #                        figWidth='1/3')
    # bij_train_plot.initializeFigure()
    # bij_train_plot.plotFigure(contourLvl=contour_lvl, zlims=bijlims)
    # bij_train_plot.axes.add_patch(next(patches))
    # bij_train_plot.finalizeFigure()






