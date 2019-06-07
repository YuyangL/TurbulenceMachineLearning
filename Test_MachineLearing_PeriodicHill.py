import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from PostProcess_FieldData import FieldData
from PostProcess_Tensor import processReynoldsStress, getBarycentricMapData
from FeatureExtraction import getFeatureSet1, splitTrainTestDataList, inputOutlierDetection
from BarycentricMap import getBarycentricMapCoordinates
import time as t
# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

from scipy.interpolate import griddata
from numba import njit, prange
from Utilities import sampleData, timer
from math import floor, ceil
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

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
# Average fields of interest for reading and processing
fields = ('U', 'k', 'p', 'omega',
          'grad_U', 'grad_k', 'grad_p')  # list/tuple(str)
# Eddy-viscosity coefficient to convert omega to epsilon via
# epsilon = Cmu*k*omega
cmu = 0.09  # float
# Whether process field data, invariants, features from scratch,
# or use raw field pickle data and process invariants and features
# or use raw field and invariants pickle data and process features
process_raw_field, process_invariants = False, False  # bool
# The following is only relevant if processInvariants is True
if process_invariants:
    # Absolute cap value for Sij and Rij; tensor basis coefficients g; tensor basis Tij
    cap_sij_rij, cap_g, cap_tij = 1e9, 1e9, 1e9  # float/int

# Save anything when possible
save_fields = True  # bool


"""
Machine Learning Settings
"""
# Whether to calculate features or directly read from pickle data
calculate_features = False  # bool
# Feature set number
fs = '1'  # '1', '12', '123'
# Whether save the trained ML model
saveEstimator = True
estimatorName = 'rf'
scalerScheme = 'robust'
# Whether to presort X for every feature before finding the best split at each node
presort = True  # bool
# Minimum number of samples at leaf
# This is used to not only prevent overfitting
min_samples_leaf = 2  # int / float 0-1
# Max depth of the tree to prevent overfitting
max_depth = 15  # int
# Best split finding scheme to speed up the process of locating the best split in samples of each node
split_finder = "brent"  # "brute", "brent", "1000"
seed = 123
# For debugging, verbose on bij reconstruction from Tij and g;
# and/or verbose on "brent"/"brute"/"1000" split finding scheme
tb_verbose, split_verbose = False, False  # bool; bool
# Fraction of data for testing
test_fraction = 0.2  # float [0-1]


"""
Process User Inputs, No Need to Change
"""
# Ensemble name of fields useful for Machine Learning
ml_field_ensemble_name = 'ML_Fields_' + rans_case_name
# Initialize case object
case = FieldData(caseName=rans_case_name, caseDir=caseDir, times=time, fields=fields, save=save_fields)


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
    # Convert 1D array to 2D so that I can hstack them to 1 array ensemble, nCell x 1
    k, epsilon = k.reshape((-1, 1)), epsilon.reshape((-1, 1))
    # Assemble all useful fields for Machine Learning
    ml_field_ensemble = np.hstack((grad_k, k, epsilon, grad_u, grad_p))
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
    grad_u = ml_field_ensemble[:, 5:14]
    grad_p = ml_field_ensemble[:, 14:17]
    # Load confined cell centers too
    cc = case.readPickleData(time, 'cc')


"""
Calculate Field Invariants
"""
if process_invariants:
    # Step 1: strain rate and rotation rate tensor Sij and Rij
    sij, rij = case.getStrainAndRotationRateTensorField(grad_u, tke = k, eps = epsilon, cap = cap_sij_rij)
    # Step 2: 10 invariant bases TB
    tb = case.getInvariantBasesField(sij, rij, quadratic_only = False, is_scale = True)
    # Since TB is nPoint x 10 x 3 x 3, reshape it to nPoint x 10 x 9
    tb = tb.reshape((tb.shape[0], tb.shape[1], 9))
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
    bij6_les_all = case.getAnisotropyTensorField(uu_prime2_les)
    # Interpolate LES bij to the same grid of RANS
    bij6_les = np.empty((len(cc), 6))
    # Go through every bij component and interpolate
    print("\nInterpolating LES bij to RANS grid...")
    for i in range(6):
        bij6_les[:, i] = griddata(cc_les[:, :2], bij6_les_all[:, i], cc[:, :2])

    # Expand bij from 6 components to its full 9 components form
    bij_les = np.zeros((len(cc), 9))
    # b11, b12, b13
    bij_les[:, :3] = bij6_les[:, :3]
    # b21, b22, b23
    bij_les[:, 3], bij_les[:, 4:6] = bij6_les[:, 1], bij6_les[:, 3:5]
    # b31, b32, b33
    bij_les[:, 6], bij_les[:, 7:] = bij6_les[:, 2], bij6_les[:, 4:]
    # If save_fields, save the processed RANS invariants and LES bij (interpolated to same grid of RANS)
    if save_fields:
        case.savePickleData(time, sij, fileNames = ('Sij'))
        case.savePickleData(time, rij, fileNames = ('Rij'))
        case.savePickleData(time, tb, fileNames = ('Tij'))
        case.savePickleData(time, bij_les, fileNames = ('bij_LES'))

# Else if read RANS invariants and LES bij data from pickle
else:
    invariants = case.readPickleData(time, fileNames = ('Sij',
                                                        'Rij',
                                                        'Tij',
                                                        'bij_LES'))
    sij = invariants['Sij']
    rij = invariants['Rij']
    tb = invariants['Tij']
    bij_les = invariants['bij_LES']


"""
Calculate Feature Sets
"""
if calculate_features:
    # Feature set 1
    fs1 = getFeatureSet1(sij, rij)
    # If only feature set 1 used for ML input, then do train test data split here
    if fs == '1':
        # FIXME: TB couldn't be split here
        # xTrain, xTest, yTrain, yTest, _ = splitTrainTestData(x = fs1, y = bij, randState = seed, scalerScheme = None)
        if save_fields:
            case.savePickleData(time, fs1, fileNames = ('FS' + fs))

# Else, directly read feature set data
else:
    if fs == '1':
        fs1 = case.readPickleData(time, fileNames = ('FS' + fs))


"""
Machine Learning Train, Test Data Preparation
"""
@timer
@njit(parallel=True)
def _transposeTensorBasis(tb):
    tb_transpose = np.empty((len(tb), tb.shape[2], tb.shape[1]))
    for i in prange(len(tb)):
        tb_transpose[i] = tb[i].T

    return tb_transpose

# Transpose Tij so that it's n_samples x 9 components x 10 bases
tb = _transposeTensorBasis(tb)
# X is RANS invariant features
if fs == '1':
    x = fs1
else:
    x = fs1

# y is LES bij interpolated to RANS grid
y = bij_les
# Train-test data split, incl. Tij
list_data_train, list_data_test = splitTrainTestDataList([x, y, tb], test_fraction=test_fraction, seed=seed)
x_train, y_train, tb_train = list_data_train[0], list_data_train[1], list_data_train[2]
x_test, y_test, tb_test = list_data_test[0], list_data_test[1], list_data_test[2]
# x_train, y_train, tb_train = sampleData((x, y, tb), floor((1 - fraction_test)*len(x)), replace=False)
# x_test, y_test, tb_test = sampleData((x, y, tb), ceil(fraction_test*len(x)), replace=False)


"""
Machine Learning Training
"""
t0 = t.time()
regressor = DecisionTreeRegressor(presort=presort, max_depth=max_depth, tb_verbose=tb_verbose, min_samples_leaf=min_samples_leaf, split_finder=split_finder, split_verbose=split_verbose)
regressor.fit(x_train, y_train, tb=tb_train)
t1 = t.time()
print('\nFinished DecisionTreeRegressor in {:.4f} s'.format(t1 - t0))
score_test = regressor.score(x_test, y_test)
score_train = regressor.score(x_train, y_train)

plt.figure(num="DBRT", figsize=(16, 10))
plot = plot_tree(regressor, fontsize=6, max_depth=5, filled=True, rounded=True, proportion=True, impurity=False)


"""
Visualize Machine Learning Predictions
"""
t0 = t.time()
y_pred = regressor.predict(x_test)
t1 = t.time()
print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))

t0 = t.time()
_, eigval, eigvec = processReynoldsStress(y_pred, make_anisotropic=False, realization_iter=0)
t1 = t.time()
print('\nFinished processing Reynolds stress in {:.4f} s'.format(t1 - t0))

t0 = t.time()
xy_bary, rgb_bary = getBarycentricMapData(eigval)
t1 = t.time()
print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))


