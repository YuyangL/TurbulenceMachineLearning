import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from FieldData import FieldData

# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

import numpy as np
from math import ceil, floor
from Utilities import sampleData, timer
from numba import jit, njit, prange
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

"""
User Inputs
"""
# Name of the flow case
caseName = 'ALM_N_H_OneTurb'  # str
# Absolute directory of this flow case
caseDir = '/media/yluan'  # str
# Which time to extract input and output for ML
time = 'last'  # str/float/int or 'last'
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


"""
Machine Learning Settings
"""
fs = '1'
# Whether save the trained ML model
saveEstimator = True
estimatorName = 'rf'
scalerScheme = 'robust'
# Minimum number of samples at leaf
# This is used to not only prevent overfitting
min_samples_leaf = 10  # int / float 0-1
# Best split finding scheme to speed up the process of locating the best split in samples of each node
split_finder = "brent"  # "brute", "brent", "1000"
seed = 12345
verbose = 1
# Fraction of data for testing
fraction_test = 0.2  # float 0-1
# Whether to use sampling to reduce size of input data
sampling = True  # bool
# Only if sampling is True
if sampling:
    # Number of samples and whether to use with replacement, i.e. same sample can be re-picked
    sample_train, replace = 10000, False  # int; bool


"""
Process User Inputs
"""
if estimatorName in ('rf', 'RF', 'RandForest'):
    estimatorName = 'RandomForest'

# Automatically select time if time is set to 'latest'
if time == 'last':
    if caseName == 'ALM_N_H_ParTurb':
        time = '22000.0918025'
    elif caseName == 'ALM_N_H_OneTurb':
        time = '24995.0788025'

else:
    time = str(time)

if boxAutoDim == 'first':
    confinedFieldNameSub += '1'
elif boxAutoDim == 'second':
    confinedFieldNameSub += '2'
elif boxAutoDim == 'third':
    confinedFieldNameSub += '3'

if not confineBox:
    confinedFieldNameSub = ''

xTrainName = 'FS' + fs + '_Train_' + confinedFieldNameSub
xTestName = 'FS' + fs + '_Test_' + confinedFieldNameSub
yTrainName = 'yTrain_' + confinedFieldNameSub
yTestName = 'yTest_' + confinedFieldNameSub
x_name = 'FS' + fs + '_' + confinedFieldNameSub
tb_name = 'TB_' + confinedFieldNameSub
bij_name = 'bij_' +  confinedFieldNameSub
# Initialize this flow case to access necessary methods and attributes
case = FieldData(caseName = caseName, caseDir = caseDir, times = time, fields = 'dummy', save = saveEstimator)
# Load train and test data
# FIXME: TB wasn't split the same way train and tests were done
# xTrain = case.readPickleData(time, fileNames = xTrainName)
# xTest = case.readPickleData(time, fileNames = xTestName)
# yTrain = case.readPickleData(time, fileNames = yTrainName)
# yTest = case.readPickleData(time, fileNames = yTestName)
x = case.readPickleData(time, fileNames=x_name)
# Both Tij and bij here are C-contiguous
# TODO: make Tij n_samples x 9 components x 10 bases
# Tij transpose is A and is n_samples x 10 bases x 9 components
tb_transpose = case.readPickleData(time, fileNames=tb_name)

@timer
@njit(parallel=True)
def _transposeTensorBasis(tb):
    tb_transpose = np.empty((len(tb), tb.shape[2], tb.shape[1]))
    for i in prange(len(tb)):
        tb_transpose[i] = tb[i].T

    return tb_transpose

tb = _transposeTensorBasis(tb_transpose)
# bij is b and is n_samples x 9 components
bij = case.readPickleData(time, fileNames=bij_name)

@timer
@jit(parallel=True, fastmath=True)
def getTTandTb(tb, bij):
    tb_tb, tb_bij = np.empty((tb.shape[0], 10, 10)), np.empty((tb.shape[0], 10))
    for i in prange(tb.shape[0]):
        tb_tb[i] = np.dot(tb[i], tb[i].T)
        tb_bij[i] = np.dot(tb[i], bij[i])

    return tb_tb, tb_bij

# # Calculate Tij^T*Tij and Tij^T*bij
# tb_tb, tb_bij = getTTandTb(tb, bij)
# Sample training and testing data
# TODO: modify train_test_split() to take not only X, y but also tb
if sampling:
    x_train, y_train, tb_train = sampleData((x, bij, tb), sample_train, replace=replace)
    x_test, y_test, tb_test = sampleData((x, bij, tb), ceil(sample_train*fraction_test), replace=replace)
else:
    x_train, y_train, tb_train = sampleData((x, bij, tb), floor((1 - fraction_test)*len(x)), replace=False)
    x_test, y_test, tb_test = sampleData((x, bij, tb), ceil(fraction_test*len(x)), replace=False)




# from sklearn.multioutput import RegressorChain
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# estimator = RegressorChain(RandomForestRegressor(n_estimators = 100, verbose = 1, n_jobs = -1), order = np.arange(0, 10, 1))
# estimator.fit(xTrain, yTrain)
# score = estimator.score(xTest, yTest)
# # yPred = estimator.predict(xTest)



regressor = DecisionTreeRegressor(presort=True, max_depth=15, tb_verbose=False, min_samples_leaf=min_samples_leaf, split_finder=split_finder)
regressor.fit(x_train, y_train, tb=tb_train)
score_test = regressor.score(x_test, y_test)
score_train = regressor.score(x_train, y_train)

plt.figure(num="DBRT", figsize=(16, 10))
plot = plot_tree(regressor, fontsize=6, max_depth=5, filled=True, rounded=True, proportion=True, impurity=False)


# """
# Setup Estimator Grid Search (Cross-Validation)
# """
# if estimatorName in ('RandomForest', 'ExtremeRandomForest'):
#     # [ADJUSTABLE] Number of decision trees, the higher the better, so not really useful to grid search, 1000 gives a fast computation
#     nEstimator = 100
#     # [ADJUSTABLE] Max number of features per split, in percentage, recommended as 1/3 of max features
#     maxFeatPercents = np.arange(3, 5, 1)
#     # [ADJUSTABLE] Min number of sample before a split is allowed, 2 is recommended
#     minSampleSplits = [2, 4, 8]
#     # [ADJUSTABLE] Criterion for split
#     criterion = ['mse']
#     # Either Random Forest or Extremely Random Forest
#     if estimatorName == 'RandomForest':
#         splitScheme = 'RF'
#     else:
#         splitScheme = 'extremeRF'
#
#     estimatorGS, tuneParams = setupRandomForestGridSearch(nEstimator, maxFeatPercents, minSampleSplits, criterion,
#                                                                            scalerScheme = scalerScheme,
#                                                                            customOobScore = None,
#                                                                            randState = seed, verbose = verbose,
#                                                                            splitScheme = splitScheme)
#
#
# """
# Train Estimator Using Grid Search (Cross-Validation)
# """
# if estimatorName in ('RandomForest', 'ExtremeRandomForest'):
#     # Train the estimator with GS (no CV)
#     bestGrid = processEstimatorGridSearch(estimatorGS, tuneParams, xTrain, yTrain, xTest, yTest,
#                                                    verbose = verbose)
#     yPred = estimatorGS.predict(xTest)
#     score = estimatorGS.score(xTest, yTest)




