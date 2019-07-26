import pickle
import os
from Preprocess.GridSearchSetup import setupDecisionTreeGridSearchCV, setupRandomForestGridSearch, setupAdaBoostGridSearch,  performEstimatorGridSearch
from joblib import dump
import time as t


"""
User Inputs
"""
casedir = 'ALM_N_H_OneTurb'
gsdata_name = 'list_data_GS_Confined'
traindata_name = 'list_data_train_Confined'
confined_zone = '1'
estimators = ('TBDT', 'TBRF', 'TBAB')  # str, list/tuple(str)


"""
Machine Learning Settings
"""
# Whether to presort X for every feature before finding the best split at each node
presort = True  # bool
# Maximum number of features to consider for best split
max_features = (1/3., 2/3., 1.)  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples at leaf
min_samples_leaf = 2  # int / float 0-1
# [DEPRECATED] L2 regularization fraction to penalize large optimal 10 g found through LS fit of min_g(bij - Tij*g)
alpha_g_fit = 0.  # float
# L2 regularization coefficient to penalize large optimal 10 g during best split finder
alpha_g_split = (0., 0.001, 0.00001)  # list/tuple(float) or float
# Best split finding scheme to speed up the process of locating the best split in samples of each node
split_finder = "brent"  # "brute", "brent", "1000", "auto"
# Cap of optimal g magnitude after LS fit
g_cap = None  # int or float or None
# Realizability iterator to shift predicted bij to be realizable
realize_iter = 0  # int
# Seed value for reproducibility
seed = 123
# For debugging, verbose on bij reconstruction from Tij and g;
# and/or verbose on "brent"/"brute"/"1000"/"auto" split finding scheme
tb_verbose, split_verbose = False, False  # bool; bool
# Whether verbose on GSCV. 0 means off, larger means more verbose
gscv_verbose = 2  # int
# Number of n-fold cross validation
cv = 5  # int or None
n_jobs = 20  # int
# For TBDT only
tree_kwargs = dict(gs_min_samples_split=(0.0005, 0.001, 0.002),
                   max_depth=None)
# For TBRF only
rf_kwargs = dict(gs_min_samples_split=(0.0005, 0.001, 0.002),
                 max_depth=None,
                 n_estimators=20,
                 oob_score=True,
                 median_predict=True,
                 bij_novelty='excl')
# For TBAB only
ab_kwargs = dict(gs_max_depth=(5, 10, 15),
                 gs_learning_rate=(0.1, 0.2, 0.4),
                 min_samples_split=0.002,
                 n_estimators=100,
                 loss='square',
                 bij_novelty='lim')


"""
Process User Inputs
"""
casedir = casedir + '/'
resdir = casedir + 'Result/'
os.makedirs(resdir, exist_ok=True)
gsdata_name = gsdata_name + str(confined_zone)
traindata_name = traindata_name + str(confined_zone)
if isinstance(estimators, str): estimators = (estimators,)
# General ML kwargs
general_kwargs = dict(presort=presort,
                      gs_max_features=max_features,
                      gs_alpha_g_split=alpha_g_split,
                      min_samples_leaf=min_samples_leaf,
                      alpha_g_fit=alpha_g_fit,
                      split_finder=split_finder,
                      g_cap=g_cap,
                      realize_iter=realize_iter,
                      rand_state=seed,
                      tb_verbose=tb_verbose,
                      split_verbose=split_verbose,
                      gscv_verbose=gscv_verbose,
                      verbose=gscv_verbose,
                      cv=cv,
                      n_jobs=n_jobs,
                      refit=False,
                      return_train_score=True)


"""
Load Train Data
"""
t0 = t.time()
list_data_gs = pickle.load(open(casedir + gsdata_name + '.p', 'rb'), encoding='ASCII')
cc_gs = list_data_gs[0]
x_gs = list_data_gs[1]
y_gs = list_data_gs[2]
tb_gs = list_data_gs[3]
del list_data_gs

list_data_train = pickle.load(open(casedir + traindata_name + '.p', 'rb'), encoding='ASCII')
cc_train = list_data_train[0]
x_train = list_data_train[1]
y_train = list_data_train[2]
tb_train = list_data_train[3]
del list_data_train
t1 = t.time()
print('\nFinished loading GS and train data in {:.4f} s'.format(t1 - t0))


"""
Machine Learning
"""
for estimator in estimators:
    # regressor_gs is the GSCV object while regressor is the actual estimator
    if estimator == 'TBDT':
        tree_kwargs_tot = {**tree_kwargs, **general_kwargs}
        regressor_gs, regressor, tuneparams = setupDecisionTreeGridSearchCV(**tree_kwargs_tot)
    elif estimator == 'TBRF':
        # For TBRF, there's no GSCV
        rf_kwargs_tot = {**rf_kwargs, **general_kwargs}
        regressor, tuneparams = setupRandomForestGridSearch(**rf_kwargs_tot)
    elif estimator == 'TBAB':
        ab_kwargs_tot = {**ab_kwargs, **general_kwargs}
        regressor_gs, regressor, tuneparams = setupAdaBoostGridSearch(**ab_kwargs_tot)

    # GS(CV)
    print(tuneparams)
    t0 = t.time()
    if estimator in ('TBDT', 'TBAB'):
        # This is a GSCV object
        regressor_gs.fit(x_gs, y_gs, tb=tb_gs)
        # Save the GSCV for further inspection
        dump(regressor_gs, resdir + 'GSCV_' + estimator + '_Confined' + str(confined_zone) + '.joblib')
        # This is the actual estimator, setting the best found hyper-parameters to it
        regressor.set_params(**regressor_gs.best_params_)
    else:
        # For TBRF, regressor_gs is regressor and is internaly updated to best hyper-parameters during GS
        performEstimatorGridSearch(regressor, tuneparams, x_gs, y_gs, tb_gs,
                                   verbose=gscv_verbose, refit=False)

    t1 = t.time()
    print('\nFinished {} GS(CV) in {:.4f} min'.format(estimator, (t1 - t0)/60.))

    # Actual training, fitting using the best found hyper-parameters
    t0 = t.time()
    regressor.fit(x_train, y_train, tb=tb_train)
    # Save estimator
    dump(regressor, resdir + estimator + '_Confined' + str(confined_zone) + '.joblib')
    t1 = t.time()
    print('\nFinished {} training in {:.4f} min'.format(estimator, (t1 - t0)/60.))

