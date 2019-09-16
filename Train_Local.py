import pickle
import os
from Preprocess.GridSearchSetup import setupDecisionTreeGridSearchCV, setupRandomForestGridSearch, setupAdaBoostGridSearchCV, setupGradientBoostGridSearchCV, performEstimatorGridSearch, performEstimatorGridSearchCV
from joblib import dump, load
import time as t

"""
User Inputs
"""
unittest = False
if unittest:
    casedir = '/tmp/ALM_N_H_OneTurb/'  #'/media/yluan/ALM_N_H_OneTurb/Fields/Result/24995.0438025/'
else:
    casedir =  '/media/yluan/RANS/N_H_OneTurb_Simple_ABL/Fields/Result/5000' #'/tmp/ALM_N_H_OneTurb/'  #'/media/yluan/ALM_N_H_OneTurb/Fields/Result/24995.0438025/'
gsdata_name = 'list_data_GS_Confined'
traindata_name = 'list_data_train_Confined'
confined_zone = '2'
estimators = 'TBDT' #('TBDT', 'TBRF', 'TBAB', 'TBGB')  # str, list/tuple(str)
# Whether skip the GSCV step, only for TBDT and TBAB
skip_gscv = 'auto'  # 'auto', bool


"""
Machine Learning Settings
"""
# Whether to presort X for every feature before finding the best split at each node
presort = True  # bool
# Maximum number of features to consider for best split
max_features = (1/3., 2/3., 1.) if not unittest else 1.  # list/tuple(int / float 0-1) or int / float 0-1
# Minimum number of samples at leaf
min_samples_leaf = 2  # int / float 0-1
# [DEPRECATED] L2 regularization fraction to penalize large optimal 10 g found through LS fit of min_g(bij - Tij*g)
alpha_g_fit = 0.  # float
# L2 regularization coefficient to penalize large optimal 10 g during best split finder
alpha_g_split = (0., 0.001, 0.00001) if not unittest else 0.  # list/tuple(float) or float
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
tb_verbose, split_verbose = False, True  # bool; bool
# Whether verbose on GSCV. 0 means off, larger means more verbose
gs_verbose = 2  # int
# Number of n-fold cross validation
cv = 4  # int or None
n_jobs = -1  # int
# Feature selection related settings
rf_selector = True
rf_selector_n_estimators = 3200 if not unittest else 800
rf_selector_threshold = '0.1*median'
# For TBDT only
tree_kwargs = dict(gs_min_samples_split=(0.0005, 0.001, 0.002) if not unittest else 0.002,
                   max_depth=None if not unittest else None)
# For TBRF only
rf_kwargs = dict(gs_min_samples_split=(0.0005, 0.001, 0.002) if not unittest else 0.002,
                 max_depth=None if not unittest else None,
                 n_estimators=32 if not unittest else 8,
                 oob_score=True,
                 median_predict=True,
                 # [DEPRECATED]
                 bij_novelty=None)
# For TBAB only
ab_kwargs = dict(gs_max_depth=(5, 10) if not unittest else 5,
                 gs_learning_rate=(0.1, 0.2, 0.4) if not unittest else 0.1,
                 min_samples_split=0.002,
                 n_estimators=64 if not unittest else 16,
                 loss='square',
                 # [DEPRECATED]
                 bij_novelty=None)
# For TBGB only
gb_kwargs = dict(gs_max_depth=(5, 10) if not unittest else 5,
                 gs_learning_rate=(0.1, 0.2, 0.4) if not unittest else 0.1,
                 min_samples_split=0.002,
                 n_estimators=64 if not unittest else 16,
                 loss='ls',
                 subsample=0.8,
                 # FIXME: n_iter_no_change causes segmentation error
                 n_iter_no_change=None,
                 tol=1e-8,
                 # [DREPRECATED]
                 bij_novelty=None)


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
                      rand_state=None,
                      tb_verbose=tb_verbose,
                      split_verbose=split_verbose,
                      gs_verbose=gs_verbose,
                      cv=cv,
                      n_jobs=n_jobs,
                      return_train_score=False,
                      rf_selector_threshold=rf_selector_threshold,
                      rf_selector_n_estimators=rf_selector_n_estimators)


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

if unittest:
    x_train = x_gs
    y_train = y_gs
    tb_train = tb_gs
else:
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
        regressor_gs, regressor, tuneparams, tbkw = setupDecisionTreeGridSearchCV(**tree_kwargs_tot)
    elif estimator == 'TBRF':
        # For TBRF, there's no GSCV
        rf_kwargs_tot = {**rf_kwargs, **general_kwargs}
        regressor_gs, regressor, tuneparams, tbkw = setupRandomForestGridSearch(**rf_kwargs_tot)
    elif estimator == 'TBAB':
        ab_kwargs_tot = {**ab_kwargs, **general_kwargs}
        regressor_gs, regressor, tuneparams, tbkw = setupAdaBoostGridSearchCV(**ab_kwargs_tot)
    elif estimator == 'TBGB':
        gb_kwargs_tot = {**gb_kwargs, **general_kwargs}
        regressor_gs, regressor, tuneparams, tbkw = setupGradientBoostGridSearchCV(**gb_kwargs_tot)

    # Load saved GSCV estimator if asked and don't do GSCV again
    if skip_gscv in ('auto', True):
        try:
            if estimator != 'TBRF':
                regressor_gs = load(resdir + 'GSCV_' + estimator + '_Confined' + str(confined_zone) + '.joblib')
            else:
                regressor_gs = load(resdir + 'GS_' + estimator + '_Confined' + str(confined_zone) + '.joblib')

            do_gscv = False
            print('\nSaved GSCV estimator found, skipping GSCV and directly to full training...')
        except:
            do_gscv = True

    else:
        do_gscv = True

    # Assign proper tb kw to fit_params
    fit_param_gs, fit_param = {}, {}
    fit_param_gs[tbkw] = tb_gs
    # fit_param[tbkw] = tb_train if not unittest else None


    """
    GS(CV) and Final Training
    """
    print(tuneparams)
    t0 = t.time()
    if estimator in ('TBDT', 'TBAB', 'TBGB'):
        regressor, best_params = performEstimatorGridSearchCV(regressor_gs, regressor, x_gs, y_gs,
                                                 tb_kw=tbkw, tb_gs=tb_gs,
                                                 x_train=x_train, y_train=y_train, tb_train=tb_train,
                                                              gs=do_gscv,
                                                 savedir=resdir, gscv_name='GSCV_' + estimator + '_Confined' + str(confined_zone),
                                                 final_name=estimator + '_Confined' + str(confined_zone),
                                                              refit=True)
        # if do_gscv:
        #     # This is a GSCV object
        #     regressor_gs.fit(x_gs, y_gs, **fit_param_gs)
        #     # Save the GSCV for further inspection
        #     dump(regressor_gs, resdir + 'GSCV_' + estimator + '_Confined' + str(confined_zone) + '.joblib')
        #
        # # This is the actual estimator, setting the best found hyper-parameters to it
        # regressor.set_params(**regressor_gs.best_params_)
    else:
        # For TBRF, regressor_gs is equivalent to regressor and is internally updated to best hyper-parameters during GS
        regressor, best_params = performEstimatorGridSearch(regressor_gs, regressor, tuneparams,
                                                            x_gs, y_gs, tb_kw=tbkw, tb_gs=tb_gs,
                                                            x_train=x_train, y_train=y_train, tb_train=tb_train,
                                                            gs=do_gscv,
                                                            savedir=resdir, gs_name='GS_' + estimator + '_Confined' + str(confined_zone),
                                                            final_name=estimator + '_Confined' + str(confined_zone),
                                                            refit=True)

    t1 = t.time()
    print('\nFinished {} GS(CV) as well as final training in {:.4f} min'.format(estimator, (t1 - t0)/60.))

    # # Actual training, fitting using the best found hyper-parameters
    # t0 = t.time()
    # if not unittest:
    #     regressor.fit(x_train, y_train, **fit_param)
    # else:
    #     regressor.fit(x_gs, y_gs, **fit_param_gs)
    #
    # # Save estimator
    # if not unittest: dump(regressor, resdir + estimator + '_Confined' + str(confined_zone) + '.joblib')
    # t1 = t.time()
    # print('\nFinished {} training in {:.4f} min'.format(estimator, (t1 - t0)/60.))

