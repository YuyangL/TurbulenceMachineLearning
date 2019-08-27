import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from PostProcess_FieldData import FieldData
from joblib import load
from PlottingTool import Plot2D_MultiAxes, Plot2D


"""
User Inputs, Anything Can Be Changed Here
"""
# Absolute directory of this flow case
casedir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'
rf_selector_threshold='median'
# The case folder name storing the estimator
estimator_folder = "TBDT/{50_0.002_2_0}_4cv" + '_' + rf_selector_threshold
estimator_folder0 = "TBDT/{50_0.002_2_0}_4cv"
estimator_dir = "RANS_Re10595"
# Feature set number
fs = 'grad(TKE)_grad(p)+'  # '1', '12', '123'
realize_iter = 0  # int
plot_varthreshold = False


"""
Process User Inputs
"""
if 'TBRF' in estimator_folder or 'tbrf' in estimator_folder:
    estimator_name = 'TBRF'
elif 'TBDT' in estimator_folder or 'tbdt' in estimator_folder:
    estimator_name = 'TBDT'
elif 'TBAB' in estimator_folder or 'tbab' in estimator_folder:
    estimator_name = 'TBAB'
elif 'TBGB' in estimator_folder or 'tbgb' in estimator_folder:
    estimator_name = 'TBGB'
else:
    estimator_name = 'TBDT'

estimator_fullpath = casedir + '/' + estimator_dir + '/Fields/Result/5000/Archive/' + estimator_folder + '/' + estimator_name
estimator_fullpath0 = casedir + '/' + estimator_dir + '/Fields/Result/5000/Archive/' + estimator_folder0 + '/' + estimator_name
# Average fields of interest for reading and processing
fields = ('U', 'k', 'p', 'omega',
          'grad_U', 'grad_k', 'grad_p')
# # Ensemble name of fields useful for Machine Learning
# ml_field_ensemble_name = 'ML_Fields_' + rans_case_name
# # Initialize case object
case = FieldData(caseName=estimator_dir, caseDir=casedir, times=time, fields=fields, save=False)
figdir = case.resultPaths[time] + '/ValidationCurve'
#
# invariants = case.readPickleData(time, fileNames = ('Sij',
#                                                    'Rij',
#                                                    'Tij',
#                                                    'bij_LES'))
# sij = invariants['Sij']
# rij = invariants['Rij']
# tb = invariants['Tij']
# bij_les = invariants['bij_LES']
# cc = case.readPickleData(time, fileNames='cc')
# fs_data = case.readPickleData(time, fileNames = ('FS_' + fs))
#
# list_data_test = case.readPickleData(time, 'list_data_test_seed' + str(seed))
# cc_test = list_data_test[0]
# ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
# x_test, y_test, tb_test = list_data_test[1:4]

print('\nLoading regressor... ')
regressor = load(estimator_fullpath + '.joblib')
# Base regressor without feature selection
regressor0 = load(estimator_fullpath0 + '.joblib')


"""
Plot Feature Importance
"""
# TBRF is not GridSearch object thus doesn't have best_estimator_ attribute
if 'TBRF' not in estimator_name:
    feat_importance_all = regressor.best_estimator_.steps[0][1].estimator_.feature_importances_
    all_std = np.std([tree.feature_importances_ for tree in regressor.best_estimator_.steps[0][1].estimator_.estimators_],
                     axis=0)
    threshold = regressor.best_estimator_.steps[0][1].threshold_

    feat_importance = regressor.best_estimator_.steps[1][1].feature_importances_
    # Feature importance without feature selection
    feat_importance0 = regressor0.best_estimator_.feature_importances_
else:
    feat_importance_all = regressor.steps[0][1].estimator_.feature_importances_
    all_std = np.std(
            [tree.feature_importances_ for tree in regressor.steps[0][1].estimator_.estimators_],
            axis=0)
    threshold = regressor.steps[0][1].threshold_

    feat_importance = regressor.steps[1][1].feature_importances_
    # Feature importance without feature selection
    feat_importance0 = regressor0.feature_importances_

# Only TBRF have std
if 'TBRF' in estimator_name:
    std = np.std([tree.feature_importances_ for tree in regressor.steps[1][1].estimators_],
                 axis=0)
    # Std of TBRF without feature selection
    std0 = np.std([tree.feature_importances_ for tree in regressor0.estimators_],
                  axis=0)
else:
    std = np.zeros_like(all_std)
    std0 = np.zeros_like(all_std)

feat_importance_actual = feat_importance_all.copy()
actual_std = np.zeros_like(all_std)
feat_importance_actual[feat_importance_actual < threshold] = 0.
i0 = 0
for i in range(len(feat_importance_all)):
    if feat_importance_actual[i] != 0.:
        feat_importance_actual[i] = feat_importance[i0]
        actual_std[i] = std[i0]
        i0 += 1

myplot = Plot2D((np.arange(1, 51),)*3, (feat_importance_all, feat_importance0, feat_importance_actual), save=True, show=False,
                xlabel='Feature', ylabel = 'Importance',
                figwitdh='1/3', figdir=figdir, name=estimator_name + '_importance_' + rf_selector_threshold,
                ylim=(0., 0.5))
myplot.initializeFigure()
myplot.plotFigure(linelabel=('TBRF feature selector', estimator_name, 'TBRF feature selector + ' + estimator_name), showmarker=True)
myplot.axes.fill_between(np.arange(1, 51),
                         np.maximum(feat_importance_all - all_std, 0.),
                         feat_importance_all + all_std, alpha=0.25, color=myplot.colors[0], lw=0.)
# Only TBRF have std
if 'TBRF' in estimator_name:
    myplot.axes.fill_between(np.arange(1, 51),
                             np.maximum(feat_importance0 - std0, 0.),
                             feat_importance0 + std0, alpha=0.25, color=myplot.colors[1], lw=0.)
    myplot.axes.fill_between(np.arange(1, 51),
                             np.maximum(feat_importance_actual - actual_std, 0.),
                             feat_importance_actual + actual_std, alpha=0.25, color=myplot.colors[2], lw=0.)

myplot.finalizeFigure(xyscale=('linear', 'linear'))


if plot_varthreshold:
    """
    TBDT
    """
    # 1e-7 is actually 0 but use 1e-7 instead for log axis to work
    varthreshold = [1e-3, 1e-4, 1e-5, 1e-7]
    train_score = [0.64569362,
                   0.6486845,
                   0.648697747,
                   0.64856944]
    test_score = [-107.5855455,
                  -89.21176822,
                  -38.361893,
                  -1.8943537]
    times = [0.631667, 1.4, 1.4, 1.01]
    features = [17, 45, 50, 50]
    plot2 = Plot2D_MultiAxes((np.array(varthreshold),)*2, (np.array(train_score), np.array(test_score)),
                             list_x2=np.array(varthreshold), list_y2=np.array(features), ax2loc='y', save=True, show=False,
                             figdir=figdir, name='TBDT_varthreshold',
                             figwidth='1/3',
                             xlabel='Var threshold', ylabel='$R^2$ score',
                             ax2label='Selected features', plot_type2='shade',
                             xlim=(1e-7, 1e-3), y2lim=(0, 100))
    plot2.initializeFigure()
    plot2.plotFigure(showmarker=True, linelabel=('Train', 'Test'))
    plot2.finalizeFigure(xyscale=('log', 'symlog'))


    """
    TBRF
    """
    varthreshold = [1e-3, 1e-4, 1e-5, 1e-7]
    train_score = [0.6479522,
                   -1.82594,
                   -0.650914,
                   -0.6509076]
    test_score = [0.6250,
                  0.58047,
                  0.590504,
                  0.592556]
    times = [3.6, 9.1, 10.3, 5.9]
    features = [17, 45, 50, 50]
    plot2 = Plot2D_MultiAxes((np.array(varthreshold),)*2, (np.array(train_score), np.array(test_score)),
                             list_x2=np.array(varthreshold), list_y2=np.array(features), ax2loc='y', save=True,
                             show=False,
                             figdir=figdir, name='TBRF_varthreshold',
                             figwidth='1/3',
                             xlabel='Var threshold', ylabel='$R^2$ score',
                             ax2label='Selected features', plot_type2='shade',
                             xlim=(1e-7, 1e-3), y2lim=(0, 100))
    plot2.initializeFigure()
    plot2.plotFigure(showmarker=True, linelabel=('Train', 'Test'))
    plot2.finalizeFigure(xyscale=('log', 'symlog'))




# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import RobustScaler, MaxAbsScaler, MinMaxScaler
# from sklearn.tree import DecisionTreeRegressor
# x_test0 = x_test.copy()
# x_test1 = x_test.copy()
# # scaler = MaxAbsScaler(copy=True)
# # selector = VarianceThreshold(threshold=1e-4)
# tree = DecisionTreeRegressor(split_finder='brent')
# # pipeline = Pipeline([('scaler', scaler),
# #                      ('selector', selector),
# #                      ('tree', tree)])
#
# selector = SelectFromModel(RandomForestRegressor(n_estimators=1000, max_depth=3, split_finder='brent', median_predict=True, oob_score=True,
#                                                  n_jobs=-1, verbose=2,
#                                                  max_features=2/3.),
#                            threshold='0.25*median')
# pipeline = Pipeline([('selector', selector),
#                      ('tree', tree)])
#
#
# # x_test = pipeline.fit_transform(x_test)
# pipeline.fit(x_test, y_test, tree__tb=tb_test)
# score = pipeline.score(x_test0, y_test, tb=tb_test)
# y_pred = pipeline.predict(x_test1, tb=tb_test)


# # np.random.seed(0)
# # X = np.random.rand(1000, 3)
# # y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)
#
# # f_test, _ = f_regression(X, y)
# # f_test /= np.max(f_test)
#
# mi = mutual_info_regression(x_test, y_test)
# mi /= np.max(mi)
#
# plt.figure(figsize=(15, 5))
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.scatter(x_test[:, i], y_test, edgecolor='black', s=20)
#     plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
#     if i == 0:
#         plt.ylabel("$y$", fontsize=14)
#     # plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
#     #           fontsize=16)
# plt.show()





# from time import time
# import matplotlib.pyplot as plt
#
# from sklearn.datasets import fetch_olivetti_faces
# from sklearn.ensemble import ExtraTreesClassifier
#
# # Number of cores to use to perform parallel fitting of the forest model
# n_jobs = 1
#
# # Load the faces dataset
# data = fetch_olivetti_faces()
# X = data.images.reshape((len(data.images), -1))
# y = data.target
#
# mask = y < 5  # Limit to 5 classes
# X = X[mask]
# y = y[mask]
#
# # Build a forest and compute the pixel importances
# print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
# t0 = time()
# forest = ExtraTreesClassifier(n_estimators=1000,
#                               max_features=128,
#                               n_jobs=n_jobs,
#                               random_state=0)
#
# forest.fit(X, y)
# print("done in %0.3fs" % (time() - t0))
# importances = forest.feature_importances_
# importances = importances.reshape(data.images[0].shape)
#
# # Plot pixel importances
# plt.matshow(importances, cmap=plt.cm.hot)
# plt.title("Pixel importances with forests of trees")
# plt.show()
