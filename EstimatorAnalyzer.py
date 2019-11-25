import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from joblib import load
from PlottingTool import BaseFigure, Plot2D, Plot2D_MultiAxes


"""
User Inputs
"""
# casename = 'ALM_N_H_OneTurb'  # 'N_H_OneTurb_LowZ_Rwall2', 'ALM_N_H_OneTurb'
casename = 'ALM_N_H_OneTurb'  # 'N_H_OneTurb_LowZ_Rwall2', 'ALM_N_H_OneTurb'
casedir = '/media/yluan/RANS'
casedir = '/media/yluan'
ml_folder = 'ML'
estimators = ('TBGB',) #('TBDT', 'TBRF', 'TBAB', 'TBGB')
domain_confinezone = 2
ylim = (0., .5)



"""
Feature Importance Bar Chart
"""
for i, estimator in enumerate(estimators):
    estimator_name = estimator + '_Confined' + str(domain_confinezone)
    # case = FieldData(casename=casename, casedir=casedir)
    ml_path = casedir + '/' + casename + '/' + ml_folder + '/'
    estimator_path = ml_path + estimator + '/'

    regressor = load(estimator_path + estimator_name + '.joblib')
    threshold = regressor.steps[0][1].threshold_
    threshold_str = regressor.steps[0][1].threshold
    threshold_str = threshold_str.replace("*", '')
    threshold_str = threshold_str.replace("m", 'M')

    # Feature selector feature importance
    fs_importance = regressor.steps[0][1].estimator_.feature_importances_
    all_std = np.std(
            [tree.feature_importances_ for tree in regressor.steps[0][1].estimator_.estimators_],
            axis=0)
    # Estimator feature importance
    importance = regressor.steps[1][1].feature_importances_

    # Fill in 0 for removed features in final regressor
    importance_actual = fs_importance.copy()
    importance_actual[importance_actual < threshold] = 0.
    j0 = 0
    for j in range(len(fs_importance)):
        if importance_actual[j] != 0.:
            importance_actual[j] = importance[j0]
            j0 += 1

    # # Feature index as x axis
    # list_x = (np.arange(1, len(importance_actual) + 1),
    #           np.arange(1, len(fs_importance) + 1),
    #           (1, len(fs_importance)))
    # list_y = (importance_actual, fs_importance, (threshold, threshold))
    # plot = Plot2D(list_x, list_y, name='FeatureImportance' + estimator, xlabel='Feature', ylabel='Importance', figdir=ml_path,
    #                  show=False, save=True,
    #              xlim=(0, len(fs_importance) + 1), ylim=ylim)
    # plot.initializeFigure()
    # plot.plotFigure(linelabel=(estimator, 'TBRF feature selector', threshold_str), showmarker=True)
    # plot.axes.fill_between(np.arange(1, len(fs_importance) + 1),
    #                        np.maximum(fs_importance - all_std, 0.),
    #                        fs_importance + all_std, alpha=0.25, color=plot.colors[1], lw=0.)
    # plot.finalizeFigure()


"""
Plot TBGB OOB Improvement
"""
loss_improve = regressor.steps[1][1].oob_improvement_
loss = regressor.steps[1][1].train_score_
list_x = np.arange(1, regressor.steps[1][1].n_estimators + 1)
list_x2 = np.arange(1, regressor.steps[1][1].n_estimators + 1)
list_y = loss
list_y2 = loss_improve

plot_loss = Plot2D_MultiAxes(list_x, list_y, list_x2, list_y2, name='Loss' + estimator, ax2loc='y', ax2label='OOB loss improvement',
                             xlabel='Iteration', ylabel='In-bag deviance', figdir=ml_path, save=True, show=False)
plot_loss.initializeFigure()
plot_loss.plotFigure(xyscale2=('linear', 'log'))
plot_loss.finalizeFigure(xyscale=('linear', 'log'))


"""
Plot TBAB Regression Error
"""
estimator_path = ml_path + 'TBAB/'
estimator_name = 'TBAB_Confined' + str(domain_confinezone)
tbab = load(estimator_path + estimator_name + '.joblib')
err = tbab._final_estimator.estimator_errors_
weight = tbab._final_estimator.estimator_weights_
list_x = np.arange(1, tbab.steps[1][1].n_estimators + 1)
list_x2 = np.arange(1, tbab.steps[1][1].n_estimators + 1)
list_y = err
list_y2 = weight

plot_err = Plot2D_MultiAxes(list_x, list_y, list_x2, list_y2, name='ErrorTBAB', ax2loc='y', ax2label='Estimator weight',
                             xlabel='Iteration', ylabel='Regression error', figdir=ml_path, save=True, show=False)
plot_err.initializeFigure()
plot_err.plotFigure(xyscale2=('linear', 'linear'))
plot_err.finalizeFigure(xyscale=('linear', 'log'))
