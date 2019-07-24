import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from joblib import load
from PostProcess_FieldData import FieldData
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor
from Preprocess.Feature import getInvariantFeatureSet
from Utility import interpolateGridData
from numba import njit, prange
from Utilities import timer
import time as t
from scipy import ndimage
import matplotlib.pyplot as plt
from PlottingTool import BaseFigure, Plot2D, Plot2D_Image
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import copy

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both ML and test
ml_casename = 'ALM_N_H_OneTurb'  # str
test_casename = 'ALM_N_H_OneTurb'  # str
# Absolute parent directory of ML and test case
casedir = '/media/yluan'  # str
# Which time to extract input and output for ML
# Slice names for prediction and visualization
slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
              'twoDupstreamTurbine', 'rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine', 'sevenDdownstreamTurbine')
time = 'last'  # str/float/int or 'last'
seed = 123  # int
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# The case folder name storing the estimator
estimator_folder = "TBRF_16t_auto_p{47_16_4_0_0_50}_s123_boots_0iter"
# Feature set string
fs = 'grad(TKE)_grad(p)'  # 'grad(TKE)_grad(p)', 'grad(TKE)', 'grad(p)', None
realize_iter = 0  # int


"""
Plot Settings
"""
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
figheight_multiplier = 1.1  # float
# Limit for bij plot
bijlims = (-1/2., 2/3.)  # (float, float)
# Save figures and show figures
save_fig, show = True, False  # bool; bool
# If save figure, figure extension and DPI
ext, dpi = 'png', 1000  # str; int


"""
Process User Inputs, Don't Change
"""
# Automatically select time if time is set to 'latest'
if time == 'last':
    if test_casename == 'ALM_N_H_ParTurb':
        time = '22000.0918025'
    elif test_casename == 'ALM_N_H_OneTurb':
        time = '24995.0788025'

else:
    time = str(time)
    
estimator_fullpath = casedir + '/' + ml_casename + '/' + '/' + estimator_folder + '/'
if 'TBRF' in estimator_folder or 'tbrf' in estimator_folder:
    estimator_name = 'TBRF'
elif 'TBDT' in estimator_folder or 'tbdt' in estimator_folder:
    estimator_name = 'TBDT'
elif 'TBAB' in estimator_folder or 'tbab' in estimator_folder:
    estimator_name = 'TBAB'
else:
    estimator_name = 'TBDT'

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

uniform_mesh_size = int(uniform_mesh_size)
# Ensemble name of fields useful for Machine Learning
ml_field_ensemble_name = 'ML_Fields_' + ml_casename
# Initialize case object for both ML and test case
# case_ml = FieldData(caseName=ml_casename, caseDir=casedir, times=time, fields=fields, save=False)
case = FieldData(caseName=test_casename, caseDir=casedir, times=time, fields=fields, save=False)


"""
Load Data and Regressor
"""
print('\nLoading regressor... ')
regressor = load(estimator_fullpath + estimator_name + '.joblib')
# Loop through each test slice, predict and visualize
for slicename in slicenames:
    list_data_test = case.readPickleData(time, 'list_data_test_' + slicename)
    ccx_test = list_data_test[0][:, 0]
    ccy_test = list_data_test[0][:, 1]
    x_test = list_data_test[1]
    y_test = list_data_test[2]
    tb_test = list_data_test[3]


    """
    Predict
    """
    t0 = t.time()
    score_test = regressor.score(x_test, y_test, tb=tb_test)
    y_pred_test = regressor.predict(x_test, tb=tb_test)
    t1 = t.time()
    print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))


    """
    Posprocess Machine Learning Predictions
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
    _, _, _, rgb_bary_predtest_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
    t1 = t.time()
    print('\nFinished interpolating mesh data for barycentric map in {:.4f} s'.format(t1 - t0))

    t0 = t.time()
    _, _, _, y_test_mesh = interpolateGridData(ccx_test, ccy_test, y_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
    _, _, _, y_predtest_mesh = interpolateGridData(ccx_test, ccy_test, y_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
    t1 = t.time()
    print('\nFinished interpolating mesh data for bij in {:.4f} s'.format(t1 - t0))


    """
    Plotting
    """
    # First barycentric maps
    figname = 'barycentric_{}_test_{}'.format(test_casename, slicename)
    xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
    extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
    barymap_test = Plot2D_Image(val=rgb_bary_test_mesh, name=figname, xlabel=xlabel,
                                ylabel=ylabel,
                                save=save_fig, show=show,
                                figdir=case.resultPaths[time],
                                figwidth='half',
                                rotate_img=True,
                                extent=extent_test,
                                figheight_multiplier=0.7)
    barymap_test.initializeFigure()
    barymap_test.plotFigure()
    barymap_test.finalizeFigure(showcb=False)

    figname = 'barycentric_{}_predtest_{}'.format(test_casename, slicename)
    barymap_predtest = Plot2D_Image(val=rgb_bary_predtest_mesh, name=figname, xlabel=xlabel,
                                    ylabel=ylabel,
                                    save=save_fig, show=show,
                                    figdir=case.resultPaths[time],
                                    figwidth='half',
                                    rotate_img=True,
                                    extent=extent_test,
                                    figheight_multiplier=0.7)
    barymap_predtest.initializeFigure()
    barymap_predtest.plotFigure()
    barymap_predtest.finalizeFigure(showcb=False)
    
    # Then bij plots
    # Create figure names for bij plots
    bijcomp = (11, 12, 13, 22, 23, 33)
    fignames_predtest, fignames_test, bijlabels, bijlabels_pred = [], [], [], []
    for ij in bijcomp:
        fignames_predtest.append('b{}_{}_predtest_{}'.format(ij, test_casename, slicename))
        fignames_test.append('b{}_{}_test_{}'.format(ij, test_casename, slicename))
        bijlabels.append('$b_{}$ [-]'.format('{' + str(ij) + '}'))
        bijlabels_pred.append('$\hat{b}_{' + str(ij) + '}$ [-]')

    # Go through each bij component
    for i in range(len(bijcomp)):
        bij_predtest_plot = Plot2D_Image(val=y_predtest_mesh[:, :, i], name=fignames_predtest[i], xlabel=xlabel,
                                         ylabel=ylabel, val_label=bijlabels_pred[i],
                                         save=save_fig, show=show,
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
                                     ylabel=ylabel, val_label=bijlabels[i],
                                     save=save_fig, show=show,
                                     figdir=case.resultPaths[time],
                                     figwidth='1/3',
                                     val_lim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test,
                                     figheight_multiplier=figheight_multiplier)
        bij_test_plot.initializeFigure()
        bij_test_plot.plotFigure()
        bij_test_plot.finalizeFigure()


