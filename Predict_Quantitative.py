import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from joblib import load
from PostProcess_FieldData import FieldData
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor
from Utility import interpolateGridData, rotateData
import time as t
from PlottingTool import BaseFigure, Plot2D, Plot2D_Image, PlotContourSlices3D, PlotSurfaceSlices3D, PlotImageSlices3D
import os
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from copy import copy
import matplotlib.pyplot as plt

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
slicenames = ('twoDupstreamTurbine', 'rotorPlane')  # str
set_types = 'auto'  # str
time = 'last'  # str/float/int or 'last'
seed = 123  # int
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# The case folder name storing the estimator
estimator_folder = "ML/TBRF"  # str
confinezone = '2'  # str
# Feature set string
fs = 'grad(TKE)_grad(p)'  # 'grad(TKE)_grad(p)', 'grad(TKE)', 'grad(p)', None
realize_iter = 0  # int


"""
Plot Settings
"""
plotslices, plotlines = False, True
figfolder = 'Result'
xlabel = 'Distance [m]'
# Field rotation for vertical slices, rad or deg
fieldrot = 30.  # float
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
# Subsample for barymap coordinates, jump every subsample
subsample = 50  # int
figheight_multiplier = 1.  # float
# Limit for bij plot
bijlims = (-1/2., 2/3.)  # (float, float)
# Save figures and show figures
save_fig, show = True, False  # bool; bool
# If save figure, figure extension and DPI
ext, dpi = 'png', 1000  # str; int
# Height limit for line plots
heightlim = (0., 700.)  # tuple(float)


"""
Process User Inputs, Don't Change
"""
# Automatically select time if time is set to 'latest'
if time == 'last':
    if test_casename == 'ALM_N_H_ParTurb':
        time = '25000.0838025'
        if slicenames == 'auto': slicenames = ('alongWindSouthernRotor', 'alongWindNorthernRotor',
                                               'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                               'twoDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines', 'threeDdownstreamTurbines',
                                               'sevenDdownstreamTurbines')
        if set_types == 'auto': set_types = ('oneDdownstreamSouthernTurbine',
                                             'threeDdownstreamSouthernTurbine',
                                             'sevenDdownstreamSouthernTurbine',
                                             'oneDdownstreamNorthernTurbine',
                                             'threeDdownstreamNorthernTurbine',
                                             'sevenDdownstreamNorthernTurbine')
        # For line plots
        # 1D downstream: southern, northern; 3D downstream: southern, northern; 7D downstream: southern, northern
        turbloc = (2165.916, 1661.916, 2024.89, 1519.776, 1733.262, 1228.148)
    elif test_casename == 'ALM_N_L_ParTurb':
        time = '23000.07'
        if slicenames == 'auto': slicenames = ('alongWindSouthernRotor', 'alongWindNorthernRotor',
                                               'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                               'twoDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines', 'threeDdownstreamTurbines',
                                               'sevenDdownstreamTurbines')
        if set_types == 'auto': set_types = ('oneDdownstreamSouthernTurbine',
                                             'threeDdownstreamSouthernTurbine',
                                             'sevenDdownstreamSouthernTurbine',
                                             'oneDdownstreamNorthernTurbine',
                                             'threeDdownstreamNorthernTurbine',
                                             'sevenDdownstreamNorthernTurbine')
        turbloc = (2165.916, 1661.916, 2024.89, 1519.776, 1733.262, 1228.148)
    elif test_casename == 'ALM_N_H_OneTurb':
        time = '24995.0438025'
        if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                               'twoDupstreamTurbine', 'rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine',
                                               'sevenDdownstreamTurbine')
        if set_types == 'auto': set_types = ('oneDdownstreamTurbine',
                                             'threeDdownstreamTurbine',
                                             'sevenDdownstreamTurbine')
        # For line plots
        # 1D downstream, 3D downstream, 7D downstream
        turbloc_h = (1913.916, 1768.424, 1477.439)

    elif test_casename == 'ALM_N_H_SeqTurb':
        time = '25000.1288025'
        if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                               'twoDupstreamTurbineOne', 'rotorPlaneOne', 'rotorPlaneTwo',
                                               'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo',
                                               'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo',
                                               'sixDdownstreamTurbineTwo')
        if set_types == 'auto': set_types = ('oneDdownstreamTurbineOne',
                                             'oneDdownstreamTurbineTwo',
                                             'threeDdownstreamTurbineOne',
                                             'threeDdownstreamTurbineTwo',
                                             'sixDdownstreamTurbineTwo')
        # For line plots
        # 1D downstream: upwind, downwind; 3D downstream: upwind, downwind; 6D downstream downwind
        turbloc_h = (1913.916, 1404.693, 1768.424, 1259.201, 1040.963)
    elif test_casename == 'ALM_N_L_SeqTurb':
        time = '23000.135'
        if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                               'twoDupstreamTurbineOne', 'rotorPlaneOne', 'rotorPlaneTwo',
                                               'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo',
                                               'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo',
                                               'sixDdownstreamTurbineTwo')
        turbloc_h = (1913.916, 1404.693, 1768.424, 1259.201, 1040.963)
    elif test_casename == 'ALM_N_L_ParTurb_Yaw':
        time = '23000.065'
        if slicenames == 'auto': slicenames = ('alongWindSouthernRotor', 'alongWindNorthernRotor',
         'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
         'twoDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines', 'threeDdownstreamTurbines',
         'sevenDdownstreamTurbines')
        turbloc = (2165.916, 1661.916, 2024.89, 1519.776, 1733.262, 1228.148)
    elif test_casename == 'ALM_N_H_ParTurb_HiSpeed':
        time = ''
        turbloc = (2165.916, 1661.916, 2024.89, 1519.776, 1733.262, 1228.148)

else:
    time = str(time)

estimator_fullpath = casedir + '/' + ml_casename + '/' + estimator_folder + '/'
if 'TBRF' in estimator_folder or 'tbrf' in estimator_folder:
    estimator_name = 'TBRF'
elif 'TBDT' in estimator_folder or 'tbdt' in estimator_folder:
    estimator_name = 'TBDT'
elif 'TBAB' in estimator_folder or 'tbab' in estimator_folder:
    estimator_name = 'TBAB'
else:
    estimator_name = 'TBDT'

estimator_name += '_Confined' + str(confinezone)
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
figdir = estimator_fullpath + '/' + figfolder
os.makedirs(figdir, exist_ok=True)
if fieldrot > 2*np.pi: fieldrot /= 180./np.pi
# Initialize case object for both ML and test case
# case_ml = FieldData(caseName=ml_casename, caseDir=casedir, times=time, fields=fields, save=False)
case = FieldData(caseName=test_casename, caseDir=casedir, times=time, fields=fields, save=False)


"""
Load Data and Regressor
"""
print('\nLoading regressor... ')
regressor = load(estimator_fullpath + estimator_name + '.joblib')


"""
Plot Slices
"""
if plotslices:
    # Loop through each test slice, predict and visualize
    for slicename in slicenames:
        # In which plane are the slices, either xy or rz
        if slicename in ('hubHeight', 'quarterDaboveHub', 'turbineApexHeight'):
            slicedir = 'xy'
        # Else if vertical, then slice is radial and z direction
        else:
            slicedir = 'rz'

        list_data_test = case.readPickleData(time, 'list_data_test_' + slicename)
        ccx_test = list_data_test[0][:, 0]
        ccy_test = list_data_test[0][:, 1]
        ccz_test = list_data_test[0][:, 2]
        # First axis is radial for vertical slice and x for horizontal slice
        if slicedir == 'rz':
            cc1_test = ccx_test/np.sin(fieldrot) if 'alongWind' not in slicename else ccx_test/np.cos(fieldrot)
        else:
            cc1_test = ccx_test

        # 2nd axis is z for vertical slices and y for horizontal
        cc2_test = ccz_test if slicedir == 'rz' else ccy_test
        del ccx_test, ccy_test, ccz_test
        # if slicedir == 'xy':
        #     cc_test = np.vstack((cc1_test, cc2_test, np.zeros_like(cc1_test))).T
        #     cc_test = rotateData(cc_test, anglez=fieldrot)
        #     cc1_test = cc_test[:, 0]
        #     cc2_test = cc_test[:, 1]
        #     del cc_test

        x_test = list_data_test[1]
        x_test[x_test > 1e10] = 1e10
        x_test[x_test < -1e10] = 1e10
        y_test_unrot = list_data_test[2]
        tb_test = list_data_test[3]
        del list_data_test
        # Rotate field
        y_test = expandSymmetricTensor(y_test_unrot).reshape((-1, 3, 3))
        y_test = rotateData(y_test, anglez=fieldrot)
        y_test = contractSymmetricTensor(y_test.reshape((-1, 9)))


        """
        Predict
        """
        t0 = t.time()
        score_test = regressor.score(x_test, y_test, tb=tb_test)
        y_pred_test_unrot = regressor.predict(x_test, tb=tb_test)
        # Rotate field
        y_pred_test = expandSymmetricTensor(y_pred_test_unrot).reshape((-1, 3, 3))
        y_pred_test = rotateData(y_pred_test, anglez=fieldrot)
        y_pred_test = contractSymmetricTensor(y_pred_test.reshape((-1, 9)))
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
        # Manually limit over range RGB values
        rgb_bary_pred_test[rgb_bary_pred_test > 1.] = 1.
        t1 = t.time()
        print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))

        if 'OneTurb' in test_casename:
            if slicedir == 'xy':
                extent_test = (633.225, 1930.298, 817.702, 1930.298)
                c1lim = (633.225, 2039.417)
                c2lim = (817.702, 1993.298)
            elif slicedir == 'rz':
                c2lim = (0., 405.)
                if 'alongWind' in slicename:
                    extent_test = (1039.051, 1039.051 + 126*10, 0., 405.)
                    c1lim = (1039.051, 1039.051 + 126*10)
                elif 'twoDupstream' in slicename:
                    extent_test = (1484.689, 1484.689 + 5*126, 0., 405.)
                    c1lim = tuple(np.array([742.344, 1057.344])/np.sin(fieldrot))
                elif 'rotorPlane' in slicename:
                    extent_test = (1671.662, 1671.662 + 5*126, 0., 405.)
                    c1lim = tuple(np.array([960.583, 1275.583])/np.sin(fieldrot))
                elif 'oneDdownstream' in slicename:
                    extent_test = (1598.916, 1598.916 + 5*126, 0., 405.)
                    c1lim = tuple(np.array([1069.702, 1384.702])/np.sin(fieldrot))
                elif 'threeDdownstream' in slicename:
                    extent_test = (1453.424, 1453.424 + 5*126, 0., 405.)
                    c1lim = tuple(np.array([1287.94, 1602.94])/np.sin(fieldrot))
                elif 'sevenDdownstream' in slicename:
                    extent_test = (1162.439, 1162.439 + 5*126, 0., 405.)
                    c1lim = tuple(np.array([1724.417, 2039.417])/np.sin(fieldrot))

        else:
            extent_test = (cc1_test.min(), cc1_test.max(), cc2_test.min(), cc2_test.max())

        t0 = t.time()
        ccx_test_mesh, ccy_test_mesh, _, rgb_bary_test_mesh = interpolateGridData(cc1_test, cc2_test, rgb_bary_test,
                                                                                  xlim=c1lim, ylim=c2lim,
                                                                                  mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
        _, _, _, rgb_bary_predtest_mesh = interpolateGridData(cc1_test, cc2_test, rgb_bary_pred_test,
                                                              xlim=c1lim, ylim=c2lim,
                                                              mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
        t1 = t.time()
        print('\nFinished interpolating mesh data for barycentric map in {:.4f} s'.format(t1 - t0))

        t0 = t.time()
        _, _, _, y_test_mesh = interpolateGridData(cc1_test, cc2_test, y_test,
                                                   xlim=c1lim,
                                                   ylim=c2lim,
                                                   mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
        _, _, _, y_predtest_mesh = interpolateGridData(cc1_test, cc2_test, y_pred_test,
                                                       xlim=c1lim, ylim=c2lim,
                                                       mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
        t1 = t.time()
        print('\nFinished interpolating mesh data for bij in {:.4f} s'.format(t1 - t0))


        """
        Plotting
        """
        # First barycentric maps
        figname = 'barycentric_{}_{}_test_{}'.format(test_casename, estimator_name, slicename)
        # x and y label is y if horizontal slice otherwise z
        if slicedir == 'xy':
            xlabel = '$x$ [m]'
            ylabel = '$y$ [m]'
        else:
            xlabel = '$r$ [m]'
            ylabel = '$z$ [m]'


        barymap_test = Plot2D_Image(val=rgb_bary_test_mesh, name=figname, xlabel=xlabel,
                                    ylabel=ylabel,
                                    save=save_fig, show=show,
                                    figdir=figdir,
                                    figwidth='half',
                                    rotate_img=True,
                                    extent=(c1lim[0], c1lim[1], c2lim[0], c2lim[1]),
                                    figheight_multiplier=1)
        barymap_test.initializeFigure()
        barymap_test.plotFigure()
        barymap_test.finalizeFigure(showcb=False)

        figname = 'barycentric_{}_{}_predtest_{}'.format(test_casename, estimator_name, slicename)
        barymap_predtest = Plot2D_Image(val=rgb_bary_predtest_mesh, name=figname, xlabel=xlabel,
                                        ylabel=ylabel,
                                        save=save_fig, show=show,
                                        figdir=figdir,
                                        figwidth='half',
                                        rotate_img=True,
                                        extent=(c1lim[0], c1lim[1], c2lim[0], c2lim[1]),
                                        figheight_multiplier=1
        )
        barymap_predtest.initializeFigure()
        barymap_predtest.plotFigure()
        barymap_predtest.finalizeFigure(showcb=False)


        # Then bij plots
        # Create figure names for bij plots
        bijcomp = (11, 12, 13, 22, 23, 33)
        fignames_predtest, fignames_test, bijlabels, bijlabels_pred = [], [], [], []
        for ij in bijcomp:
            fignames_predtest.append('b{}_{}_{}_predtest_{}'.format(ij, test_casename, estimator_name, slicename))
            fignames_test.append('b{}_{}_{}_test_{}'.format(ij, test_casename, estimator_name, slicename))
            bijlabels.append('$b_{}$ [-]'.format('{' + str(ij) + '}'))
            bijlabels_pred.append('$\hat{b}_{' + str(ij) + '}$ [-]')

        # Go through each bij component
        for i in range(len(bijcomp)):
            bij_predtest_plot = Plot2D_Image(val=y_predtest_mesh[:, :, i], name=fignames_predtest[i], xlabel=xlabel,
                                             ylabel=ylabel, val_label=bijlabels_pred[i],
                                             save=save_fig, show=show,
                                             figdir=figdir,
                                             figwidth='half',
                                             val_lim=bijlims,
                                             rotate_img=True,
                                             extent=(c1lim[0], c1lim[1], c2lim[0], c2lim[1]),
                                             figheight_multiplier=figheight_multiplier)
            bij_predtest_plot.initializeFigure()
            bij_predtest_plot.plotFigure()
            bij_predtest_plot.finalizeFigure()

            bij_test_plot = Plot2D_Image(val=y_test_mesh[:, :, i], name=fignames_test[i], xlabel=xlabel,
                                         ylabel=ylabel, val_label=bijlabels[i],
                                         save=save_fig, show=show,
                                         figdir=figdir,
                                         figwidth='half',
                                         val_lim=bijlims,
                                         rotate_img=True,
                                         extent=(c1lim[0], c1lim[1], c2lim[0], c2lim[1]),
                                         figheight_multiplier=figheight_multiplier)
            bij_test_plot.initializeFigure()
            bij_test_plot.plotFigure()
            bij_test_plot.finalizeFigure()


"""
Plot Lines
"""
if plotlines:
    lineresult_folder = figdir + '/Lines/'
    os.makedirs(lineresult_folder, exist_ok=True)
    triverts = [(0., 0.), (1., 0.), (0.5, np.sqrt(3)/2.), (0., 0.)]
    tripath = Path(triverts)
    tripatch = PathPatch(tripath, fill=False, edgecolor=(89/255.,)*3, ls='-', zorder=-1)
    tripatches = []
    for i in range(100):
        tripatches.append(copy(tripatch))

    tripatches = iter(tripatches)
    # Go through each line type
    for iset, set_type in enumerate(set_types):
        y_test_all, y_pred_all, y_all = [], [], []
        x_bary_test_all, y_bary_test_all = [], []
        x_bary_pred_all, y_bary_pred_all = [], []
        x_bary_all, y_bary_all = [], []
        for orient in ('H', 'V'):
            list_data_test = case.readPickleData(time, 'list_data_test_' + set_type + '_' + orient)
            distance_test = list_data_test[0]
            distance_test_subsamp = distance_test[::subsample]
            x_test = list_data_test[1]
            x_test[x_test > 1e10] = 1e10
            x_test[x_test < -1e10] = 1e10
            y_test_unrot = list_data_test[2]
            tb_test = list_data_test[3]
            del list_data_test
            # Rotate field
            y_test = expandSymmetricTensor(y_test_unrot).reshape((-1, 3, 3))
            y_test = rotateData(y_test, anglez=fieldrot)
            y_test = contractSymmetricTensor(y_test.reshape((-1, 9)))
            # y_test_all.append(y_test)
            y_all.append(y_test)


            """
            Predict
            """
            t0 = t.time()
            score_test = regressor.score(x_test, y_test, tb=tb_test)
            y_pred_test_unrot = regressor.predict(x_test, tb=tb_test)
            # Rotate field
            y_pred_test = expandSymmetricTensor(y_pred_test_unrot).reshape((-1, 3, 3))
            y_pred_test = rotateData(y_pred_test, anglez=fieldrot)
            y_pred_test = contractSymmetricTensor(y_pred_test.reshape((-1, 9)))
            t1 = t.time()
            print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))
            # y_pred_all.append(y_pred_test)
            y_all.append(y_pred_test)

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
            # Manually limit over range RGB values
            rgb_bary_pred_test[rgb_bary_pred_test > 1.] = 1.
            xy_bary_test[xy_bary_test > 1.] = 1.
            xy_bary_test[xy_bary_test < 0.] = 0.
            xy_bary_pred_test[xy_bary_pred_test > 1.] = 1.
            xy_bary_pred_test[xy_bary_pred_test < 0.] = 0.
            t1 = t.time()
            print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))
            # x_bary_test_all.append(xy_bary_test[::subsample, 0])
            # y_bary_test_all.append(xy_bary_test[::subsample, 1])
            # x_bary_pred_all.append(xy_bary_pred_test[::subsample, 0])
            # y_bary_pred_all.append(xy_bary_pred_test[::subsample, 1])
            x_bary_all.append(xy_bary_test[::subsample, 0])
            y_bary_all.append(xy_bary_test[::subsample, 1])
            x_bary_all.append(xy_bary_pred_test[::subsample, 0])
            y_bary_all.append(xy_bary_pred_test[::subsample, 1])


            """
            Plotting bij for Each Line
            """
            # Create figure names for bij plots
            bijcomp = (11, 12, 13, 22, 23, 33)
            fignames_bij, fignames_test, bijlabels, bijlabels_pred = [], [], [], []
            for ij in bijcomp:
                fignames_bij.append('b{}_{}_{}_{}_{}'.format(ij, test_casename, estimator_name, set_type, orient))
                bijlabels.append('$b_{}$ [-]'.format('{' + str(ij) + '}'))

            # Go through each bij component
            sortidx = np.argsort(distance_test)
            for i in range(len(bijcomp)):
                if orient == 'H':
                    list_x = (distance_test.take(sortidx),)*2
                    list_y = (y_test[:, i].take(sortidx), y_pred_test[:, i].take(sortidx))
                    xlim = None
                    ylim = bijlims
                    turblocs = (turbloc_h[iset] - 63., turbloc_h[iset] + 63.)
                    xlabel, ylabel = 'Range [m]', bijlabels[i]
                else:
                    list_y = (distance_test.take(sortidx),)*2
                    list_x = (y_test[:, i].take(sortidx), y_pred_test[:, i].take(sortidx))
                    ylim = heightlim
                    xlim = bijlims
                    turblocs = (90. - 63., 90. + 63.)
                    xlabel, ylabel = bijlabels[i], 'Range [m]'

                bij_plot = Plot2D(list_x=list_x,
                                           list_y=list_y,
                                           name=fignames_bij[i], xlabel=xlabel,  ylabel=ylabel,
                                           save=save_fig, show=show,
                                           figwidth='1/3',
                                           figheight_multiplier=figheight_multiplier,
                                           figdir=lineresult_folder,
                                           plot_type='infer',
                                           xlim=xlim, ylim=ylim)
                bij_plot.initializeFigure()
                if orient == 'H':
                    bij_plot.axes.fill_between(turblocs, bijlims[0], bijlims[1],
                                               alpha=0.5, facecolor=bij_plot.colors[2], zorder=-1)
                else:
                    bij_plot.axes.fill_between(bijlims, turblocs[0], turblocs[1],
                                               alpha=0.5, facecolor=bij_plot.colors[2], zorder=-1)
                    bij_plot.axes.fill_between(bijlims, 700., 800.,
                                               alpha=0.5, facecolor=bij_plot.colors[3], zorder=-1)

                bij_plot.plotFigure(linelabel=('Truth', 'Prediction'))
                bij_plot.finalizeFigure()


        """
        Plotting Barycentric Maps of Multiple Lines at Once
        """
        # First barycentric maps
        figname = 'barycentric_{}_{}_test_{}'.format(test_casename, estimator_name, set_type)

        barymap = Plot2D(list_x=x_bary_all,
                                   list_y=y_bary_all,
                                   name=figname, xlabel=None, ylabel=None,
                                   save=save_fig, show=show,
                                   figwidth='half',
                                   figheight_multiplier=figheight_multiplier,
                                   figdir=lineresult_folder,
                                   plot_type='scatter',
                         equalaxis=True,
                         val_label='Normalized range')
        barymap.initializeFigure()
        barymap.axes.add_patch(next(tripatches))
        barymap.plotFigure(linelabel=('H, truth', 'H, prediction', 'V, truth', 'V, prediction'), showmarker=True, markercolors=distance_test_subsamp/distance_test_subsamp.max())
        plt.axis('off')
        barymap.axes.annotate(r'$\textbf{X}_{2c}$', (0., 0.), (-0.1, 0.))
        barymap.axes.annotate(r'$\textbf{X}_{3c}$', (0.5, np.sqrt(3)/2.))
        barymap.axes.annotate(r'$\textbf{X}_{1c}$', (1., 0.))
        barymap.finalizeFigure(cbar_orient='vertical', legloc='upper left')

        # # Then bij plots
        # # Create figure names for bij plots
        # bijcomp = (11, 12, 13, 22, 23, 33)
        # sortidx = np.argsort(distance_test)
        # fignames, bijlabels, bijlabels_pred = [], [], []
        # for ij in bijcomp:
        #     fignames.append('b{}_{}_{}_{}'.format(ij, test_casename, estimator_name, set_type))
        #     bijlabels.append('$b_{}$ [-]'.format('{' + str(ij) + '}'))
        #     bijlabels_pred.append('$\hat{b}_{' + str(ij) + '}$ [-]')
        #
        # # Go through each bij component
        # for i in range(len(bijcomp)):
        #     bi = []
        #     for y in y_all:
        #         bi.append(y[:, i].take(sortidx, axis=0))
        #
        #     bij_predtest_plot = Plot2D(list_x=(distance_test.take(sortidx)/1000.,)*4,
        #                                list_y=bi,
        #                                name=fignames[i], xlabel=xlabel,  ylabel=bijlabels[i],
        #                                      save=save_fig, show=show,
        #                                      figwidth='1/3',
        #                                      figheight_multiplier=figheight_multiplier,
        #                                figdir=lineresult_folder,
        #                                plot_type='infer',
        #                                ylim=bijlims,
        #                                equalaxis=False)
        #     bij_predtest_plot.initializeFigure()
        #     patch = next(patches)
        #     bij_predtest_plot.axes.add_patch(patch)
        #     bij_predtest_plot.plotFigure(linelabel=('H, truth', 'H, prediction', 'V, truth', 'V, prediction'))
        #     bij_predtest_plot.finalizeFigure()


