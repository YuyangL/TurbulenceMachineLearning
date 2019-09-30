import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from joblib import load
from FieldData import FieldData
from SliceData import SliceProperties
from DataBase import *
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor, makeRealizable
from Utility import interpolateGridData, rotateData, gaussianFilter, fieldSpatialSmoothing
import time as t
from PlottingTool import BaseFigure, Plot2D, Plot2D_Image, PlotContourSlices3D, PlotSurfaceSlices3D, PlotImageSlices3D, plotTurbineLocations
import os
import numpy as np
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from copy import copy
from scipy.ndimage import gaussian_filter
from Postprocess.Filter import nan_helper

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both ML and test
ml_casename = 'ALM_N_H_OneTurb'  # str
test_casename = 'ALM_N_H_ParTurb2'  # str
# Absolute parent directory of ML and test case
casedir = '/media/yluan'  # str
# Which time to extract input and output for ML
# Slice names for prediction and visualization
slicenames = ('oneDupstreamTurbine', 'rotorPlane', 'oneDdownstreamTurbine')  # str
slicenames = ('oneDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines')  # str
# slicenames = ('threeDdownstreamTurbine', 'fiveDdownstreamTurbine', 'sevenDdownstreamTurbine')  # str
# slicenames = ('threeDdownstreamTurbines', 'fiveDdownstreamTurbines', 'sevenDdownstreamTurbines')  # str
time = 'latestTime'  # str/float/int or 'latestTime'
seed = 123  # int
# Interpolation method when interpolating mesh grids
interp_method = "linear"  # "nearest", "linear", "cubic"
# The case folder name storing the estimator
estimator_folder = "ML/TBDT"  # str
confinezone = '2'  # str
# Feature set string
fs = 'grad(TKE)_grad(p)+'  # 'grad(TKE)_grad(p)', 'grad(TKE)', 'grad(p)', None
filter = True  # bool
# Multiplier to realizable bij limits [-1/2, 1/2] off-diagonally and [-1/3, 2/3] diagonally.
# Whatever is outside bounds is treated as NaN.
# Whatever between bounds and realizable limits are made realizable
bijbnd_multiplier = 2.  # float
# Height of the horizontal slices, only used for 3D horizontal slices plot
horslice_offsets = (90., 121.5, 153.)
save_data = False
bij_novelty = 'excl'  # 'excl', 'reset', None


"""
Plot Settings
"""
plot_property = '*'  # 'bary', 'bij', '*'
plotslices, plotlines = True, False
figfolder = 'Result'
# Field rotation for vertical slices, rad or deg
fieldrot = 30.  # float
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e5  # int
contour_lvl = 200
figheight_multiplier = 1.  # float
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
if 'ParTurb' in test_casename:
    if time == 'latestTime':
        if test_casename == 'ALM_N_H_ParTurb2':
            time = '25000.0838025'
        elif test_casename == 'ALM_N_L_ParTurb':
            time = '23000.07'
        elif test_casename == 'ALM_N_L_ParTurb_Yaw':
            time = '23000.065'
        elif test_casename == 'ALM_N_H_ParTurb_HiSpeed':
            # FIXME:
            time = ''
    if slicenames == 'auto': slicenames = ('alongWindSouthernRotor', 'alongWindNorthernRotor',
                                           'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'oneDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines',
                                           'threeDdownstreamTurbines', 'fiveDdownstreamTurbines',
                                           'sevenDdownstreamTurbines')
    if 'oneDupstream' in slicenames[0] or 'rotorPlane' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
        # Read coor info from database
        turb_borders, turb_centers_frontview, confinebox, confinebox2 = ParTurb('vert')
        if 'threeDdownstream' in slicenames[0]:
            confinebox = confinebox2
            turb_centers_frontview = turb_centers_frontview[6:]
        else:
            turb_centers_frontview = turb_centers_frontview[:6]

    elif 'hubHeight' in slicenames[0] or 'groundHeight' in slicenames[0]:
        turb_borders, turb_centers_frontview, confinebox, _ = ParTurb('hor')

elif test_casename == 'ALM_N_H_OneTurb':
    time = '24995.0438025'
    if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'oneDupstreamTurbine', 'rotorPlane', 'oneDdownstreamTurbine',
                                           'threeDdownstreamTurbine', 'fiveDdownstreamTurbine', 'sevenDdownstreamTurbine')
    # For front view vertical slices, usually either "oneDupstream*" or "threeDdownstream*" is forefront slice
    if 'oneDupstream' in slicenames[0] or 'rotorPlane' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
        # Read coor info from database
        turb_borders, turb_centers_frontview, confinebox, confinebox2 = OneTurb('vert')
        # If "threeDdownstream*" is 1st slice, then there're only 3 slices in total:
        # "threeDdownstream*", "fiveDdownstream", "sevenDdownstream"
        if 'threeDdownstream' in slicenames[0]:
            confinebox = confinebox2
            turb_centers_frontview = turb_centers_frontview[3:]
        else:
            turb_centers_frontview = turb_centers_frontview[:3]

    # For horizontal slices, usually "hubHeight*" or "groundHeight*" is bottom slice
    elif 'hubHeight' in slicenames[0] or 'groundHeight' in slicenames[0]:
        turb_borders, turb_centers_frontview, confinebox, _ = OneTurb('hor')

elif test_casename == 'ALM_N_H_SeqTurb':
    time = '25000.1288025'
    # FIXME: update
    if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'twoDupstreamTurbineOne', 'rotorPlaneOne', 'rotorPlaneTwo',
                                           'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo',
                                           'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo',
                                           'sixDdownstreamTurbineTwo')
elif test_casename == 'ALM_N_L_SeqTurb':
    time = '23000.135'
    # FIXME: update
    if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'twoDupstreamTurbineOne', 'rotorPlaneOne', 'rotorPlaneTwo',
                                           'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo',
                                           'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo',
                                           'sixDdownstreamTurbineTwo')

elif test_casename == 'ALM_N_L_ParTurb_Yaw':
    time = '23000.065'
    # FIXME: update
    if slicenames == 'auto': slicenames = ('alongWindSouthernRotor', 'alongWindNorthernRotor',
                                           'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'twoDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines', 'threeDdownstreamTurbines',
                                           'sevenDdownstreamTurbines')
elif test_casename == 'ALM_N_H_ParTurb_HiSpeed':
    # FIXME: update
    time = ''

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
fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'uuPrime2',
          'grad_UAvg', 'grad_p_rghAvg', 'grad_kResolved', 'grad_kSGSmean', 'UAvg')

# Automatic view angle and figure settings, only for 3D plots
# For front view vertical plots
if 'oneDupstream' in slicenames[0] or 'rotorPlane' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
    if 'OneTurb' in test_casename:
        view_angle = (20, -80) if 'one' in slicenames[0] else (20, -95)
        # Equal axis option and figure width
        equalaxis, figwidth = True,  'half'
    elif 'ParTurb' in test_casename:
        view_angle = (20, -80) if 'one' in slicenames[0] else (20, -90)
        # view_angle = (20, -215) if 'one' in slicenames[0] else (20, -205)
        # Equal axis option and figure width
        equalaxis, figwidth = True, 'half'
# For horizontal plots
elif 'groundHeight' in slicenames[0] or 'hubHeight' in slicenames[0]:
    view_angle = (25, -115)
    equalaxis, figwidth = False, 'half'
# Default settings for other cases
else:
    view_angle = (20, -100)
    equalaxis, figwidth = True, 'full'

# Value limit for plotting
# Usually (-1/2, 2/3) for anisotropy tensor bij
bij_lim = (-0.5, 2/3.)
bij_label = (r'$\langle b_{11} \rangle$ [-]', r'$\langle b_{12} \rangle$ [-]', r'$\langle b_{13} \rangle$ [-]',
             r'$\langle b_{22} \rangle$ [-]', r'$\langle b_{23} \rangle$ [-]',
             r'$\langle b_{33} \rangle$ [-]')
bij_pred_label = (r'$\hat{b}_{11}$ [-]', r'$\hat{b}_{12}$ [-]', r'$\hat{b}_{13}$ [-]',
             r'$\hat{b}_{22}$ [-]', r'$\hat{b}_{23}$ [-]',
             r'$\hat{b}_{33}$ [-]')
if fieldrot > np.pi/2.: fieldrot /= 180./np.pi
# Initialize case field instance for test case
case = FieldData(casename=test_casename, casedir=casedir, times=time, fields=fields, save=False)
# Initialize case slice instance
case_slice = SliceProperties(time=time, casedir=casedir, casename=test_casename, rot_z=fieldrot, result_folder=figfolder)


"""
Load Data and Regressor
"""
print('\nLoading regressor... ')
regressor = load(estimator_fullpath + estimator_name + '.joblib')


"""
Plot Slices
"""
# Initialize lists to store multiple slices coor and value data
list_x, list_y, list_z = [], [], []
list_rgb, list_bij, list_rij = [], [], []
cc1_all, cc2_all = [], []
rgb_pred_all, rgb_test_all = [], []
bij_pred_all, bij_test_all = [], []
# Loop through each test slice, predict and visualize
for i, slicename in enumerate(slicenames):
    # In which plane are the slices, either xy or rz
    if slicename in ('hubHeight', 'quarterDaboveHub', 'turbineApexHeight'):
        slicedir = 'horizontal'
    # Else if vertical, then slice is radial and z direction
    else:
        slicedir = 'vertical'

    list_data_test = case.readPickleData(time, 'list_data_test_' + slicename)
    ccx_test = list_data_test[0][:, 0]
    ccy_test = list_data_test[0][:, 1]
    ccz_test = list_data_test[0][:, 2]

    x_test = list_data_test[1]
    x_test[x_test > 1e10] = 1e10
    x_test[x_test < -1e10] = 1e10
    y_test = list_data_test[2]
    tb_test = list_data_test[3]
    del list_data_test
    # # Rotate field
    # y_test = expandSymmetricTensor(y_test_unrot).reshape((-1, 3, 3))
    # y_test = rotateData(y_test, anglez=fieldrot)
    # y_test = contractSymmetricTensor(y_test)


    """
    Predict
    """
    t0 = t.time()
    score_test = regressor.score(x_test, y_test, tb=tb_test)
    # y_pred_test_unrot = regressor.predict(x_test, tb=tb_test)
    y_pred_test = regressor.predict(x_test, tb=tb_test, bij_novelty=bij_novelty)
    # Remove NaN predictions
    if bij_novelty == 'excl':
        print("Since bij_novelty is 'excl', removing NaN and making y_pred_test realizable...")
        nan_mask = np.isnan(y_pred_test).any(axis=1)
        ccx_test = ccx_test[~nan_mask]
        ccy_test = ccy_test[~nan_mask]
        ccz_test = ccz_test[~nan_mask]
        y_pred_test = y_pred_test[~nan_mask]
        for _ in range(2):
            y_pred_test = makeRealizable(y_pred_test)

    # Rotate field
    y_pred_test = expandSymmetricTensor(y_pred_test).reshape((-1, 3, 3))
    y_pred_test = rotateData(y_pred_test, anglez=fieldrot)
    y_pred_test = contractSymmetricTensor(y_pred_test)
    t1 = t.time()
    print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))


    """
    Postprocess Machine Learning Predictions
    """
    # Filter the result if requested.
    # This requires:
    # 1. Remove any component outside bound and set to NaN
    # 2. Interpolate to 2D slice mesh grid with nearest method
    # 3. Use 2D Gaussian filter to smooth the mesh grid while ignoring NaN, for every component
    # 4. Make whatever between bound and limits realizable
    if filter:
        cc2_test = ccz_test if slicedir == 'vertical' else ccy_test
        ccx_test_mesh, cc2_test_mesh, _, y_predtest_mesh = fieldSpatialSmoothing(y_pred_test, ccx_test, cc2_test, is_bij=True, bij_bnd_multiplier=bijbnd_multiplier,
                                                                                 xlim=(None,)*2, ylim=(None,)*2, mesh_target=uniform_mesh_size)
        # Collapse mesh grid
        y_pred_test = y_predtest_mesh.reshape((-1, 6))
        # Interpolate 3rd axis, either y or z, to mesh grid size and flatten it
        if slicedir == 'horizontal':
            _, ccz_test_mesh = np.mgrid[min(ccx_test):max(ccx_test):ccx_test_mesh.shape[0]*1j,
                          min(ccz_test):max(ccz_test):ccx_test_mesh.shape[1]*1j]
            ccy_test = cc2_test_mesh.ravel()
            ccz_test = ccz_test_mesh.ravel()
        else:
            ccy_test_mesh, _ = np.mgrid[min(ccy_test):max(ccy_test):ccx_test_mesh.shape[0]*1j,
                           min(ccz_test):max(ccz_test):ccx_test_mesh.shape[1]*1j]
            # In case the vertical slice is at a negative angle,
            # i.e. when x goes from low to high, y goes from high to low,
            # flip y2d from low to high to high to low
            ccy_test_mesh = np.flipud(ccy_test_mesh)
            ccy_test = ccy_test_mesh.ravel()
            ccz_test = cc2_test_mesh.ravel()

        ccx_test = ccx_test_mesh.ravel()

    t0 = t.time()
    _, eigval_test, _ = processReynoldsStress(y_test, make_anisotropic=False, realization_iter=0, to_old_grid_shape=False)
    # If filter was True, eigval_pred_test is a mesh grid
    _, eigval_pred_test, _ = processReynoldsStress(y_pred_test, make_anisotropic=False, realization_iter=0, to_old_grid_shape=False)
    t1 = t.time()
    print('\nFinished processing Reynolds stress in {:.4f} s'.format(t1 - t0))

    t0 = t.time()
    xy_bary_test, rgb_bary_test = getBarycentricMapData(eigval_test)
    # If filter was True, both xy_bary_pred_test and rgb_bary_pred_test are mesh grids
    xy_bary_pred_test, rgb_bary_pred_test = getBarycentricMapData(eigval_pred_test, to_old_grid_shape=True)
    t1 = t.time()
    print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))

    t0 = t.time()
    # Interpolate to desired uniform mesh
    if 'bary' in plot_property or '*' in plot_property:
        ccx_test_mesh, ccy_test_mesh, ccz_test_mesh, rgb_pred_test_mesh = case_slice.interpolateDecomposedSliceData_Fast(ccx_test, ccy_test, ccz_test, rgb_bary_pred_test, slice_orient=slicedir, target_meshsize=uniform_mesh_size,
                                                                                                             interp_method='nearest', confinebox=confinebox[i])
        list_rgb.append(rgb_pred_test_mesh)
        if save_data: case.savePickleData(time, (list_rgb,), ('Pred_' + str(slicenames) + '_list_rgb',))

    if 'bij' in plot_property or '*' in plot_property:
        ccx_test_mesh, ccy_test_mesh, ccz_test_mesh, bij_pred_test_mesh = case_slice.interpolateDecomposedSliceData_Fast(ccx_test,
                                                                                                             ccy_test,
                                                                                                             ccz_test,
                                                                                                             y_pred_test,
                                                                                                             slice_orient=slicedir,
                                                                                                             target_meshsize=uniform_mesh_size,
                                                                                                             interp_method=interp_method,
                                                                                                             confinebox=confinebox[i])
        list_bij.append(bij_pred_test_mesh)
        if save_data: case.savePickleData(time, (list_bij,), ('Pred_' + str(slicenames) + '_list_bij',))

    # If Gaussian filter has been used, then ccy_test_mesh needs to manually flipped upside down
    # as it was not working in interpolateDecomposedSliceData
    if filter and slicedir == 'vertical': ccy_test_mesh = np.flipud(ccy_test_mesh)
    t1 = t.time()
    print('\nFinished interpolating mesh data for barycentric map and/or bij in {:.4f} s'.format(t1 - t0))
    # Aggregate each slice's mesh grid
    list_x.append(ccx_test_mesh)
    list_y.append(ccy_test_mesh)
    list_z.append(ccz_test_mesh)
    if save_data:
        case.savePickleData(time, (list_x,), ('Pred_' + str(slicenames) + '_list_x',))
        case.savePickleData(time, (list_y,), ('Pred_' + str(slicenames) + '_list_y',))
        case.savePickleData(time, (list_z,), ('Pred_' + str(slicenames) + '_list_z',))


"""
Plotting
"""
if slicedir == 'horizontal':
    show_xylabel = (False, False)
    show_zlabel = True
    show_ticks = (False, False, True)
else:
    show_xylabel = (True, True)
    show_zlabel = False
    show_ticks = (True, True, False)

figdir = estimator_fullpath + '/' + figfolder
os.makedirs(figdir, exist_ok=True)
# First barycentric maps
if 'bary' in plot_property or '*' in plot_property:
    if 'OneTurb' in test_casename:
        figheight_multiplier = 2 if slicedir == 'horizontal' else 0.75
    else:
        figheight_multiplier = 2 if slicedir == 'horizontal' else 0.75

    figname = 'barycentric_{}_{}_predtest_{}'.format(test_casename, estimator_name, slicenames)
    # Initialize image slice plot instance
    barymap_predtest = PlotImageSlices3D(list_x=list_x, list_y=list_y, list_z=list_z,
                                         list_rgb=list_rgb, name=figname, xlabel='$x$ [m]',
                                         ylabel='$y$ [m]', zlabel='$z$ [m]',
                                         save=save_fig, show=show,
                                         figdir=figdir,
                                         figwidth=figwidth,
                                         figheight_multiplier=figheight_multiplier,
                                         viewangle=view_angle, equalaxis=equalaxis)
    barymap_predtest.initializeFigure(constrained_layout=True)
    barymap_predtest.plotFigure()
    plotTurbineLocations(barymap_predtest, slicedir, horslice_offsets, turb_borders, turb_centers_frontview)
    barymap_predtest.finalizeFigure(tight_layout=False, show_xylabel=(False,)*2, show_ticks=(False,)*3, show_zlabel=False)

# Then bij plots
if 'bij' in plot_property or '*' in plot_property:
    # Create figure names for bij plots
    bijcomp = (11, 12, 13, 22, 23, 33)
    fignames_predtest, fignames_test, bijlabels, bijlabels_pred = [], [], [], []
    for ij in bijcomp:
        fignames_predtest.append('b{}_{}_{}_predtest_{}'.format(ij, test_casename, estimator_name, slicenames))
        fignames_test.append('b{}_{}_{}_test_{}'.format(ij, test_casename, estimator_name, slicenames))
        bijlabels.append(r'$\langle b_{} \rangle$ [-]'.format('{' + str(ij) + '}'))
        bijlabels_pred.append('$\hat{b}_{' + str(ij) + '}$ [-]')

    # Go through each bij component
    for i in range(len(bijcomp)):
        # Extract ij component from each slice's bij
        list_bij_i = [bij_i[..., i] for _, bij_i in enumerate(list_bij)]
        # If horizontal slices 3D plot
        if slicedir == 'horizontal':
            bij_slice3d = PlotContourSlices3D(list_x, list_y, list_bij_i, horslice_offsets,
                                              name=fignames_predtest[i],
                                              xlabel=r'$x$ [m]',
                                              ylabel=r'$y$ [m]', zlabel=r'$z$ [m]', val_label=bijlabels_pred[i],
                                              cbar_orient='vertical',
                                              save=save_fig, show=show,
                                              figdir=figdir, viewangle=view_angle, figwidth=figwidth,
                                              figheight_multiplier=figheight_multiplier,
                                              val_lim=bij_lim, equalaxis=equalaxis)
        # Else if vertical front view slices 3D plot
        else:
            bij_slice3d = PlotSurfaceSlices3D(list_x, list_y, list_z, list_bij_i,
                                              xlabel='$x$ [m]', ylabel='$y$ [m]', zlabel='$z$ [m]',
                                              val_label=bijlabels_pred[i],
                                              name=fignames_predtest[i],
                                              save=save_fig, show=show,
                                              figdir=figdir, viewangle=view_angle, figwidth=figwidth,
                                              equalaxis=equalaxis, cbar_orient='horizontal', val_lim=bij_lim)

        bij_slice3d.initializeFigure()
        bij_slice3d.plotFigure(contour_lvl=contour_lvl)
        plotTurbineLocations(bij_slice3d, slicedir, horslice_offsets, turb_borders,
                             turb_centers_frontview)
        bij_slice3d.finalizeFigure(show_xylabel=show_xylabel, show_zlabel=show_zlabel, show_ticks=show_ticks)




