import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from joblib import load
from FieldData import FieldData
from SliceData import SliceProperties
from DataBase import *
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor,makeRealizable
from Utility import interpolateGridData, rotateData, fieldSpatialSmoothing
import time as t
from PlottingTool import BaseFigure, Plot2D, Plot2D_Image, PlotContourSlices3D, PlotSurfaceSlices3D, PlotImageSlices3D
import os
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from copy import copy
import matplotlib.pyplot as plt
from Postprocess.PostProcessFlowProperty import computeDivDevR_2D

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
slicenames = ('hubHeight', 'quarterDaboveHub', 'turbineApexHeight')  # str
time = 'latestTime'  # str/float/int or 'last'
seed = 123  # int
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# The case folder name storing the estimator
estimator_folder = "ML"  # str
estimator_name = 'TBDT'  # 'TBDT', 'TBRF', 'TBAB', 'TBGB'
confinezone = '2'  # '', '1', '2'
# Feature set string
fs = 'grad(TKE)_grad(p)+'  # 'grad(TKE)_grad(p)+', 'grad(TKE)_grad(p)', 'grad(TKE)', 'grad(p)'
# Iteration to make predictions realizable
realize_iter = 0  # int
# What to do with prediction too far from realizable range
# If bij_novelty is 'excl', 2 realize_iter is automatically used
bij_novelty = None  # None, 'excl', 'reset'
# Whether filter the prediction field with Gaussian filter
filter = False
# Multiplier to realizable bij limits [-1/2, 1/2] off-diagonally and [-1/3, 2/3] diagonally.
# Whatever is outside bounds is treated as NaN.
# Whatever between bounds and realizable limits are made realizable
bijbnd_multiplier = 2.


"""
Plot Settings
"""
plotslices = False
plotlines = True
figfolder = 'Result'
xlabel = 'Distance [m]'
# Field rotation for vertical slices, rad or deg
fieldrot = 30.  # float
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e4  # int
# Subsample for barymap coordinates, jump every "subsample"
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
# Height of the horizontal slices, only used for 3D horizontal slices plot
horslice_offsets = (90., 121.5, 153.)


"""
Process User Inputs, Don't Change
"""
# Default case settings
if 'ParTurb' in test_casename:
    # Read coor info from database
    turb_borders, turb_centers_frontview, confinebox, _ = ParTurb('hor')
    set_types = ('oneDdownstreamSouthernTurbine',
                 'threeDdownstreamSouthernTurbine',
                 'sevenDdownstreamSouthernTurbine',
                 'oneDdownstreamNorthernTurbine',
                 'threeDdownstreamNorthernTurbine',
                 'sevenDdownstreamNorthernTurbine')
    # For line plots
    # 1D downstream: southern, northern; 3D downstream: southern, northern; 7D downstream: southern, northern
    turbloc = (2165.916, 1661.916, 2024.89, 1519.776, 1733.262, 1228.148)
elif 'OneTurb' in test_casename:
    turb_borders, turb_centers_frontview, confinebox, _ = OneTurb('hor')
    set_types = ('oneDdownstreamTurbine',
                 'threeDdownstreamTurbine',
                 'sevenDdownstreamTurbine')
    # For line plots
    # 1D downstream, 3D downstream, 7D downstream
    turbloc_h = (1913.916, 1768.424, 1477.439)
elif 'SeqTurb' in test_casename:
    # TODO: SeqTurb database
    set_types = ('oneDdownstreamTurbineOne',
                 'oneDdownstreamTurbineTwo',
                 'threeDdownstreamTurbineOne',
                 'threeDdownstreamTurbineTwo',
                 'sixDdownstreamTurbineTwo')
    # For line plots
    # 1D downstream: upwind, downwind; 3D downstream: upwind, downwind; 6D downstream downwind
    turbloc_h = (1913.916, 1404.693, 1768.424, 1259.201, 1040.963)
else:
    turb_borders, turb_centers_frontview, confinebox, _ = OneTurb('hor')
    set_types = ('oneDdownstreamTurbine',
                 'threeDdownstreamTurbine',
                 'sevenDdownstreamTurbine')
    # For line plots
    # 1D downstream, 3D downstream, 7D downstream
    turbloc_h = (1913.916, 1768.424, 1477.439)

estimator_fullpath = casedir + '/' + ml_casename + '/' + estimator_folder + '/' + estimator_name + '/'
estimator_name += '_Confined' + str(confinezone)
# Average fields of interest for reading and processing
if 'grad(TKE)_grad(p)' in fs:
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_p_rghAvg', 'grad_kResolved', 'grad_kSGSmean', 'UAvg',
              'GAvg', 'divDevR', 'dDevRab_db')
elif fs == 'grad(TKE)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_kResolved', 'grad_kSGSmean',
              'GAvg', 'divDevR', 'dDevRab_db')
elif fs == 'grad(p)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_p_rghAvg', 'UAvg',
              'GAvg', 'divDevR', 'dDevRab_db')
else:
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg',
              'GAvg', 'divDevR', 'dDevRab_db')

uniform_mesh_size = int(uniform_mesh_size)
figdir = estimator_fullpath + '/' + figfolder
os.makedirs(figdir, exist_ok=True)
if fieldrot > 2*np.pi: fieldrot /= 180./np.pi
# Initialize case object for test case
case = FieldData(casename=test_casename, casedir=casedir, times=time, fields=fields, save=False)
# Update time to actual detected time if it was 'latestTime'
time = case.times[0]
# Also initialize test case slice instance
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
    # Load truth div(dev(R)), where R is -ui'uj' here
    divdev_r = case.readPickleData(time, 'div(dev(R))_' + slicename)
    # Furthermore load ddev(Rab)/db to complete predicted div(dev(R)) later, where R is ui'uj' here
    ddevr_dj = case.readPickleData(time, 'ddev(Rab)_db_' + slicename)
    # Only ddev(Ri3)/dz is used here if horizontal slice
    if slicedir == 'horizontal':
        ddevri3_dz = np.vstack((ddevr_dj[:, 2], ddevr_dj[:, 5], ddevr_dj[:, 8])).T
    # Else if vertical slice, only ddev(Ri2)/dy is used
    else:
        ddevri3_dz = np.vstack((ddevr_dj[:, 1], ddevr_dj[:, 4], ddevr_dj[:, 7])).T

    k = case.readPickleData(time, 'TKE_' + slicename)
    
    """
    Predict
    """
    t0 = t.time()
    y_pred = regressor.predict(x_test, tb=tb_test, bij_novelty=bij_novelty)
    # Remove NaN predictions
    if bij_novelty == 'excl':
        print("Since bij_novelty is 'excl', removing NaN and making y_pred realizable...")
        nan_mask = np.isnan(y_pred).any(axis=1)
        ccx_test = ccx_test[~nan_mask]
        ccy_test = ccy_test[~nan_mask]
        ccz_test = ccz_test[~nan_mask]
        y_pred = y_pred[~nan_mask]
        for _ in range(2):
            y_pred = makeRealizable(y_pred)

    # Rotate field
    y_pred = expandSymmetricTensor(y_pred).reshape((-1, 3, 3))
    y_pred = rotateData(y_pred, anglez=fieldrot)
    y_pred = contractSymmetricTensor(y_pred)
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
        ccx_test_mesh, cc2_test_mesh, _, y_pred_mesh = fieldSpatialSmoothing(y_pred, ccx_test, cc2_test,
                                                                                 is_bij=True,
                                                                                 bij_bnd_multiplier=bijbnd_multiplier,
                                                                                 xlim=(None,)*2, ylim=(None,)*2,
                                                                                 mesh_target=uniform_mesh_size)
        # Collapse mesh grid
        y_pred = y_pred_mesh.reshape((-1, 6))
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

    # Interpolate to 2D mesh grid
    ccx_test_mesh, ccy_test_mesh, ccz_test_mesh, y_pred_mesh = case_slice.interpolateDecomposedSliceData_Fast(
        ccx_test, ccy_test, ccz_test, y_pred, slice_orient=slicedir, target_meshsize=uniform_mesh_size,
        interp_method='nearest', confinebox=confinebox[i])
    _, _, _, k_mesh = case_slice.interpolateDecomposedSliceData_Fast(
            ccx_test, ccy_test, ccz_test, k, slice_orient=slicedir, target_meshsize=uniform_mesh_size,
            interp_method='nearest', confinebox=confinebox[i])
    # Since k_mesh has shape (nx, ny, 1), collapse last dim
    k_mesh = k_mesh.reshape((k_mesh.shape[0], k_mesh.shape[1]))
    _, _, _, divdev_r_mesh = case_slice.interpolateDecomposedSliceData_Fast(
            ccx_test, ccy_test, ccz_test, divdev_r, slice_orient=slicedir, target_meshsize=uniform_mesh_size,
            interp_method='nearest', confinebox=confinebox[i])
    _, _, _, ddevri3_dz_mesh = case_slice.interpolateDecomposedSliceData_Fast(
            ccx_test, ccy_test, ccz_test, ddevri3_dz, slice_orient=slicedir, target_meshsize=uniform_mesh_size,
            interp_method='nearest', confinebox=confinebox[i])

    dx = (max(ccx_test) - min(ccx_test))/(ccx_test_mesh.shape[1] - 1)
    # Spacing of the 2nd axis -- z for vertical slice and y for horizontal slice
    d2 = (max(ccy_test) - min(ccy_test))/(ccy_test_mesh.shape[1] - 1) if slicedir == 'horizontal' else (max(ccz_test) - min(ccz_test))/(ccz_test_mesh.shape[1] - 1)
    divdev_r_pred_mesh = computeDivDevR_2D(y_pred_mesh, k_mesh, ddevri3_dz_mesh, dx=dx, dy=d2)

    plt.figure(slicename + 'truth')
    plt.quiver(divdev_r_mesh[:, :, 0], divdev_r_mesh[:, :, 1])
    plt.figure(slicename + 'prediction')
    plt.quiver(divdev_r_pred_mesh[:, :, 0], divdev_r_pred_mesh[:, :, 1])
