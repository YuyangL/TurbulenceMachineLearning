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
from pyevtk.hl import pointsToVTK

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both ML and test
ml_casename = 'ALM_N_H_OneTurb'  # str
test_casename = 'ALM_N_H_ParTurb2'  # str
# Absolute parent directory of ML and test case
casedir = '/media/yluan'  # str
time = 'latestTime'  # str/float/int or 'latestTime'
seed = 123  # int
# Interpolation method when interpolating mesh grids
interp_method = "linear"  # "nearest", "linear", "cubic"
# The case folder name storing the estimator
estimator_folder = "ML/TBRF"  # str
confinezone = '2'  # str
# Feature set string
fs = 'grad(TKE)_grad(p)+'  # 'grad(TKE)_grad(p)', 'grad(TKE)', 'grad(p)', None
# Height of the horizontal slices, only used for 3D horizontal slices plot
horslice_offsets = (90., 121.5, 153.)
save_data = False
bij_novelty = 'excl'  # 'excl', 'reset', None
# Field rotation for vertical slices, rad or deg
fieldrot = 30.  # float
result_folder = 'Result'

"""
Process User Inputs
"""
if fieldrot > np.pi/2.: fieldrot /= 180./np.pi
# Initialize case field instance for test case
case = FieldData(casename=test_casename, casedir=casedir, times=time, fields='uuPrime2', save=False)
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
ccname = 'CC'
data_testname = 'list_data_train'

fullcase_dir = casedir + '/' + test_casename + '/'
ccname += '_Confined' + confinezone
data_testname += '_Confined' + confinezone

"""
Load Data and Regressor
"""
# Loading cell centers and test X data
datadict = case.readPickleData(case.times[-1], (ccname, data_testname))
cc = datadict[ccname]
x_test = datadict[data_testname][1]
# Limit X from very positive or negative values
x_test[x_test > 1e10] = 1e10
x_test[x_test < -1e10] = 1e10
y_test = datadict[data_testname][2]
tb_test = datadict[data_testname][3]
del datadict
print('\nLoading regressor... ')
regressor = load(estimator_fullpath + estimator_name + '.joblib')


"""
Predict
"""
t0 = t.time()
# score_test = regressor.score(x_test, y_test, tb=tb_test)
del y_test
# y_pred_test_unrot = regressor.predict(x_test, tb=tb_test)
y_pred_test = regressor.predict(x_test, tb=tb_test, bij_novelty=bij_novelty)
# Remove NaN predictions
if bij_novelty == 'excl':
    print("Since bij_novelty is 'excl', removing NaN and making y_pred_test realizable...")
    nan_mask = np.isnan(y_pred_test).any(axis=1)
    cc = cc[~nan_mask]
    # ccx_test = cc[:, 0][~nan_mask]
    # ccy_test = cc[:, 1][~nan_mask]
    # ccz_test = cc[:, 2][~nan_mask]
    y_pred_test = y_pred_test[~nan_mask]
    for _ in range(2):
        y_pred_test = makeRealizable(y_pred_test)

# Rotate field
y_pred_test = expandSymmetricTensor(y_pred_test).reshape((-1, 3, 3))
y_pred_test = rotateData(y_pred_test, anglez=fieldrot)
y_pred_test = contractSymmetricTensor(y_pred_test)
t1 = t.time()
print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))

t0 = t.time()
_, eigval_pred_test, eigvec_pred_test = processReynoldsStress(y_pred_test, make_anisotropic=False, realization_iter=0, to_old_grid_shape=False)
t1 = t.time()
print('\nFinished processing Reynolds stress in {:.4f} s'.format(t1 - t0))


"""
Convert to VTK
"""
os.makedirs(estimator_fullpath + '/' + result_folder, exist_ok=True)
ccx = np.ascontiguousarray(cc[:, 0])
ccy = np.ascontiguousarray(cc[:, 1])
ccz = np.ascontiguousarray(cc[:, 2])
del cc
# eigval_pred_test = np.ascontiguousarray(eigval_pred_test)
# eigvec_pred_test = np.ascontiguousarray(eigvec_pred_test)
for i in range(3):

    eigval_pred_test_i = np.ascontiguousarray(eigval_pred_test[:, i])
    pointsToVTK(estimator_fullpath + '/' + result_folder + '/' + 'Pred_' + estimator_name + '_' + test_casename + '_eigval' + str(i) + '_' + 'Confined' + confinezone + '_' + bij_novelty,
                ccx, ccy, ccz, data={"eigval" + str(i): eigval_pred_test_i})
    # pointsToVTK(estimator_fullpath + '/' + result_folder + '/' + 'Pred_' + test_casename + '_eigvec_' + 'Confined' + confinezone + '_' + bij_novelty,
    #             cc[:, 0], cc[:, 1], cc[:, 2], data={"eigvec": eigvec_pred_test})


"""
Paraview Scheme
"""
# from paraview.simple import *
# paraview.simple._DisableFirstRenderCameraReset()
#
# points_vtu = XMLUnstructuredGridReader( FileName=['/media/yluan/points.vtu'] )
#
# points_vtu.PointArrayStatus = ['bij_0']
# points_vtu.CellArrayStatus = []
#
# RenderView1 = GetRenderView()
# DataRepresentation1 = Show()
# DataRepresentation1.ScaleFactor = 94.75000000000009
# DataRepresentation1.ScalarOpacityUnitDistance = 7.440556709746787
# DataRepresentation1.SelectionPointFieldDataArrayName = 'bij_0'
# DataRepresentation1.EdgeColor = [0.0, 0.0, 0.5000076295109483]
#
# RenderView1.CenterOfRotation = [1280.0, 1372.5, 107.50000000000003]
#
# IsoVolume1 = IsoVolume()
#
# RenderView1.CameraPosition = [1280.0, 1372.5, 2492.627883768477]
# RenderView1.CameraFocalPoint = [1280.0, 1372.5, 107.50000000000003]
# RenderView1.CameraClippingRange = [2149.8391049307925, 2687.0610520250043]
# RenderView1.CameraParallelScale = 617.3165213243534
#
# IsoVolume1.ThresholdRange = [-0.09138601077922912, 0.4393011691507959]
# IsoVolume1.InputScalars = ['POINTS', 'bij_0']
#
# DataRepresentation2 = Show()
# DataRepresentation2.ScaleFactor = 92.00000000000001
# DataRepresentation2.ScalarOpacityUnitDistance = 42.63390594991192
# DataRepresentation2.SelectionPointFieldDataArrayName = 'bij_0'
# DataRepresentation2.EdgeColor = [0.0, 0.0, 0.5000076295109483]
#
# DataRepresentation1.Visibility = 0
#
# IsoVolume1.ThresholdRange = [0.3, 0.33]
#
# a1_bij_0_PVLookupTable = GetLookupTableForArray( "bij_0", 1, RGBPoints=[0.30000205311372546, 0.23, 0.299, 0.754, 0.3299976259870521, 0.706, 0.016, 0.15], VectorMode='Magnitude', NanColor=[0.25, 0.0, 0.0], ColorSpace='Diverging', ScalarRangeInitialized=1.0, AllowDuplicateScalars=1 )
#
# a1_bij_0_PiecewiseFunction = CreatePiecewiseFunction( Points=[0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0] )
#
# RenderView1.CameraViewUp = [-0.4195190969171189, 0.4760571059374443, 0.7728993202276154]
# RenderView1.CameraPosition = [1166.2219777886596, 569.2643853485366, 540.4852028737565]
# RenderView1.CameraClippingRange = [105.34354085367897, 2002.323921139287]
# RenderView1.CameraFocalPoint = [1279.9999999999993, 1372.4999999999993, 107.50000000000011]
#
# DataRepresentation2.ScalarOpacityFunction = a1_bij_0_PiecewiseFunction
# DataRepresentation2.ColorArrayName = ('POINT_DATA', 'bij_0')
# DataRepresentation2.LookupTable = a1_bij_0_PVLookupTable
#
# a1_bij_0_PVLookupTable.ScalarOpacityFunction = a1_bij_0_PiecewiseFunction
#
# Render()

