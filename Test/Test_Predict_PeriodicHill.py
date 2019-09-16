import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from joblib import load
from FieldData import FieldData
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
# Name of the flow case in both RANS and LES
rans_case_name = 'RANS_Re700'  # str
les_case_name = 'LES_Breuer/Re_700'  # str
# LES data name to read
les_data_name = 'Hill_Re_700_Breuer.csv'  # str
# Absolute directory of this flow case
casedir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'
seed = 321  # int
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# The case folder name storing the estimator
estimator_folder = "TBRF_16t_auto_p{47_16_4_0_0_50}_s123_boots_0iter"
estimator_dir = "RANS_Re10595"
# Feature set number
fs = 'grad(TKE)_grad(p)'  # '1', '12', '123'
realize_iter = 0  # int


"""
Plot Settings
"""
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
# Limit for bij plot
bijlims = (-1/3., 2/3.)  # (float, float)
# Save anything when possible
save_fields = True  # bool
# Save figures and show figures
save_fig, show = True, False  # bool; bool
if save_fig:
    # Figure extension and DPI
    ext, dpi = 'png', 1000  # str; int


"""
Process User Inputs, Don't Change
"""
estimator_fullpath = casedir + '/' + estimator_dir + '/Fields/Result/5000/Archive/' + estimator_folder + '/'
if 'TBRF' in estimator_folder or 'tbrf' in estimator_folder:
    estimator_name = 'TBRF'
elif 'TBDT' in estimator_folder or 'tbdt' in estimator_folder:
    estimator_name = 'TBDT'
else:
    estimator_name = 'TBDT'

# Average fields of interest for reading and processing
fields = ('U', 'k', 'p', 'omega',
          'grad_U', 'grad_k', 'grad_p')
# if fs == "grad(TKE)_grad(p)":
#     fields = ('U', 'k', 'p', 'omega',
#               'grad_U', 'grad_k', 'grad_p')
# elif fs == "grad(TKE)":
#     fields = ('k', 'omega',
#               'grad_U', 'grad_k')
# elif fs == "grad(p)":
#     fields = ('U', 'k', 'p', 'omega',
#               'grad_U', 'grad_p')
# else:
#     fields = ('k', 'omega',
#               'grad_U')

# Ensemble name of fields useful for Machine Learning
ml_field_ensemble_name = 'ML_Fields_' + rans_case_name
# Initialize case object
case = FieldData(caseName=rans_case_name, caseDir=casedir, times=time, fields=fields, save=False)


"""
Load Data
"""
@timer
@njit(parallel=True)
def transposeTensorBasis(tb):
    tb_transpose = np.empty((len(tb), tb.shape[2], tb.shape[1]))
    for i in prange(len(tb)):
        tb_transpose[i] = tb[i].T

    return tb_transpose

invariants = case.readPickleData(time, fileNames = ('Sij',
                                                    'Rij',
                                                    'Tij',
                                                    'bij_LES'))
sij = invariants['Sij']
rij = invariants['Rij']
tb = invariants['Tij']
bij_les = invariants['bij_LES']
cc = case.readPickleData(time, fileNames='cc')
fs_data = case.readPickleData(time, fileNames = ('FS_' + fs))

list_data_test = case.readPickleData(time, 'list_data_test_seed' + str(seed))
cc_test = list_data_test[0]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_test, y_test, tb_test = list_data_test[1:4]
# Contract symmetric tensors to 6 components in case of 9 components
y_test = contractSymmetricTensor(y_test)
tb_test = transposeTensorBasis(tb_test)
tb_test = contractSymmetricTensor(tb_test)
tb_test = transposeTensorBasis(tb_test)

print('\nLoading regressor... ')
regressor = load(estimator_fullpath + estimator_name + '.joblib')


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
_, _, _, rgb_bary_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for barycentric map in {:.4f} s'.format(t1 - t0))

t0 = t.time()
_, _, _, y_test_mesh = interpolateGridData(ccx_test, ccy_test, y_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, y_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for bij in {:.4f} s'.format(t1 - t0))


"""
Plotting
"""
figname = 'barycentric_periodichill_test_seed' + str(seed)
xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
barymap_test = Plot2D_Image(val=rgb_bary_test_mesh[:, :], name=figname, xLabel=xlabel,
                            yLabel=ylabel,
                            save=save_fig, show=show,
                            figDir=case.resultPaths[time],
                            figWidth='half',
                            rotate_img=True,
                            extent=extent_test)
geometry = np.genfromtxt(casedir + '/' + rans_case_name + '/' + "geometry.csv", delimiter=",")[:, :2]
path = Path(geometry)
patch = PathPatch(path, linewidth=0., facecolor=barymap_test.gray)
# patch is considered "a single artist" so have to make copy to use more than once
patches = []
for _ in range(14):
    patches.append(copy(patch))

patches = iter(patches)
barymap_test.initializeFigure()
barymap_test.plotFigure()
barymap_test.axes.add_patch(next(patches))
barymap_test.finalizeFigure(showcb=False)

figname = 'barycentric_periodichill_pred_test_seed' + str(seed)
barymap_predtest = Plot2D_Image(val=rgb_bary_pred_test_mesh[:, :], name=figname, xLabel=xlabel,
                                yLabel=ylabel,
                                save=save_fig, show=show,
                                figDir=case.resultPaths[time],
                                figWidth='half',
                                rotate_img=True,
                                extent=extent_test)
barymap_predtest.initializeFigure()
barymap_predtest.plotFigure()
barymap_predtest.axes.add_patch(next(patches))
barymap_predtest.finalizeFigure(showcb=False)

fignames_predtest = ('b11_periodichill_pred_test_seed' + str(seed),
                     'b12_periodichill_pred_test_seed' + str(seed),
                     'b13_periodichill_pred_test_seed' + str(seed),
                     'b22_periodichill_pred_test_seed' + str(seed),
                     'b23_periodichill_pred_test_seed' + str(seed),
                     'b33_periodichill_pred_test_seed' + str(seed))
fignames_test = ('b11_periodichill_test_seed' + str(seed),
                 'b12_periodichill_test_seed' + str(seed),
                 'b13_periodichill_test_seed' + str(seed),
                 'b22_periodichill_test_seed' + str(seed),
                 'b23_periodichill_test_seed' + str(seed),
                 'b33_periodichill_test_seed' + str(seed))
zlabels = (r'$b_{11} [-]$', r'$b_{12} [-]$', '$b_{13} [-]$', '$b_{22} [-]$', '$b_{23} [-]$', '$b_{33} [-]$')
for i in range(len(zlabels)):
    bij_predtest_plot = Plot2D_Image(val=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xLabel=xlabel,
                                     yLabel=ylabel, zlabel=zlabels[i],
                                     save=save_fig, show=show,
                                     figDir=case.resultPaths[time],
                                     figWidth='1/3',
                                     zlim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test)
    bij_predtest_plot.initializeFigure()
    bij_predtest_plot.plotFigure()
    bij_predtest_plot.axes.add_patch(next(patches))
    bij_predtest_plot.finalizeFigure()

    bij_test_plot = Plot2D_Image(val=y_test_mesh[:, :, i], name=fignames_test[i], xLabel=xlabel,
                                 yLabel=ylabel, zlabel=zlabels[i],
                                 save=save_fig, show=show,
                                 figDir=case.resultPaths[time],
                                 figWidth='1/3',
                                 zlim=bijlims,
                                 rotate_img=True,
                                 extent=extent_test)
    bij_test_plot.initializeFigure()
    bij_test_plot.plotFigure()
    bij_test_plot.axes.add_patch(next(patches))
    bij_test_plot.finalizeFigure()


