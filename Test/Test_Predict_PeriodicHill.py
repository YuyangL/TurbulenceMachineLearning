import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
sys.path.append('/home/yluan/Documents/ML/TBRF_MK')
from tbdt_v8 import TBDT
from tbrf_v4 import TBRF
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
from scipy.interpolate import griddata

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both RANS and LES
# Only Re = 700 is DNS, the rest is LES
re = 5600  # 700, 5600, 10595
rans_case_name = 'RANS_Re' + str(re)  # str
les_case_name = 'LES_Breuer/Re_' + str(re)  # str
# LES data name to read
les_data_name = 'Hill_Re_' + str(re) + '_Breuer.csv'  # str
# Absolute directory of this flow case
casedir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'
seed = 123  # int
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# The case folder name storing the estimator
# estimator_folder = 'TBAB/16_{50_0.002_2_0_10}_4cv_ls_0.1rate'  #"TBRF_16t_auto_p{47_16_4_0_0_50}_s123_boots_0iter"
estimator_folder = 'TBDT/Kaandorp_{50_0.002_0.001_0}'
estimator_dir = "RANS_Re10595"
# Feature set number
fs = 'grad(TKE)_grad(p)+'  # '1', '12', '123'
realize_iter = 2  # int
bij_novelty = None  # 'excl', None


"""
Plot Settings
"""
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 2e5  # int
# Limit for bij plot
bijlims = (-1/2., 2/3.)  # (float, float)
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
elif 'TBAB' in estimator_folder or 'tbab' in estimator_folder:
    estimator_name = 'TBAB'
elif 'TBGB' in estimator_folder or 'tbgb' in estimator_folder:
    estimator_name = 'TBGB'

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
case = FieldData(casename=rans_case_name, casedir=casedir, times=time, fields=fields, save=False)


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

# Load rotated and/or confined field data useful for Machine Learning
ml_field_ensemble = case.readPickleData(time, ml_field_ensemble_name)
grad_k, k = ml_field_ensemble[:, :3], ml_field_ensemble[:, 3]
epsilon = ml_field_ensemble[:, 4]
grad_u, u = ml_field_ensemble[:, 5:14], ml_field_ensemble[:, 14:17]
grad_p = ml_field_ensemble[:, 17:20]
invariants = case.readPickleData(time, filenames=('Sij',
                                                    'Rij',
                                                    'Tij',
                                                    'bij_LES'))
sij = invariants['Sij']
rij = invariants['Rij']
tb = invariants['Tij']
bij_les = invariants['bij_LES']
uuprime2_les = case.readPickleData(time, 'uuPrime2_LES')[0]
cc = case.readPickleData(time, filenames='CC')
fs_data = case.readPickleData(time, filenames=('FS_' + fs))
# Also read LES TKE production G related fields
g_les = case.readPickleData(time, 'G_LES')
gradu_les = case.readPickleData(time, 'grad(U)_LES')
k_les = case.readPickleData(time, 'k_LES')
div_uuprime2_les = case.readPickleData(time, 'div(uuPrime2)_LES')

list_data_test = case.readPickleData(time, 'list_data_test_seed' + str(seed))
cc_test = list_data_test[0]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_test, y_test, tb_test = list_data_test[1:4]
# Contract symmetric tensors to 6 components in case of 9 components
y_test = contractSymmetricTensor(y_test)
tb_test = transposeTensorBasis(tb_test)
tb_test = contractSymmetricTensor(tb_test)
tb_test = transposeTensorBasis(tb_test)

# Sort g_les, gradu_les, k_les that were based on cc to be based on cc_test by "interpolation"
g_les = griddata(cc[:, :2], g_les, cc_test[:, :2], method=interp_method)
# k_*_sorted are based on cc that is a sorted RANS grid coor array
k_les_sorted, k_sorted = k_les.copy(), k.copy()
k_les = griddata(cc[:, :2], k_les, cc_test[:, :2], method=interp_method)
k = griddata(cc[:, :2], k, cc_test[:, :2], method=interp_method)
for i in range(9):
    gradu_les[:, i] = griddata(cc[:, :2], gradu_les[:, i], cc_test[:, :2], method=interp_method)
    grad_u[:, i] = griddata(cc[:, :2], grad_u[:, i], cc_test[:, :2], method=interp_method)


print('\nLoading regressor... ')
regressor = load(estimator_fullpath + estimator_name + '.joblib')


"""
Predict
"""
t0 = t.time()
# score_test = regressor.score(x_test, y_test, tb=tb_test)
if 'Kaandorp' in estimator_fullpath:
    x_test_copy = np.swapaxes(x_test, 0, 1)
    tb_test_copy = np.swapaxes(tb_test, 1, 2)
    tb_test_copy = expandSymmetricTensor(tb_test_copy)
    tb_test_copy = np.swapaxes(tb_test_copy, 0, 2)
    tree_struct = regressor.copy()
    if 'TBDT' in estimator_name:
        regressor = TBDT(tree_filename='TBDT_Kaandorp', regularization=False, splitting_features='all',
                         regularization_lambda=0., optim_split=True, optim_threshold=2,
                         min_samples_leaf=24)
    elif 'TBRF' in estimator_name:
        regressor = TBRF(min_samples_leaf=24, tree_filename='TBRF_Kaandorp_%i', n_trees=8,
                         regularization=False,
                         regularization_lambda=0., splitting_features='all',
                         optim_split=True, optim_threshold=2, read_from_file=False)

    # Take median
    if 'TBRF' in estimator_name:
        _, y_pred_test, _ = regressor.predict(x_test_copy, tb_test_copy, tree_struct)
        y_pred_test = np.median(y_pred_test, axis=2)
    else:
        y_pred_test, _ = regressor.predict(x_test_copy, tb_test_copy, tree_struct)

    y_pred_test = np.swapaxes(y_pred_test, 0, 1)
else:
    y_pred_test = regressor.predict(x_test, tb=tb_test, bij_novelty=bij_novelty)
t1 = t.time()
print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))



"""
Calculate Turbulent Shear Stress Gradient -div(ui'uj')
"""
@njit(parallel=True, fastmath=True)
def calcSymmTensorDivergence2D(tensor_mesh, ccx_mesh, ccy_mesh):
    drdx = np.empty_like(tensor_mesh)
    drdy = np.empty_like(tensor_mesh)
    for i in prange(tensor_mesh.shape[2]):
        # First dRij/dx
        # Upwind diff for first cell
        drdx[:, 0, i] = (tensor_mesh[:, 1, i] - tensor_mesh[:, 0, i])/(
                ccx_mesh[:, 1] - ccx_mesh[:, 0])
        # Central diff for inner cells
        drdx[:, 1:-1, i] = (tensor_mesh[:, 2:, i] - tensor_mesh[:, :-2, i])/(
                ccx_mesh[:, 2:] - ccx_mesh[:, :-2])
        # Downwind diff for last cell
        drdx[:, -1, i] = (tensor_mesh[:, -1, i] - tensor_mesh[:, -2, i])/(
                ccx_mesh[:, -1] - ccx_mesh[:, -2])

        # Repeat for dRij/dy
        drdy[0, :, i] = (tensor_mesh[1, :, i] - tensor_mesh[0, :, i])/(
                ccy_mesh[1, :] - ccy_mesh[0, :])
        drdy[1:-1, :, i] = (tensor_mesh[2:, :, i] - tensor_mesh[:-2, :, i])/(
                ccy_mesh[2:, :] - ccy_mesh[:-2, :])
        drdy[-1, :, i] = (tensor_mesh[-1, :, i] - tensor_mesh[-2, :, i])/(
                ccy_mesh[-1, :] - ccy_mesh[-2, :])

    # Now perform the divergence
    tensor_div_mesh = np.empty((tensor_mesh.shape[0], tensor_mesh.shape[1], 3))
    # 1st row: dR11/dx + dR12/dy + dR13/dz
    # Assuming no component vary with z
    tensor_div_mesh[..., 0] = drdx[..., 0] + drdy[..., 1] + 0.
    # 2nd row: dR12/dx + dR22/dy + dR23/dz
    tensor_div_mesh[..., 1] = drdx[..., 1] + drdy[..., 3] + 0.
    # 3rd row: dR13/dx + dR23/dy + dR33/dz
    tensor_div_mesh[..., 2] = drdx[..., 2] + drdy[..., 4] + 0.
    return tensor_div_mesh

# Sort predicted bij according to cc so that div(ui'uj') can be calculated
y_pred_sorted = np.empty((len(cc), 6))
for i in range(6):
    y_pred_sorted[:, i] = griddata(cc_test[:, :2], y_pred_test[:, i], cc[:, :2])

for i in range(1, len(cc)):
    if cc[i, 0] < cc[i - 1, 0]:
        nx = i
        break

ccx_mesh = cc[:, 0].reshape((-1, nx))
ccy_mesh = cc[:, 1].reshape((-1, nx))
y_pred_sorted_mesh = y_pred_sorted.reshape((-1, nx, 6))
k_sorted_mesh = k_sorted.reshape((-1, nx))
k_les_sorted_mesh = k_les_sorted.reshape((-1, nx))
# Calculate Rij = ui'uj' = 2/3k*I + 2k*bij
r_sorted_mesh = np.empty_like(y_pred_sorted_mesh)
r_les_sorted_mesh = np.empty_like(y_pred_sorted_mesh)
for i in range(6):
    # Rij based on RANS k
    r_sorted_mesh[..., i] = 2/3.*k_sorted_mesh + 2.*k_sorted_mesh*y_pred_sorted_mesh[..., i] if i in (0, 3, 5) else 2.*k_sorted_mesh*y_pred_sorted_mesh[..., i]
    # Rij based on DNS/LES k
    r_les_sorted_mesh[..., i] = 2/3.*k_les_sorted_mesh + 2.*k_les_sorted_mesh*y_pred_sorted_mesh[..., i] if i in (
    0, 3, 5) else 2.*k_les_sorted_mesh*y_pred_sorted_mesh[..., i]

# Compute divergence, don't forget - sign at front
div_r_sorted_mesh = -calcSymmTensorDivergence2D(r_sorted_mesh, ccx_mesh, ccy_mesh)
div_r_les_sorted_mesh = -calcSymmTensorDivergence2D(r_les_sorted_mesh, ccx_mesh, ccy_mesh)
# Lastly, interpolate the result to cc_test, component by component
div_r = np.empty((len(cc_test), 6))
div_r_les = np.empty((len(cc_test), 6))
for i in range(3):
    # This is predicted bij with RANS k
    div_r[:, i] = griddata(cc[:, :2], div_r_sorted_mesh[..., i].ravel(), cc_test[:, :2])
    # This is predicted bij with DNS/LES k
    div_r_les[:, i] = griddata(cc[:, :2], div_r_les_sorted_mesh[..., i].ravel(), cc_test[:, :2])
    # This is ground truth
    div_uuprime2_les[:, i] = griddata(cc[:, :2], div_uuprime2_les[:, i], cc_test[:, :2])


"""
Calculate TKE Production G 
"""
y_pred_full = expandSymmetricTensor(y_pred_test).reshape((len(cc_test), 3, 3))
g_pred_les = np.empty(len(cc_test))
g_pred = np.empty(len(cc_test))
for i in range(len(cc_test)):
    g_pred_les[i] = -2.*k_les[i]*np.tensordot(y_pred_full[i], gradu_les[i].reshape((3, 3)))
    g_pred[i] = -2.*k[i]*np.tensordot(y_pred_full[i], grad_u[i].reshape((3, 3)))


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

# Interpolate truth and predicted xy_bary to meshgrid
t0 = t.time()
_, _, _, xy_bary_test_mesh = interpolateGridData(ccx_test, ccy_test, xy_bary_test, mesh_target=uniform_mesh_size, interp=interp_method)
_, _, _, xy_bary_pred_mesh = interpolateGridData(ccx_test, ccy_test, xy_bary_pred_test, mesh_target=uniform_mesh_size, interp=interp_method)
t1 = t.time()
print('\nFinished interpolating mesh data for barycentric x, y coordinates in {:.4f} s'.format(t1 - t0))

t0 = t.time()
_, _, _, y_test_mesh = interpolateGridData(ccx_test, ccy_test, y_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, y_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for bij in {:.4f} s'.format(t1 - t0))

# Interpolate truth and predicted G to meshgrid
t0 = t.time()
_, _, _, g_les_mesh = interpolateGridData(ccx_test, ccy_test, g_les, mesh_target=uniform_mesh_size, interp=interp_method)
_, _, _, g_pred_les_mesh = interpolateGridData(ccx_test, ccy_test, g_pred_les, mesh_target=uniform_mesh_size, interp=interp_method)
_, _, _, g_pred_mesh = interpolateGridData(ccx_test, ccy_test, g_pred, mesh_target=uniform_mesh_size, interp=interp_method)
t1 = t.time()
print('\nFinished interpolating mesh data for TKE production G in {:.4f} s'.format(t1 - t0))

# Interpolate truth and predicted -div(ui'uj') to meshgrid
t0 = t.time()
_, _, _, divr_les_mesh = interpolateGridData(ccx_test, ccy_test, div_uuprime2_les, mesh_target=uniform_mesh_size, interp=interp_method)
_, _, _, divr_pred_les_mesh = interpolateGridData(ccx_test, ccy_test, div_r_les, mesh_target=uniform_mesh_size, interp=interp_method)
_, _, _, divr_pred_mesh = interpolateGridData(ccx_test, ccy_test, div_r, mesh_target=uniform_mesh_size, interp=interp_method)
t1 = t.time()
print('\nFinished interpolating mesh data for turbulent shear stress gradient -div(uiuj) in {:.4f} s'.format(t1 - t0))


"""
Plot Barycentric Map Image
"""
figname = 'barycentric_periodichill_test_seed' + str(seed)
xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
barymap_test = Plot2D_Image(val=rgb_bary_test_mesh[:, :], name=figname, xlabel=xlabel,
                            ylabel=ylabel,
                            save=save_fig, show=show,
                            figdir=case.result_paths[time],
                            figwidth='half',
                            rotate_img=True,
                            extent=extent_test)
geometry = np.genfromtxt(casedir + '/' + rans_case_name + '/' + "geometry.csv", delimiter=",")[:, :2]
path = Path(geometry)
patch = PathPatch(path, linewidth=0., facecolor=barymap_test.gray)
# patch is considered "a single artist" so have to make copy to use more than once
patches = []
for _ in range(30):
    patches.append(copy(patch))

patches = iter(patches)
barymap_test.initializeFigure()
barymap_test.plotFigure()
barymap_test.axes.add_patch(next(patches))
barymap_test.finalizeFigure(showcb=False)

figname = 'barycentric_periodichill_pred_test_seed' + str(seed)
barymap_predtest = Plot2D_Image(val=rgb_bary_pred_test_mesh, name=figname, xlabel=xlabel,
                                ylabel=ylabel,
                                save=save_fig, show=show,
                                figdir=case.result_paths[time],
                                figwidth='half',
                                rotate_img=True,
                                extent=extent_test)
barymap_predtest.initializeFigure()
barymap_predtest.plotFigure()
barymap_predtest.axes.add_patch(next(patches))
barymap_predtest.finalizeFigure(showcb=False)


"""
Plot Barycentric Triangle Along 3 Vertical Lines
"""
@njit
def findNearest(arr, val):
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return arr[idx]

# Make the triangle shape for plotting
triverts = [(0., 0.), (1., 0.), (0.5, np.sqrt(3)/2.), (0., 0.)]
tripath = Path(triverts)
tripatch = PathPatch(tripath, fill=False, edgecolor=(89/255.,)*3, ls='-', zorder=-1)
tripatches = []
for i in range(100):
    tripatches.append(copy(tripatch))

tripatches = iter(tripatches)

list_g_les_v = []
list_g_pred_v = []
list_g_pred_les_v = []

list_divr_les_v = []
list_divr_pred_les_v = []
list_divr_pred_v = []

list_ccy_v = []
# 3 vertical lines of interest, namely y = 1, y = 4.5, y = 8
xloc = (1., 4.5, 8.)
ccx_v1 = findNearest(ccx_test_mesh.ravel(), xloc[0])
ccx_v4_5 = findNearest(ccx_test_mesh.ravel(), xloc[1])
ccx_v8 = findNearest(ccx_test_mesh.ravel(), xloc[2])
ccx_v_all = (ccx_v1, ccx_v4_5, ccx_v8)

gmax = -1e9
gmin = 1e9
divr_min = 1e9
divr_max = -1e9
for i in range(3):
    # ccx_v_idx = np.where((ccx_test > vline_bnd[i][0]) & (ccx_test < vline_bnd[i][1]))
    # ccx_v = ccx_test[ccx_v_idx]
    # ccy_v = ccy_test[ccx_v_idx]
    # x_test_v = x_test[ccx_v_idx]
    # y_test_v = y_test[ccx_v_idx]
    # tb_test_v = tb_test[ccx_v_idx]
    ccx_v = ccx_test_mesh[ccx_test_mesh == ccx_v_all[i]]
    ccy_v = ccy_test_mesh[ccx_test_mesh == ccx_v_all[i]]
    # Find corresponding xy bary too
    xy_bary_test_v = xy_bary_test_mesh[ccx_test_mesh == ccx_v_all[i]]
    xy_bary_pred_v = xy_bary_pred_mesh[ccx_test_mesh == ccx_v_all[i]]
    # Sort ccx, ccy, xy bary according to ccy
    ccy_v_sortidx = np.argsort(ccy_v)
    ccx_v = ccx_v[ccy_v_sortidx]
    ccy_v = ccy_v[ccy_v_sortidx]
    xy_bary_test_v = xy_bary_test_v[ccy_v_sortidx]
    xy_bary_pred_v = xy_bary_pred_v[ccy_v_sortidx]
    # If y = 1 or 8, then y should > ~0.5
    if i in (0, 1, 2):
        y_idx = np.where((ccy_v > .5) & (ccy_v < 3.))[0]
        xy_bary_test_v = xy_bary_test_v[y_idx]
        xy_bary_pred_v = xy_bary_pred_v[y_idx]
        list_ccy_v.append(ccy_v[y_idx])

    # Limit predicted barycentric x, y coor
    xy_bary_pred_v[xy_bary_pred_v < -0.1] = -0.1
    xy_bary_pred_v[xy_bary_pred_v > 1.] = 1.

    # Plot Barycentric triangle
    list_x = (xy_bary_test_v[:, 0], xy_bary_pred_v[:, 0])
    list_y = (xy_bary_test_v[:, 1], xy_bary_pred_v[:, 1])
    figname = 'triangle{}'.format(i)
    barymap = Plot2D(list_x, list_y, name=figname, xlabel=None, ylabel=None,
                     save=save_fig, show=show, figwidth='1/3', figheight_multiplier=1,
                     figdir=case.result_paths[time], equalaxis=True)
    barymap.initializeFigure()
    barymap.axes.add_patch(next(tripatches))
    barymap.plotFigure(linelabel=('Truth', 'Prediction'), showmarker=True)
    plt.axis('off')
    barymap.axes.annotate(r'$\textbf{X}_{2c}$', (0., 0.), (-0.15, 0.))
    barymap.axes.annotate(r'$\textbf{X}_{3c}$', (0.5, np.sqrt(3)/2.))
    barymap.axes.annotate(r'$\textbf{X}_{1c}$', (1., 0.))
    barymap.finalizeFigure(legloc='upper left')

    # Retrieve TKE Production G in Vertical Lines
    # Find corresponding G at the 3 vertical lines as above
    g_les_v = g_les_mesh[ccx_test_mesh == ccx_v_all[i]]
    g_pred_les_v = g_pred_les_mesh[ccx_test_mesh == ccx_v_all[i]]
    g_pred_v = g_pred_mesh[ccx_test_mesh == ccx_v_all[i]]
    # Sort
    g_les_v = g_les_v[ccy_v_sortidx]
    g_pred_les_v = g_pred_les_v[ccy_v_sortidx]
    g_pred_v = g_pred_v[ccy_v_sortidx]
    # We're only interested in y between (0.5, 2.5)
    g_les_v = g_les_v[y_idx]
    g_pred_les_v = g_pred_les_v[y_idx]
    g_pred_v = g_pred_v[y_idx]
    # Find limit and store array
    if gmin > min(g_les_v): gmin = min(g_les_v)
    if gmax < max(g_les_v): gmax = max(g_les_v)
    list_g_les_v.append(g_les_v)
    list_g_pred_v.append(g_pred_v)
    list_g_pred_les_v.append(g_pred_les_v)

    # Repeat again for turbulent shear stress gradient -div(ui'uj')
    divr_les_v = divr_les_mesh[ccx_test_mesh == ccx_v_all[i]]
    divr_pred_les_v = divr_pred_les_mesh[ccx_test_mesh == ccx_v_all[i]]
    divr_pred_v = divr_pred_mesh[ccx_test_mesh == ccx_v_all[i]]
    # Sort
    divr_les_v = divr_les_v[ccy_v_sortidx]
    divr_pred_les_v = divr_pred_les_v[ccy_v_sortidx]
    divr_pred_v = divr_pred_v[ccy_v_sortidx]
    # Select y range in [0.5, inf]
    divr_les_v = divr_les_v[y_idx]
    divr_pred_les_v = divr_pred_les_v[y_idx]
    divr_pred_v = divr_pred_v[y_idx]
    # Find limit and store array
    if divr_min > min(divr_les_v[:, :2].ravel()): divr_min = min(divr_les_v[:, :2].ravel())
    if divr_max < max(divr_les_v[:, :2].ravel()): divr_max = max(divr_les_v[:, :2].ravel())
    list_divr_les_v.append(divr_les_v)
    list_divr_pred_les_v.append(divr_pred_les_v)
    list_divr_pred_v.append(divr_pred_v)





"""
Plot TKE production G Along 3 Vertical Lines in 1 Plot
"""
g_range_multiplier = 2.
g_range = (gmax - gmin)*g_range_multiplier
# Max available x span for a g line plot is 4 [m], thus rescale G to an x range
g_multiplier = 4./g_range
for i in range(3):
    # Limit if prediction is too off
    list_g_pred_v[i][list_g_pred_v[i] < gmin*g_range_multiplier] = gmin*g_range_multiplier
    list_g_pred_v[i][list_g_pred_v[i] > gmax*g_range_multiplier] = gmax*g_range_multiplier
    list_g_pred_les_v[i][list_g_pred_les_v[i] < gmin*g_range_multiplier] = gmin*g_range_multiplier
    list_g_pred_les_v[i][list_g_pred_les_v[i] > gmax*g_range_multiplier] = gmax*g_range_multiplier

    list_g_les_v[i] *= g_multiplier
    list_g_pred_v[i] *= g_multiplier
    list_g_pred_les_v[i] *= g_multiplier
    # Shift them to their respective x location
    list_g_les_v[i] += xloc[i]
    list_g_pred_les_v[i] += xloc[i]
    list_g_pred_v[i] += xloc[i]

figname = 'g_seed' + str(seed)
xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
gplot = Plot2D_Image(val=rgb_bary_test_mesh, name=figname, xlabel=xlabel,
                            ylabel=ylabel,
                            save=save_fig, show=show,
                            figdir=case.result_paths[time],
                            figwidth='half',
                            rotate_img=True,
                            extent=extent_test, alpha=0.25,
                     xlim=(0, 9))
gplot.initializeFigure()
gplot.plotFigure()
# Only Re = 700 is DNS, the rest is LES
truthtype = 'DNS' if re == 700 else 'LES'
for i in range(3):
    label0 = truthtype if i == 0 else None
    label1 = 'RANS prediction' if i == 0 else None
    label2 = truthtype + ' prediction' if i == 0 else None
    gplot.axes.plot(list_g_les_v[i], list_ccy_v[i], label=label0, color=gplot.colors[0], alpha=0.75, ls=':')
    gplot.axes.plot(list_g_pred_v[i], list_ccy_v[i], label=label1, color=gplot.colors[1], alpha=0.75, ls='-')
    gplot.axes.plot(list_g_pred_les_v[i], list_ccy_v[i], label=label2, color=gplot.colors[2], alpha=0.75, ls='-.')
    gplot.axes.plot(list_g_pred_v[i]*0 + xloc[i], list_ccy_v[i], color=gplot.gray, alpha=1, ls='--', zorder=-10)

gplot.axes.add_patch(next(patches))
gplot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
gplot.finalizeFigure(showcb=False)



"""
Plot Turbulent Shear Stress Gradient Along 3 Vertical Lines
"""
divr_range_multiplier = 2.
divr_range = (divr_max + abs(divr_min))*divr_range_multiplier
# Max available x span for a div(ui'uj') line plot is 4 [m], thus rescale it to an x range
divr_multiplier = 4./divr_range
# Outer loop is the div component, only plotting u and v component
for i0 in range(2):
    # Inner loop is 3 vertical lines for each div component
    for i in range(3):
        # Limit if prediction is too off
        list_divr_pred_v[i][:, i0][list_divr_pred_v[i][:, i0] < divr_min*divr_range_multiplier] = divr_min*divr_range_multiplier
        list_divr_pred_v[i][:, i0][list_divr_pred_v[i][:, i0] > divr_max*divr_range_multiplier] = divr_max*divr_range_multiplier
        list_divr_pred_les_v[i][:, i0][list_divr_pred_les_v[i][:, i0] < divr_min*divr_range_multiplier] = divr_min*divr_range_multiplier
        list_divr_pred_les_v[i][:, i0][list_divr_pred_les_v[i][:, i0] > divr_max*divr_range_multiplier] = divr_max*divr_range_multiplier

        list_divr_les_v[i][:, i0] *= divr_multiplier
        list_divr_pred_v[i][:, i0] *= divr_multiplier
        list_divr_pred_les_v[i][:, i0] *= divr_multiplier
        # Shift them to their respective x location
        list_divr_les_v[i][:, i0] += xloc[i]
        list_divr_pred_les_v[i][:, i0] += xloc[i]
        list_divr_pred_v[i][:, i0] += xloc[i]

    figname = 'div(uiuj)_' + str(i0)
    xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
    divr_plot = Plot2D_Image(val=rgb_bary_test_mesh, name=figname, xlabel=xlabel,
                         ylabel=ylabel,
                         save=save_fig, show=show,
                         figdir=case.result_paths[time],
                         figwidth='half',
                         rotate_img=True,
                         extent=extent_test, alpha=0.25,
                         xlim=(0, 9))
    divr_plot.initializeFigure()
    divr_plot.plotFigure()
    # Only Re = 700 is DNS, the rest is LES
    truthtype = 'DNS' if re == 700 else 'LES'
    for i in range(3):
        label0 = truthtype if i == 0 else None
        label1 = 'RANS prediction' if i == 0 else None
        label2 = truthtype + ' prediction' if i == 0 else None
        divr_plot.axes.plot(list_divr_les_v[i][:, i0], list_ccy_v[i], label=label0, color=divr_plot.colors[0], alpha=0.75, ls=':')
        divr_plot.axes.plot(list_divr_pred_v[i][:, i0], list_ccy_v[i], label=label1, color=divr_plot.colors[1], alpha=0.75, ls='-')
        divr_plot.axes.plot(list_divr_pred_les_v[i][:, i0], list_ccy_v[i], label=label2, color=divr_plot.colors[2], alpha=0.75, ls='-.')
        divr_plot.axes.plot(list_divr_pred_v[i][:, i0]*0 + xloc[i], list_ccy_v[i], color=divr_plot.gray, alpha=1, ls='--', zorder=-10)

    divr_plot.axes.add_patch(next(patches))
    divr_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
    divr_plot.finalizeFigure(showcb=False)



# fignames_predtest = ('b11_periodichill_pred_test_seed' + str(seed),
#                      'b12_periodichill_pred_test_seed' + str(seed),
#                      'b13_periodichill_pred_test_seed' + str(seed),
#                      'b22_periodichill_pred_test_seed' + str(seed),
#                      'b23_periodichill_pred_test_seed' + str(seed),
#                      'b33_periodichill_pred_test_seed' + str(seed))
# fignames_test = ('b11_periodichill_test_seed' + str(seed),
#                  'b12_periodichill_test_seed' + str(seed),
#                  'b13_periodichill_test_seed' + str(seed),
#                  'b22_periodichill_test_seed' + str(seed),
#                  'b23_periodichill_test_seed' + str(seed),
#                  'b33_periodichill_test_seed' + str(seed))
# zlabels = (r'$b_{11} [-]$', r'$b_{12} [-]$', '$b_{13} [-]$', '$b_{22} [-]$', '$b_{23} [-]$', '$b_{33} [-]$')
# for i in range(len(zlabels)):
#     bij_predtest_plot = Plot2D_Image(val=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xlabel=xlabel,
#                                      ylabel=ylabel, zlabel=zlabels[i],
#                                      save=save_fig, show=show,
#                                      figdir=case.result_paths[time],
#                                      figwidth='1/3',
#                                      zlim=bijlims,
#                                      rotate_img=True,
#                                      extent=extent_test)
#     bij_predtest_plot.initializeFigure()
#     bij_predtest_plot.plotFigure()
#     bij_predtest_plot.axes.add_patch(next(patches))
#     bij_predtest_plot.finalizeFigure()
# 
#     bij_test_plot = Plot2D_Image(val=y_test_mesh[:, :, i], name=fignames_test[i], xlabel=xlabel,
#                                  ylabel=ylabel, zlabel=zlabels[i],
#                                  save=save_fig, show=show,
#                                  figdir=case.result_paths[time],
#                                  figwidth='1/3',
#                                  zlim=bijlims,
#                                  rotate_img=True,
#                                  extent=extent_test)
#     bij_test_plot.initializeFigure()
#     bij_test_plot.plotFigure()
#     bij_test_plot.axes.add_patch(next(patches))
#     bij_test_plot.finalizeFigure()


