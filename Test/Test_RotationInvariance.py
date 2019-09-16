import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from FieldData import FieldData
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor
from Preprocess.Feature import getInvariantFeatureSet
from Utility import interpolateGridData
from Preprocess.FeatureExtraction import splitTrainTestDataList
import time as t
# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

from Utilities import timer
import matplotlib.pyplot as plt
from PlottingTool import BaseFigure, Plot2D, Plot2D_Image
from scipy import ndimage
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import copy
from joblib import load, dump
import os

"""
User Inputs, Anything Can Be Changed Here
"""
# Rotation angle around x, y, z axis
rot_x, rot_y, rot_z = 10, 20, 30  # deg
# Name of the flow case in both RANS and LES
rans_case_name = 'RANS_Re10595'  # str
les_case_name = 'LES_Breuer/Re_10595'  # str
# LES data name to read
les_data_name = 'Hill_Re_10595_Breuer.csv'  # str
# Absolute directory of this flow case
caseDir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# Eddy-viscosity coefficient to convert omega to epsilon via
# epsilon = Cmu*k*omega
cmu = 0.09  # float
# Whether process field data, invariants, features from scratch,
# or use raw field pickle data and process invariants and features
# or use raw field and invariants pickle data and process features
process_raw_field, process_invariants = False, False  # bool
# The following is only relevant if processInvariants is True
if process_invariants:
    # Absolute cap value for Sij and Rij; tensor basis Tij
    cap_sij_rij, cap_tij = 1e9, 1e9  # float/int


"""
Machine Learning Setting
"""
estimator_name = 'tbdt'
fs = 'grad(TKE)_grad(p)'
test_fraction = 0.2
# Seed value for reproducibility
seed = 123
realize_iter = 0
bij_novelty = 'lim'


"""
Plot Settings
"""
# Whether to plot velocity magnitude to verify readFieldData()
plot_u = False  # bool
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
# Number of contour levels in bij's plot
contour_lvl = 50  # int
# Limit for bij plot
bijlims = (-1/2., 2/3.)  # (float, float)
figheight_multiplier = 1.1
# Save anything when possible
save_fields = True  # bool
# Save figures and show figures
save_fig, show = True, False  # bool; bool
if save_fig:
    # Figure extension and DPI
    ext, dpi = 'png', 1000  # str; int



"""
Process User Inputs
"""
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

# Convert rotation angle from degree to radian
rot_x_rad = rot_x/180.*np.pi if rot_x > 2.*np.pi else rot_x
rot_y_rad = rot_y/180.*np.pi if rot_y > 2.*np.pi else rot_y
rot_z_rad = rot_z/180.*np.pi if rot_z > 2.*np.pi else rot_z
# Ensemble name of fields useful for Machine Learning
ml_field_ensemble_name = 'ML_Fields_' + rans_case_name
if estimator_name == "tbdt":
    estimator_name = "TBDT"
elif estimator_name == "tbrf":
    estimator_name = "TBRF"
elif estimator_name == 'tbab':
    estimator_name = 'TBAB'
elif estimator_name in ('TBRC', 'tbrc'):
    estimator_name = 'TBRC'

# Initialize case object
case = FieldData(caseName=rans_case_name, caseDir=caseDir, times=time, fields=fields, save=save_fields)


"""
Read RANS and LES Data and Rotate
"""
# Read raw RANS unrotated flow field
ml_field_ensemble = case.readPickleData(time, ml_field_ensemble_name)
grad_k, k = ml_field_ensemble[:, :3], ml_field_ensemble[:, 3]
epsilon = ml_field_ensemble[:, 4]
grad_u, u = ml_field_ensemble[:, 5:14], ml_field_ensemble[:, 14:17]
grad_p = ml_field_ensemble[:, 17:20]
del ml_field_ensemble
# Read Sij, Rij, Tij, bij
invariants = case.readPickleData(time, fileNames = ('Sij',
                                                    'Rij',
                                                    'Tij',
                                                    'bij_LES'))
sij = invariants['Sij']
rij = invariants['Rij']
tb = invariants['Tij']
bij_les = invariants['bij_LES']
del invariants
cc = case.readPickleData(time, fileNames='cc')
# Rotation matrix around x, y, and z axis
rotij_z = np.array([[np.cos(rot_z_rad), -np.sin(rot_z_rad), 0],
                    [np.sin(rot_z_rad), np.cos(rot_z_rad), 0],
                    [0, 0, 1]])
rotij_y = np.array([[np.cos(rot_y_rad), 0, np.sin(rot_y_rad)],
                    [0, 1, 0],
                    [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)]])
rotij_x = np.array([[1, 0, 0],
                    [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)],
                    [0, np.sin(rot_x_rad), np.cos(rot_x_rad)]])
rotij = rotij_z @ (rotij_y @ rotij_x)

# Make sure Tij is shape (n_samples, n_bases, n_outputs)
tb_t = np.swapaxes(tb, 1, 2) if tb.shape[2] == 10 else tb
# Expand symmetric tensors from 6 components to full 9 components.
# This excl. Rij since it's anti-symmetric
sij_full, tb_t_full, bij_les_full = expandSymmetricTensor(sij), expandSymmetricTensor(tb_t), expandSymmetricTensor(bij_les)
# Reshape to matrix shape
grad_u = grad_u.reshape((-1, 3, 3))
sij_full = sij_full.reshape((sij.shape[0], 3, 3))
rij_full = rij.reshape((rij.shape[0], 3, 3))
tb_t_full = tb_t_full.reshape((tb.shape[0], tb_t.shape[1], 3, 3))
bij_les_full = bij_les_full.reshape((bij_les.shape[0], 3, 3))
# Perform rotation: R*Tensor*R^T and R*Vec
# by going through every sample
u_rot = np.empty_like(u)
grad_u_rot = np.empty_like(grad_u)
grad_k_rot, grad_p_rot = np.empty_like(grad_k), np.empty_like(grad_p)
sij_rot, rij_rot = np.empty_like(sij_full), np.empty_like(rij_full)
tb_t_rot, tb_rot = np.empty_like(tb_t_full), np.empty((tb.shape[0], 6, tb_t.shape[1]))
bij_les_rot = np.empty_like((bij_les_full))
for i in range(len(sij)):
    u_rot[i] = np.dot(rotij, u[i])
    grad_u_rot[i] = rotij @ (grad_u[i] @ rotij.T)
    grad_k_rot[i] = np.dot(rotij, grad_k[i])
    grad_p_rot[i] = np.dot(rotij, grad_p[i])
    sij_rot[i] = rotij @ (sij_full[i] @ rotij.T)
    rij_rot[i] = rotij @ (rij_full[i] @ rotij.T)
    bij_les_rot[i] = rotij @ (bij_les_full[i] @ rotij.T)
    # For Tij, go through every basis too
    for j in range(tb_t.shape[1]):
        tb_t_rot[i, j] = rotij @ (tb_t_full[i, j] @ rotij.T)

# Collapse matrix form from 3 x 3 to 9 x 1
tb_t_rot = tb_t_rot.reshape((tb_t.shape[0], tb_t.shape[1], 9))
bij_les_rot = bij_les_rot.reshape((-1, 9))
# Contract rotated symmetric tensors from 9 full components back to 6 components
tb_t_rot = contractSymmetricTensor(tb_t_rot)
bij_les_rot = contractSymmetricTensor(bij_les_rot)
# Change Tij back to shape (n_samples, n_outputs, n_bases)
tb_rot = np.swapaxes(tb_t_rot, 1, 2)


"""
Calculate Rotated Feature Sets
"""
if fs == 'grad(TKE)':
    fs_data_rot, labels = getInvariantFeatureSet(sij_rot, rij_rot, grad_k_rot, k=k, eps=epsilon)
elif fs == 'grad(p)':
    fs_data_rot, labels = getInvariantFeatureSet(sij_rot, rij_rot, grad_p=grad_p_rot, u=u_rot, grad_u=grad_u_rot)
elif fs == 'grad(TKE)_grad(p)':
    fs_data_rot, labels = getInvariantFeatureSet(sij_rot, rij_rot,
                                                 grad_k=grad_k_rot, grad_p=grad_p_rot, k=k, eps=epsilon,
                                                 u=u_rot,
                                             grad_u=grad_u_rot)


"""
Load Estimator and Predict Rotated Inputs
"""
# Assign rotated inputs x and outputs y
x_rot = fs_data_rot
y_rot = bij_les_rot
list_data_train_rot, list_data_test_rot = splitTrainTestDataList([cc, x_rot, y_rot, tb_rot],
                                                                 test_fraction=test_fraction, seed=seed)

cc_train, cc_test = list_data_train_rot[0], list_data_test_rot[0]
ccx_train, ccy_train, ccz_train = cc_train[:, 0], cc_train[:, 1], cc_train[:, 2]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_train_rot, y_train_rot, tb_train_rot = list_data_train_rot[1:4]
x_test_rot, y_test_rot, tb_test_rot = list_data_test_rot[1:4]
del list_data_test_rot, list_data_train_rot
# Load regressor
regressor = load(case.resultPaths[time] + estimator_name + '.joblib')
# Score rotated input predictions
score_test_rot = regressor.score(x_test_rot, y_test_rot, tb=tb_test_rot)
score_train_rot = regressor.score(x_train_rot, y_train_rot, tb=tb_train_rot)
# Predict rotated inputs
t0 = t.time()
y_pred_test_rot = regressor.predict(x_test_rot, tb=tb_test_rot, bij_novelty=bij_novelty)
y_pred_train_rot = regressor.predict(x_train_rot, tb=tb_train_rot, bij_novelty=bij_novelty)
t1 = t.time()
print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))

# Rotate predictions back to base reference frame.
# From R*f(Tensor)*R^T = f(R*Tensor*R^T),
# y_pred_rot = f(R*Tensor*R^T), we want to get
# y_pred = f(Tensor) = R^(-1)*f(R*Tensor*R^T)*(R^T)^(-1).
# Since rotation tensor R has property R^T = R^(-1),
# y_pred = R^T*f(R*Tesor*R^T)*R^T = R^T*y_pred_rot*R
y_pred_test_rot, y_pred_train_rot = expandSymmetricTensor(y_pred_test_rot), expandSymmetricTensor(y_pred_train_rot)
y_pred_test_rot, y_pred_train_rot = y_pred_test_rot.reshape((-1, 3, 3)), y_pred_train_rot.reshape((-1, 3, 3))
y_pred_test, y_pred_train = np.empty_like(y_pred_test_rot), np.empty_like(y_pred_train_rot)
for i in range(len(y_pred_test_rot)):
    y_pred_test[i] = rotij.T @ (y_pred_test_rot[i] @ rotij)

for i in range(len(y_pred_train_rot)):
    y_pred_train[i] = rotij.T @ (y_pred_train_rot[i] @ rotij)

y_pred_test, y_pred_train = y_pred_test.reshape((-1, 9)), y_pred_train.reshape((-1, 9))
y_pred_test, y_pred_train = contractSymmetricTensor(y_pred_test), contractSymmetricTensor(y_pred_train)


"""
Postprocess Machine Learning Predictions
"""
t0 = t.time()
# _, eigval_test, _ = processReynoldsStress(y_test, make_anisotropic=False, realization_iter=0)
# _, eigval_train, _ = processReynoldsStress(y_train, make_anisotropic=False, realization_iter=0)
y_pred_test3, eigval_pred_test, _ = processReynoldsStress(y_pred_test, make_anisotropic=False, realization_iter=realize_iter)
y_pred_train3, eigval_pred_train, _ = processReynoldsStress(y_pred_train, make_anisotropic=False, realization_iter=realize_iter)
t1 = t.time()
print('\nFinished processing Reynolds stress in {:.4f} s'.format(t1 - t0))

t0 = t.time()
# xy_bary_test, rgb_bary_test = getBarycentricMapData(eigval_test)
# xy_bary_train, rgb_bary_train = getBarycentricMapData(eigval_train)
xy_bary_pred_test, rgb_bary_pred_test = getBarycentricMapData(eigval_pred_test)
xy_bary_pred_train, rgb_bary_pred_train = getBarycentricMapData(eigval_pred_train)
t1 = t.time()
print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))

t0 = t.time()
# ccx_test_mesh, ccy_test_mesh, _, rgb_bary_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
# ccx_train_mesh, ccy_train_mesh, _, rgb_bary_train_mesh = interpolateGridData(ccx_train, ccy_train, rgb_bary_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
ccx_test_mesh, ccy_test_mesh, _, rgb_bary_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
ccx_train_mesh, ccy_train_mesh, _, rgb_bary_pred_train_mesh = interpolateGridData(ccx_train, ccy_train, rgb_bary_pred_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for barycentric map in {:.4f} s'.format(t1 - t0))

t0 = t.time()
_, _, _, y_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, y_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_pred_train_mesh = interpolateGridData(ccx_train, ccy_train, y_pred_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for bij in {:.4f} s'.format(t1 - t0))


"""
Plotting
"""
figdir = case.resultPaths[time] + 'RotationInvariance/' + estimator_name
os.makedirs(figdir, exist_ok=True)
# rgb_b2ry_train_mesh = ndimage.rotate(rgb_bary_train_mesh, 90)
rgb_bary_pred_test_mesh = ndimage.rotate(rgb_bary_pred_test_mesh, 90)
rgb_bary_pred_train_mesh = ndimage.rotate(rgb_bary_pred_train_mesh, 90)
xlabel, ylabel = ('$\bm{S}^2$', r'$y$ [m]')
geometry = np.genfromtxt(caseDir + '/' + rans_case_name + '/'  + "geometry.csv", delimiter=",")[:, :2]
figname = 'barycentric_predtest_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z)
bary_map = BaseFigure((None,), (None,), name=figname, xlabel=xlabel,
                      ylabel=ylabel, save=save_fig, show=show,
                      figdir=case.resultPaths[time],
                      figheight_multiplier=0.7)
path = Path(geometry)
patch = PathPatch(path, linewidth=0., facecolor=bary_map.gray)
# patch is considered "a single artist" so have to make copy to use more than once
patches = []
for _ in range(14):
    patches.append(copy(patch))

patches = iter(patches)
extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
extent_train = (ccx_train.min(), ccx_train.max(), ccy_train.min(), ccy_train.max())
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(figdir + '/' + figname + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_predtrain_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_train_mesh, origin='upper', aspect='equal', extent=extent_train)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(figdir + '/' + bary_map.name + '.' + ext, dpi=dpi)

plt.close()

fignames_predtest = ('b11_predtest_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                     'b12_predtest_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                     'b13_predtest_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                     'b22_predtest_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                     'b23_predtest_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                     'b33_predtest_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z))
fignames_predtrain = ('b11_predtrain_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                      'b12_predtrain_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                      'b13_predtrain_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                      'b22_predtrain_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                      'b23_predtrain_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z),
                      'b33_predtrain_rot_{0}_{1}_{2}'.format(rot_x, rot_y, rot_z))
zlabels = ('$b_{11}$ [-]', '$b_{12}$ [-]', '$b_{13}$ [-]', '$b_{22}$ [-]', '$b_{23}$ [-]', '$b_{33}$ [-]')
zlabels_pred = ('$\hat{b}_{11}$ [-]', '$\hat{b}_{12}$ [-]', '$\hat{b}_{13}$ [-]', '$\hat{b}_{22}$ [-]', '$\hat{b}_{23}$ [-]', '$\hat{b}_{33}$ [-]')
# Go through eveyr bij component and plot
for i in range(len(zlabels)):
    bij_predtest_plot = Plot2D_Image(val=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xlabel=xlabel,
                                     ylabel=ylabel, val_label=zlabels_pred[i],
                                     save=save_fig, show=show,
                                     figdir=figdir,
                                     figwidth='1/3',
                                     val_lim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test,
                                     figheight_multiplier=figheight_multiplier)
    bij_predtest_plot.initializeFigure()
    bij_predtest_plot.plotFigure()
    bij_predtest_plot.axes.add_patch(next(patches))
    bij_predtest_plot.finalizeFigure()

    bij_predtrain_plot = Plot2D_Image(val=y_pred_train_mesh[:, :, i], name=fignames_predtrain[i], xlabel=xlabel,
                                      ylabel=ylabel, val_label=zlabels_pred[i],
                                      save=save_fig, show=show,
                                      figdir=figdir,
                                      figwidth='1/3',
                                      val_lim=bijlims,
                                      rotate_img=True,
                                      extent=extent_test,
                                      figheight_multiplier=figheight_multiplier)
    bij_predtrain_plot.initializeFigure()
    bij_predtrain_plot.plotFigure()
    bij_predtrain_plot.axes.add_patch(next(patches))
    bij_predtrain_plot.finalizeFigure()
