import numpy as np
import sys
# See https://github.com/mikaelk/TBRF
sys.path.append('/home/yluan/Documents/ML/TBRF_MK')
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from FieldData import FieldData
from tbdt_v8 import TBDT
from tbrf_v4 import TBRF
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor
from Utility import interpolateGridData
import matplotlib.pyplot as plt
from PlottingTool import BaseFigure, Plot2D_Image
from scipy import ndimage
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import copy
import time as t
from joblib import dump
from sklearn.metrics import r2_score
from math import ceil


"""
User Inputs, Anything Can Be Changed Here
"""
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
# Kinematic viscosity for Re = 10595
nu = 9.438414346389807e-05  # float
seed = 123  # Seed for data


"""
Machine Learning Settings
"""
estimator_name = 'tbrf'  # 'tbdt', 'tbrf'
n_estimators = 8  # int
regularization = False  # bool
regularization_lambda = 0.  # float
write_g = False  # Whether write tensor basis coefficient fields
splitting_features = 'all'  # 'all', 'sqrt', 'div3'
min_samples_leaf = 'default'  # 'default', int
realize_iter = 0  # int


"""
Plot Settings
"""
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
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
Process User Inputs
"""
# Average fields of interest for reading and processing
fields = ('U', 'k', 'p', 'omega',
          'grad_U', 'grad_k', 'grad_p')
# Ensemble name of fields useful for Machine Learning
ml_field_ensemble_name = 'ML_Fields_' + rans_case_name
# Initialize case object
case = FieldData(caseName=rans_case_name, caseDir=caseDir, times=time, fields=fields, save=save_fields)
if estimator_name == "tbdt":
    estimator_name = "TBDT_Kaandorp"
elif estimator_name == "tbrf":
    estimator_name = "TBRF_Kaandorp"


"""
Read Data
"""
list_data_train = case.readPickleData(time, 'list_data_train_seed' + str(seed))
list_data_test = case.readPickleData(time, 'list_data_test_seed' + str(seed))
cc_train, cc_test = list_data_train[0], list_data_test[0]
ccx_train, ccy_train, ccz_train = cc_train[:, 0], cc_train[:, 1], cc_train[:, 2]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_train, y_train, tb_train = list_data_train[1:4]
x_test, y_test, tb_test = list_data_test[1:4]

# Expand symmetric tensors to full size
y_train = expandSymmetricTensor(y_train)
y_test = expandSymmetricTensor(y_test)
# Swap n_bases and n_outputs axes as expandSymmetricTensor requires Tij of shape (n_samples, n_bases, n_outputs)
tb_train = np.swapaxes(tb_train, 1, 2)
tb_test = np.swapaxes(tb_test, 1, 2)
tb_train = expandSymmetricTensor(tb_train)
tb_test = expandSymmetricTensor(tb_test)
# Since TBDT and TBRF from M. Kaandorp requires X of shape (n_features, n_samples),
# y of shape (9, n_samples), and Tij of shape (9, n_bases, n_samples)
y_train = np.swapaxes(y_train, 0, 1)
y_test = np.swapaxes(y_test, 0, 1)
x_train = np.swapaxes(x_train, 0, 1)
x_test = np.swapaxes(x_test, 0, 1)
tb_train = np.swapaxes(tb_train, 0, 2)
tb_test = np.swapaxes(tb_test, 0, 2)
if min_samples_leaf == 'default': min_samples_leaf = ceil(0.001*x_train.shape[1])


"""
Machine Learning Training and Prediction
"""
if 'TBDT' in estimator_name:
    regressor = TBDT(tree_filename=estimator_name, regularization=regularization, splitting_features=splitting_features,
                regularization_lambda=regularization_lambda, optim_split=True, optim_threshold=2,
                min_samples_leaf=min_samples_leaf)
elif 'TBRF' in estimator_name:
    regressor = TBRF(min_samples_leaf=min_samples_leaf, tree_filename='TBRF_Kaandorp_%i', n_trees=n_estimators, regularization=regularization,
         regularization_lambda=regularization_lambda, splitting_features=splitting_features,
         optim_split=True, optim_threshold=2, read_from_file=False)

t0 = t.time()
tree_struct = regressor.fit(x_train, y_train, tb_train)
dump(tree_struct, case.resultPaths[time] + estimator_name + '.joblib')
t1 = t.time()
print('\nFinished training in {:.4f} s'.format(t1 - t0))
if 'TBDT' in estimator_name:
    y_pred_test, g_test = regressor.predict(x_test, tb_test, tree_struct)
    y_pred_train, g_train = regressor.predict(x_train, tb_train, tree_struct)
else:
    # This is ensemble of all tree's predictions
    _, y_pred_test, g_test = regressor.predict(x_test, tb_test, tree_struct)
    _, y_pred_train, g_train = regressor.predict(x_train, tb_train, tree_struct)
    # Take median
    y_pred_test = np.median(y_pred_test, axis=2)
    y_pred_train = np.median(y_pred_train, axis=2)

y_pred_train = np.swapaxes(y_pred_train, 0, 1)
y_pred_test = np.swapaxes(y_pred_test, 0, 1)
y_train = np.swapaxes(y_train, 0, 1)
y_test = np.swapaxes(y_test, 0, 1)
y_pred_train = contractSymmetricTensor(y_pred_train)
y_pred_test = contractSymmetricTensor(y_pred_test)
y_train = contractSymmetricTensor(y_train)
y_test = contractSymmetricTensor(y_test)
score_train = r2_score(y_train, y_pred_train)
score_test = r2_score(y_test, y_pred_test)


"""
Postprocess Machine Learning Predictions
"""
t0 = t.time()
_, eigval_test, _ = processReynoldsStress(y_test, make_anisotropic=False, realization_iter=0)
_, eigval_train, _ = processReynoldsStress(y_train, make_anisotropic=False, realization_iter=0)
y_pred_test3, eigval_pred_test, _ = processReynoldsStress(y_pred_test, make_anisotropic=False, realization_iter=realize_iter)
y_pred_train3, eigval_pred_train, _ = processReynoldsStress(y_pred_train, make_anisotropic=False, realization_iter=realize_iter)
t1 = t.time()
print('\nFinished processing Reynolds stress in {:.4f} s'.format(t1 - t0))

t0 = t.time()
xy_bary_test, rgb_bary_test = getBarycentricMapData(eigval_test)
xy_bary_train, rgb_bary_train = getBarycentricMapData(eigval_train)
xy_bary_pred_test, rgb_bary_pred_test = getBarycentricMapData(eigval_pred_test)
xy_bary_pred_train, rgb_bary_pred_train = getBarycentricMapData(eigval_pred_train)

# Manually limit RGB values
rgb_bary_pred_test[rgb_bary_pred_test > 1.] = 1.
rgb_bary_pred_train[rgb_bary_pred_train > 1.] = 1.
t1 = t.time()
print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))

t0 = t.time()
ccx_test_mesh, ccy_test_mesh, _, rgb_bary_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
ccx_train_mesh, ccy_train_mesh, _, rgb_bary_train_mesh = interpolateGridData(ccx_train, ccy_train, rgb_bary_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, rgb_bary_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, rgb_bary_pred_train_mesh = interpolateGridData(ccx_train, ccy_train, rgb_bary_pred_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for barycentric map in {:.4f} s'.format(t1 - t0))

t0 = t.time()
_, _, _, y_test_mesh = interpolateGridData(ccx_test, ccy_test, y_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_train_mesh = interpolateGridData(ccx_train, ccy_train, y_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_pred_test_mesh = interpolateGridData(ccx_test, ccy_test, y_pred_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
_, _, _, y_pred_train_mesh = interpolateGridData(ccx_train, ccy_train, y_pred_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=0.3)
t1 = t.time()
print('\nFinished interpolating mesh data for bij in {:.4f} s'.format(t1 - t0))


"""
Plotting
"""
rgb_bary_test_mesh = ndimage.rotate(rgb_bary_test_mesh, 90)
rgb_bary_train_mesh = ndimage.rotate(rgb_bary_train_mesh, 90)
rgb_bary_pred_test_mesh = ndimage.rotate(rgb_bary_pred_test_mesh, 90)
rgb_bary_pred_train_mesh = ndimage.rotate(rgb_bary_pred_train_mesh, 90)
xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
geometry = np.genfromtxt(caseDir + '/' + rans_case_name + '/'  + "geometry.csv", delimiter=",")[:, :2]
figname = 'barycentric_periodichill_test_seed' + str(seed)
bary_map = BaseFigure((None,), (None,), name=figname, xlabel=xlabel,
                      ylabel=ylabel, save=save_fig, show=show,
                      figdir=case.resultPaths[time],
                      figheight_multiplier=0.7)
path = Path(geometry)
patch = PathPatch(path, linewidth=0., facecolor=bary_map.gray)
# patch is considered "a single artist" so have to make copy to use more than once
patches = []
for _ in range(28):
    patches.append(copy(patch))

patches = iter(patches)
extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
extent_train = (ccx_train.min(), ccx_train.max(), ccy_train.min(), ccy_train.max())
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.resultPaths[time] + figname + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_pred_test_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.resultPaths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_train_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_train_mesh, origin='upper', aspect='equal', extent=extent_train)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.resultPaths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()

bary_map.name = 'barycentric_periodichill_pred_train_seed' + str(seed)
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_pred_train_mesh, origin='upper', aspect='equal', extent=extent_train)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.resultPaths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()

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
fignames_predtrain = ('b11_periodichill_pred_train_seed' + str(seed),
                      'b12_periodichill_pred_train_seed' + str(seed),
                      'b13_periodichill_pred_train_seed' + str(seed),
                      'b22_periodichill_pred_train_seed' + str(seed),
                      'b23_periodichill_pred_train_seed' + str(seed),
                      'b33_periodichill_pred_train_seed' + str(seed))
fignames_train = ('b11_periodichill_train_seed' + str(seed),
                  'b12_periodichill_train_seed' + str(seed),
                  'b13_periodichill_train_seed' + str(seed),
                  'b22_periodichill_train_seed' + str(seed),
                  'b23_periodichill_train_seed' + str(seed),
                  'b33_periodichill_train_seed' + str(seed))
zlabels = ('$b_{11}$ [-]', '$b_{12}$ [-]', '$b_{13}$ [-]', '$b_{22}$ [-]', '$b_{23}$ [-]', '$b_{33}$ [-]')
zlabels_pred = ('$\hat{b}_{11}$ [-]', '$\hat{b}_{12}$ [-]', '$\hat{b}_{13}$ [-]', '$\hat{b}_{22}$ [-]', '$\hat{b}_{23}$ [-]', '$\hat{b}_{33}$ [-]')
figheight_multiplier = 1.1
# Go through every bij component and plot
for i in range(len(zlabels)):
    bij_predtest_plot = Plot2D_Image(val=y_pred_test_mesh[:, :, i], name=fignames_predtest[i], xlabel=xlabel,
                                     ylabel=ylabel, val_label=zlabels_pred[i],
                                     save=save_fig, show=show,
                                     figdir=case.resultPaths[time],
                                     figwidth='1/3',
                                     val_lim=bijlims,
                                     rotate_img=True,
                                     extent=extent_test,
                                     figheight_multiplier=figheight_multiplier)
    bij_predtest_plot.initializeFigure()
    bij_predtest_plot.plotFigure()
    bij_predtest_plot.axes.add_patch(next(patches))
    bij_predtest_plot.finalizeFigure()

    bij_test_plot = Plot2D_Image(val=y_test_mesh[:, :, i], name=fignames_test[i], xlabel=xlabel,
                                 ylabel=ylabel, val_label=zlabels[i],
                                 save=save_fig, show=show,
                                 figdir=case.resultPaths[time],
                                 figwidth='1/3',
                                 val_lim=bijlims,
                                 rotate_img=True,
                                 extent=extent_test,
                                 figheight_multiplier=figheight_multiplier)
    bij_test_plot.initializeFigure()
    bij_test_plot.plotFigure()
    bij_test_plot.axes.add_patch(next(patches))
    bij_test_plot.finalizeFigure()

    bij_predtrain_plot = Plot2D_Image(val=y_pred_train_mesh[:, :, i], name=fignames_predtrain[i], xlabel=xlabel,
                                      ylabel=ylabel, val_label=zlabels_pred[i],
                                      save=save_fig, show=show,
                                      figdir=case.resultPaths[time],
                                      figwidth='1/3',
                                      val_lim=bijlims,
                                      rotate_img=True,
                                      extent=extent_test,
                                      figheight_multiplier=figheight_multiplier)
    bij_predtrain_plot.initializeFigure()
    bij_predtrain_plot.plotFigure()
    bij_predtrain_plot.axes.add_patch(next(patches))
    bij_predtrain_plot.finalizeFigure()

    bij_train_plot = Plot2D_Image(val=y_train_mesh[:, :, i], name=fignames_train[i], xlabel=xlabel,
                                  ylabel=ylabel, val_label=zlabels[i],
                                  save=save_fig, show=show,
                                  figdir=case.resultPaths[time],
                                  figwidth='1/3',
                                  val_lim=bijlims,
                                  rotate_img=True,
                                  extent=extent_test,
                                  figheight_multiplier=figheight_multiplier)
    bij_train_plot.initializeFigure()
    bij_train_plot.plotFigure()
    bij_train_plot.axes.add_patch(next(patches))
    bij_train_plot.finalizeFigure()
