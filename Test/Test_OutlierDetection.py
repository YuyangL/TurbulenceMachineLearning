import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from FieldData import FieldData
from Postprocess.OutlierAndNoveltyDetection import InputOutlierDetection
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor
from Utility import interpolateGridData
from joblib import load, dump
import time as t
from PlottingTool import BaseFigure, Plot2D, Plot2D_Image, PlotContourSlices3D
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import copy
import mpl_toolkits.mplot3d.art3d as art3d
import os


"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both RANS and LES
rans_case_name = 'RANS_Re10595'  # str
test_case_name = 'RANS_Re5600'  # str
les_case_name = 'LES_Breuer/Re_5600'  # str
# LES data name to read
les_data_name = 'Hill_Re_5600_Breuer.csv'  # str
# Absolute directory of this flow case
casedir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"

"""
Machine Learning Settings
"""
isoforest_name = 'IsoForest'
estimator_name = 'tbdt'
fs = 'grad(TKE)_grad(p)'
# Seed value for reproducibility
seed = 123
outlier_percent1, outlier_percent2, outlier_percent3, outlier_percent4, outlier_percent5 = \
    0.1, 0.08, 0.06, 0.04, 0.02
load_isoforest, save_estimator = True, False


"""
Plot Settings
"""
plot_tb, plot_g, plot_x = False, False, False
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
# Limit for bij plot
bijlims = (-1/2., 2/3.)  # (float, float)
contour_lvl = 50  # int
alpha = 0.6  # int, float [0, 1]
gray = (80/255.,)*3
figheight_multiplier = 1.1
# Save anything when possible
save_fields = True  # bool
# Save figures and show figures
save_fig, show = True, False  # bool; bool
if save_fig:
    # Figure extension and DPI
    ext, dpi = 'png', 1000  # str; int


"""
Process User Inputs, No Need to Change
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

if estimator_name == 'tbdt': estimator_name = 'TBDT'
# Ensemble name of fields useful for Machine Learning
ml_field_ensemble_name = 'ML_Fields_' + rans_case_name
# Initialize case object
case = FieldData(casename=rans_case_name, casedir=casedir, times=time, fields=fields)
case_test = FieldData(casename=test_case_name, casedir=casedir, times=time, fields=fields)


"""
Machine Learning with Isolation Forest
"""
list_data_train = case.readPickleData(time, 'list_data_train_seed' + str(seed))
list_data_test = case_test.readPickleData(time, 'list_data_test_seed' + str(seed))
if '10595' not in test_case_name: list_data_train = list_data_test

cc_train, cc_test = list_data_train[0], list_data_test[0]
ccx_train, ccy_train, ccz_train = cc_train[:, 0], cc_train[:, 1], cc_train[:, 2]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_train, y_train, tb_train = list_data_train[1:4]
x_test, y_test, tb_test = list_data_test[1:4]

isoforest = load(case.result_paths[time] + isoforest_name + '.joblib') if load_isoforest else None
isoforest2 = load(case.result_paths[time] + isoforest_name + '2.joblib') if load_isoforest else None
isoforest3 = load(case.result_paths[time] + isoforest_name + '2.joblib') if load_isoforest else None
isoforest4 = load(case.result_paths[time] + isoforest_name + '2.joblib') if load_isoforest else None
isoforest5 = load(case.result_paths[time] + isoforest_name + '5.joblib') if load_isoforest else None

_, _, _, _, outliers, isoforest = InputOutlierDetection(x_train, x_test, y_train, y_test, outlier_percent=outlier_percent1, randstate=seed, isoforest=isoforest, n_estimators=1000)
_, _, _, _, outliers2, isoforest2 = InputOutlierDetection(x_train, x_test, y_train, y_test, outlier_percent=outlier_percent2, randstate=seed, isoforest=isoforest2, n_estimators=1000)
_, _, _, _, outliers3, isoforest3 = InputOutlierDetection(x_train, x_test, y_train, y_test, outlier_percent=outlier_percent3, randstate=seed, isoforest=isoforest3, n_estimators=1000)
_, _, _, _, outliers4, isoforest4 = InputOutlierDetection(x_train, x_test, y_train, y_test, outlier_percent=outlier_percent4, randstate=seed, isoforest=isoforest4, n_estimators=1000)
_, _, _, _, outliers5, isoforest5 = InputOutlierDetection(x_train, x_test, y_train, y_test, outlier_percent=outlier_percent5, randstate=seed, isoforest=isoforest5, n_estimators=1000)

if save_estimator and not load_isoforest:
    dump(isoforest, case.result_paths[time] + isoforest_name + '.joblib')
    dump(isoforest2, case.result_paths[time] + isoforest_name + '2.joblib')
    dump(isoforest3, case.result_paths[time] + isoforest_name + '3.joblib')
    dump(isoforest4, case.result_paths[time] + isoforest_name + '4.joblib')
    dump(isoforest5, case.result_paths[time] + isoforest_name + '5.joblib')

x_train_out = outliers['xtrain']
x_test_out = outliers['xtest']
y_train_out = outliers['ytrain']
y_test_out = outliers['ytest']
anomaly_idx_train = outliers['anomaly_idx_train']
anomaly_idx_test = outliers['anomaly_idx_test']
anomaly_idx_train2 = outliers2['anomaly_idx_train']
anomaly_idx_test2 = outliers2['anomaly_idx_test']
anomaly_idx_train3 = outliers3['anomaly_idx_train']
anomaly_idx_test3 = outliers3['anomaly_idx_test']
anomaly_idx_train4 = outliers4['anomaly_idx_train']
anomaly_idx_test4 = outliers4['anomaly_idx_test']
anomaly_idx_train5 = outliers5['anomaly_idx_train']
anomaly_idx_test5 = outliers5['anomaly_idx_test']


"""
Postprocess Machine Learning Predictions
"""
t0 = t.time()
_, eigval_test, _ = processReynoldsStress(y_test, make_anisotropic=False, realization_iter=0)
_, eigval_train, _ = processReynoldsStress(y_train, make_anisotropic=False, realization_iter=0)
t1 = t.time()
print('\nFinished processing Reynolds stress in {:.4f} s'.format(t1 - t0))

t0 = t.time()
xy_bary_test, rgb_bary_test = getBarycentricMapData(eigval_test)
xy_bary_train, rgb_bary_train = getBarycentricMapData(eigval_train)
rgb_bary_test_out = rgb_bary_test.copy()
rgb_bary_test_out[anomaly_idx_test], rgb_bary_test_out[anomaly_idx_test2], rgb_bary_test_out[anomaly_idx_test3], rgb_bary_test_out[anomaly_idx_test4], rgb_bary_test_out[anomaly_idx_test5] \
    = (0.4,)*3, (0.3,)*3, (0.2,)*3, (0.1,)*3, (0,)*3
rgb_bary_train_out = rgb_bary_train.copy()
rgb_bary_train_out[anomaly_idx_train], rgb_bary_train_out[anomaly_idx_train2], rgb_bary_train_out[anomaly_idx_train3], rgb_bary_train_out[anomaly_idx_train4], rgb_bary_train_out[anomaly_idx_train5] \
    = (0.4,)*3, (0.3,)*3, (0.2,)*3, (0.1,)*3, (0,)*3
t1 = t.time()
print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))

t0 = t.time()
ccx_test_mesh, ccy_test_mesh, _, rgb_bary_test_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_test, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=89/255.)
_, _, _, rgb_bary_test_out_mesh = interpolateGridData(ccx_test, ccy_test, rgb_bary_test_out, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=89/255.)
ccx_train_mesh, ccy_train_mesh, _, rgb_bary_train_mesh = interpolateGridData(ccx_train, ccy_train, rgb_bary_train, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=89/255.)
_, _, _, rgb_bary_train_out_mesh = interpolateGridData(ccx_train, ccy_train, rgb_bary_train_out, mesh_target=uniform_mesh_size, interp=interp_method, fill_val=89/255.)
t1 = t.time()
print('\nFinished interpolating mesh data for barycentric map in {:.4f} s'.format(t1 - t0))


"""
Visualize Outliers and Novelties in Barycentric Map
"""
rgb_bary_test_mesh = ndimage.rotate(rgb_bary_test_mesh, 90)
rgb_bary_train_mesh = ndimage.rotate(rgb_bary_train_mesh, 90)
rgb_bary_test_out_mesh = ndimage.rotate(rgb_bary_test_out_mesh, 90)
rgb_bary_train_out_mesh = ndimage.rotate(rgb_bary_train_out_mesh, 90)

xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]')
geometry = np.genfromtxt(casedir + '/' + rans_case_name + '/'  + "geometry.csv", delimiter=",")[:, :2]
# Test barycentric map
figname = 'barycentric_periodichill_test_out'
bary_map = BaseFigure((None,), (None,), name=figname, xlabel=xlabel,
                      ylabel=ylabel, save=save_fig, show=show,
                      figdir=case.result_paths[time],
                      figheight_multiplier=0.7)
path = Path(geometry)
patch = PathPatch(path, linewidth=0., facecolor=bary_map.gray)
# patch is considered "a single artist" so have to make copy to use more than once
patches = []
for _ in range(236):
    patches.append(copy(patch))

patches = iter(patches)
extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
extent_train = (ccx_train.min(), ccx_train.max(), ccy_train.min(), ccy_train.max())
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_test_mesh, origin='upper', aspect='equal', extent=extent_test)
bary_map.axes.imshow(rgb_bary_test_out_mesh, origin='upper', aspect='equal', extent=extent_test, alpha=alpha)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.result_paths[time] + figname + '.' + ext, dpi=dpi)

plt.close()
# Train barycentric map
bary_map.name = 'barycentric_periodichill_train_out'
bary_map.initializeFigure()
bary_map.axes.imshow(rgb_bary_train_mesh, origin='upper', aspect='equal', extent=extent_train)
bary_map.axes.imshow(rgb_bary_train_out_mesh, origin='upper', aspect='equal', extent=extent_test, alpha=alpha)
bary_map.axes.set_xlabel(bary_map.xlabel)
bary_map.axes.set_ylabel(bary_map.ylabel)
bary_map.axes.add_patch(next(patches))
if save_fig:
    plt.savefig(case.result_paths[time] + bary_map.name + '.' + ext, dpi=dpi)

plt.close()


"""
Visualize Tij Along with X Outlier and Novelties
"""
if plot_tb:
    t0 = t.time()
    # Create Tij mesh ensemble, shape (n_x, n_y, n_bases, n_outputs)
    tb_test_mesh = np.empty((ccx_test_mesh.shape[0], ccy_test_mesh.shape[1], tb_test.shape[2], tb_test.shape[1]))
    tb_train_mesh = np.empty((ccx_train_mesh.shape[0], ccy_train_mesh.shape[1], tb_train.shape[2], tb_train.shape[1]))
    # Go through each Tij component, interpolate bases
    for i in range(tb_test.shape[1]):
        _, _, _, tb_test_mesh_i = interpolateGridData(ccx_test, ccy_test, tb_test[:, i], mesh_target=uniform_mesh_size,
                                                      interp=interp_method)
        _, _, _, tb_train_mesh_i = interpolateGridData(ccx_train, ccy_train, tb_train[:, i],
                                                       mesh_target=uniform_mesh_size, interp=interp_method)
        # Assign each component to an ensemble
        tb_test_mesh[..., i], tb_train_mesh[..., i] = tb_test_mesh_i, tb_train_mesh_i

    # Swap n_outputs and n_bases axis back so that (grid, n_outputs, n_bases)
    tb_test_mesh = np.swapaxes(tb_test_mesh, -2, -1)
    tb_train_mesh = np.swapaxes(tb_train_mesh, -2, -1)
    t1 = t.time()
    print('\nFinished interpolating mesh data for Tij in {:.4f} s'.format(t1 - t0))

    # Make an individual directory
    tijdir = case.result_paths[time] + '/Tij'
    os.makedirs(tijdir, exist_ok=True)
    comps = ('11', '12', '13', '22', '23', '33')
    tbcomps = ('T_{11}', 'T_{12}', 'T_{13}', 'T_{22}', 'T_{23}', 'T_{33}')
    extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
    extent_train = (ccx_train.min(), ccx_train.max(), ccy_train.min(), ccy_train.max())
    # Go through every component then every basis
    for i in range(tb_test.shape[1]):
        for j in range(tb_test.shape[2]):
            zlabel = '${0}^{1}$ [-]'.format(tbcomps[i], '{(' + str(j + 1) + ')}')
            # Tij is limited individually so that only test and train have the same limit
            tblim = (min(min(tb_test[:, i, j]), min(tb_train[:, i, j])),
                     max(max(tb_test[:, i, j]), max(tb_train[:, i, j])))
            # First, plot test Tij
            figname = 'T' + comps[i] + '_' + str(j + 1) + '_test'
            # Initialize PLot2D_Image object
            tbplot_test = Plot2D_Image(val=tb_test_mesh[..., i, j],
                                       name=figname,
                                       xlabel=xlabel, ylabel=ylabel, val_label=zlabel,
                                       save=save_fig,
                                       show=show,
                                       figdir=tijdir,
                                       rotate_img=True,
                                       extent=extent_test,
                                       figwidth='1/3',
                                       val_lim=tblim,
                                       figheight_multiplier=figheight_multiplier)
            tbplot_test.initializeFigure()
            tbplot_test.plotFigure()
            tbplot_test.axes.ticklabel_format(scilimits=(-4, 5))
            tbplot_test.axes.add_patch(next(patches))
            tbplot_test.finalizeFigure()
    
            # Then plot train Tij
            figname = 'T' + comps[i] + '_' + str(j + 1) + '_train'
            tbplot_train = Plot2D_Image(val=tb_train_mesh[..., i, j],
                                       name=figname,
                                       xlabel=xlabel, ylabel=ylabel, val_label=zlabel,
                                       save=save_fig,
                                       show=show,
                                       figdir=tijdir,
                                       rotate_img=True,
                                       extent=extent_train,
                                       figwidth='1/3',
                                       val_lim=tblim,
                                        figheight_multiplier=figheight_multiplier)
            tbplot_train.initializeFigure()
            tbplot_train.plotFigure()
            tbplot_train.axes.add_patch(next(patches))
            tbplot_train.finalizeFigure()


"""
Visualize g Using Trained TBDT
"""
if plot_g:
    # Load trained TBDT
    regressor = load(case.result_paths[time] + estimator_name + '.joblib')
    # Predict g
    g_test = regressor.predict(x_test)
    g_train = regressor.predict(x_train)
    # Interpolate to mesh
    _, _, _, g_test_mesh = interpolateGridData(ccx_test, ccy_test, g_test,
                                                                              mesh_target=uniform_mesh_size,
                                                                              interp=interp_method)
    _, _, _, g_train_mesh = interpolateGridData(ccx_train, ccy_train, g_train,
                                                           mesh_target=uniform_mesh_size, interp=interp_method)
    # Make an individual directory
    gdir = case.result_paths[time] + '/g'
    os.makedirs(gdir, exist_ok=True)
    extent_test = (ccx_test.min(), ccx_test.max(), ccy_test.min(), ccy_test.max())
    extent_train = (ccx_train.min(), ccx_train.max(), ccy_train.min(), ccy_train.max())
    # glim = (-10000, 10000)
    glim = None
    # g diff = |g_train - g_test|
    g_diff_mesh = np.abs(g_train_mesh - g_test_mesh)
    # Go through every basis
    for i in range(g_test.shape[1]):
        zlabel = "$|g'^{}|$ [-]".format('{(' + str(i + 1) + ')}')
        cbticks = [0, 100, 10000, 1e6, 1e8, 1e10]
        # g diff = |g_train - g_test|
        figname = 'g' + str(i + 1) + '_diff'
        gplot_diff = Plot2D_Image(val=g_diff_mesh[:, i],
                                  name=figname,
                                  xlabel=xlabel, ylabel=ylabel, val_label=zlabel,
                                  save=save_fig,
                                  show=show,
                                  figdir=gdir,
                                  rotate_img=True,
                                  extent=extent_test,
                                  figwidth='1/3',
                                  val_lim=glim,
                                  figheight_multiplier=figheight_multiplier)
        gplot_diff.initializeFigure()
        gplot_diff.plotFigure(norm='symlog')
        gplot_diff.axes.add_patch(next(patches))
        gplot_diff.finalizeFigure(cbticks=cbticks)

        # # First test g
        # zlabel = '$g^{}$'.format('{(' + str(i + 1) + ')}')
        # # First test g
        # figname = 'g' + str(i + 1) + '_test'
        # gplot_test = Plot2D_Image(val=g_test_mesh[:, i],
        #                           name=figname,
        #                           xLabel=xlabel, yLabel=ylabel, zLabel=zlabel,
        #                           save=save_fig,
        #                           show=show,
        #                           figDir=gdir,
        #                           rotate_img=True,
        #                           extent=extent_test,
        #                           figWidth='1/3',
        #                           zlim=glim)
        # gplot_test.initializeFigure()
        # gplot_test.plotFigure(norm='symlog')
        # gplot_test.axes.add_patch(next(patches))
        # gplot_test.finalizeFigure()
        #
        # # Then train g
        # figname = 'g' + str(i + 1) + '_train'
        # gplot_train = Plot2D_Image(val=g_train_mesh[:, i],
        #                           name=figname,
        #                           xLabel=xlabel, yLabel=ylabel, zLabel=zlabel,
        #                           save=save_fig,
        #                           show=show,
        #                           figDir=gdir,
        #                           rotate_img=True,
        #                           extent=extent_test,
        #                           figWidth='1/3',
        #                           zlim=glim)
        # gplot_train.initializeFigure()
        # gplot_train.plotFigure(norm='symlog')
        # gplot_train.axes.add_patch(next(patches))
        # gplot_train.finalizeFigure()


if plot_x:
    xdir = case.result_paths[time] + '/X'
    os.makedirs(xdir, exist_ok=True)
    # Interpolate to mesh
    _, _, _, x_test_mesh = interpolateGridData(ccx_test, ccy_test, x_test,
                                               mesh_target=uniform_mesh_size,
                                               interp=interp_method)
    _, _, _, x_train_mesh = interpolateGridData(ccx_train, ccy_train, x_train,
                                                mesh_target=uniform_mesh_size, interp=interp_method)

    if fs == 'grad(TKE)_grad(p)':
        labels = ['S^2', 'S^3', '\Omega^2', '\Omega^2S', '\Omega^2S^2', '\Omega^2S\Omega S^2',
                  'A_k^2', 'A_k^2S', 'A_k^2S^2', 'A_k^2SA_kS^2', '\Omega A_k', '\Omega A_kS', '\Omega A_kS^2',
                  '\Omega^2A_kS', 'A_k^2\Omega S', '\Omega^2A_kS^2', 'A_k^2\Omega S^2', '\Omega^2SA_kS^2', 'A_k^2S\Omega S^2',
                  'A_p^2', 'A_p^2S', 'A_p^2S^2', 'A_p^2SA_pS^2', '\Omega A_p', '\Omega A_pS', '\Omega A_pS^2',
                  '\Omega^2A_pS', 'A_p^2\Omega S', '\Omega^2A_pS^2', 'A_p^2\Omega S^2', '\Omega^2SA_pS^2', 'A_p^2S\Omega S^2',
                  'A_kA_p', 'A_kA_pS', 'A_kA_pS^2', 'A_k^2A_pS', 'A_p^2A_kS', 'A_k^2A_pS^2', 'A_p^2A_kS^2',
                  'A_k^2SA_pS^2', 'A_p^2SA_kS^2',
                  '\Omega A_kA_p', '\Omega A_kA_pS', '\Omega A_pA_kS', '\Omega A_kA_pS^2', '\Omega A_pA_kS^2', '\Omega A_kSA_pS^2']
        zlabel = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6',
                  '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '2.10', '2.11', '2.12', '2.13',
                  '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '3.10', '3.11', '3.12', '3.13',
                  '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9', '4.10', '4.11', '4.12', '4.13', '4.14', '4.15']
    elif 'grad(TKE)' in fs:
        labels = ['S^2', 'S^3', '\Omega^2', '\Omega^2S', '\Omega^2S^2', '\Omega^2S\Omega S^2',
                  'A_k^2', 'A_k^2S', 'A_k^2S^2', 'A_k^2SA_kS^2', '\Omega A_k', '\Omega A_kS', '\Omega A_kS^2',
                  '\Omega^2A_kS', 'A_k^2\Omega S', '\Omega^2A_kS^2', 'A_k^2\Omega S^2', '\Omega^2SA_kS^2', 'A_k^2S\Omega S^2']
        zlabel = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6',
                  '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '2.10', '2.11', '2.12', '2.13']
    else:
        labels = ['S^2', 'S^3', '\Omega^2', '\Omega^2S', '\Omega^2S^2', '\Omega^2S\Omega S^2']
        zlabel = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6']

    for i in range(len(zlabel)):
        zlabel[i] = 'Feature ' + zlabel[i] + ' [-]'
        labels[i] = '$' + labels[i] + '$ [-]'

    for i in range(x_test.shape[1]):
        # For test features
        figname = 'X' + str(i + 1) + 'test'
        xlim = min(min(x_train[:, i]), min(x_test[:, i])), max(max(x_train[:, i]), max(x_test[:, i]))
        xtest_plot = Plot2D_Image(val=x_test_mesh[..., i],
                                  name=figname,
                                  xlabel=xlabel, ylabel=ylabel, val_label=labels[i],
                                  save=save_fig,
                                  show=show,
                                  figdir=xdir,
                                  rotate_img=True,
                                  extent=extent_test,
                                  figwidth='1/3',
                                  val_lim=xlim,
                                  figheight_multiplier=figheight_multiplier)
        xtest_plot.initializeFigure()
        xtest_plot.plotFigure()
        xtest_plot.axes.add_patch(next(patches))
        xtest_plot.finalizeFigure()

        # For train features
        figname = 'X' + str(i + 1) + 'train'
        xtrain_plot = Plot2D_Image(val=x_train_mesh[..., i],
                                  name=figname,
                                  xlabel=xlabel, ylabel=ylabel, val_label=labels[i],
                                  save=save_fig,
                                  show=show,
                                  figdir=xdir,
                                  rotate_img=True,
                                  extent=extent_train,
                                  figwidth='1/3',
                                  val_lim=xlim,
                                  figheight_multiplier=figheight_multiplier)
        xtrain_plot.initializeFigure()
        xtrain_plot.plotFigure()
        xtrain_plot.axes.add_patch(next(patches))
        xtrain_plot.finalizeFigure()












# # First, take Frobenius norm on Tij to collapse n_output axis
# # so that Tij becomes shape (n_samples, n_bases)
# # 4 times the mesh grid, ignoring b13 and b23
# x_test_contour, y_test_contour = (ccx_test_mesh,)*4, (ccy_test_mesh,)*4
# x_train_contour, y_train_contour = (ccx_train_mesh,)*4, (ccy_train_mesh,)*4
#
# # Go through each basis
# for i in range(1):#tb_test.shape[2]):
#     # Same with interpolated Tij of shape (nx, ny, n_bases, n_outputs), ignore b13, b23
#     vals_test = (tb_test_mesh[..., i, 0], tb_test_mesh[..., i, 1], tb_test_mesh[..., i, 3], tb_test_mesh[..., i, 5])
#     vals_train = (tb_train_mesh[..., i, 0], tb_train_mesh[..., i, 1], tb_train_mesh[..., i, 3], tb_train_mesh[..., i, 5])
#
#     figname = 'Tij_' + str(i + 1)
#     tbplot_test = PlotContourSlices3D(x_test_contour, y_test_contour, vals_test, np.arange(4), contourLvl=contour_lvl, gradientBg=False, equalAxis=True, zDir='x',
#                                       name=figname, xLabel=xlabel, yLabel=ylabel, save=save_fig, show=show,
#                                       figDir=case.result_paths[time],
#                                       viewAngles=(45, 90),
#                                       cbarOrientate='vertical',
#                                       zLabel=None)
#     tbplot_test.initializeFigure(figSize=(2, 1))
#     tbplot_test.plotFigure()
#     for j in range(4):
#         patch_i = next(patches)
#         tbplot_test.axes.add_patch(patch_i)
#         art3d.pathpatch_2d_to_3d(patch_i, z=j, zdir='x')
#
#     # plt.show()
#     tbplot_test.finalizeFigure()
#     # plt.savefig(tbplot_test.figDir + '/' + tbplot_test.name + '.png',
#     #             dpi=1000)
#     print('\nFigure ' + tbplot_test.name + '.png saved in ' + tbplot_test.figDir)
#
#     # tbplot_train = PlotContourSlices3D(x_train_contour, y_train_contour, vals_train, np.arange(4), contourLvl=contour_lvl, gradientBg=False, equalAxis=True,
#     #                              name=figname, xLabel=xlabel, yLabel=ylabel,save=save_fig, show=show, figDir=case.result_paths[time])
#     # tbplot_train.initializeFigure()
#     # tbplot_train.plotFigure()
#     # tbplot_train.axes.add_patch(next(patches))
#     # tbplot_train.finalizeFigure()
