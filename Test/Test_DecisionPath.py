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
from PlottingTool import BaseFigure, Plot2D, Plot2D_Image, PlotContourSlices3D, Plot2D_MultiAxes
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import copy
import os
from sklearn.tree import plot_tree

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both RANS and LES
rans_case_name = 'RANS_Re10595'  # str
les_case_name = 'LES_Breuer/Re_10595'  # str
# LES data name to read
les_data_name = 'Hill_Re_10595_Breuer.csv'  # str
# Absolute directory of this flow case
casedir = '/media/yluan/DNS/PeriodicHill'  # str
# Which time to extract input and output for ML
time = '5000'  # str/float/int or 'last'
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"


"""
Machine Learning Settings
"""
estimator_name = 'tbdt'
# Seed value for reproducibility
seed = 123


"""
Plot Settings
"""
# When plotting, the mesh has to be uniform by interpolation, specify target size
uniform_mesh_size = 1e6  # int
# Limit for bij plot
bijlims = (-1/3., 2/3.)  # (float, float)
contour_lvl = 50  # int
alpha = 0.6  # int, float [0, 1]
gray = (80/255.,)*3
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


"""
Load Data, Trained Estimator and Predict
"""
list_data_train = case.readPickleData(time, 'list_47data_train_seed' + str(seed))
list_data_test = case.readPickleData(time, 'list_47data_test_seed' + str(seed))

cc_train, cc_test = list_data_train[0], list_data_test[0]
ccx_train, ccy_train, ccz_train = cc_train[:, 0], cc_train[:, 1], cc_train[:, 2]
ccx_test, ccy_test, ccz_test = cc_test[:, 0], cc_test[:, 1], cc_test[:, 2]
x_train, y_train, tb_train = list_data_train[1:4]
x_test, y_test, tb_test = list_data_test[1:4]


print('\n\nLoading regressor... \n')
regressor = load(case.result_paths[time] + estimator_name + '_full.joblib')
score_test = regressor.score(x_test, y_test, tb=tb_test)
score_train = regressor.score(x_train, y_train, tb=tb_train)

t0 = t.time()
# Predict bij as well as g
g_test = regressor.predict(x_test)
g_train = regressor.predict(x_train)
y_pred_test = regressor.predict(x_test, tb=tb_test)
y_pred_train = regressor.predict(x_train, tb=tb_train)
t1 = t.time()
print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))


"""
Postprocess to Track down 1 Novelty
"""
# # According to b11 prediction, 1 novelty is located in x in (1.6, 1.9) and y in (0.9, 1.2)
# # Bracket the aforementioned x and y range respectively
# ccx_test_outidx = np.where((1.6 < ccx_test) & (ccx_test < 1.9))
# ccy_test_outidx = np.where((0.9 < ccy_test) & (ccy_test < 1.2))
# # Then find intersection considering both x and y
# cc_test_outidx = np.intersect1d(ccx_test_outidx, ccy_test_outidx)
# # Get b11 values in this bracketed region
# b11_pred_out = y_pred_test[cc_test_outidx, 0]
# b11_out = y_test[cc_test_outidx, 0]
# # Get the index of the worst outlier
# b11_outmax_idx = np.where(y_pred_test == max(b11_pred_out))[0]
# 3 biggest b11 novelties at index: 1961, 488, 1702
b11_outmax_idx = 1961  # 1961, 488, 1702

# Tell me the exact x, y coordinate of this novelty
ccx_out, ccy_out = ccx_test[b11_outmax_idx], ccy_test[b11_outmax_idx]
# Then tell me which index this coordinate is (approximately) in training data
proximity_x = 0.035 if b11_outmax_idx == 1702 else 0.01  # 0.01 for 1961 and 488, 0.035 for 1702
proximity_y = 0.035 if b11_outmax_idx == 1702 else 0.01  # 0.01 for 1961 and 488, 0.035 for 1702
ccx_train_idx = np.where((ccx_train > ccx_out - proximity_x) & (ccx_train < ccx_out + proximity_x))
ccy_train_idx = np.where((ccy_train > ccy_out - proximity_y) & (ccy_train < ccy_out + proximity_y))
# Only take one if more than one fits
cc_train_idx = np.intersect1d(ccx_train_idx, ccy_train_idx)[0]
# Again, get the exact coordinate in train coordinate
ccx_train_out, ccy_train_out = ccx_train[cc_train_idx], ccy_train[cc_train_idx]
print('\nTest novelty coor: [{0}, {1}]'.format(ccx_out, ccy_out))
print('Corresponding train coor: [{0}, {1}]'.format(ccx_train_out, ccy_train_out))
# Get train and test features for this specific outlier
x_train_out = x_train[cc_train_idx]
x_test_out = x_test[b11_outmax_idx]
# 2 samples in x. 1st is train x at this outlier coordinate; 2nd is test x
x_out = np.vstack((x_train_out, x_test_out))

# Get T11 as well as g at that location
t11_test, t11_train = tb_test[b11_outmax_idx, 0], tb_train[cc_train_idx, 0]
t11_diff = np.abs(t11_train - t11_test)
print('\nT11 for test: {}'.format(t11_test))
print('T11 for train: {}'.format(t11_train))
print('diff(T11): {}'.format(t11_diff))
g_test_out, g_train_out = g_test[b11_outmax_idx], g_train[cc_train_idx]
g_diff = np.abs(g_train_out - g_test_out)
print('\ng for test: {}'.format(g_test_out))
print('g for train: {}'.format(g_train_out))
# Cumulative (T11*g)'
t11g_diff_cumu = np.empty_like(t11_test)
t11g_diff_sum = 0.
for i in range(len(t11_test)):
    t11g_diff = t11_test[i]*g_test_out[i] - t11_train[i]*g_train_out[i]
    t11g_diff_sum += t11g_diff
    t11g_diff_cumu[i] = abs(t11g_diff_sum)


"""
Decision Path
"""
# Using those arrays, we can parse the tree structure:
n_nodes = regressor.tree_.node_count
children_left = regressor.tree_.children_left
children_right = regressor.tree_.children_right
feature = regressor.tree_.feature
threshold = regressor.tree_.threshold

# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
# while len(stack) > 0:
#     node_id, parent_depth = stack.pop()
#     node_depth[node_id] = parent_depth + 1
#
#     # If we have a test node
#     if (children_left[node_id] != children_right[node_id]):
#         stack.append((children_left[node_id], parent_depth + 1))
#         stack.append((children_right[node_id], parent_depth + 1))
#     else:
#         is_leaves[node_id] = True
#
# print("The binary tree structure has %s nodes and has "
#       "the following tree structure:"
#       % n_nodes)
# for i in range(n_nodes):
#     if is_leaves[i]:
#         print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
#     else:
#         print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
#               "node %s."
#               % (node_depth[i] * "\t",
#                  i,
#                  children_left[i],
#                  feature[i],
#                  threshold[i],
#                  children_right[i],
#                  ))
# print()

# First let's retrieve the decision path of each sample. The decision_path
# method allows to retrieve the node indicator functions. A non zero element of
# indicator matrix at the position (i, j) indicates that the sample i goes
# through the node j.
node_indicator = regressor.decision_path(x_out)

# Similarly, we can also have the leaves ids reached by each sample.
leave_id = regressor.apply(x_out)
feature_out = np.zeros(x_test.shape[1])

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample(s).
# Go through every sample provided, there should be 2 in total, 1st train, 2nd test
cnt = 0
for i in range(x_out.shape[0]):
    sample_id = i
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('\nRules used to predict [train, test] sample: sample %s: '%sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (x_out[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        feature_out[feature[node_id]] += 1
        cnt += 1
        print("decision id node %s : (x_out[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 x_out[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))

# For a group of samples, we have the following common node.
sample_ids = [0, 1]
common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                len(sample_ids))

common_node_id = np.arange(n_nodes)[common_nodes]

print("\nThe following samples %s share the node %s in the tree"
      % (sample_ids, common_node_id))
print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))


# plt.figure(num="DBRT", figsize=(16, 10), constrained_layout=False)
# try:
#     plot = plot_tree(regressor.best_estimator_, fontsize=6, max_depth=5, filled=True, rounded=True, proportion=True, impurity=False)
# except AttributeError:
#     plot = plot_tree(regressor, fontsize=6, max_depth=5, filled=True, rounded=True, proportion=True, impurity=False)


"""
Visualize Features with Importance as Shade
"""
figname = 'NoveltyFeatures_i' + str(b11_outmax_idx)
xdir = case.result_paths[time] + '/X'
os.makedirs(xdir, exist_ok=True)
xlabel, ylabel = 'Feature', 'Value'
list_x, list_y = (np.arange(x_test.shape[1]) + 1,)*2, (x_out[0], x_out[1])
list_x2 = np.arange(x_test.shape[1]) + 1
list_y2 = feature_out/x_test.shape[1]*100.
xlim = x2lim = None  # min(list_x2), max(list_x2)
y2lim = 0, 2*max(list_y2)
xplot = Plot2D_MultiAxes(list_x, list_y, list_x2, list_y2, ax2loc='y', plot_type2='shade',
                         ax2label='Importance $\%$', figwidth='1/3',
                         xlim=xlim, x2lim=x2lim, y2lim=y2lim,
                         xlabel=xlabel, ylabel=ylabel,
               name=figname, save=save_fig, show=show, figdir=xdir)
xplot.initializeFigure()
xplot.plotFigure(linelabel=('Train', 'Test'), showmarker=True)
xplot.finalizeFigure(xyscale=('linear', 'linear'))


"""
Visualize g', Tij' and Cumulative (Tij*g)' as Shade
"""
figname = 'T11g_diff_i' + str(b11_outmax_idx)
gdir = case.result_paths[time] + '/g'
os.makedirs(gdir, exist_ok=True)
list_x = (np.arange(len(t11_test)) + 1,)*2
list_y = t11_diff, g_diff
list_x2 = np.arange(len(t11_test)) + 1
list_y2 = t11g_diff_cumu
xlabel = 'Basis $i$'
ylabel = 'Error'  # "$|T'^{(i)}_{11}|$ \& $|g'^{(i)}|$"
xlim = x2lim = (1, 10)  # min(list_x2), max(list_x2)
ylim = None
y2lim = 0., max(list_y2)**2.
t11gplot = Plot2D_MultiAxes(list_x, list_y, list_x2, list_y2, ax2loc='y', plot_type2='shade',
                            ax2label="$|\sum_i(T^{(i)}_{11}g^{(i)})'|$", figwidth='1/3',
                            xlim=xlim, x2lim=x2lim, ylim=ylim, y2lim=y2lim,
                            xlabel=xlabel, ylabel=ylabel,
                            name=figname, save=save_fig, show=show, figdir=gdir)
t11gplot.initializeFigure()
t11gplot.plotFigure(linelabel=("$|T'^{(i)}_{11}|$", "$|g'^{(i)}|$"), xyscale2=('linear', 'symlog'),  showmarker=True)
t11gplot.finalizeFigure(xyscale=('linear', 'linear'))
