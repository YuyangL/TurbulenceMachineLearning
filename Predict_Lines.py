import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from SetData import SetProperties
import time as t
from PlottingTool import BaseFigure, Plot2D, Plot2D_Image, PlotContourSlices3D, PlotSurfaceSlices3D, PlotImageSlices3D
import os
import numpy as np
import matplotlib.pyplot as plt


"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both ML and test
casename = 'ALM_N_L_SeqTurb'  # str
# Absolute parent directory of ML and test case
casedir = '/media/yluan'  # str
time = 'latestTime'  # str/float/int or 'latestTime'
# Orientation of the lines, either vertical or horizontal
line_orient = '_H_'  # '_H_', '_V_'
# Line offsets w.r.t. D
offset_d = (-1, 1, 3, 6, 8, 10)
remove_tbrfexcl = True


"""
Plot Settings
"""
figfolder = 'Result'
xlabel = 'Horizontal distance [m]'
# Subsample for barymap coordinates, jump every "subsample"
subsample = 50  # int
figheight_multiplier = 1. if 'SeqTurb' not in casename else .5  # float
# Save figures and show figures
save_fig, show = True, False  # bool; bool
# If save figure, figure extension and DPI
ext, dpi = 'pdf', 300  # str; int
# Height limit for line plots
heightlim = (0., 700.)  # tuple(float)
ylim = (-1., 1.)


"""
Process User Inputs
"""
property_types = ('G', 'k', 'epsilon', 'XYbary', 'divDevR', 'U')
property_names = ('GAvg_G_pred_TBDT_G_pred_TBRF_G_pred_TBAB_G_pred_TBGB_kSGSmean_kTotal_epsilonSGSmean_epsilonTotal',
              'XYbary_XYbary_pred_TBDT_XYbary_pred_TBRF_XYbary_pred_TBAB_XYbary_pred_TBGB_divDevR_divDevR_pred_TBDT_divDevR_pred_TBRF_divDevR_pred_TBAB_divDevR_pred_TBGB_UAvg')
# property_names = ('GAvg_G_pred_TBDT_G_pred_TBRF_G_pred_TBRFexcl_G_pred_TBAB_G_pred_TBGB_kSGSmean_kTotal_epsilonSGSmean_epsilonTotal',
#                   'XYbary_XYbary_pred_TBDT_XYbary_pred_TBRF_XYbary_pred_TBRFexcl_XYbary_pred_TBAB_XYbary_pred_TBGB_divDevR_divDevR_pred_TBDT_divDevR_pred_TBRF_divDevR_pred_TBRFexcl_divDevR_pred_TBAB_divDevR_pred_TBGB_UAvg')
nlines = 6 if 'TBRFexcl' in property_names[0] else 5
# 6D, 8D, 10D downstream upwind turbine are exclusive to SeqTurb
if 'SeqTurb' in casename:
    set_types = ('oneDupstreamTurbine',
                 'oneDdownstreamTurbine',
                 'threeDdownstreamTurbine',
                 'sixDdownstreamTurbine',
                 'eightDdownstreamTurbine',
                 'tenDdownstreamTurbine')
else:
    set_types = ('oneDupstreamTurbine',
                 'oneDdownstreamTurbine',
                 'threeDdownstreamTurbine')

# Set object initialization
case = SetProperties(casename=casename, casedir=casedir, time=time)
# Read sets
case.readSets(orientation_kw=line_orient)


"""
Re-group Flow Properties
"""
# Flow property from 1st property
g = {}
k = {}
epsilon = {}
gmin, kmin, epsmin = (1e9,)*3
gmax, kmax, epsmax = (-1e9,)*3
# Flow property from 2nd property
xybary = {}
divdev_r = {}
u = {}
# Horizonal-z component split
divdev_r_xy, divdev_r_z = {}, {}
uxy, uz = {}, {}
divdev_r_xy_min, divdev_r_z_min, uxy_min, uz_min = (1e9,)*4
divdev_r_xy_max, divdev_r_z_max, uxy_max, uz_max = (-1e9,)*4
gcol = 6 if 'TBRFexcl' in property_names[0] else 5
xybary_col = 18 if 'TBRFexcl' in property_names[1] else 15
divdev_r_col = 18 if 'TBRFexcl' in property_names[1] else 15
# Go through each set location
for type in set_types:
    # GAvg, G_TBDT, G_TBRF, (G_TBRFexcl), G_TBAB, G_TBGB
    g[type] = case.data[type + line_orient + property_names[0]][:, :gcol]
    gmin_tmp, gmax_tmp = min(g[type][:, 0]), max(g[type][:, 0])
    if gmin_tmp < gmin: gmin = gmin_tmp
    if gmax_tmp > gmax: gmax = gmax_tmp
    # (kSGS), kTotal, skipping kSGS as it's useless here
    k[type] = case.data[type + line_orient + property_names[0]][:, gcol + 1]
    kmin_tmp, kmax_tmp = min(k[type]), max(k[type])
    if kmin_tmp < kmin: kmin = kmin_tmp
    if kmax_tmp > kmax: kmax = kmax_tmp
    # (epsilonSGS), epsilonTotal, skipping epsilonSGS as it's useless here
    epsilon[type] = case.data[type + line_orient + property_names[0]][:, gcol + 3]
    epsmin_tmp, epsmax_tmp = min(epsilon[type].ravel()), max(epsilon[type].ravel())
    if epsmin_tmp < epsmin: epsmin = epsmin_tmp
    if epsmax_tmp > epsmax: epsmax = epsmax_tmp

    # Note 3rd D is dummy
    xybary[type] = case.data[type + line_orient + property_names[1]][:, :xybary_col]
    divdev_r[type] = case.data[type + line_orient + property_names[1]][:, xybary_col:xybary_col + divdev_r_col]
    # Initialize horizontal-z component array for each location
    divdev_r_xy[type], divdev_r_z[type] = np.empty((len(divdev_r[type]), nlines)), np.empty((len(divdev_r[type]), nlines))
    # Assign xy and z components of div(dev(Rij))
    i, j = 0, 0
    while i < divdev_r_col:
        divdev_r_xy[type][:, j] = np.sqrt(divdev_r[type][:, i]**2 + divdev_r[type][:, i + 1]**2)
        divdev_r_z[type][:, j] = divdev_r[type][:, i + 2]
        j += 1
        i += 3

    # Find min and max of div(dev(Rij))
    divdev_r_xy_min_tmp, divdev_r_xy_max_tmp = min(divdev_r_xy[type][:, 0]), max(divdev_r_xy[type][:, 0])
    if divdev_r_xy_min_tmp < divdev_r_xy_min: divdev_r_xy_min = divdev_r_xy_min_tmp
    if divdev_r_xy_max_tmp > divdev_r_xy_max: divdev_r_xy_max = divdev_r_xy_max_tmp
    divdev_r_z_min_tmp, divdev_r_z_max_tmp = min(divdev_r_z[type][:, 0]), max(divdev_r_z[type][:, 0])
    if divdev_r_z_min_tmp < divdev_r_z_min: divdev_r_z_min = divdev_r_z_min_tmp
    if divdev_r_z_max_tmp > divdev_r_z_max: divdev_r_z_max = divdev_r_z_max_tmp

    u[type] = case.data[type + line_orient + property_names[1]][:, -3:]
    # Assign xy and z component of U
    uxy[type] = np.sqrt(u[type][:, 0]**2 + u[type][:, 1]**2)
    uz[type] = u[type][:, 2]
    # Min and max of U
    uxy_min = min(min(uxy[type]), uxy_min)
    uxy_max = max(max(uxy[type]), uxy_max)
    uz_min = min(min(uz[type]), uz_min)
    uz_max = max(max(uz[type]), uz_max)


"""
Plot G Lines for Several Locations
"""
def prepareLineValues(set_types, xlocs, val_dict, val_lim, lim_multiplier=4., linespan=8., normalize=False):
    """
    Prepare values line dictionary by scaling it to within the linespan.
    """
    relaxed_lim = (val_lim[1] - val_lim[0])*lim_multiplier if not normalize else max(val_lim[1], abs(val_lim[0]))
    scale = linespan/relaxed_lim if not normalize else 2./relaxed_lim
    print('Scale is {}'.format(scale))
    for i, type in enumerate(set_types):
        # # Limit if prediction is too off
        # val_dict[type][
        #     val_dict[type] < val_lim[0]*lim_multiplier] = val_lim[0]*lim_multiplier
        # val_dict[type][
        #     val_dict[type] > val_lim[1]*lim_multiplier] = val_lim[1]*lim_multiplier
        val_dict[type] = val_dict[type]*scale
        # Shift them to their respective x location
        val_dict[type] = val_dict[type] + xlocs[i]

    return val_dict

def plotOneTurbAuxiliary(plot_obj, xlim, ylim):
    """
    Plot shaded area for OneTurb case since some areas are not predicted.
    Also plot turbine location.
    """
    # plot_obj.axes.fill_between(xlim, ylim[1] - 252, ylim[1], facecolor=plot_obj.gray, alpha=0.25, zorder=100)
    # plot_obj.axes.fill_between(xlim, ylim[0], ylim[0] + 252, facecolor=plot_obj.gray, alpha=0.25, zorder=100)
    # plot_obj.axes.plot(np.zeros(2), [ylim[1] - 378, ylim[1] - 504], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.zeros(2), [-.5, .5], color=plot_obj.gray, alpha=1., ls='-', zorder=-10)

def plotParTurbYawAuxiliary(plot_obj, ylim):
    """
    Plot turbine locations for ParTurb with 10 deg yaw in southern turbine and 20 deg yaw in northern turbine.
    """
    plot_obj.axes.plot((63/200.*np.sin(1/18.*np.pi), -63/200.*np.sin(1/18.*np.pi)), [126, 252], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot((63/200.*np.sin(1/9.*np.pi), -63/200.*np.sin(1/9.*np.pi)), [630, 756], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    # plot_obj.axes.plot(np.zeros(2), [-2.5, -1.5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    # plot_obj.axes.plot(np.zeros(2), [1.5, 2.5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)

def plotParTurbAuxiliary(plot_obj, ylim):
    """
    Plot turbine locations for ParTurb.
    """
    # plot_obj.axes.plot(np.zeros(2), [ylim[1] - 126, ylim[1] - 252], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    # plot_obj.axes.plot(np.zeros(2), [ylim[1] - 630, ylim[1] - 756], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.zeros(2), [-2.5, -1.5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.zeros(2), [1.5, 2.5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)

def plotSeqTurbAuxiliary(plot_obj, xlim, ylim):
    """
    Plot shaded area for SeqTurb case since some areas are not predicted.
    Also plot turbine locations.
    """
    # plot_obj.axes.fill_between(xlim, ylim[1] - 252, ylim[1], facecolor=plot_obj.gray, alpha=0.25, zorder=100)
    # plot_obj.axes.fill_between(xlim, ylim[0], ylim[0] + 252, facecolor=plot_obj.gray, alpha=0.25, zorder=100)
    # # Flip y
    # plot_obj.axes.plot(np.zeros(2), [ylim[1] - 378., ylim[1] - 504.], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    # plot_obj.axes.plot(np.ones(2)*7, [ylim[1] - 378., ylim[1] - 504.], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.zeros(2), [-.5, .5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.ones(2)*7, [-.5, .5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)

figwidth = 'full' if 'SeqTurb' in casename else 'half'
labels = ('LES', 'TBDT', 'TBRF')
labels2 = ('LES', 'TBAB', 'TBGB')
ls = ('-', '-.', '--')
xlim = (-2, 5) if figwidth == 'half' else (-2, 12)

# Normalized G = G/|G|_max
g = prepareLineValues(set_types, offset_d, g, (gmin, gmax), normalize=True)
figname, figname2 = 'G1', 'G2'
# Normalized G, namely P_k/|P_k|_max
xlabel, ylabel = (r'$\langle P_k \rangle/0.5|\langle P_k \rangle|_\mathrm{max} + d/D$', r'$d/D$')

# Normalized y
list_y = [case.coor[set_types[i] + line_orient + property_names[0]]/126. for i in range(len(set_types))]
# Mean of y for the front/one/parallel and rear turbine
ymean = (max(list_y[0]) + min(list_y[0]))/2.
ymean1 = (max(list_y[-1]) + min(list_y[-1]))/2.
# Center y around 0 y/D
for i in range(len(list_y)):
    list_y[i] -= ymean if i < 3 else ymean1
    
# if figwidth == 'full':
#     yoffset = 255.
#     for i in range(3, 6):
#         list_y[i] += yoffset

list_x = [g[set_types[i]] for i in range(len(set_types))]
# First plot of G for LES, TBDT, TBRF
gplot = Plot2D(list_x=[None], list_y=[None], name=figname, xlabel=xlabel, ylabel=ylabel,
               save=save_fig, show=show,
               figdir=case.result_path,
               figwidth=figwidth, xlim=xlim, figheight_multiplier=figheight_multiplier, 
               ylim=ylim)
gplot.initializeFigure()
gplot.markercolors = None
# Go through each line location
for i in range(len(list_x)):
    # Then for each location, go through each line
    # for j in range(3):
    #     label = labels[j] if i == 0 else None
    #     # Plot each prediction and flip y
    #     gplot.axes.plot(list_x[i][:, j], -list_y[i], label=label, color=gplot.colors[j], alpha=.8, ls=ls[j])

    # Plot LES, TBRF, TBAB
    gplot.axes.plot(list_x[i][:, 0], -list_y[i], label='LES', color=gplot.colors[0], alpha=.8, ls=ls[0])
    gplot.axes.plot(list_x[i][:, 2], -list_y[i], label='TBRF', color=gplot.colors[1], alpha=.8, ls=ls[1])
    gplot.axes.plot(list_x[i][:, 3], -list_y[i], label='TBAB', color=gplot.colors[2], alpha=.8, ls=ls[2])

# Plot turbine and shade unpredicted area too
if 'OneTurb' in casename:
    plotOneTurbAuxiliary(gplot, xlim, (min(list_y[0]), max(list_y[0])))
elif 'ParTurb' in casename:
    if 'Yaw' in casename:
        plotParTurbYawAuxiliary(gplot, (min(list_y[0]), max(list_y[0])))
    else:
        plotParTurbAuxiliary(gplot, (min(list_y[0]), max(list_y[0])))

else:
    plotSeqTurbAuxiliary(gplot, xlim, (min(list_y[0]), max(list_y[0])))

# Plot vertical dotted lines to indicate where the lines are samples
gplot.axes.plot((-1, -1), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')
gplot.axes.plot((1, 1), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')
gplot.axes.plot((3, 3), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')
gplot.axes.plot((6, 6), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')
gplot.axes.plot((8, 8), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')
gplot.axes.plot((10, 10), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')
# uxy_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
plt.xticks(np.arange(-1, 12), ('-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'))
# plt.xticks((-1, 1, 3, 5, 6, 8, 10), ('0', '1', '', '', '', '', '', '', '', '', ''))
gplot.axes.grid(which='major', alpha=.25)
gplot.axes.set_xlabel(gplot.xlabel)
gplot.axes.set_ylabel(gplot.ylabel)
gplot.axes.set_ylim(gplot.ylim)
# Don't plot legend cuz no space
# gplot.axes.legend(loc='best')
plt.savefig(gplot.figdir + '/' + gplot.name + '.' + ext, transparent=False,
            dpi=dpi)
print('\nFigure ' + gplot.name + '.' + ext + ' saved in ' + gplot.figdir)
plt.show() if gplot.show else plt.close()
# gplot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
# gplot.finalizeFigure(showleg=False)

# Second plot for LES, TBAB, TBGB
gplot2 = Plot2D(list_x=list_x, list_y=list_y, name=figname2, xlabel=xlabel, ylabel=ylabel,
               save=save_fig, show=show,
               figdir=case.result_path,
               figwidth=figwidth, xlim=xlim, figheight_multiplier=figheight_multiplier)
gplot2.initializeFigure()
gplot2.markercolors = None
# Go through each line location
for i in range(len(list_x)):
    # Then for each location, go through each (LES, TBAB, TBGB) line
    labels2_cpy = labels2 if i == 0 else (None, None, None)
    # Flip y
    gplot2.axes.plot(list_x[i][:, 0], max(list_y[0]) - list_y[i], label=labels2_cpy[0], color=gplot2.colors[0], alpha=.8, ls=ls[0])
    gplot2.axes.plot(list_x[i][:, 3], max(list_y[0]) - list_y[i], label=labels2_cpy[1], color=gplot2.colors[1], alpha=.8, ls=ls[1])
    gplot2.axes.plot(list_x[i][:, 4], max(list_y[0]) - list_y[i], label=labels2_cpy[2], color=gplot2.colors[2], alpha=.8, ls=ls[2])

# Plot turbine too
if 'OneTurb' in casename:
    plotOneTurbAuxiliary(gplot2, xlim, (min(list_y[0]), max(list_y[0])))
elif 'ParTurb' in casename:
    if 'Yaw' in casename:
        plotParTurbYawAuxiliary(gplot2, (min(list_y[0]), max(list_y[0])))
    else:
        plotParTurbAuxiliary(gplot2, (min(list_y[0]), max(list_y[0])))

else:
    plotSeqTurbAuxiliary(gplot2, xlim, (min(list_y[0]), max(list_y[0])))

gplot2.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
gplot2.finalizeFigure(showleg=False)


"""
Plot divDevR Lines for Several Locations
"""
# Prepare values to scale with x in the plot
# for i, type in enumerate(set_types):
#     divdev_r_xy[type][divdev_r_xy[type] > .1*divdev_r_xy_max] = .1*divdev_r_xy_max
#
# divdev_r_xy_max *= .1

divdev_r_xy = prepareLineValues(set_types, offset_d, divdev_r_xy, (divdev_r_xy_min, divdev_r_xy_max),
                                lim_multiplier=2., linespan=4.)
divdev_r_z = prepareLineValues(set_types, offset_d, divdev_r_z, (divdev_r_z_min, divdev_r_z_max),
                                lim_multiplier=4., linespan=4.)

figname, figname2, figname3, figname4 = 'divDevRxy1', 'divDevRxy2', 'divDevRz1', 'divDevRz2'
list_y = [case.coor[set_types[i] + line_orient + property_names[0]][::10] for i in range(len(set_types))]

list_x = [divdev_r_xy[set_types[i]][::10] for i in range(len(set_types))]
# Limit HiSpeed LES noise
if 'HiSpeed' in casename: list_x[0][0, 0] = 1.
list_x2 = [divdev_r_z[set_types[i]][::10] for i in range(len(set_types))]
# First plot of divDevR in xy for LES, TBDT, TBRF
divdevr_xy_plot = Plot2D(list_x=list_x, list_y=list_y, name=figname, xlabel=xlabel, ylabel=ylabel,
               save=save_fig, show=show,
               figdir=case.result_path,
               figwidth=figwidth, xlim=xlim, figheight_multiplier=figheight_multiplier)
divdevr_xy_plot.initializeFigure()
divdevr_xy_plot.markercolors = None
# Go through each line location
for i in range(len(list_x)):
    # Then for each location, go through each line
    for j in range(3):
        label = labels[j] if i == 0 else None
        divdevr_xy_plot.axes.plot(list_x[i][:, j], max(list_y[0]) - list_y[i], label=label, color=divdevr_xy_plot.colors[j], alpha=.8, ls=ls[j])

# Plot turbine and shade unpredicted area too
if 'OneTurb' in casename:
    plotOneTurbAuxiliary(divdevr_xy_plot, xlim, (min(list_y[0]), max(list_y[0])))
elif 'ParTurb' in casename:
    if 'Yaw' in casename:
        plotParTurbYawAuxiliary(divdevr_xy_plot, (min(list_y[0]), max(list_y[0])))
    else:
        plotParTurbAuxiliary(divdevr_xy_plot, (min(list_y[0]), max(list_y[0])))

else:
    plotSeqTurbAuxiliary(divdevr_xy_plot, xlim, (min(list_y[0]), max(list_y[0])))

divdevr_xy_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
divdevr_xy_plot.finalizeFigure(showleg=False)

# Second plot of divDevR in xy for LES, TBAB, TBGB
divdevr_xy_plot2 = Plot2D(list_x=list_x, list_y=list_y, name=figname2, xlabel=xlabel, ylabel=ylabel,
                save=save_fig, show=show,
                figdir=case.result_path,
                figwidth=figwidth, xlim=xlim, figheight_multiplier=figheight_multiplier)
divdevr_xy_plot2.initializeFigure()
divdevr_xy_plot2.markercolors = None
# Go through each line location
for i in range(len(list_x)):
    # Then for each location, go through each (LES, TBAB, TBGB) line
    labels2_cpy = labels2 if i == 0 else (None, None, None)
    divdevr_xy_plot2.axes.plot(list_x[i][:, 0], max(list_y[0]) - list_y[i], label=labels2_cpy[0], color=divdevr_xy_plot2.colors[0], alpha=.8, ls=ls[0])
    divdevr_xy_plot2.axes.plot(list_x[i][:, 3], max(list_y[0]) - list_y[i], label=labels2_cpy[1], color=divdevr_xy_plot2.colors[1], alpha=.8, ls=ls[1])
    divdevr_xy_plot2.axes.plot(list_x[i][:, 4], max(list_y[0]) - list_y[i], label=labels2_cpy[2], color=divdevr_xy_plot2.colors[2], alpha=.8, ls=ls[2])

if 'OneTurb' in casename:
    plotOneTurbAuxiliary(divdevr_xy_plot2, xlim, (min(list_y[0]), max(list_y[0])))
elif 'ParTurb' in casename:
    if 'Yaw' in casename:
        plotParTurbYawAuxiliary(divdevr_xy_plot2, (min(list_y[0]), max(list_y[0])))
    else:
        plotParTurbAuxiliary(divdevr_xy_plot2, (min(list_y[0]), max(list_y[0])))

else:
    plotSeqTurbAuxiliary(divdevr_xy_plot2, xlim, (min(list_y[0]), max(list_y[0])))

divdevr_xy_plot2.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
divdevr_xy_plot2.finalizeFigure(showleg=False)

# First plot of divDevR in z for LES, TBDT, TBRF
divdevr_z_plot = Plot2D(list_x=list_x2, list_y=list_y, name=figname3, xlabel=xlabel, ylabel=ylabel,
                         save=save_fig, show=show,
                         figdir=case.result_path,
                         figwidth=figwidth, xlim=xlim, figheight_multiplier=figheight_multiplier)
divdevr_z_plot.initializeFigure()
divdevr_z_plot.markercolors = None
# Go through each line location
for i in range(len(list_x2)):
    # Then for each location, go through each line
    for j in range(3):
        label = labels[j] if i == 0 else None
        divdevr_z_plot.axes.plot(list_x2[i][:, j], max(list_y[0]) - list_y[i], label=label, color=divdevr_xy_plot.colors[j], alpha=.8, ls=ls[j])

# Plot turbine and shade unpredicted area too
if 'OneTurb' in casename:
    plotOneTurbAuxiliary(divdevr_z_plot, xlim, (min(list_y[0]), max(list_y[0])))
elif 'ParTurb' in casename:
    if 'Yaw' in casename:
        plotParTurbYawAuxiliary(divdevr_z_plot, (min(list_y[0]), max(list_y[0])))
    else:
        plotParTurbAuxiliary(divdevr_z_plot, (min(list_y[0]), max(list_y[0])))

else:
    plotSeqTurbAuxiliary(divdevr_z_plot, xlim, (min(list_y[0]), max(list_y[0])))

divdevr_z_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
divdevr_z_plot.finalizeFigure(showleg=False)

# Second plot of divDevR in z for LES, TBAB, TBGB
divdevr_z_plot2 = Plot2D(list_x=list_x2, list_y=list_y, name=figname4, xlabel=xlabel, ylabel=ylabel,
                          save=save_fig, show=show,
                          figdir=case.result_path,
                          figwidth=figwidth, xlim=xlim, figheight_multiplier=figheight_multiplier)
divdevr_z_plot2.initializeFigure()
divdevr_z_plot2.markercolors = None
# Go through each line location
for i in range(len(list_x2)):
    # Then for each location, go through each (LES, TBAB, TBGB) line
    labels2_cpy = labels2 if i == 0 else (None, None, None)
    divdevr_z_plot2.axes.plot(list_x2[i][:, 0], max(list_y[0]) - list_y[i], label=labels2_cpy[0], color=divdevr_xy_plot2.colors[0], alpha=.8, ls=ls[0])
    divdevr_z_plot2.axes.plot(list_x2[i][:, 3], max(list_y[0]) - list_y[i], label=labels2_cpy[1], color=divdevr_xy_plot2.colors[1], alpha=.8, ls=ls[1])
    divdevr_z_plot2.axes.plot(list_x2[i][:, 4], max(list_y[0]) - list_y[i], label=labels2_cpy[2], color=divdevr_xy_plot2.colors[2], alpha=.8, ls=ls[2])

if 'OneTurb' in casename:
    plotOneTurbAuxiliary(divdevr_z_plot2, xlim, (min(list_y[0]), max(list_y[0])))
elif 'ParTurb' in casename:
    if 'Yaw' in casename:
        plotParTurbYawAuxiliary(divdevr_z_plot2, (min(list_y[0]), max(list_y[0])))
    else:
        plotParTurbAuxiliary(divdevr_z_plot2, (min(list_y[0]), max(list_y[0])))
else:
    plotSeqTurbAuxiliary(divdevr_z_plot2, xlim, (min(list_y[0]), max(list_y[0])))

divdevr_z_plot2.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
divdevr_z_plot2.finalizeFigure(showleg=False)


