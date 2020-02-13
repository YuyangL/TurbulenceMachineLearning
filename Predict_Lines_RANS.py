import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from joblib import load
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
casename = 'N_H_ParTurb_LowZ_Rwall'  # 'N_H_OneTurb_LowZ_Rwall2', 'N_H_ParTurb_LowZ_Rwall'
casename_les = 'ALM_N_H_ParTurb2'  # 'ALM_N_H_OneTurb', 'ALM_N_H_ParTurb2'
# Absolute parent directory of ML and test case
casedir = '/media/yluan/RANS'  # str
casedir_les = '/media/yluan'  # str
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
# Field rotation for vertical slices, rad or deg
fieldrot = 30.  # float
# Subsample for barymap coordinates, jump every "subsample"
subsample = 50  # int
figheight_multiplier = 1.  # float
# Save figures and show figures
save_fig, show = True, False  # bool; bool
# If save figure, figure extension and DPI
ext, dpi = 'pdf', 300  # str; int
# Height limit for line plots
heightlim = (0., 700.)  # (float, float)


"""
Process User Inputs, Don't Change
"""
property_types = ('G', 'k', 'epsilon', 'XYbary', 'divDevR_blend', 'U')
property_types_les = ('GAvg', 'kTotal', 'epsilonTotal', 'XYbary', 'divDevR', 'UAvg')
property_names = ('G_G_pred_TBDT_G_pred_TBRF_G_pred_TBAB_G_pred_TBGB_k_epsilon',
                  'XYbary_XYbary_pred_TBDT_XYbary_pred_TBRF_XYbary_pred_TBAB_XYbary_pred_TBGB_divDevR_blend_divDevR_pred_TBDT_divDevR_pred_TBRF_divDevR_pred_TBAB_divDevR_pred_TBGB_U')
property_names_blend = ('G_k_epsilon', 'divDevR_blend_U')
property_names_les = ('GAvg_G_pred_TBDT_G_pred_TBRF_G_pred_TBAB_G_pred_TBGB_kSGSmean_kTotal_epsilonSGSmean_epsilonTotal',
                      'XYbary_XYbary_pred_TBDT_XYbary_pred_TBRF_XYbary_pred_TBAB_XYbary_pred_TBGB_divDevR_divDevR_pred_TBDT_divDevR_pred_TBRF_divDevR_pred_TBAB_divDevR_pred_TBGB_UAvg')
nlines = 6 if 'TBRFexcl' in property_names[0] else 5
# 6D, 8D, 10D downstream upwind turbine are exclusive to SeqTurb
if 'SeqTurb' in casename:
    set_locs = ('oneDupstreamTurbine',
                 'oneDdownstreamTurbine',
                 'threeDdownstreamTurbine',
                 'sixDdownstreamTurbine',
                 'eightDdownstreamTurbine',
                 'tenDdownstreamTurbine')
else:
    set_locs = ('oneDupstreamTurbine',
                 'oneDdownstreamTurbine',
                 'threeDdownstreamTurbine')
    # Final time step of pure RANS, bij injected RANS, and pure LES
    if 'ParTurb' in casename:
        time, time_blend, time_les = '10000', 'latestTime', 'latestTime'
    else:
        time, time_blend, time_les = '5000', '24000', 'latestTime'


# Set object initialization
# Pure RANS
case = SetProperties(casename=casename, casedir=casedir, time=time)
# bij injected RANS
case_blend = SetProperties(casename=casename, casedir=casedir, time=time_blend)
# Pure LES
case_les = SetProperties(casename=casename_les, casedir=casedir_les, time=time_les)
# Read sets
case.readSets(orientation_kw=line_orient)
case_blend.readSets(orientation_kw=line_orient)
case_les.readSets(orientation_kw=line_orient)


"""
Re-group Flow Properties
"""
def readDataFromSet(set_obj, set_locs, line_orient, property_names, n_estimators=5):
    """
    Read and split columns to corresponding variable arrays from an ensemble of columns
    """
    g, k, epsilon = {}, {}, {}
    gmin, kmin, epsmin = (1e9,)*3
    gmax, kmax, epsmax = (-1e9,)*3
    # Flow property from 2nd property
    xybary, divdev_r, u = {}, {}, {}
    # Horizonal-z component split
    divdev_r_xy, divdev_r_z = {}, {}
    uxy, uz = {}, {}
    divdev_r_xy_min, divdev_r_z_min, uxy_min, uz_min = (1e9,)*4
    divdev_r_xy_max, divdev_r_z_max, uxy_max, uz_max = (-1e9,)*4
    gcol = n_estimators
    xybary_col = n_estimators*3
    divdev_r_col = n_estimators*3
    for setloc in set_locs:
    # GAvg, G_TBDT, G_TBRF, (G_TBRFexcl), G_TBAB, G_TBGB
        g[setloc] = set_obj.data[setloc + line_orient + property_names[0]][:, :gcol]
        gmin_tmp, gmax_tmp = min(g[setloc][:, 0]), max(g[setloc][:, 0])
        if gmin_tmp < gmin: gmin = gmin_tmp
        if gmax_tmp > gmax: gmax = gmax_tmp
        # (kSGS), kTotal, skipping kSGS as it's useless here
        kcol = gcol + 1 if 'SGS' in property_names[0] else gcol
        k[setloc] = set_obj.data[setloc + line_orient + property_names[0]][:, kcol]
        kmin_tmp, kmax_tmp = min(k[setloc]), max(k[setloc])
        if kmin_tmp < kmin: kmin = kmin_tmp
        if kmax_tmp > kmax: kmax = kmax_tmp
        # (epsilonSGS), epsilonTotal, skipping epsilonSGS as it's useless here
        epscol = gcol + 3 if 'SGS' in property_names[0] else gcol + 1
        epsilon[setloc] = set_obj.data[setloc + line_orient + property_names[0]][:, epscol]
        epsmin_tmp, epsmax_tmp = min(epsilon[setloc].ravel()), max(epsilon[setloc].ravel())
        if epsmin_tmp < epsmin: epsmin = epsmin_tmp
        if epsmax_tmp > epsmax: epsmax = epsmax_tmp

        # Note 3rd D is dummy
        xybary[setloc] = set_obj.data[setloc + line_orient + property_names[1]][:, :xybary_col]
        divdev_r[setloc] = set_obj.data[setloc + line_orient + property_names[1]][:, xybary_col:xybary_col + divdev_r_col]
        # Initialize horizontal-z component array for each location
        divdev_r_xy[setloc], divdev_r_z[setloc] = np.empty((len(divdev_r[setloc]), nlines)), np.empty(
                (len(divdev_r[setloc]), nlines))
        # Assign xy and z components of div(dev(Rij))
        i, j = 0, 0
        while i < divdev_r_col:
            divdev_r_xy[setloc][:, j] = np.sqrt(divdev_r[setloc][:, i]**2 + divdev_r[setloc][:, i + 1]**2)
            divdev_r_z[setloc][:, j] = divdev_r[setloc][:, i + 2]
            j += 1
            i += 3

        # Find min and max of div(dev(Rij))
        divdev_r_xy_min_tmp, divdev_r_xy_max_tmp = min(divdev_r_xy[setloc][:, 0]), max(divdev_r_xy[setloc][:, 0])
        if divdev_r_xy_min_tmp < divdev_r_xy_min: divdev_r_xy_min = divdev_r_xy_min_tmp
        if divdev_r_xy_max_tmp > divdev_r_xy_max: divdev_r_xy_max = divdev_r_xy_max_tmp
        divdev_r_z_min_tmp, divdev_r_z_max_tmp = min(divdev_r_z[setloc][:, 0]), max(divdev_r_z[setloc][:, 0])
        if divdev_r_z_min_tmp < divdev_r_z_min: divdev_r_z_min = divdev_r_z_min_tmp
        if divdev_r_z_max_tmp > divdev_r_z_max: divdev_r_z_max = divdev_r_z_max_tmp

        u[setloc] = set_obj.data[setloc + line_orient + property_names[1]][:, -3:]
        # Assign xy and z component of U
        uxy[setloc] = np.sqrt(u[setloc][:, 0]**2 + u[setloc][:, 1]**2)
        uz[setloc] = u[setloc][:, 2]
        # Min and max of U
        uxy_min = min(min(uxy[setloc]), uxy_min)
        uxy_max = max(max(uxy[setloc]), uxy_max)
        uz_min = min(min(uz[setloc]), uz_min)
        uz_max = max(max(uz[setloc]), uz_max)

    glim = [gmin, gmax]
    klim = [kmin, kmax]
    epslim = [epsmin, epsmax]
    divdev_r_lim = [[divdev_r_xy_min, divdev_r_xy_max],
                    [divdev_r_z_min, divdev_r_z_max]]
    ulim = [[uxy_min, uxy_max], [uz_min, uz_max]]
    return g, k, epsilon, divdev_r_xy, divdev_r_z, uxy, uz, glim, klim, epslim, divdev_r_lim, ulim

# Go through each set location
# Pure RANS
g, k, epsilon, divdev_r_xy, divdev_r_z, uxy, uz, glim, klim, epslim, divdev_r_lim, ulim = readDataFromSet(case, set_locs, line_orient, property_names)
# Pure LES
g_les, k_les, epsilon_les, divdev_r_xy_les, divdev_r_z_les, uxy_les, uz_les, glim_les, klim_les, epslim_les, divdev_r_lim_les, ulim_les = readDataFromSet(case_les,
                                                                                                                                                          set_locs,
                                                                                                                                                          line_orient,
                                                                                                                                                          property_names_les)
# bij injected RANS
g_blend, k_blend, epsilon_blend = {}, {}, {}
divdev_r_blend, divdev_r_xy_blend, divdev_r_z_blend = {}, {}, {}
u_blend, uxy_blend, uz_blend = {}, {}, {}
for setloc in set_locs:
    g_blend[setloc] = case_blend.data[setloc + line_orient + property_names_blend[0]][:, 0]
    k_blend[setloc] = case_blend.data[setloc + line_orient + property_names_blend[0]][:, 1]
    epsilon_blend[setloc] = case_blend.data[setloc + line_orient + property_names_blend[0]][:, 2]
    divdev_r_blend[setloc] = case_blend.data[setloc + line_orient + property_names_blend[1]][:, :3]
    divdev_r_xy_blend[setloc] = np.sqrt(divdev_r_blend[setloc][:, 0]**2 + divdev_r_blend[setloc][:, 1]**2)
    divdev_r_z_blend[setloc] = divdev_r_blend[setloc][:, 2]
    u_blend[setloc] = case_blend.data[setloc + line_orient + property_names_blend[1]][:, 3:]
    uxy_blend[setloc] = np.sqrt(u_blend[setloc][:, 0]**2. + u_blend[setloc][:, 1]**2.)
    uz_blend[setloc] = u_blend[setloc][:, 2]

glim[0], glim[1] = min(glim[0], glim_les[0]), max(glim[1], glim_les[1])
klim[0], klim[1] = min(klim[0], klim_les[0]), max(klim[1], klim_les[1])
epslim[0], epslim[1] = min(epslim[0], epslim_les[0]), max(epslim[1], epslim_les[1])
for i in range(2):
    # First div(dev(Rij))_xy then div(dev(Rij))_z
    divdev_r_lim[i][0], divdev_r_lim[i][1] = min(divdev_r_lim[i][0], divdev_r_lim_les[i][0]), max(divdev_r_lim[i][1], divdev_r_lim_les[i][1])
    # First u_xy then u_z
    ulim[i][0], ulim[i][1] = min(ulim[i][0], ulim_les[i][0]), max(ulim[i][1],
                                                                                                  ulim_les[i][1])


"""
Prepare Plots By Rearranging Arrays 
"""
def regroupValueDictionary(set_locs, val_dict_rans, val_dict_blend, val_dict_les):
    """
    Regroup a scalar variable into the order of LES -> 1 estimator / blended injection -> RANS, for each set location. There can be no estimators, useful for variables e.g. U, TKE, epsilon.
    The return is a dictionary of set locations while, at each location, there's a list containing [LES data, ML/blended data, RANS data]
    """
    val_plots = {}
    for loc in set_locs:
        val_plots[loc] = []
        try:
            # Both pure RANS and ML prediction results are stored in val_dict_rans
            val_rans = val_dict_rans[loc][:, 0]
            val_ml = val_dict_rans[loc][:, 1:]
            # Number of ML estimators is the number of columns
            n_estimators = val_ml.shape[1]
            # LES data also has n_estimators + 1 columns representing truth -> ML, but we only need truth
            val_les = val_dict_les[loc][:, 0]
        except:
            val_rans = val_dict_rans[loc]
            val_les = val_dict_les[loc]
            n_estimators = 0

        val_blend = val_dict_blend[loc]
        for i in range(n_estimators):
            val_plots[loc].append([val_les, val_ml[:, i], val_rans])

        val_plots[loc].append([val_les, val_blend, val_rans])

    return val_plots




def prepareLineValues(set_locs, xlocs, val_dict, val_lim, lim_multiplier=4., linespan=8., normalize=False):
    """
    Prepare values line dictionary by scaling it to within the linespan.
    If normalize is True, lim_multiplier and linespan are ignored.
    """
    from functools import partial
    def forloop(jk_inputs, val_dict, setloc, scale, xloc):
        j, k = jk_inputs
        # Inplace update
        val_dict[setloc][j][k] = val_dict[setloc][j][k]*scale
        val_dict[setloc][j][k] = val_dict[setloc][j][k] + xloc
        # return val_dict

    relaxed_lim = (val_lim[1] - val_lim[0])*lim_multiplier if not normalize else max(val_lim[1], abs(val_lim[0])) #(val_lim[1] - val_lim[0])
    scale = linespan/relaxed_lim if not normalize else 2./relaxed_lim

    print('Scale is {}'.format(scale))
    for i, setloc in enumerate(set_locs):
        # # Limit if prediction is too off
        # val_dict[setloc][
        #     val_dict[setloc] < val_lim[0]*lim_multiplier] = val_lim[0]*lim_multiplier
        # val_dict[setloc][
        #     val_dict[setloc] > val_lim[1]*lim_multiplier] = val_lim[1]*lim_multiplier
        if not isinstance(val_dict[setloc], (list, tuple)):
            val_dict[setloc] = val_dict[setloc]*scale
            # Shift them to their respective x location
            val_dict[setloc] = val_dict[setloc] + xlocs[i]
        else:
            jklst = []
            for j in range(len(val_dict[setloc])):
                for k in range(len(val_dict[setloc][j])):
                    jklst.append((j, k))

            # Mapping all possible j and k indices
            list(map(partial(forloop, val_dict=val_dict, setloc=setloc, scale=scale, xloc=xlocs[i]), jklst))

            # # For a set location, go through all figures
            # for j in range(len(val_dict[setloc])):
            #     # For each figure, go through all lines
            #     for k in range(len(val_dict[setloc][j])):
            #         val_dict[setloc][j][k] = val_dict[setloc][j][k]*scale
            #         val_dict[setloc][j][k] = val_dict[setloc][j][k] + xlocs[i]

    return val_dict

def plotOneTurbAuxiliary(plot_obj, xlim, ylim):
    """
    Plot shaded area for OneTurb case since some areas are not predicted.
    Also plot turbine location.
    """
    # plot_obj.axes.fill_between(xlim, ylim[1] - 252, ylim[1], facecolor=plot_obj.gray, alpha=0.25, zorder=100)
    # plot_obj.axes.fill_between(xlim, ylim[0], ylim[0] + 252, facecolor=plot_obj.gray, alpha=0.25, zorder=100)
    # plot_obj.axes.plot(np.zeros(2), [378, 504], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.zeros(2), [-.5, .5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)

def plotParTurbAuxiliary(plot_obj):
    """
    Plot turbine locations for ParTurb.
    """
    # plot_obj.axes.plot(np.zeros(2), [126, 252], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    # plot_obj.axes.plot(np.zeros(2), [630, 756], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.zeros(2), [-2.5, -1.5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.zeros(2), [1.5, 2.5], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)

def plotSeqTurbAuxiliary(plot_obj, xlim, ylim):
    """
    Plot shaded area for SeqTurb case since some areas are not predicted.
    Also plot turbine locations.
    """
    plot_obj.axes.fill_between(xlim, ylim[1] - 252, ylim[1], facecolor=plot_obj.gray, alpha=0.25, zorder=100)
    plot_obj.axes.fill_between(xlim, ylim[0], ylim[0] + 252, facecolor=plot_obj.gray, alpha=0.25, zorder=100)
    plot_obj.axes.plot(np.zeros(2), [378, 504], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)
    plot_obj.axes.plot(np.ones(2)*7, [378, 504], color=plot_obj.gray, alpha=1, ls='-', zorder=-10)

figwidth = 'full' if 'SeqTurb' in casename else 'half'
labels = (('LES', 'TBDT', 'RANS'),
('LES', 'TBRF', 'RANS'),
('LES', 'TBAB', 'RANS'),
('LES', 'TBGB', 'RANS'),
('LES', 'Data-driven RANS', 'RANS'))
# Because of the above plot arrangement, regroup each variable to have LES, ML, RANS order
g_plots = regroupValueDictionary(set_locs, g, g_blend, g_les)
k_plots = regroupValueDictionary(set_locs, k, k_blend, k_les)
epsilon_plots = regroupValueDictionary(set_locs, epsilon, epsilon_blend, epsilon_les)
divdev_r_xy_plots = regroupValueDictionary(set_locs, divdev_r_xy, divdev_r_xy_blend, divdev_r_xy_les)
divdev_r_z_plots = regroupValueDictionary(set_locs, divdev_r_z, divdev_r_z_blend, divdev_r_z_les)
uxy_plots = regroupValueDictionary(set_locs, uxy, uxy_blend, uxy_les)
uz_plots = regroupValueDictionary(set_locs, uz, uz_blend, uz_les)
ls = ('-.', '-', '--')
xlim = (-2, 5) if figwidth == 'half' else (-2, 12)
ylim = (-1., 1.) if 'ParTurb' not in casename else (-3.5, 3.5)
# Prepare line values
# g = prepareLineValues(set_locs, offset_d, g_plots, glim)
# Normalized G = G/|G|_max
g = prepareLineValues(set_locs, offset_d, g_plots, glim, normalize=True)
k = prepareLineValues(set_locs, offset_d, k_plots, klim, linespan=4.)
epsilon = prepareLineValues(set_locs, offset_d, epsilon_plots, epslim, linespan=4.)
# Normalized divDevR_hor = divDevR_hor/|divDevR_hor|_max
# divdev_r_xy = prepareLineValues(set_locs, offset_d, divdev_r_xy_plots, divdev_r_lim[0])
divdev_r_xy = prepareLineValues(set_locs, offset_d, divdev_r_xy_plots, divdev_r_lim[0], normalize=True)
divdev_r_z = prepareLineValues(set_locs, offset_d, divdev_r_z_plots, divdev_r_lim[1])
uxy = prepareLineValues(set_locs, offset_d, uxy_plots, ulim[0], linespan=2.)
# Normalized uz = uz/|uz|_max
# uz = prepareLineValues(set_locs, offset_d, uz_plots, ulim[1])
uz = prepareLineValues(set_locs, offset_d, uz_plots, ulim[1], normalize=True)


"""
Plot G for 3 Horizontal Lines in 5 figures
"""
# 5 plots for each variables, starting with G
fignames = ["G" + str(i) for i in range(5)]
# Normalized G, renamed to P_k
xlabel, ylabel = (r'$P_k/|P_k|_\mathrm{max}$', r'$d/D$')
# Normalized y
list_y = [case.coor[set_locs[i] + line_orient + property_names[0]]/126.
          for i in range(len(set_locs))]
# Normalized y mean
ymean = (max(list_y[0]) + min(list_y[0]))/2.
ymean1 = (max(list_y[-1]) + min(list_y[-1]))/2.
# Center y around 0 y/D
# list_y = [list_y[i] - ymean if i < 3 else ymean1 for i in range(len(list_y))]
for i in range(len(list_y)):
    list_y[i] -= ymean if i < 3 else ymean1


# [DEPRECATED]
if figwidth == 'full':
    yoffset = 255.
    for i in range(3, 6):
        list_y[i] += yoffset

for i0 in range(len(fignames)):
    list_x = [np.array(g[set_locs[i]][i0]).T for i in range(len(set_locs))]
    gplot = Plot2D(list_x=list_x, list_y=list_y, name=fignames[i0], xlabel=xlabel, ylabel=ylabel,
                   save=save_fig, show=show,
                   figdir=case.result_path,
                   figwidth=figwidth, xlim=xlim, ylim=ylim)
    gplot.initializeFigure()
    gplot.markercolors = None
    # Go through each line location
    for i in range(len(list_x)):
        # Then for each location, go through each line
        for j in range(3):
            label = labels[i0][j] if i == 0 else None
            # Flip y
            gplot.axes.plot(list_x[i][:, j], -list_y[i], label=label, color=gplot.colors[j], alpha=.8, ls=ls[j])

    # Plot turbine and shade unpredicted area too
    if 'OneTurb' in casename:
        plotOneTurbAuxiliary(gplot, xlim, (min(list_y[0]), max(list_y[0])))
    elif 'ParTurb' in casename:
        plotParTurbAuxiliary(gplot)
    else:
        plotSeqTurbAuxiliary(gplot, xlim, (min(list_y[0]), max(list_y[0])))

    # Plot vertical dotted lines to indicate where the lines are samples
    gplot.axes.plot((-1, -1), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')
    gplot.axes.plot((1, 1), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')
    gplot.axes.plot((3, 3), (min(list_y[0]), max(list_y[0])), color=gplot.gray, alpha=.8, ls=':')

    plt.xticks(np.arange(-1, 5), ('0', '', '1', '', '', '', '', '', '', '', '', ''))
    plt.xticks((-1, 1, 3, 5), ('0', '1', '', '', '', '', '', '', '', '', ''))
    gplot.axes.grid(alpha=.25)
    gplot.axes.set_xlabel(gplot.xlabel)
    gplot.axes.set_ylabel(gplot.ylabel)
    gplot.axes.set_ylim(gplot.ylim)
    # Disable legend and provide it in caption
    # gplot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
    plt.savefig(gplot.figdir + '/' + gplot.name + '.' + ext, transparent=False,
                dpi=dpi)
    print('\nFigure ' + gplot.name + '.' + ext + ' saved in ' + gplot.figdir)
    plt.show() if gplot.show else plt.close()
    # gplot.finalizeFigure(showleg=False)


"""
Plot div(dev(Rij))_xy and div(dev(Rij))_z Lines for 3 Locations in 5 Figures
"""
# 5 plots for each variables, currently div(dev(R))_xy
fignames = ["divDevRxy" + str(i) for i in range(5)]
xlabel = r'$\left( \nabla \cdot R_{ij}^D \right)_\mathrm{hor}/\left( \nabla \cdot R_{ij}^D \right)_\mathrm{hor, max}$'
list_y = [case.coor[set_locs[i] + line_orient + property_names[0]][::3]/126.
          for i in range(len(set_locs))]
# Normalized y mean
ymean = (max(list_y[0]) + min(list_y[0]))/2.
ymean1 = (max(list_y[-1]) + min(list_y[-1]))/2.
# Center y around 0 y/D
for i in range(len(list_y)):
    list_y[i] -= ymean if i < 3 else ymean1

# Go through every figure
for i0 in range(len(fignames)):
    list_x = [np.array(divdev_r_xy[set_locs[i]][i0]).T[::3] for i in range(len(set_locs))]
    # First plot of G for LES, TBDT, TBRF
    divdev_r_xy_plot = Plot2D(list_x=list_x, list_y=list_y, name=fignames[i0], xlabel=xlabel, ylabel=ylabel,
                   save=save_fig, show=show,
                   figdir=case.result_path,
                   figwidth=figwidth, xlim=xlim, ylim=ylim)
    divdev_r_xy_plot.initializeFigure()
    divdev_r_xy_plot.markercolors = None
    # Go through each line location
    for i in range(len(list_x)):
        # Then for each location, go through each line
        for j in range(3):
            label = labels[i0][j] if i == 0 else None
            # Flip y
            divdev_r_xy_plot.axes.plot(list_x[i][:, j], -list_y[i], label=label, color=divdev_r_xy_plot.colors[j], alpha=.8, ls=ls[j])

    # Plot turbine and shade unpredicted area too
    if 'OneTurb' in casename:
        plotOneTurbAuxiliary(divdev_r_xy_plot, xlim, (min(list_y[0]), max(list_y[0])))
    elif 'ParTurb' in casename:
        plotParTurbAuxiliary(divdev_r_xy_plot)
    else:
        plotSeqTurbAuxiliary(divdev_r_xy_plot, xlim, (min(list_y[0]), max(list_y[0])))

    # Plot vertical dotted lines to indicate where the lines are samples
    divdev_r_xy_plot.axes.plot((-1, -1), (min(list_y[0]), max(list_y[0])), color=divdev_r_xy_plot.gray, alpha=.8, ls=':')
    divdev_r_xy_plot.axes.plot((1, 1), (min(list_y[0]), max(list_y[0])), color=divdev_r_xy_plot.gray, alpha=.8, ls=':')
    divdev_r_xy_plot.axes.plot((3, 3), (min(list_y[0]), max(list_y[0])), color=divdev_r_xy_plot.gray, alpha=.8, ls=':')

    plt.xticks(np.arange(-1, 5), ('0', '', '1', '', '', '', '', '', '', '', '', ''))
    plt.xticks((-1, 1, 3, 5), ('0', '1', '', '', '', '', '', '', '', '', ''))
    divdev_r_xy_plot.axes.grid(alpha=.25)
    divdev_r_xy_plot.axes.set_xlabel(divdev_r_xy_plot.xlabel)
    divdev_r_xy_plot.axes.set_ylabel(divdev_r_xy_plot.ylabel)
    divdev_r_xy_plot.axes.set_ylim(divdev_r_xy_plot.ylim)
    plt.savefig(divdev_r_xy_plot.figdir + '/' + divdev_r_xy_plot.name + '.' + ext, transparent=False,
                dpi=dpi)
    print('\nFigure ' + divdev_r_xy_plot.name + '.' + ext + ' saved in ' + divdev_r_xy_plot.figdir)
    plt.show() if divdev_r_xy_plot.show else plt.close()

    # divdev_r_xy_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
    # divdev_r_xy_plot.finalizeFigure(showleg=False)

# 5 plots for each variables, currently div(dev(R))_z
fignames = ["divDevRz" + str(i) for i in range(5)]
list_y = [case.coor[set_locs[i] + line_orient + property_names[0]][::5] for i in range(len(set_locs))]
# Go through every figure
for i0 in range(len(fignames)):
    list_x = [np.array(divdev_r_z[set_locs[i]][i0]).T[::5] for i in range(len(set_locs))]
    # First plot of G for LES, TBDT, TBRF
    divdev_r_z_plot = Plot2D(list_x=list_x, list_y=list_y, name=fignames[i0], xlabel=xlabel, ylabel=ylabel,
                              save=save_fig, show=show,
                              figdir=case.result_path,
                              figwidth=figwidth, xlim=xlim)
    divdev_r_z_plot.initializeFigure()
    divdev_r_z_plot.markercolors = None
    # Go through each line location
    for i in range(len(list_x)):
        # Then for each location, go through each line
        for j in range(3):
            label = labels[i0][j] if i == 0 else None
            divdev_r_z_plot.axes.plot(list_x[i][:, j], max(list_y[0]) - list_y[i], label=label, color=divdev_r_z_plot.colors[j], alpha=0.75, ls=ls[j])

    # Plot turbine and shade unpredicted area too
    if 'OneTurb' in casename:
        plotOneTurbAuxiliary(divdev_r_z_plot, xlim, (min(list_y[0]), max(list_y[0])))
    elif 'ParTurb' in casename:
        plotParTurbAuxiliary(divdev_r_z_plot)
    else:
        plotSeqTurbAuxiliary(divdev_r_z_plot, xlim, (min(list_y[0]), max(list_y[0])))

    divdev_r_z_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
    divdev_r_z_plot.finalizeFigure(showleg=False)


"""
Plot TKE Lines for 3 Locations in 1 Figure
"""
# 1 plot for TKE
fignames = ["k"]
labels2 = ('LES', 'Data-driven RANS', 'RANS')
list_y = [case.coor[set_locs[i] + line_orient + property_names[0]][::5] for i in range(len(set_locs))]
# Go through every figure
for i0 in range(len(fignames)):
    list_x = [np.array(k[set_locs[i]][i0]).T[::5] for i in range(len(set_locs))]
    # First plot of G for LES, TBDT, TBRF
    k_plot = Plot2D(list_x=list_x, list_y=list_y, name=fignames[i0], xlabel=xlabel, ylabel=ylabel,
                             save=save_fig, show=show,
                             figdir=case.result_path,
                             figwidth=figwidth, xlim=xlim)
    k_plot.initializeFigure()
    k_plot.markercolors = None
    # Go through each line location
    for i in range(len(list_x)):
        # Then for each location, go through each line
        for j in range(3):
            label = labels2[j] if i == 0 else None
            k_plot.axes.plot(list_x[i][:, j], max(list_y[0]) - list_y[i], label=label, color=k_plot.colors[j], alpha=0.75, ls=ls[j])

    # Plot turbine and shade unpredicted area too
    if 'OneTurb' in casename:
        plotOneTurbAuxiliary(k_plot, xlim, (min(list_y[0]), max(list_y[0])))
    elif 'ParTurb' in casename:
        plotParTurbAuxiliary(k_plot)
    else:
        plotSeqTurbAuxiliary(k_plot, xlim, (min(list_y[0]), max(list_y[0])))

    # Plot vertical dotted lines to indicate where the lines are samples
    k_plot.axes.plot((-1, -1), (min(list_y[0]), max(list_y[0])), color=k_plot.gray, alpha=0.5, ls=':')
    k_plot.axes.plot((1, 1), (min(list_y[0]), max(list_y[0])), color=k_plot.gray, alpha=0.5, ls=':')
    k_plot.axes.plot((3, 3), (min(list_y[0]), max(list_y[0])), color=k_plot.gray, alpha=0.5, ls=':')
    k_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
    k_plot.finalizeFigure(showleg=False)


"""
Plot Epsilon Lines for 3 Locations in 1 Figure
"""
# 1 plot for epsilon
fignames = ["epsilon"]
labels2 = ('LES', 'Data-driven RANS', 'RANS')
list_y = [case.coor[set_locs[i] + line_orient + property_names[0]][::5] for i in range(len(set_locs))]
# Go through every figure
for i0 in range(len(fignames)):
    list_x = [np.array(epsilon[set_locs[i]][i0]).T[::5] for i in range(len(set_locs))]
    # First plot of G for LES, TBDT, TBRF
    epsilon_plot = Plot2D(list_x=list_x, list_y=list_y, name=fignames[i0], xlabel=xlabel, ylabel=ylabel,
                    save=save_fig, show=show,
                    figdir=case.result_path,
                    figwidth=figwidth, xlim=xlim)
    epsilon_plot.initializeFigure()
    epsilon_plot.markercolors = None
    # Go through each line location
    for i in range(len(list_x)):
        # Then for each location, go through each line
        for j in range(3):
            label = labels2[j] if i == 0 else None
            epsilon_plot.axes.plot(list_x[i][:, j], max(list_y[0]) - list_y[i], label=label, color=epsilon_plot.colors[j], alpha=0.75, ls=ls[j])

    # Plot turbine and shade unpredicted area too
    if 'OneTurb' in casename:
        plotOneTurbAuxiliary(epsilon_plot, xlim, (min(list_y[0]), max(list_y[0])))
    elif 'ParTurb' in casename:
        plotParTurbAuxiliary(epsilon_plot)
    else:
        plotSeqTurbAuxiliary(epsilon_plot, xlim, (min(list_y[0]), max(list_y[0])))

    # Plot vertical dotted lines to indicate where the lines are samples
    epsilon_plot.axes.plot((-1, -1), (min(list_y[0]), max(list_y[0])), color=epsilon_plot.gray, alpha=0.5, ls=':')
    epsilon_plot.axes.plot((1, 1), (min(list_y[0]), max(list_y[0])), color=epsilon_plot.gray, alpha=0.5, ls=':')
    epsilon_plot.axes.plot((3, 3), (min(list_y[0]), max(list_y[0])), color=epsilon_plot.gray, alpha=0.5, ls=':')
    epsilon_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
    epsilon_plot.finalizeFigure(showleg=False)


"""
Plot Uxy and Uz Lines for 3 Locations in 1 Figure
"""
# 1 plot for Uxy
fignames = ["Uxy"]
labels2 = ('LES', 'Data-driven RANS', 'RANS')
list_y = [case.coor[set_locs[i] + line_orient + property_names[0]][::5] for i in range(len(set_locs))]
# Go through every figure
for i0 in range(len(fignames)):
    list_x = [np.array(uxy[set_locs[i]][i0]).T[::5] for i in range(len(set_locs))]
    # First plot of G for LES, TBDT, TBRF
    uxy_plot = Plot2D(list_x=list_x, list_y=list_y, name=fignames[i0], xlabel=xlabel, ylabel=ylabel,
                   save=save_fig, show=show,
                   figdir=case.result_path,
                   figwidth=figwidth, xlim=xlim)
    uxy_plot.initializeFigure()
    uxy_plot.markercolors = None
    # Go through each line location
    for i in range(len(list_x)):
        # Then for each location, go through each line
        for j in range(3):
            label = labels2[j] if i == 0 else None
            uxy_plot.axes.plot(list_x[i][:, j], max(list_y[0]) - list_y[i], label=label, color=uxy_plot.colors[j], alpha=0.75, ls=ls[j])

    # Plot turbine and shade unpredicted area too
    if 'OneTurb' in casename:
        plotOneTurbAuxiliary(uxy_plot, xlim, (min(list_y[0]), max(list_y[0])))
    elif 'ParTurb' in casename:
        plotParTurbAuxiliary(uxy_plot)
    else:
        plotSeqTurbAuxiliary(uxy_plot, xlim, (min(list_y[0]), max(list_y[0])))

    # Plot vertical dotted lines to indicate where the lines are samples
    uxy_plot.axes.plot((-1, -1), (min(list_y[0]), max(list_y[0])), color=uxy_plot.gray, alpha=0.5, ls=':')
    uxy_plot.axes.plot((1, 1), (min(list_y[0]), max(list_y[0])), color=uxy_plot.gray, alpha=0.5, ls=':')
    uxy_plot.axes.plot((3, 3), (min(list_y[0]), max(list_y[0])), color=uxy_plot.gray, alpha=0.5, ls=':')
    uxy_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
    uxy_plot.finalizeFigure(showleg=False)

# 1 plot for Uz
fignames = ["Uz"]
list_y = [case.coor[set_locs[i] + line_orient + property_names[0]][::3]/126.
          for i in range(len(set_locs))]
# Normalized y mean
ymean = (max(list_y[0]) + min(list_y[0]))/2.
ymean1 = (max(list_y[-1]) + min(list_y[-1]))/2.

# Center y around 0 y/D
for i in range(len(list_y)):
    list_y[i] -= ymean if i < 3 else ymean1

xlabel = r'$u_z/|u_z|_\mathrm{max}$'
# Go through every figure
for i0 in range(len(fignames)):
    list_x = [np.array(uz[set_locs[i]][i0]).T[::3] for i in range(len(set_locs))]
    # First plot of G for LES, TBDT, TBRF
    uz_plot = Plot2D(list_x=list_x, list_y=list_y, name=fignames[i0], xlabel=xlabel, ylabel=ylabel,
                              save=save_fig, show=show,
                              figdir=case.result_path,
                              figwidth=figwidth, xlim=xlim, ylim=ylim)
    uz_plot.initializeFigure()
    uz_plot.markercolors = None
    # Go through each line location
    for i in range(len(list_x)):
        # Then for each location, go through each line
        for j in range(3):
            label = labels2[j] if i == 0 else None
            uz_plot.axes.plot(list_x[i][:, j], -list_y[i], label=label, color=uz_plot.colors[j], alpha=.8, ls=ls[j])

    # Plot turbine and shade unpredicted area too
    if 'OneTurb' in casename:
        plotOneTurbAuxiliary(uz_plot, xlim, (min(list_y[0]), max(list_y[0])))
    elif 'ParTurb' in casename:
        plotParTurbAuxiliary(uz_plot)
    else:
        plotSeqTurbAuxiliary(uz_plot, xlim, (min(list_y[0]), max(list_y[0])))

    # Plot vertical dotted lines to indicate where the lines are samples
    uz_plot.axes.plot((-1, -1), (min(list_y[0]), max(list_y[0])), color=uz_plot.gray, alpha=.8, ls=':')
    uz_plot.axes.plot((1, 1), (min(list_y[0]), max(list_y[0])), color=uz_plot.gray, alpha=.8, ls=':')
    uz_plot.axes.plot((3, 3), (min(list_y[0]), max(list_y[0])), color=uz_plot.gray, alpha=.8, ls=':')

    plt.xticks(np.arange(-1, 5), ('0', '', '1', '', '', '', '', '', '', '', '', ''))
    plt.xticks((-1, 1, 3, 5), ('0', '1', '', '', '', '', '', '', '', '', ''))
    uz_plot.axes.grid(alpha=.25)
    uz_plot.axes.set_xlabel(uz_plot.xlabel)
    uz_plot.axes.set_ylabel(uz_plot.ylabel)
    uz_plot.axes.set_ylim(uz_plot.ylim)
    plt.savefig(uz_plot.figdir + '/' + uz_plot.name + '.' + ext, transparent=False,
                dpi=dpi)
    print('\nFigure ' + uz_plot.name + '.' + ext + ' saved in ' + uz_plot.figdir)
    plt.show() if uz_plot.show else plt.close()

    # uz_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
    # uz_plot.finalizeFigure(showleg=False)

# # Prepare values to scale with x in the plot
# divdev_r_xy = prepareLineValues(set_locs, offset_d, divdev_r_xy, (divdev_r_xy_min, divdev_r_xy_max),
#                                 lim_multiplier=4., linespan=6.)
# divdev_r_z = prepareLineValues(set_locs, offset_d, divdev_r_z, (divdev_r_z_min, divdev_r_z_max),
#                                lim_multiplier=4., linespan=4.)
#
# figname, figname2, figname3, figname4 = 'divDevRxy1', 'divDevRxy2', 'divDevRz1', 'divDevRz2'
# list_y = [case.coor[set_locs[i] + line_orient + property_names[0]][::10] for i in range(len(set_locs))]
# # if figwidth == 'full':
# #     for i in range(3, 6):
# #         list_y[i] += yoffset
#
# list_x = [divdev_r_xy[set_locs[i]][::10] for i in range(len(set_locs))]
# list_x2 = [divdev_r_z[set_locs[i]][::10] for i in range(len(set_locs))]
# # First plot of divDevR in xy for LES, TBDT, TBRF
# divdevr_xy_plot = Plot2D(list_x=list_x, list_y=list_y, name=figname, xlabel=xlabel, ylabel=ylabel,
#                          save=save_fig, show=show,
#                          figdir=case.result_path,
#                          figwidth=figwidth, xlim=xlim)
# divdevr_xy_plot.initializeFigure()
# divdevr_xy_plot.markercolors = None
# # Go through each line location
# for i in range(len(list_x)):
#     # Then for each location, go through each line
#     for j in range(3):
#         label = labels[j] if i == 0 else None
#         divdevr_xy_plot.axes.plot(list_x[i][:, j], list_y[i], label=label, color=divdevr_xy_plot.colors[j], alpha=0.75, ls=ls[j])
#
# # Plot turbine and shade unpredicted area too
# if 'OneTurb' in casename:
#     plotOneTurbAuxiliary(divdevr_xy_plot, xlim, (min(list_y[0]), max(list_y[0])))
# elif 'ParTurb' in casename:
#     plotParTurbAuxiliary(divdevr_xy_plot)
# else:
#     plotSeqTurbAuxiliary(divdevr_xy_plot, xlim, (min(list_y[0]), max(list_y[0])))
#
# divdevr_xy_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
# divdevr_xy_plot.finalizeFigure(showleg=False)
#
# # Second plot of divDevR in xy for LES, TBAB, TBGB
# divdevr_xy_plot2 = Plot2D(list_x=list_x, list_y=list_y, name=figname2, xlabel=xlabel, ylabel=ylabel,
#                           save=save_fig, show=show,
#                           figdir=case.result_path,
#                           figwidth=figwidth, xlim=xlim)
# divdevr_xy_plot2.initializeFigure()
# divdevr_xy_plot2.markercolors = None
# # Go through each line location
# for i in range(len(list_x)):
#     # Then for each location, go through each (LES, TBAB, TBGB) line
#     labels2_cpy = labels2 if i == 0 else (None, None, None)
#     divdevr_xy_plot2.axes.plot(list_x[i][:, 0], list_y[i], label=labels2_cpy[0], color=divdevr_xy_plot2.colors[0], alpha=0.75, ls=ls[0])
#     divdevr_xy_plot2.axes.plot(list_x[i][:, 3], list_y[i], label=labels2_cpy[1], color=divdevr_xy_plot2.colors[1], alpha=0.75, ls=ls[1])
#     divdevr_xy_plot2.axes.plot(list_x[i][:, 4], list_y[i], label=labels2_cpy[2], color=divdevr_xy_plot2.colors[2], alpha=0.75, ls=ls[2])
#
# if 'OneTurb' in casename:
#     plotOneTurbAuxiliary(divdevr_xy_plot2, xlim, (min(list_y[0]), max(list_y[0])))
# elif 'ParTurb' in casename:
#     plotParTurbAuxiliary(divdevr_xy_plot2)
# else:
#     plotSeqTurbAuxiliary(divdevr_xy_plot2, xlim, (min(list_y[0]), max(list_y[0])))
#
# divdevr_xy_plot2.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
# divdevr_xy_plot2.finalizeFigure(showleg=False)
#
# # First plot of divDevR in z for LES, TBDT, TBRF
# divdevr_z_plot = Plot2D(list_x=list_x2, list_y=list_y, name=figname3, xlabel=xlabel, ylabel=ylabel,
#                         save=save_fig, show=show,
#                         figdir=case.result_path,
#                         figwidth=figwidth, xlim=xlim)
# divdevr_z_plot.initializeFigure()
# divdevr_z_plot.markercolors = None
# # Go through each line location
# for i in range(len(list_x2)):
#     # Then for each location, go through each line
#     for j in range(3):
#         label = labels[j] if i == 0 else None
#         divdevr_z_plot.axes.plot(list_x2[i][:, j], list_y[i], label=label, color=divdevr_xy_plot.colors[j], alpha=0.75, ls=ls[j])
#
# # Plot turbine and shade unpredicted area too
# if 'OneTurb' in casename:
#     plotOneTurbAuxiliary(divdevr_z_plot, xlim, (min(list_y[0]), max(list_y[0])))
# elif 'ParTurb' in casename:
#     plotParTurbAuxiliary(divdevr_z_plot)
# else:
#     plotSeqTurbAuxiliary(divdevr_z_plot, xlim, (min(list_y[0]), max(list_y[0])))
#
# divdevr_z_plot.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
# divdevr_z_plot.finalizeFigure(showleg=False)
#
# # Second plot of divDevR in z for LES, TBAB, TBGB
# divdevr_z_plot2 = Plot2D(list_x=list_x2, list_y=list_y, name=figname4, xlabel=xlabel, ylabel=ylabel,
#                          save=save_fig, show=show,
#                          figdir=case.result_path,
#                          figwidth=figwidth, xlim=xlim)
# divdevr_z_plot2.initializeFigure()
# divdevr_z_plot2.markercolors = None
# # Go through each line location
# for i in range(len(list_x2)):
#     # Then for each location, go through each (LES, TBAB, TBGB) line
#     labels2_cpy = labels2 if i == 0 else (None, None, None)
#     divdevr_z_plot2.axes.plot(list_x2[i][:, 0], list_y[i], label=labels2_cpy[0], color=divdevr_xy_plot2.colors[0], alpha=0.75, ls=ls[0])
#     divdevr_z_plot2.axes.plot(list_x2[i][:, 3], list_y[i], label=labels2_cpy[1], color=divdevr_xy_plot2.colors[1], alpha=0.75, ls=ls[1])
#     divdevr_z_plot2.axes.plot(list_x2[i][:, 4], list_y[i], label=labels2_cpy[2], color=divdevr_xy_plot2.colors[2], alpha=0.75, ls=ls[2])
#
# if 'OneTurb' in casename:
#     plotOneTurbAuxiliary(divdevr_z_plot2, xlim, (min(list_y[0]), max(list_y[0])))
# elif 'ParTurb' in casename:
#     plotParTurbAuxiliary(divdevr_z_plot2)
# else:
#     plotSeqTurbAuxiliary(divdevr_z_plot2, xlim, (min(list_y[0]), max(list_y[0])))
#
# divdevr_z_plot2.axes.legend(loc='lower left', shadow=False, fancybox=False, ncol=3)
# divdevr_z_plot2.finalizeFigure(showleg=False)


