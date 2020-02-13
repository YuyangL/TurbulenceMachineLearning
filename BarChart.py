import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from PlottingTool import BaseFigure, Plot2D, Plot2D_MultiAxes
import matplotlib.pyplot as plt
from math import sqrt

bar_comp = ('LES precursor', 'Wind plant LES', 'Wind plant RANS', r'LES $b_{ij}$ injection')

# CPU hour
xlabel = ('H-OneTurb LES', 'H-OneTurb RANS', 'H-ParTurb LES', 'H-ParTurb RANS', 'L-ParTurb', 'L-ParTurb-Yaw', 'H-ParTurb-HiSpeed', 'L-SeqTurb')
# Precursor CPU hour in the order of xlabel
precursor = (5231, 5231, 5231, 5231, 4685, 4685, 5559, 4685)
# SOWFA LES, 0 for RANS cases
alm = (36078, 0, 40470, 0, 48999, 53274, 72410, 72146)
precursor_alm = np.empty(len(xlabel))
for i in range(len(xlabel)):
    precursor_alm[i] = precursor[i] + alm[i]

# SOWFA pure RANS, 0 for LES cases
rans = (0, 486, 0, 703, 0, 0, 0, 0)
precursor_alm_rans = np.empty(len(xlabel))
for i in range(len(xlabel)):
    precursor_alm_rans[i] = precursor_alm[i] + rans[i]

# SOWFA bij injected RANS, 0 for LES cases
rans_bij = (0, 671, 0, 1308, 0, 0, 0, 0)
t_tot = np.empty(len(xlabel))
for i in range(len(xlabel)):
    t_tot[i] = precursor_alm_rans[i] + rans_bij[i]

# x locations
ind = np.arange(len(xlabel))
# Width of bars
width = 0.3
plot = BaseFigure([], [], 'CPUhour', figdir='/media/yluan/', show=False, save=True,
                  ylabel='CPU hour', xlabel='')
plot.initializeFigure()
p0 = plot.axes.bar(ind, precursor, width, zorder=10)
p1 = plot.axes.bar(ind, alm, width, bottom=precursor, zorder=10)
p2 = plot.axes.bar(ind, rans, width, bottom=precursor_alm, zorder=10)
p3 = plot.axes.bar(ind, rans_bij, width, bottom=precursor_alm_rans, zorder=10)
plt.xticks(ind, xlabel, rotation='30')
plt.legend((p0[0], p1[0], p2[0], p3[0]), bar_comp, shadow=False, fancybox=False)
# plt.show()
plot.finalizeFigure(show_xylabel=(False, True), xyscale=('bla', 'bla'))



"""
Power to Cost Ratio
"""
# Power at generator
rans_oneturb = 2379.
rans_oneturb_bij = 2347.
les_oneturb = 2276.
rans_parturb = 2431.
rans_parturb_bij = 2353.
les_parturb = 2223.

xlabel = ('OneTurb LES', ' Data-driven OneTurb RANS', ' OneTurb RANS',
          'ParTurb LES', 'Data-driven ParTurb RANS', 'ParTurb RANS')
ylabel = r'$\frac{\mathrm{Average \ power \ accuracy}}{\mathrm{Normalized \ CPU \ hour}}$'
# Power to CPU hour ratio
p_cost = (1.,
          (1. - (rans_oneturb_bij - les_oneturb)/les_oneturb)/(t_tot[1]/t_tot[0]),
          (1. - (rans_oneturb - les_oneturb)/les_oneturb)/(precursor_alm_rans[1]/t_tot[0]))
p_cost2 = (1.,
           (1. - (rans_parturb_bij - les_parturb)/les_parturb)/(t_tot[3]/t_tot[2]),
           (1. - (rans_parturb - les_parturb)/les_parturb)/(precursor_alm_rans[3]/t_tot[2]))

# x locations
ind = np.arange(len(p_cost))
ind2 = np.arange(len(p_cost), len(p_cost) + len(p_cost2))
plot = BaseFigure([], [], 'PowerToCPUhour', figdir='/media/yluan/', show=False, save=True,
                  ylabel=ylabel, xlabel='')
plot.initializeFigure()
p0 = plot.axes.bar(ind, p_cost, width, zorder=100)
p1 = plot.axes.bar(ind2, p_cost2, width, zorder=100)
plt.legend((p0[0], p1[0]), ('N-H-OneTurbine', 'N-H-ParallelTurbines'), shadow=False, fancybox=False)
plt.xticks(np.arange(len(p_cost) + len(p_cost2)), xlabel, rotation='20')
# plt.show()
plot.finalizeFigure(show_xylabel=(False, True), xyscale=('bla', 'bla'))


"""
Compare to Kaandorp
"""
xlabel = ('TBDT', 'Kaandorp TBDT', 'TBRF', 'Kaandorp TBRF', 'Kaandorp TBRF theoretical best')
ylabel = ('Train time [min]')

times = (0.8467, 26.015)
times2 = (5.9, 206.299, 206.299/8.)
ind = (0, 1)
ind2 = (2, 3, 4)
plot = BaseFigure([], [], 'MLtime', figdir='/media/yluan/DNS', show=False, save=True,
                  ylabel=ylabel, xlabel='')
plot.initializeFigure()
p0 = plot.axes.bar(ind, times, width, zorder=100)
p1 = plot.axes.bar(ind2, times2, width, zorder=100)
plt.legend((p0[0], p1[0]), ('TBDT', 'TBRF w/ 8 DT'), shadow=False, fancybox=False)
plt.xticks(np.arange(5), xlabel, rotation='15')
# plt.show()
plot.finalizeFigure(show_xylabel=(False, True), xyscale=('bla', 'bla'))


