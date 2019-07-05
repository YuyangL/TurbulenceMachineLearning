import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from warnings import warn
from Utilities import timer
from numba import njit, jit, prange





class BaseFigure:
    def __init__(self, list_x, list_y, name='UntitledFigure', font_size=8, x_labels=('$x$',), y_labels=('$y$',), 
                 save_dir='./', show=True, save=True, equal_ax=False, x_lims=(None,), y_lims=(None,), 
                 fig_span='infer', fig_height_multiplier=1., subplots='infer', colors=('tableau10',)):
        self.list_x, self.list_y = ((list_x,), (list_y,)) if isinstance(list_x, np.ndarray) else (list_x, list_y)
        # Assume number of plots is number of x provided by default
        self.nplots = len(list_x)
        self.name, self.save_dir, self.save, self.show = name, save_dir, save, show
        if not self.show:
            plt.ioff()
        else:
            plt.ion()

        self.x_labels = self._ensureEqualTupleElements(x_labels)
        self.y_labels = self._ensureEqualTupleElements(y_labels) 
        self.equal_ax = equal_ax
        self.x_lims, self.y_lims = x_lims, y_lims
        self.colors, self.gray = self._setColors(which=colors[0]) if colors[0] in ('tableau10', 'tableau20') else (colors, (89/255., 89/255., 89/255.))

        # If subplots is provided, use it, otherwise use nplots as number of plot columns
        self.subplots = tuple(subplots) if isinstance(subplots, (list, tuple)) else (1, self.nplots)
        # If only (1, 1) subplots and fig_span is "infer", then set figure span to half page width
        self.fig_span = 'half' if (subplots == (1, 1) and fig_span == 'infer') else 'full'
        self.fig_span, self.fig_height_multiplier, self.font_size = fig_span, fig_height_multiplier, font_size
        # By default number of lines in a plot is 1 (dummy value)
        self.nlines = self._ensureEqualTupleElements(1)


    def _ensureEqualTupleElements(self, input):
        """
        Make sure the input is converted not only a tuple but also having the length of number of plots.
        If input is not a list/tuple already, str/int/float/ndarray type will be detected.

        :param input: Input to be converted to tuple of length equal to number of plots.
        :type input: list/tuple/str/int/float/ndarray
        :return: Tuple equal to number of plots.
        :rtype: tuple
        """

        # If input is one of str/int/float/ndarray type, then put it in a tuple equal to number of plots
        if isinstance(input, (str, int, float, np.ndarray)):
            outputs = (input,)*self.nplots
        # Else if input is list/tuple of 1 element while number of plots > 1,
        # then make it equal number of plots
        elif len(input) == 1 and self.nplots > 1:
            outputs = tuple(input*self.nplots)
        # Otherwise, return input
        else:
            outputs = tuple(input)

        return outputs


    @staticmethod
    def _setColors(which='qualitative'):
        """
        Set colors (not colormap) preference. 
        Choices are "qualitative", "tableau10", and "tableau20".
        
        :param which: Which set of colors to use.
        :type which: "qualitative"/"tableau10"/"tableau20", optional (default="qualitative")
        :return: Color set of preference and specific color for gray.
        :rtype: (list, tuple)
        """
        
        # These are the "Tableau 20" colors as RGB.
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        tableau10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (23, 190, 207), (214, 39, 40), (188, 189, 34), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127)]
        # Orange, blue, magenta, cyan, red, teal, grey
        qualitative = [(238, 119, 51), (0, 119, 187), (238, 51, 119), (51, 187, 238), (204, 51, 117), (0, 153, 136), (187, 187, 187)]
        colors_dict = {'tableau20': tableau20,
                      'tableau10': tableau10,
                      'qualitative': qualitative}
        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        colors = colors_dict[which]
        for i in range(len(colors)):
            r, g, b = colors[i]
            colors[i] = (r/255., g/255., b/255.)

        tableau_gray = (89/255., 89/255., 89/255.)
        
        return colors, tableau_gray


    @staticmethod
    def _latexify(fig_width=None, fig_height=None, fig_span='half', linewidth=1, font_size=8, subplots=(1, 1), fig_height_multiplier= 1.):
        """Set up matplotlib's RC params for LaTeX plotting.
        Call this before plotting a figure.

        Parameters
        ----------
        fig_width : float, optional (default=None), inches
        fig_height : float,  optional (default=None), inches
        """
        # Code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

        # Width and max height in inches for IEEE journals taken from
        # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
        if fig_width is None:
            if subplots[1] == 1:
                fig_width = 3.39 if fig_span is 'half' else 6.9  # inches
            else:
                fig_width = 6.9  # inches

        if fig_height is None:
            golden_mean = (np.sqrt(5) - 1.0)/2.0  # Aesthetic ratio
            # In case subplots option is not applicable e.g. normal Plot2D and you still want elongated height
            fig_height = fig_width*golden_mean*fig_height_multiplier  # height in inches
            # fig_height *= (0.25 + (subplots[0] - 1)) if subplots[0] > 1 else 1
            fig_height *= subplots[0]

        MAX_HEIGHT_INCHES = 8.0
        if fig_height > MAX_HEIGHT_INCHES:
            warn("\nfig_height too large:" + str(fig_height) +
                  ". Will reduce to " + str(MAX_HEIGHT_INCHES) + " inches", stacklevel=2)
            fig_height = MAX_HEIGHT_INCHES

        tableauGray = (89/255., 89/255., 89/255.)
        mpl.rcParams.update({
            'backend':             'Qt5Agg',
            'text.latex.preamble': [r"\usepackage{gensymb,amsmath}"],
            'axes.labelsize':      font_size,  # fontsize for x and y labels (was 10)
            'axes.titlesize':      font_size + 2.,
            'font.size':           font_size,  # was 10
            'legend.fontsize':     font_size - 2.,  # was 10
            'xtick.labelsize':     font_size - 2.,
            'ytick.labelsize':     font_size - 2.,
            'xtick.color':         tableauGray,
            'ytick.color':         tableauGray,
            'xtick.direction':     'out',
            'ytick.direction':     'out',
            'text.usetex':         True,
            'figure.figsize':      (fig_width, fig_height),
            'font.family':         'serif',
            "legend.framealpha":   0.5,
            'legend.edgecolor':    'none',
            'lines.linewidth':     linewidth,
            'lines.markersize':    2,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            'axes.edgecolor':      tableauGray,
            'lines.antialiased':   True,
            'patch.antialiased':   True,
            'text.antialiased':    True})


    def initializeFigure(self):
        self._latexify(font_size=self.font_size, fig_span=self.fig_span, subplots=self.subplots, fig_height_multiplier=self.fig_height_multiplier)

        self.fig, self.axes = plt.subplots(self.subplots[0], self.subplots[1], num=self.name, constrained_layout=True)
        # If no multiple axes, still make self.axes index-able
        if not isinstance(self.axes, np.ndarray): self.axes = (self.axes,)
        print('\nFigure ' + self.name + ' initialized')


    def plotFigure(self):
        print('\nPlotting ' + self.name + '...')


    def _ensureMeshGrid(self):
        if len(np.array(self.list_x[0]).shape) == 1:
            warn('\nX and Y are 1D, contour/contourf requires mesh grid. Converting X and Y to mesh grid '
                    'automatically...\n',
                    stacklevel = 2)
            # Convert tuple to list
            self.list_x, self.list_y = list(self.list_x), list(self.list_y)
            self.list_x[0], self.list_y[0] = np.meshgrid(self.list_x[0], self.list_y[0], sparse = False)


    def finalizeFigure(self, xy_scale=(None,), grid=True,
                       transparent_bg=False, legloc='best'):
        for i in range(self.nplots):
            if self.nlines[i] > 1 and legloc is not None:
                ncol = 2 if self.nlines[i] > 3 else 1
                self.axes[i].legend(loc=legloc, shadow=False, fancybox=False, ncol=ncol)
    
            if grid:
                self.axes[i].grid(which = 'major', alpha = 0.25)
    
            self.axes[i].set_xlabel(self.x_labels)
    
            self.axes[i].set_ylabel(self.y_labels)
    
            if self.equal_ax:
                # Only execute 2D equal axis if the figure is actually 2D
                try:
                    self.view_angles
                except AttributeError:
                    self.axes[i].set_aspect('equal', 'box')
    
            if self.x_lims[0] is not None:
                self.axes[i].set_xlim(self.x_lims)
    
            if self.y_lims[0] is not None:
                self.axes[i].set_ylim(self.y_lims)
    
            if xy_scale[0] is not None:
                self.axes[i].set_xscale(xy_scale[0]), self.axes[i].set_yscale(xy_scale[1])

        print('\nFigure ' + self.name + ' finalized')
        if self.save:
            # plt.savefig(self.save_dir + '/' + self.name + '.png', transparent = transparent_bg, bbox_inches = 'tight', dpi = 1000)
            plt.savefig(self.save_dir + '/' + self.name + '.png', transparent=transparent_bg,
                        dpi=1000)
            print('\nFigure ' + self.name + '.png saved in ' + self.save_dir)

        if self.show:
            plt.show()
        # Close current figure window
        # so that the next figure will be based on a new figure window even if the same name
        else:
            plt.close()


class Plot2D(BaseFigure):
    def __init__(self, list_x, list_y, z2D = (None,), type = 'infer', alpha = 0.75, zLabel = '$z$', cmap = 'plasma', gradientBg = False, gradientBgRange = (None, None), gradientBgDir = 'x', **kwargs):
        self.z2D = z2D
        self.lines, self.markers = ("-", "--", "-.", ":")*5, ('o', 'D', 'v', '^', '<', '>', 's', '8', 'p')*3
        self.alpha, self.cmap = alpha, cmap
        self.zLabel = zLabel
        self.gradientBg, self.gradientBgRange, self.gradientBgDir = gradientBg, gradientBgRange, gradientBgDir

        super().__init__(list_x, list_y, **kwargs)

        # If multiple data provided, make sure type is a tuple of the same length
        if type == 'infer':
            self.type = ('contourf',)*len(list_x) if z2D[0] is not None else ('line',)*len(list_x)
        else:
            self.type = (type,)*len(list_x) if isinstance(type, str) else type


    def plotFigure(self, plotsLabel = (None,), contourLvl = 10):
        # Gradient background, only for line and scatter plots
        if self.gradientBg and self.type[0] in ('line', 'scatter'):
            x2D, y2D = np.meshgrid(np.linspace(self.x_lims[0], self.x_lims[1], 3), np.linspace(self.y_lims[0], self.y_lims[1], 3))
            z2D = (np.meshgrid(np.linspace(self.x_lims[0], self.x_lims[1], 3), np.arange(3)))[0] if self.gradientBgDir is 'x' else (np.meshgrid(np.arange(3), np.linspace(self.y_lims[0], self.y_lims[1], 3)))[1]
            self.axes[0].contourf(x2D, y2D, z2D, 500, cmap = 'gray', alpha = 0.33, vmin = self.gradientBgRange[0], vmax = self.gradientBgRange[1])

        super().plotFigure()

        self.plotsLabel = np.arange(1, len(self.list_x) + 1) if plotsLabel[0] is None else plotsLabel
        self.plots = [None]*len(self.list_x)
        for i in range(len(self.list_x)):
            if self.type[i] == 'line':
                self.plots[i] = self.axes[0].plot(self.list_x[i], self.list_y[i], ls = self.lines[i], label = str(self.plotsLabel[i]), color = self.colors[i], alpha = self.alpha)
            elif self.type[i] == 'scatter':
                self.plots[i] = self.axes[0].scatter(self.list_x[i], self.list_y[i], lw = 0, label = str(self.plotsLabel[i]), alpha = self.alpha, color = self.colors[i], marker = self.markers[i])
            elif self.type[i] == 'contourf':
                self._ensureMeshGrid()
                self.plots[i] = self.axes[0].contourf(self.list_x[i], self.list_y[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both', antialiased = False)
            elif self.type[i] == 'contour':
                self._ensureMeshGrid()
                self.plots[i] = self.axes[0].contour(self.list_x[i], self.list_y[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            else:
                warn("\nUnrecognized plot type! type must be one/list of ('infer', 'line', 'scatter', 'contourf', 'contour').\n", stacklevel = 2)
                return


    def finalizeFigure(self, cbarOrientate = 'horizontal', **kwargs):
        if self.type in ('contourf', 'contour') and len(self.axes) == 1:
            cb = plt.colorbar(self.plots[0], ax = self.axes[0], orientation = cbarOrientate)
            cb.set_label(self.zLabel)
            super().finalizeFigure(grid = False, **kwargs)
        else:
            super().finalizeFigure(**kwargs)



class Plot2D_InsetZoom(Plot2D):
    def __init__(self, list_x, list_y, zoomBox, subplots = (2, 1), **kwargs):
        super().__init__(list_x, list_y, fig_width = 'full', subplots = subplots, **kwargs)

        self.zoomBox = zoomBox


    @staticmethod
    def _mark_inset(parent_axes, inset_axes, loc1a = 1, loc1b = 1, loc2a = 2, loc2b = 2, **kwargs):
        # Draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        # loc1, loc2 : {1, 2, 3, 4}
        rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

        pp = BboxPatch(rect, fill = False, **kwargs)
        parent_axes.add_patch(pp)

        p1 = BboxConnector(inset_axes.bbox, rect, loc1 = loc1a, loc2 = loc1b, **kwargs)
        inset_axes.add_patch(p1)
        p1.set_clip_on(False)
        p2 = BboxConnector(inset_axes.bbox, rect, loc1 = loc2a, loc2 = loc2b, **kwargs)
        inset_axes.add_patch(p2)
        p2.set_clip_on(False)

        print('\nInset created')
        return pp, p1, p2


    def plotFigure(self, plotsLabel = (None,), contourLvl = 10):
        super().plotFigure(plotsLabel, contourLvl)
        for i in range(len(self.list_x)):
            if self.type is 'line':
                self.axes[1].plot(self.list_x[i], self.list_y[i], ls = self.lines[i], label = str(self.plotsLabel[i]), alpha = self.alpha, color = self.colors[i])
            elif self.type is 'scatter':
                self.axes[1].scatter(self.list_x[i], self.list_y[i], lw = 0, label = str(self.plotsLabel[i]), alpha = self.alpha, marker = self.markers[i])
            elif self.type is 'contourf':
                self.axes[1].contourf(self.list_x[i], self.list_y[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            elif self.type is 'contour':
                self.axes[1].contour(self.list_x[i], self.list_y[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')


    def finalizeFigure(self, cbarOrientate = 'vertical', setXYlabel = (False, True), xy_scale = ('linear', 'linear'), **kwargs):
        self.axes[1].set_xlim(self.zoomBox[0], self.zoomBox[1]), self.axes[1].set_ylim(self.zoomBox[2], self.zoomBox[3])
        self.axes[1].set_xlabel(self.x_labels), self.axes[1].set_ylabel(self.y_labels)
        if self.equal_ax:
            self.axes[1].set_aspect('equal', 'box')

        self.axes[1].set_xscale(xy_scale[0]), self.axes[1].set_yscale(xy_scale[1])
        self._mark_inset(self.axes[0], self.axes[1], loc1a = 1, loc1b = 4, loc2a = 2, loc2b = 3, fc = "none",
                         ec = self.gray, ls = ':')
        if self.type in ('contour', 'contourf'):
            for ax in self.axes:
                ax.tick_params(axis = 'both', direction = 'out')

        else:
            self.axes[1].grid(which = 'both', alpha = 0.25)
            if len(self.list_x) > 1:
                ncol = 2 if len(self.list_x) > 3 else 1
                self.axes[1].legend(loc = 'best', shadow = False, fancybox = False, ncol = ncol)

        for spine in ('top', 'bottom', 'left', 'right'):
            if self.type in ('contour', 'contourf'):
                self.axes[0].spines[spine].set_visible(False)
            self.axes[1].spines[spine].set_visible(True)
            self.axes[1].spines[spine].set_linestyle(':')

        # plt.draw()
        # Single colorbar
        if self.type in ('contour', 'contourf'):
            self.fig.subplots_adjust(bottom = 0.1, top = 0.9, left = 0.1, right = 0.8)  # , wspace = 0.02, hspace = 0.2)
            cbar_ax = self.fig.add_axes((0.83, 0.1, 0.02, 0.8))
            cb = plt.colorbar(self.plots[0], cax = cbar_ax, orientation = 'vertical')
            cb.set_label(self.zLabel)
            cb.ax.tick_params(axis = 'y', direction = 'out')

        super().finalizeFigure(tightLayout = False, cbarOrientate = cbarOrientate, setXYlabel = setXYlabel, xy_scale = xy_scale, grid = False, **kwargs)


class BaseFigure3D(BaseFigure):
    def __init__(self, list_x2D, list_y2D, zLabel = '$z$', alpha = 1, viewAngles = (15, -115), zLim = (None,), cmap = 'plasma', cmapLabel = '$U$', grid = True, cbarOrientate = 'horizontal', **kwargs):
        super(BaseFigure3D, self).__init__(list_x = list_x2D, list_y = list_y2D, **kwargs)
        # The name of list_x and list_y becomes list_x2D and list_y2D since they are 2D
        self.list_x2D, self.list_y2D = self.list_x, self.list_y
        self.zLabel, self.zLim = zLabel, zLim
        self.cmapLabel, self.cmap = cmapLabel, cmap
        self.alpha, self.grid, self.viewAngles = alpha, grid, viewAngles
        self.plot, self.cbarOrientate = None, cbarOrientate


    def initializeFigure(self, figSize = (1, 1)):
        # Update Matplotlib rcparams
        self.latexify(font_size = self.font_size, fig_width = self.fig_width, subplots = figSize)
        self.fig = plt.figure(self.name)
        self.axes = (self.fig.gca(projection = '3d'),)


    def plotFigure(self):
        super(BaseFigure3D, self).plotFigure()

        self._ensureMeshGrid()


    def finalizeFigure(self, fraction = 0.06, pad = 0.08, showCbar = True, reduceNtick = True, tightLayout = True,
                       **kwargs):
        self.axes[0].set_zlabel(self.zLabel)
        self.axes[0].set_zlim(self.zLim)
        # Color bar
        if showCbar:
            cb = plt.colorbar(self.plot, fraction = fraction, pad = pad, orientation = self.cbarOrientate, extend = 'both', aspect = 25, shrink = 0.75)
            cb.set_label(self.cmapLabel)

        # Turn off background on all three panes
        self._format3D_Axes(self.axes[0])
        # Equal axes
        # [REQUIRES SOURCE CODE MODIFICATION] Equal axis
        # Edit the get_proj function inside site-packages\mpl_toolkits\mplot3d\axes3d.py:
        # try: self.localPbAspect=self.pbaspect
        # except AttributeError: self.localPbAspect=[1,1,1]
        # xmin, xmax = np.divide(self.get_xlim3d(), self.localPbAspect[0])
        # ymin, ymax = np.divide(self.get_ylim3d(), self.localPbAspect[1])
        # zmin, zmax = np.divide(self.get_zlim3d(), self.localPbAspect[2])
        if self.equal_ax:
            try:
                arZX = abs((self.zLim[1] - self.zLim[0])/(self.x_lims[1] - self.x_lims[0]))
                arYX = abs((self.y_lims[1] - self.y_lims[0])/(self.x_lims[1] - self.x_lims[0]))

                # Constrain AR from getting too large
                arYX, arZX = np.min((arYX, 2)), np.min((arZX, 2))
                # Axes aspect ratio doesn't really work properly
                self.axes[0].pbaspect = (1, arYX, arZX)
                # auto_scale_xyz is not preferable since it does it by setting a cubic box
                # scaling = np.array([getattr(self.axes[0], 'get_{}lim'.format(dim))() for dim in 'xyz'])
                # self.axes[0].auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
            except AttributeError:
                warn('\nTo set custom aspect ratio of the 3D plot, you need modification of the source code axes3d.py. The aspect ratio might be incorrect for ' + self.name + '\n', stacklevel = 2)
                pass

        if reduceNtick:
            self.axes[0].set_xticks(np.linspace(self.x_lims[0], self.x_lims[1], 3))
            self.axes[0].set_yticks(np.linspace(self.y_lims[0], self.y_lims[1], 3))
            self.axes[0].set_zticks(np.linspace(self.zLim[0], self.zLim[1], 3))

        # # Strictly equal axis of all three axis
        # _, _, _, _, _, _ = self.get3D_AxesLimits(self.axes[0])
        # 3D grid
        self.axes[0].grid(self.grid)
        self.axes[0].view_init(self.viewAngles[0], self.viewAngles[1])
        # # View distance
        # self.axes[0].dist = 11

        super().finalizeFigure(grid = False, legShow = False, **kwargs)
        

    @timer
    @jit(parallel = True, fastmath = True)
    def getSlicesLimits(self, list_x2D, list_y2D, listZ2D = np.empty(100), listOtherVals = np.empty(100)):
        getXlim = True if self.x_lims[0] is None else False
        getYlim = True if self.y_lims[0] is None else False
        getZlim = True if self.zLim[0] is None else False
        self.x_lims = [1e20, -1e20] if self.x_lims[0] is None else self.x_lims
        self.y_lims = [1e20, -1e20] if self.y_lims[0] is None else self.y_lims
        self.zLim = [1e20, -1e20] if self.zLim[0] is None else self.zLim
        otherValsLim = [1e20, -1e20]
        for i in prange(len(list_x2D)):
            if getXlim:
                xmin, xmax = np.min(list_x2D[i]), np.max(list_x2D[i])
                # Replace old limits with new ones if better limits found
                self.x_lims[0] = xmin if xmin < self.x_lims[0] else self.x_lims[0]
                self.x_lims[1] = xmax if xmax > self.x_lims[1] else self.x_lims[1]

            if getYlim:
                ymin, ymax = np.min(list_y2D[i]), np.max(list_y2D[i])
                self.y_lims[0] = ymin if ymin < self.y_lims[0] else self.y_lims[0]
                self.y_lims[1] = ymax if ymax > self.y_lims[1] else self.y_lims[1]

            if getZlim:
                zmin, zmax = np.min(listZ2D[i]), np.max(listZ2D[i])
                self.zLim[0] = zmin if zmin < self.zLim[0] else self.zLim[0]
                self.zLim[1] = zmax if zmax > self.zLim[1] else self.zLim[1]

            otherVals_min, otherVals_max = np.nanmin(listOtherVals[i]), np.nanmax(listOtherVals[i])
            otherValsLim[0] = otherVals_min if otherVals_min < otherValsLim[0] else otherValsLim[0]
            otherValsLim[1] = otherVals_max if otherVals_max > otherValsLim[1] else otherValsLim[1]

        return otherValsLim
         

    @staticmethod
    def _format3D_Axes(ax):
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_xaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_yaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_zaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})


    @staticmethod
    def get3D_AxesLimits(ax, setAxesEqual = True):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a Matplotlib axis, e.g., as output from plt.gca().
        '''
        x_limsits = ax.get_xlim3d()
        y_limsits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limsits[1] - x_limsits[0])
        x_middle = np.mean(x_limsits)
        y_range = abs(y_limsits[1] - y_limsits[0])
        y_middle = np.mean(y_limsits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        # The plot bounding box is a sphere in the sense of the infinity norm,
        # hence I call half the max range the plot radius.
        if setAxesEqual:
            plot_radius = 0.5*max([x_range, y_range, z_range])
            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([0, z_middle + plot_radius])

        return x_range, y_range, z_range, x_limsits, y_limsits, z_limits


class PlotContourSlices3D(BaseFigure3D):
    def __init__(self, contourX2D, contourY2D, listSlices2D, sliceOffsets, zDir = 'z', contourLvl = 10, gradientBg = True, equal_ax = False, **kwargs):
        super(PlotContourSlices3D, self).__init__(list_x2D = contourX2D, list_y2D = contourY2D, equal_ax = equal_ax, **kwargs)

        self.listSlices2D = (listSlices2D,) if isinstance(listSlices2D, np.ndarray) else listSlices2D
        self.sliceOffsets, self.zDir = iter(sliceOffsets), zDir
        self.x_lims = (min(sliceOffsets), max(sliceOffsets)) if (self.x_lims[0] is None) and (zDir == 'x') else self.x_lims
        self.y_lims = (min(sliceOffsets), max(sliceOffsets)) if (self.y_lims[0] is None) and (zDir == 'y') else self.y_lims
        self.zLim = (min(sliceOffsets), max(sliceOffsets)) if (self.zLim[0] is None) and (zDir == 'z') else self.zLim
        # If axis limits are still not set, infer
        # if self.x_lims[0] is None:
        #     self.x_lims = (np.min(contourX2D), np.max(contourX2D))
        #
        # if self.y_lims[0] is None:
        #     self.y_lims = (np.min(contourY2D), np.max(contourY2D))
        #
        # if self.zLim[0] is None:
        #     self.zLim = (np.min(listSlices2D), np.max(listSlices2D))
        _ = self.getSlicesLimits(list_x2D = self.list_x2D, list_y2D = self.list_y2D, listZ2D = self.listSlices2D)
        # self.sliceMin, self.sliceMax = self.zLim
        self.sliceMin, self.sliceMax = np.amin(listSlices2D), np.amax(listSlices2D)
        self.contourLvl, self.gradientBg = contourLvl, gradientBg
        # # Good initial view angle
        # self.viewAngles = (20, -115) if zDir is 'z' else (15, -60)
        self.cbarOrientate = 'vertical' if zDir is 'z' else 'horizontal'


    # def initializeFigure(self, figSize = (1, 1)):
    #     # If zDir is 'z', then the figure height is twice width, else, figure width is twice height
    #     # figSize = (2.75, 1) if self.zDir is 'z' else (1, 2)
    #
    #     super().initializeFigure(figSize = figSize)


    def plotFigure(self):
        super(PlotContourSlices3D, self).plotFigure()

        # Currently, gradient background feature is only available for zDir = 'x'
        if self.gradientBg:
            if self.zDir is 'x':
                x2Dbg, y2Dbg = np.meshgrid(np.linspace(self.x_lims[0], self.x_lims[1], 3), np.linspace(self.y_lims[0], self.y_lims[1], 3))
                z2Dbg, _ = np.meshgrid(np.linspace(self.x_lims[0], self.x_lims[1], 3), np.linspace(self.zLim[0], self.zLim[1], 3))
                self.axes[0].contourf(x2Dbg, y2Dbg, z2Dbg, 500, zdir = 'z', offset = 0, cmap = 'gray', alpha = 0.5, antialiased = True)
                # # Uncomment below to enable gradient background of all three planes
                # self.axes[0].contourf(x2Dbg, z2Dbg, y2Dbg, 500, zdir = 'y', offset = 300, cmap = 'gray', alpha = 0.5, antialiased = True)
                # Y3, Z3 = np.meshgrid(np.linspace(self.y_lims[0], self.y_lims[1], 3),
                #                      np.linspace(self.zLim[0], self.zLim[1], 3))
                # X3 = np.ones(Y3.shape)*self.x_lims[0]
                # self.axes[0].plot_surface(X3, Y3, Z3, color = 'gray', alpha = 0.5)
            else:
                warn('\nGradient background only supports zDir = "x"!\n', stacklevel = 2)

        # Actual slice plots
        for i, slice in enumerate(self.listSlices2D):
            if self.zDir is 'x':
                X, Y, Z = slice, self.list_x2D[i], self.list_y2D[i]
            elif self.zDir is 'y':
                X, Y, Z = self.list_x2D[i], slice, self.list_y2D[i]
            else:
                X, Y, Z = self.list_x2D[i], self.list_y2D[i], slice

            # "levels" makes sure all slices are in same cmap range
            self.plot = self.axes[0].contourf(X, Y, Z, self.contourLvl, zdir = self.zDir,
                                              offset = next(self.sliceOffsets), alpha = self.alpha, cmap = self.cmap,
                                              levels = np.linspace(self.sliceMin, self.sliceMax, 100), antialiased = False)


    def finalizeFigure(self, **kwargs):
        # Custom color bar location in the figure
        (fraction, pad) = (0.046, 0.04) if self.zDir is 'z' else (0.06, 0.08)
        # if self.zDir is 'z':
        #     ar = abs((self.y_lims[1] - self.y_lims[0])/(self.x_lims[1] - self.x_lims[0]))
        #     pbaspect = (1, ar, 1)
        # elif self.zDir is 'x':
        #     ar = abs((self.zLim[1] - self.zLim[0])/(self.y_lims[1] - self.y_lims[0]))
        #     pbaspect = (1, 1, ar)
        # else:
        #     ar = abs((self.zLim[1] - self.zLim[0])/(self.x_lims[1] - self.x_lims[0]))
        #     pbaspect = (1, 1, ar)

        super(PlotContourSlices3D, self).finalizeFigure(fraction = fraction, pad = pad, **kwargs)


class PlotSurfaceSlices3D(BaseFigure3D):
    def __init__(self, list_x2D, list_y2D, listZ2D, listSlices2D, **kwargs):
        super(PlotSurfaceSlices3D, self).__init__(list_x2D = list_x2D, list_y2D = list_y2D, **kwargs)

        self.listZ2D = (listZ2D,) if isinstance(listZ2D, np.ndarray) else listZ2D
        self.listSlices2D = (listSlices2D,) if isinstance(listSlices2D, np.ndarray) else listSlices2D
        # self.x_lims = (np.min(list_x2D), np.max(list_x2D)) if self.x_lims[0] is None else self.x_lims
        # self.y_lims = (np.min(list_y2D), np.max(list_y2D)) if self.y_lims[0] is None else self.y_lims
        # self.zLim = (np.min(listZ2D), np.max(listZ2D)) if self.zLim[0] is None else self.zLim
        self.cmapLim = self.getSlicesLimits(list_x2D = self.list_x2D, list_y2D = self.list_y2D, listZ2D = listZ2D,
                                           listOtherVals = listSlices2D)
        # self.list_x2D, self.list_y2D = iter(self.list_x2D), iter(self.list_y2D)

        # Find minimum and maximum of the slices values for color, ignore NaN
        # self.cmapLim = (np.nanmin(listSlices2D), np.nanmax(listSlices2D))
        self.cmapNorm = mpl.colors.Normalize(self.cmapLim[0], self.cmapLim[1])
        self.cmapVals = plt.cm.ScalarMappable(norm = self.cmapNorm, cmap = self.cmap)
        self.cmapVals.set_array([])
        # For colorbar mappable
        self.plot = self.cmapVals


    def plotFigure(self):
        for i, slice in enumerate(self.listSlices2D):
            print('\nPlotting ' + self.name + '...')
            fColors = self.cmapVals.to_rgba(slice)
            self.axes[0].plot_surface(self.list_x2D[i], self.list_y2D[i], self.listZ2D[i], cstride = 1,
                                      rstride = 1, facecolors = fColors, vmin = self.cmapLim[0], vmax = self.cmapLim[1], shade = False)


    # def finalizeFigure(self, **kwargs):
    #     arZX = abs((self.zLim[1] - self.zLim[0])/(self.x_lims[1] - self.x_lims[0]))
    #     arYX = abs((self.y_lims[1] - self.y_lims[0])/(self.x_lims[1] - self.x_lims[0]))
    #     # Axes aspect ratio doesn't really work properly
    #     pbaspect = (1., arYX, arZX*2)
    #
    #     super(PlotSurfaceSlices3D, self).finalizeFigure(pbaspect = pbaspect, **kwargs)


class PlotImageSlices3D(BaseFigure3D):
    def __init__(self, list_x2D, list_y2D, listZ2D, listRGB, **kwargs):
        super(PlotImageSlices3D, self).__init__(list_x2D = list_x2D, list_y2D = list_y2D, **kwargs)

        # Convert listZ2D to tuple if it's np.ndarray
        self.listZ2D = (listZ2D,) if isinstance(listZ2D, np.ndarray) else listZ2D
        self.listRGB = (listRGB,) if isinstance(listRGB, np.ndarray) else listRGB
        # Make sure list of RGB arrays are between 0 and 1
        for i, rgbVals in enumerate(self.listRGB):
            self.listRGB[i][rgbVals > 1] = 1
            self.listRGB[i][rgbVals < 0] = 0
                
        # Axes limits
        # self.x_lims = (np.min(list_x2D), np.max(list_x2D)) if self.x_lims[0] is None else self.x_lims
        # self.y_lims = (np.min(list_y2D), np.max(list_y2D)) if self.y_lims[0] is None else self.y_lims
        # self.zLim = (np.min(listZ2D), np.max(listZ2D)) if self.zLim[0] is None else self.zLim
        _ = self.getSlicesLimits(list_x2D = self.list_x2D, list_y2D = self.list_y2D, listZ2D = self.listZ2D)



    @timer
    @jit(parallel = True)
    def plotFigure(self):
        print('\nPlotting {}...'.format(self.name))
        # For gauging progress
        milestone = 33
        for i in prange(len(self.listRGB)):
            self.axes[0].plot_surface(self.list_x2D[i], self.list_y2D[i], self.listZ2D[i], cstride = 1, rstride = 1, 
                                      facecolors = self.listRGB[i], shade = False)
            progress = (i + 1)/len(self.listRGB)*100.
            if progress >= milestone:
                print(' {0}%... '.format(milestone))
                milestone += 33
            
    
    def finalizeFigure(self, **kwargs):
        super(PlotImageSlices3D, self).finalizeFigure(showCbar = False, **kwargs)
            
    
    
        
        

        











if __name__ == '__main__':
    x = np.linspace(0, 300, 100)
    y = np.linspace(0, 100, 100)
    y2 = np.linspace(10, 80, 100)

    z2D = np.linspace(1, 10, x.size*y.size).reshape((y.size, x.size))
    z2D2 = np.linspace(10, 30, x.size*y.size).reshape((y.size, x.size))
    z2D3 = np.linspace(30, 60, x.size*y.size).reshape((y.size, x.size))

    # myplot = Plot2D_InsetZoom((x, x), (y, y2), z2D = (None,), zoomBox = (10, 60, 20, 40), save = True, equal_ax = True, save_dir = 'R:/', name = 'newFig')

    # myplot = Plot2D_InsetZoom(x, y, z2D = z2D, zoomBox = (10, 70, 10, 30), save = True, equal_ax = True,
    #                           save_dir = 'R:/', name = 'newFig2')

    # myplot = PlotSlices3D(x, y, [z2D, z2D2, z2D3], sliceOffsets = [0, 20, 50], name = '3d2', save_dir = 'R:/', x_lims = (0, 150), zDir = 'x')
    myplot = PlotContourSlices3D(x, y, [z2D, z2D2, z2D3], sliceOffsets = [20000, 20500, 21000], name = '3d2', save_dir = 'R:/', zDir = 'x', x_labels = '$x$', y_labels = '$y$', zLabel = r'$z$ [m]', zLim = (0, 100), y_lims = (0, 300), gradientBg = True)

    myplot.initializeFigure()

    myplot.plotFigure()

    myplot.finalizeFigure()


