
class BaseFigure:
    def __init__(self, listX, listY, name = 'UntitledFigure', fontSize = 8, xLabel = '$x$', yLabel = '$y$', figDir = './', show = True, save = True, equalAxis = False, xLim = (None,), yLim = (None,), figWidth = 'half', figHeightMultiplier = 1., subplots = (1, 1), colors = ('tableau10',)):
        self.listX, self.listY = ((listX,), (listY,)) if isinstance(listX, np.ndarray) else (listX, listY)
        self.name, self.figDir, self.save, self.show = name, figDir, save, show
        if not self.show:
            plt.ioff()
        else:
            plt.ion()

        self.xLabel, self.yLabel, self.equalAxis = xLabel, yLabel, equalAxis
        self.xLim, self.yLim = xLim, yLim
        (self.colors, self.gray) = self.setColors(which = colors[0]) if colors[0] in ('tableau10', 'tableau20') else (colors, (89/255., 89/255., 89/255.))

        self.subplots, self.figWidth, self.figHeightMultiplier, self.fontSize = subplots, figWidth, figHeightMultiplier, fontSize


    @staticmethod
    def setColors(which = 'qualitative'):
        # These are the "Tableau 20" colors as RGB.
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        tableau10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (23, 190, 207), (214, 39, 40), (188, 189, 34), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127)]
        # Orange, blue, magenta, cyan, red, teal, grey
        qualitative = [(238, 119, 51), (0, 119, 187), (238, 51, 119), (51, 187, 238), (204, 51, 117), (0, 153, 136), (187, 187, 187)]
        colorsDict = {'tableau20': tableau20,
                      'tableau10': tableau10,
                      'qualitative': qualitative}
        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        colors = colorsDict[which]
        for i in range(len(colors)):
            r, g, b = colors[i]
            colors[i] = (r/255., g/255., b/255.)

        tableauGray = (89/255., 89/255., 89/255.)
        return colors, tableauGray


    @staticmethod
    def latexify(fig_width = None, fig_height = None, figWidth = 'half', linewidth = 1, fontSize = 8, subplots = (1, 1), figHeightMultiplier = 1.):
        """Set up matplotlib's RC params for LaTeX plotting.
        Call this before plotting a figure.

        Parameters
        ----------
        fig_width : float, optional, inches
        fig_height : float,  optional, inches
        """
        # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

        # Width and max height in inches for IEEE journals taken from
        # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
        if fig_width is None:
            if subplots[1] == 1:
                fig_width = 3.39 if figWidth is 'half' else 6.9  # inches
            else:
                fig_width = 6.9  # inches

        if fig_height is None:
            golden_mean = (np.sqrt(5) - 1.0)/2.0  # Aesthetic ratio
            # In case subplots option is not applicable e.g. normal Plot2D and you still want elongated height
            fig_height = fig_width*golden_mean*figHeightMultiplier  # height in inches
            # fig_height *= (0.25 + (subplots[0] - 1)) if subplots[0] > 1 else 1
            fig_height *= subplots[0]

        MAX_HEIGHT_INCHES = 8.0
        if fig_height > MAX_HEIGHT_INCHES:
            warn("\nfig_height too large:" + str(fig_height) +
                 ". Will reduce to " + str(MAX_HEIGHT_INCHES) + " inches", stacklevel = 2)
            fig_height = MAX_HEIGHT_INCHES

        tableauGray = (89/255., 89/255., 89/255.)
        mpl.rcParams.update({
            'backend':             'Qt5Agg',
            'text.latex.preamble': [r"\usepackage{gensymb,amsmath}"],
            'axes.labelsize':      fontSize,  # fontsize for x and y labels (was 10)
            'axes.titlesize':      fontSize + 2.,
            'font.size':           fontSize,  # was 10
            'legend.fontsize':     fontSize - 2.,  # was 10
            'xtick.labelsize':     fontSize - 2.,
            'ytick.labelsize':     fontSize - 2.,
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
        self.latexify(fontSize=self.fontSize, figWidth=self.figWidth, subplots=self.subplots, figHeightMultiplier=self.figHeightMultiplier)

        self.fig, self.axes = plt.subplots(self.subplots[0], self.subplots[1], num=self.name, constrained_layout=True)
        if not isinstance(self.axes, np.ndarray): self.axes = (self.axes,)
        print('\nFigure ' + self.name + ' initialized')


    def plotFigure(self):
        print('\nPlotting ' + self.name + '...')


    def _ensureMeshGrid(self):
        if len(np.array(self.listX[0]).shape) == 1:
            warn('\nX and Y are 1D, contour/contourf requires mesh grid. Converting X and Y to mesh grid '
                 'automatically...\n',
                 stacklevel = 2)
            # Convert tuple to list
            self.listX, self.listY = list(self.listX), list(self.listY)
            self.listX[0], self.listY[0] = np.meshgrid(self.listX[0], self.listY[0], sparse = False)


    def finalizeFigure(self, xyScale = (None,), tightLayout = True, setXYlabel = (True, True), grid = True,
                       transparentBg = False, legLoc = 'best', legShow = True):
        if len(self.listX) > 1 and legShow:
            nCol = 2 if len(self.listX) > 3 else 1
            self.axes[0].legend(loc = legLoc, shadow = False, fancybox = False, ncol = nCol)

        if grid:
            self.axes[0].grid(which = 'major', alpha = 0.25)

        if setXYlabel[0]:
            self.axes[0].set_xlabel(self.xLabel)

        if setXYlabel[1]:
            self.axes[0].set_ylabel(self.yLabel)

        if self.equalAxis:
            # Only execute 2D equal axis if the figure is acutally 2D
            try:
                self.viewAngles
            except AttributeError:
                self.axes[0].set_aspect('equal', 'box')

        if self.xLim[0] is not None:
            self.axes[0].set_xlim(self.xLim)

        if self.yLim[0] is not None:
            self.axes[0].set_ylim(self.yLim)

        if xyScale[0] is not None:
            self.axes[0].set_xscale(xyScale[0]), self.axes[0].set_yscale(xyScale[1])

        if tightLayout:
            plt.tight_layout()

        print('\nFigure ' + self.name + ' finalized')
        if self.save:
            # plt.savefig(self.figDir + '/' + self.name + '.png', transparent = transparentBg, bbox_inches = 'tight', dpi = 1000)
            plt.savefig(self.figDir + '/' + self.name + '.png', transparent = transparentBg,
                        dpi = 1000)
            print('\nFigure ' + self.name + '.png saved in ' + self.figDir)

        if self.show:
            plt.show()
        # Close current figure window
        # so that the next figure will be based on a new figure window even if the same name
        else:
            plt.close()

def plotAnisotropyTensor():

    return
