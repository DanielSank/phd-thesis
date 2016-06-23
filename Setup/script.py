import numpy as np
import matplotlib.pyplot as plt

import pyle.plotting.plothelper as ph


LABEL_FONT_SIZE = 24
TICK_LABEL_FONT_SIZE = 20
SUBPLOT_LABEL_FONT_SIZE = 28
TICK_WIDTH = 4
TICK_LENGTH = 4
CB_TICK_LABEL_FONT_SIZE = 16
MARKER_SIZE = 8
LEGEND_FONT_SIZE = 18
LINE_WIDTH = 4


def makePresentable(ax, xlabel=None, ylabel=None,
                    labelFontSize=LABEL_FONT_SIZE,
                    tickWidth=TICK_WIDTH, tickLength=TICK_LENGTH, tickColor='k',
                    tickLabelFontSize=TICK_LABEL_FONT_SIZE,
                    yTicks=None, xTicks=None,
                    gridLineWidth=2, gridColor='k',
                    yLimits=None, xLimits=None):
    """One stop shop for adjusting plot parameters"""
    for axis, ticks, label in zip((ax.xaxis, ax.yaxis), (xTicks, yTicks),
                                  (xlabel, ylabel)):
        if label is not None:
            axis.set_label_text(label)
        # ticks
        if ticks is not None:
            axis.set_ticks(ticks)
            axis.set_ticklabels([str(t) for t in ticks])
        axis.set_tick_params(length=tickLength ,width=tickWidth,
            color=tickColor, labelsize=tickLabelFontSize)
        # Label font
        label = axis.get_label()
        label.set_fontsize(labelFontSize)
    # grid
    ax.grid(linewidth=gridLineWidth, color=gridColor)
    # x and y limits
    if xLimits is not None:
        left, right = xLimits
        ax.set_xlim(left=left, right=right)
    if yLimits is not None:
        bottom, top = yLimits
        ax.set_ylim(bottom=bottom, top=top)


def thermalNoiseRatio():
    def thermalNoise(x):
        return 1.0 / (np.exp(1.0/x) - 1)
    
    # Define x = kT/hf.
    # Set f = 6GHz.
    # At 4K, x is approximately 13.8
    
    x = np.logspace(-2.1, 0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(x, thermalNoise(13.8*x)/thermalNoise(13.8), linewidth=LINE_WIDTH)
    ax.set_xlim([0.009, 1.2])
    ax.set_ylim([6E-6, 2])
    makePresentable(ax, r"Relative temperature $\alpha$", "Relative power",
        tickLabelFontSize=32, labelFontSize=36)
