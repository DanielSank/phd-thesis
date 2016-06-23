import itertools
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

import labrad
import labrad.units as units
ns, MHz, GHz, dBm= (units.Unit(s) for s in ['ns', 'MHz', 'GHz', 'dBm'])

import pyle.datavault as dv
import pyle.plotting.plothelper as ph
import pyle.analysis.readout as roa
import pyle.envelopes as env
import pyle.dataking.envelopehelpers as eh
import pyle.fitting.fitting as fitting
import pyle.analysis.readout as roa

LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 24
SUBPLOT_LABEL_FONT_SIZE = 28
TICK_WIDTH = 4
TICK_LENGTH = 4
CB_TICK_LABEL_FONT_SIZE = 18
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
        

# Photons versus time

PHOTONS_VS_TIME_PATH = ['', 'Evan', 'Qubit', 'Purcell5', '131005', 7]

def analyze_resonator_power(ds, toffset=0.0):
    """
    Extracts information related to readout power from data set
    
    This is a copy of code from Evan's dev directory.
    """
    import pyle.dataking.adjust as adjust
    qubit_name = ds.parameters['measure'][0]
    if isinstance(qubit_name, int): # Old style measure
        qubit_name = ds.parameters['config'][qubit_name]
    q = ds.parameters[qubit_name]
    x = fitting.minimaCuts(ds, 'columns', 10, invert=True)
    idx = np.argsort(x[:,0])
    delays_ns = x[idx,0] - toffset
    freqs_GHz = x[idx,1]
    df_GHz = q['f10']['GHz'] - freqs_GHz

    chi = 0.5*(q['readoutResonatorFreq1'] - q['readoutResonatorFreq0'])
    chi_GHz = chi['GHz']
    photons = -df_GHz / (2*chi_GHz)

    eta = (q['f21'] - q['f10'])
    delta = (q['f10'] - q['readoutFrequency'])
    g = (chi * delta * (1+delta/eta)).sqrt()
    
    ncrit = 0.25*((delta/g)**2)['']
    
    # print "ncrit: %s" % (ncrit,)
    start_idx = np.argmax(photons)
    start_guess = delays_ns[start_idx+3]

    # adj = adjust.Adjust()
    # adj.plot(delays_ns, photons, 'b.')
    # adj.x_param('tstart', start_guess)
    # adj.x_param('tstop', np.max(delays_ns))
    # result = adj.run()
    result = {}
    result['tstart'] = start_guess
    result['tstop'] = np.max(delays_ns)

    if not result:
        return
    tstart = result['tstart']
    tstop = result['tstop']
    mask = np.logical_and(delays_ns > tstart, delays_ns < tstop)
    result = fitting.fitCurve('exponential', delays_ns[mask], photons[mask], (max(photons), tstop-tstart, min(photons)))
    gamma0 = result[0][0]
    tau_power = result[0][1]
    offset = result[0][2]
    Q = tau_power * 2*np.pi*q['readoutFrequency']['GHz']
    # plt.figure()
    # plt.xlabel('Time (ns)')
    # plt.ylabel('Photons')
    # plt.plot(delays_ns[mask], fitting.exponential(delays_ns[mask], *result[0]), 'b-')
    # #plt.plot(delays_ns, np.ones_like(delays_ns)*4*ncrit, 'r-', linewidth=2, label='$4 n_{crit}$')
    # #plt.plot(delays_ns, np.ones_like(delays_ns)*ncrit, 'g--', linewidth=2, label='$n_{crit}$')
    # plt.plot(delays_ns, photons, 'bo', label='Data')
    # #plt.legend()
    phase_error = 2*chi_GHz*tau_power * fitting.exponential(delays_ns[mask], gamma0, tau_power, 0)
    P_max = labrad.units.hbar * (2 * np.pi  * q['readoutFrequency'])**2 * max(photons) / Q
    P_max_dB = 10*np.log10(P_max['mW'])*dBm
    # print "Max power: ", P_max_dB
    return [delays_ns, photons, delays_ns[mask], fitting.exponential(delays_ns[mask], *result[0]), Q]


def photonsVsTime_raw(dataset, ax):
    """Raw data (2D plot) photons versus time"""
    
    delay, photons, delay_mask, fit_data, Q = \
        analyze_resonator_power(dataset, toffset=50*ns)

    # Find x/y values
    xValues = set(dataset[:,0])
    yValues = set(dataset[:,1])
    xLen = len(xValues)
    yLen = len(yValues)
    
    tMin_ns = min(xValues)
    tMax_ns = max(xValues)
    fMin_GHz = min(yValues)
    fMax_GHz = max(yValues)
    
    # plot
    z = np.reshape(dataset[:,-1], (xLen, yLen))
    z = np.transpose(z)
    z = z[::-1,:]
    extent=[min(delay), max(delay), fMin_GHz, fMax_GHz]
    image = ax.imshow(z, aspect=350, interpolation='none', extent=extent)
    
    # Ticks
    times_ns = np.array([x for x in xValues])
    times_ns.sort()
    frequencies_GHz = np.array([y for y in yValues])
    frequencies_GHz.sort()
    xAxis = ax.xaxis
    yAxis = ax.yaxis
    
    xAxis.set_tick_params(width=TICK_WIDTH, color='w')
    yAxis.set_tick_params(width=TICK_WIDTH, color='w')
    
    # Grid lines
    ax.grid(b=True, which='major', color='w', linestyle='--')
    
    # Axis labels
    ax.set_ylabel('Frequency [GHz]')
    
    return ax, image


def photonsVsTime_fitted(dataset, ax):
    """Fit raw photons vs time data"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    delay, photons, delay_mask, fit_data, Q = \
        analyze_resonator_power(dataset, toffset=50*ns)
    
    q = dataset.parameters[dataset.parameters['measure'][0]]
    ax.plot(delay, photons, 'b.', markersize=10)
    ax.plot(delay_mask, fit_data, 'r-', linewidth=2)
    t = np.linspace(min(delay), max(delay), 1000)
    ax.set_ylabel('Resonator\nPhotons')
    ax.set_xlabel(r'$\tau$ [ns]')
    ax.grid()
    return ax


def photonsVsTime_pulseSequence(dataset, ax):
    """Show pulse sequence for photons vs time"""
    
    delay, photons, delay_mask, fit_data, Q = \
        analyze_resonator_power(dataset, toffset=50*ns)
    
    q = dataset.parameters[dataset.parameters['measure'][0]]
    rr = env.flattop(0.5*q['readoutWidth'], q['readoutLen'], q['readoutWidth'])+\
        (q['preEmphFactor'] - 1.0) * env.flattop(0.5 * q['readoutWidth'],
         q['preEmphLen'], q['readoutWidth'])
    rr = rr / (q['preEmphFactor'])
    rr = rr + env.shift(rr, 500)
    rr = env.mix(rr, df=150*MHz)
    
    xy = env.cosine(100, q['piLen']['ns']/2)
    xy = env.mix(xy, df=150*MHz)
    xy = xy/2
    
    xy_left = env.shift(xy, -40)
    xy_right = env.shift(xy, 40)
    
    t = np.linspace(min(delay), max(delay)+500, 2000)
    ax.plot(t, rr(t, fourier=False), 'g-', linewidth=3, label='Measurement')
    ax.plot(t, xy(t, fourier=False) + 1, 'b-', linewidth=3, label='XY')
    #ax.plot(t, xy_left(t, fourier=False) + 1, 'b-', alpha=0.5, linewidth=3)
    #ax.plot(t, xy_right(t, fourier=False) + 1, 'b-', alpha=0.5, linewidth=3)
    ax.arrow(0, 1.5, 80, 0, head_width=0.15, head_length=20, fc='k', ec='k',
             linewidth=3)
    
    ax.xaxis.set_label_text('Time [ns]')
    ax.yaxis.set_label_text('Pulse amplitude [a.u.]')
    ax.text(50, 1.55, r'$\tau$', fontsize=30)
    return ax


def photonsVsTime(cxn, dataPath=None, dataNum=None,
                  labelFontSize=LABEL_FONT_SIZE,
                  tickLabelFontSize=TICK_LABEL_FONT_SIZE,
                  subplotLabelFontSize=SUBPLOT_LABEL_FONT_SIZE,
                  markerEdgeWidth=TICK_WIDTH,
                  tickSize=TICK_LENGTH,
                  colorBarTickLabelFontSize=CB_TICK_LABEL_FONT_SIZE):
    """Generate complete photons vs time figure"""
    
    if dataPath is None:
        dataPath = PHOTONS_VS_TIME_PATH[:-1]
    if dataNum is None:
        dataNum = PHOTONS_VS_TIME_PATH[-1]
    
    # Fetch data
    dvw = dv.DataVaultWrapper(dataPath, cxn)
    dataset = dvw[dataNum]
    
    # Arrange axes and plot
    f = plt.figure()
    # Upper panel - control sequence
    gsControl = gridspec.GridSpec(1, 1)
    gsControl.update(left=0.1, right=0.95, top=0.95, bottom=0.7)
    axControl = plt.subplot(gsControl[:, :])
    axControl = photonsVsTime_pulseSequence(dataset, axControl)
    # Lower panel - data
    gsData = gridspec.GridSpec(6, 30)
    gsData.update(left=0.1, right=0.95, top=0.62, bottom=0.1,
        hspace=0)
    # Raw data
    axRaw = plt.subplot(gsData[0:4, 0:29])
    axRaw, imageRaw = photonsVsTime_raw(dataset, axRaw)
    axColorbar = plt.subplot(gsData[0:4, 29])
    cb = plt.colorbar(imageRaw, cax=axColorbar, orientation='vertical')
    cb.solids.set_edgecolor("face")
    # Fit
    axFit = plt.subplot(gsData[4:, 0:29], sharex=axRaw)
    axFit = photonsVsTime_fitted(dataset, axFit)
    
    # Adjust text and tick sizes
    for ax in (axControl, axRaw, axFit):
        ax.grid(linewidth=2)
        yAxis = ax.yaxis
        xAxis = ax.xaxis
        # Set y axis label and y tick label sizes
        label = yAxis.get_label()
        label.set_fontsize(labelFontSize)
        for label in yAxis.get_ticklabels():
            label.set_fontsize(tickLabelFontSize)
        # Set tick sizes
        for axis in [xAxis, yAxis]:
            for line in axis.get_ticklines():
                line.set_markeredgewidth(markerEdgeWidth)
                line.set_markersize(tickSize)
    
    for ax in [axControl, axFit]:
        ax.xaxis.get_label().set_fontsize(labelFontSize)
        for label in ax.xaxis.get_ticklabels():
            label.set_fontsize(tickLabelFontSize)
    
    cbTicks=[0.2, 0.4, 0.6, 0.8]
    cb.set_ticks(cbTicks)
    cb.ax.tick_params(labelsize=colorBarTickLabelFontSize, color='k',
                      width=markerEdgeWidth/2.0, length=markerEdgeWidth*1.5/2.0) 
    
    # Set axes limits
    axControl.set_ylim(bottom=-1.2, top=1.88)
    axControl.set_xlim(left=-45, right=725)
    axRaw.set_xlim(left=-47, right=460)
    axFit.set_ylim(bottom=-21, top=167)
    
    # Label subplots
    f.text(0.12, 0.89, 'a)', fontsize=subplotLabelFontSize)
    f.text(0.12, 0.5, 'b)', fontsize=subplotLabelFontSize, color='w')
    f.text(0.12, 0.225, 'c)', fontsize=subplotLabelFontSize)
    

# Lollipop

LOLLIPOP_PATH = ['','Daniel','Transmon','Purcell5','130826',19]

def lollipopDips(cxn, dataPath=None, dataNum=None, fig=None):
    """Show resonance dips for 0, 1, 2 states"""
    
    if dataPath is None:
        dataPath = LOLLIPOP_PATH[:-1]
    if dataNum is None:
        dataNum = LOLLIPOP_PATH[-1]
    
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    
    dvw = dv.DataVaultWrapper(dataPath, cxn)
    dataset = dvw[dataNum]
    
    colors = ['b','r','g']
    
    freq_GHz = dataset[:,0]
    
    kets = (r'$\left| 0 \rangle \right.$', r'$\left| 1 \rangle \right.$',
            r'$\left| 2 \rangle \right.$')
    for i, color, ket, in zip([0,1,2], colors, kets):
        ax.plot(freq_GHz, np.sqrt(dataset[:,2*i+1]**2+dataset[:,2*i+2]**2),
                color+'-', linewidth=3, label=ket)
    ax.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
    ax.yaxis.set_ticklabels(['', '0', '5', '10', '15', '20', ''])
    ax.grid()
    makePresentable(ax, xlabel='Frequency [GHz]',
                    ylabel=r'$\left| S_{21} \right|$ [au]',
                    xLimits=[6.786, 6.812])
    return ax


# AC Stark

AC_STARK_PATH = ['', 'Daniel', 'Transmon', 'Purcell5', '130928', 43]

def acStark(cxn, dataPath=None, dataNum=None,
            labelFontSize=LABEL_FONT_SIZE,
            tickLabelFontSize=TICK_LABEL_FONT_SIZE,
            subplotLabelFontSize=SUBPLOT_LABEL_FONT_SIZE,
            markerEdgeWidth=TICK_WIDTH,
            tickSize=TICK_LENGTH,
            colorBarTickLabelFontSize=CB_TICK_LABEL_FONT_SIZE):
    
    from matplotlib import rc
    
    if dataPath is None:
        dataPath = AC_STARK_PATH[:-1]
    if dataNum is None:
        dataNum = AC_STARK_PATH[-1]
    
    dvw = dv.DataVaultWrapper(dataPath, cxn)
    dataset = dvw[dataNum]
    
    dacAmpSquared = dataset[:,0]
    frequency_GHz = dataset[:,1]
    
    dacAmpSquared = np.array([x for x in set(dacAmpSquared)])
    dacAmpSquared.sort()
    
    frequency_GHz = np.array([x for x in set(frequency_GHz)])
    frequency_GHz.sort()
    
    z = dataset[:,2].reshape((len(dacAmpSquared), len(frequency_GHz)))
    z = z.transpose()[::-1, :]
    
    f = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.95, top=0.95, bottom=0.1)
    axData = plt.subplot(gs[:, :])
    # axColorbar = plt.subplot(gs[:,19])
    
    extent=[min(dacAmpSquared), max(dacAmpSquared), min(frequency_GHz),
        max(frequency_GHz)]
    image = axData.imshow(z, interpolation='none', extent=extent, aspect=0.1)
    axData.set_xlabel('DAC amplitude squared [a.u.]')
    axData.set_ylabel('Frequency [GHz]')
    
    cb = plt.colorbar(image, ax=axData, orientation='vertical')
    cb.solids.set_edgecolor("face")
    cbTicks=[0.2, 0.4, 0.6, 0.8]
    cb.set_ticks(cbTicks)
    cb.ax.tick_params(labelsize=CB_TICK_LABEL_FONT_SIZE, color='k',
                      width=TICK_WIDTH/2.0, length=TICK_WIDTH*1.5/2.0) 
    
    # Grid
    axData.grid(b=True, which='major', color='w', linestyle='--', linewidth=2)
    
    for ax in [axData.yaxis, axData.xaxis]:
        # Set y axis label and y tick label sizes
        label = ax.get_label()
        label.set_fontsize(labelFontSize)
        for label in ax.get_ticklabels():
            label.set_fontsize(tickLabelFontSize)
        # Set tick sizes
        for line in ax.get_ticklines():
            line.set_markeredgewidth(markerEdgeWidth)
            line.set_markersize(tickSize)
            line.set_color('white')
    
    cbTicks=[0.2, 0.4, 0.6, 0.8]
    cb.set_ticks(cbTicks)
    cb.ax.tick_params(labelsize=colorBarTickLabelFontSize, color='k',
                      width=markerEdgeWidth/2.0, length=markerEdgeWidth*1.5/2.0) 
    
    # Set axes limits
    # axData.set_ylim(bottom=5.65, top=5.8)
    
    return image


# Stimulated transitions

STIMULATED_TRANSITIONS_2D_PATH = ['', 'Daniel', 'Transmon', 'Purcell5', '130826',
                               32]

def stimulatedTransitions2D_pulseSequence(dataset, ax, varPower=False):
    """Show pulse sequence for photons vs time"""
    
    delay = np.linspace(-100, 800, 2000)
    
    q = dataset.parameters['q2']
    rr = env.flattop(0.5*q['readoutWidth'], q['readoutLen'], q['readoutWidth'])+\
        (q['preEmphFactor'] - 1.0) * env.flattop(0.5 * q['readoutWidth'],
         q['preEmphLen'], q['readoutWidth'])
    rr = rr / (q['preEmphFactor'])
    if not varPower:
        rr = rr + env.shift(rr, 500)
    else:
        rr = rr + env.shift(rr, 500)/2.0
    rr = env.mix(rr, df=150*MHz)
    
    xy = env.cosine(-40, q['piLen']['ns']/2)
    xy = env.mix(xy, df=150*MHz)
    xy = xy/2
    
    xy_left = env.shift(xy, -40)
    xy_right = env.shift(xy, 40)
    t = np.linspace(min(delay), 700, 2000)
    ax.plot(t, rr(t, fourier=False), 'g-', linewidth=3, label='Measurement')
    ax.plot(t, xy(t, fourier=False) + 1, 'b-', linewidth=3, label='XY')
    
    makePresentable(ax, xlabel='Time [ns]', ylabel='Pulse amp. [au]',
                    labelFontSize=LABEL_FONT_SIZE,
                    tickWidth=TICK_WIDTH, tickLength=TICK_LENGTH,
                    tickColor='k', yTicks=(-1.0, 0, 1.0),
                    tickLabelFontSize=TICK_LABEL_FONT_SIZE,
                    gridLineWidth=2, gridColor='k',
                    yLimits=(-1.2, 1.75))
    return ax


def stimulatedTransitions2D(cxn, dataPath=None, dataNum=None,
            labelFontSize=LABEL_FONT_SIZE,
            tickLabelFontSize=TICK_LABEL_FONT_SIZE,
            subplotLabelFontSize=SUBPLOT_LABEL_FONT_SIZE,
            markerEdgeWidth=TICK_WIDTH,
            tickSize=TICK_LENGTH,
            colorBarTickLabelFontSize=CB_TICK_LABEL_FONT_SIZE):
    
    if dataPath is None:
        dataPath = STIMULATED_TRANSITIONS_2D_PATH[:-1]
    if dataNum is None:
        dataNum = STIMULATED_TRANSITIONS_2D_PATH[-1]
    
    dvw = dv.DataVaultWrapper(dataPath, cxn)
    dataset = dvw[dataNum]
    
    f = plt.figure()
    numDataRows = 2
    
    # Upper panel - control sequence
    gsControl = gridspec.GridSpec(1, 1)
    gsControl.update(left=0.15, right=0.95, top=0.95, bottom=0.8)
    axControl = plt.subplot(gsControl[:, :])
    axControl = stimulatedTransitions2D_pulseSequence(dataset, axControl)
    # Lower panel - data
    gsData = gridspec.GridSpec(numDataRows, 1)
    gsData.update(left=0.15, right=0.95, top=0.72, bottom=0.1, hspace=0.1)
    
    drivePower_dBm = dataset[:,0]
    driveLen_ns = dataset[:,1]
    
    drivePower_dBm = np.array([x for x in set(drivePower_dBm)])
    drivePower_dBm.sort()
    driveLen_ns = np.array([x for x in set(driveLen_ns)])
    driveLen_ns.sort()
    
    # Convert power photons
    q = dataset.parameters['q2']
    chi_GHz = 0.5*(q['resonatorFreq1'] - q['resonatorFreq0'])['GHz']
    calAcStark = dataset.parameters['q2']['calAcStark']
    dacAmpSquared = np.array([eh.power2amp(p_dBm*dBm) \
        for p_dBm in drivePower_dBm])**2
    resonatorPhotons = (calAcStark * dacAmpSquared / (2 * chi_GHz))
    
    zUpward = dataset[:,2].reshape((len(drivePower_dBm), len(driveLen_ns)))
    zDownward = dataset[:,3].reshape((len(drivePower_dBm), len(driveLen_ns)))
    zUpward = zUpward.transpose()[::-1, :]
    zDownward = zDownward.transpose()[::-1, :]
    
    # Plot data
    axUpward = plt.subplot(gsData[0,:])
    axDownward = plt.subplot(gsData[1,:])
    aspect = 0.05
    extent=[min(drivePower_dBm), max(drivePower_dBm), min(driveLen_ns),
        max(driveLen_ns)]
    imageUpward = axUpward.imshow(zUpward, interpolation='none',
        extent=extent,
        aspect=aspect)
    imageDownward = axDownward.imshow(zDownward, interpolation='none',
        extent=extent,
        aspect=aspect)
    
    # nMin = min(resonatorPhotons)
    # nMax = max(resonatorPhotons)
    # #photonTicks = np.logspace(np.log10(nMin), np.log10(nMax), 2)
    # photonTicks = [10, 33, 100, 200, 300]
    # axUpward.xaxis.set_ticks(photonTicks)
    # axDownward.xaxis.set_ticks(photonTicks)
    # axDownward.xaxis.set_ticklabels(photonTicks)
    
    # color bars
    for ax, image in zip([axUpward, axDownward], [imageUpward, imageDownward]):
        cb = plt.colorbar(image, ax=ax, orientation='vertical', fraction=0.02)
        cb.solids.set_edgecolor("face")
        cbTicks = [0.2, 0.4, 0.6, 0.8]
        cb.set_ticks(cbTicks)
        cb.ax.tick_params(labelsize=CB_TICK_LABEL_FONT_SIZE, color='k',
                          width=TICK_WIDTH/2.0, length=TICK_WIDTH*1.5/2.0)
    
    axUpward.xaxis.set_ticklabels([])
    axDownward.set_xlabel('Power [dBm]')
    for ax in (axUpward, axDownward):
        # y label
        ax.set_ylabel('Pulse length [ns]')
        # grid
        ax.grid(b=True, which='major', color='w', linestyle='--', linewidth=2)
        for axis in [ax.yaxis, ax.xaxis]:
            # ticks
            axis.set_tick_params(length=TICK_LENGTH ,width=TICK_WIDTH,
                color='w', labelsize=TICK_LABEL_FONT_SIZE)
            # Label font
            label = axis.get_label()
            label.set_fontsize(labelFontSize)
        
        # y ticks
        yTicks = [100, 140, 180]
        ax.yaxis.set_ticks(yTicks)
        ax.yaxis.set_ticklabels([str(t) for t in yTicks])
    
    # Subplot labels
    f.text(0.04, 0.925, 'a)', fontsize=SUBPLOT_LABEL_FONT_SIZE)
    f.text(0.04, 0.675, 'b)', fontsize=SUBPLOT_LABEL_FONT_SIZE)
    f.text(0.04, 0.35, 'c)', fontsize=SUBPLOT_LABEL_FONT_SIZE)


def stimulatedTransitions2D_noControlSequence(cxn, dataPath=None, dataNum=None,
            labelFontSize=LABEL_FONT_SIZE,
            tickLabelFontSize=TICK_LABEL_FONT_SIZE,
            subplotLabelFontSize=SUBPLOT_LABEL_FONT_SIZE,
            markerEdgeWidth=TICK_WIDTH,
            tickSize=TICK_LENGTH,
            colorBarTickLabelFontSize=CB_TICK_LABEL_FONT_SIZE):
    
    if dataPath is None:
        dataPath = STIMULATED_TRANSITIONS_PATH[:-1]
    if dataNum is None:
        dataNum = STIMULATED_TRANSITIONS_PATH[-1]
    
    dvw = dv.DataVaultWrapper(dataPath, cxn)
    dataset = dvw[dataNum]
    
    f = plt.figure()
    numDataRows = 2
    
    # plot panels
    gsData = gridspec.GridSpec(numDataRows, 1)
    gsData.update(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0.1)
    
    drivePower_dBm = dataset[:,0]
    driveLen_ns = dataset[:,1]
    
    drivePower_dBm = np.array([x for x in set(drivePower_dBm)])
    drivePower_dBm.sort()
    driveLen_ns = np.array([x for x in set(driveLen_ns)])
    driveLen_ns.sort()
    
    # Convert power photons
    q = dataset.parameters['q2']
    chi_GHz = 0.5*(q['resonatorFreq1'] - q['resonatorFreq0'])['GHz']
    calAcStark = dataset.parameters['q2']['calAcStark']
    dacAmpSquared = np.array([eh.power2amp(p_dBm*dBm) \
        for p_dBm in drivePower_dBm])**2
    resonatorPhotons = (calAcStark * dacAmpSquared / (2 * chi_GHz))
    
    zUpward = dataset[:,2].reshape((len(drivePower_dBm), len(driveLen_ns)))
    zDownward = dataset[:,3].reshape((len(drivePower_dBm), len(driveLen_ns)))
    zUpward = zUpward.transpose()[::-1, :]
    zDownward = zDownward.transpose()[::-1, :]
    
    # Plot data
    axUpward = plt.subplot(gsData[0,:])
    axDownward = plt.subplot(gsData[1,:])
    aspect = 0.08
    extent=[min(drivePower_dBm), max(drivePower_dBm), min(driveLen_ns),
        max(driveLen_ns)]
    imageUpward = axUpward.imshow(zUpward, interpolation='none',
        extent=extent,
        aspect=aspect)
    imageDownward = axDownward.imshow(zDownward, interpolation='none',
        extent=extent,
        aspect=aspect)
    
    # nMin = min(resonatorPhotons)
    # nMax = max(resonatorPhotons)
    # #photonTicks = np.logspace(np.log10(nMin), np.log10(nMax), 2)
    # photonTicks = [10, 33, 100, 200, 300]
    # axUpward.xaxis.set_ticks(photonTicks)
    # axDownward.xaxis.set_ticks(photonTicks)
    # axDownward.xaxis.set_ticklabels(photonTicks)
    
    # color bars
    for ax, image in zip([axUpward, axDownward], [imageUpward, imageDownward]):
        cb = plt.colorbar(image, ax=ax, orientation='vertical', fraction=0.02)
        cb.solids.set_edgecolor("face")
        cbTicks = [0.2, 0.4, 0.6, 0.8]
        cb.set_ticks(cbTicks)
        cb.ax.tick_params(labelsize=CB_TICK_LABEL_FONT_SIZE, color='k',
                          width=TICK_WIDTH/2.0, length=TICK_WIDTH*1.5/2.0)
    
    axUpward.xaxis.set_ticklabels([])
    axDownward.set_xlabel('Power [dBm]')
    for ax in (axUpward, axDownward):
        # y label
        ax.set_ylabel('Pulse length [ns]')
        # grid
        ax.grid(b=True, which='major', color='w', linestyle='--', linewidth=2)
        for axis in [ax.yaxis, ax.xaxis]:
            # ticks
            axis.set_tick_params(length=TICK_LENGTH ,width=TICK_WIDTH,
                color='w', labelsize=TICK_LABEL_FONT_SIZE)
            # Label font
            label = axis.get_label()
            label.set_fontsize(labelFontSize)
        
        # y ticks
        yTicks = [100, 140, 180]
        ax.yaxis.set_ticks(yTicks)
        ax.yaxis.set_ticklabels([str(t) for t in yTicks])
    
    # Subplot labels
    f.text(0.04, 0.925, 'a)', fontsize=SUBPLOT_LABEL_FONT_SIZE)
    f.text(0.04, 0.475, 'b)', fontsize=SUBPLOT_LABEL_FONT_SIZE)


# 131005, 37 and 54 have stimulated transition data with ac Stark cal


STIMULATED_TRANSITIONS_1D_PATH = ['', 'Daniel', 'Transmon', 'Purcell5',
                                  '131005']

def stimulatedTransitions1D(cxn):
    dvw = dv.DataVaultWrapper(STIMULATED_TRANSITIONS_1D_PATH, cxn)
    dataset37 = dvw[37]
    dataset54 = dvw[54]
    
    f = plt.figure()
    # Pulse sequence
    gsControl = gridspec.GridSpec(1, 1)
    gsControl.update(left=0.15, right=0.95, top=0.95, bottom=0.8)
    axControl = plt.subplot(gsControl[:, :])
    axControl = stimulatedTransitions2D_pulseSequence(dataset37, axControl,
        varPower=True)
    # Data
    gsData = gridspec.GridSpec(1, 1)
    gsData.update(left=0.15, right=0.95, top=0.72, bottom=0.1, hspace=0.1)
    axData37 = plt.subplot(gsData[0, :])
    #axData54 = plt.subplot(gsData[1, :])
    
    legendLabels = [r'$| %d \rangle \rightarrow | %d \rangle$'%(x,y) for x,y in itertools.product((0,1),(0,1,2))]

    dataset = dataset37
    ax = axData37
    
    drivePower_dBm = dataset[:,0]
    # ac Stark slope read off of screen
    m_GHzPerDacAmpSquared = (5.568-5.780)/(0.1482 - 0.0194)
    q = dataset.parameters['q2']
    chi_GHz = 0.5*(q['resonatorFreq1'] - q['resonatorFreq0'])['GHz']
    
    dacAmpSquared = np.array([eh.power2amp(p_dBm*dBm) \
        for p_dBm in drivePower_dBm])**2
    resonatorPhotons = (m_GHzPerDacAmpSquared * \
        dacAmpSquared / (2 * chi_GHz))
    
    for i, label in zip(range(dataset.shape[1]-1), legendLabels):
        ax.plot(resonatorPhotons, dataset[:,i+1], 'o-',
                markersize=12, markeredgewidth=0, linewidth=4,
                label=label)
        ax.legend(numpoints=1, fontsize=LEGEND_FONT_SIZE)
        makePresentable(ax, xlabel='Resonator Photons',
            ylabel='Probability',
            labelFontSize=LABEL_FONT_SIZE,
            tickWidth=TICK_WIDTH, tickLength=TICK_LENGTH,
            tickColor='k', yTicks=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
            tickLabelFontSize=TICK_LABEL_FONT_SIZE,
            gridLineWidth=2, gridColor='k',
            xLimits=(0, 300))
    f.text(0.04, 0.925, 'a)', fontsize=SUBPLOT_LABEL_FONT_SIZE)
    f.text(0.04, 0.7, 'b)', fontsize=SUBPLOT_LABEL_FONT_SIZE)


# Time domain fidelity (two qubits)

ERROR_VS_TIME_PATH = ['', 'Daniel', 'Transmon', 'Purcell5', '131010', 44]

def errorVsTime(cxn):
    """
    data should be as indicated in figureData.txt
    [Daniel,Transmon,Purcell5,131010,44]"
    """
    
    dvw = dv.DataVaultWrapper(ERROR_VS_TIME_PATH[:-1], cxn)
    data = dvw[-1]
    
    heraldRanges = ((280, 535, 1410, 2000),
                    (280, 535, 1410, 2000))
    result = roa.IQTrace_multiQubit(data, debug=True,
                                    heraldRanges=heraldRanges,
                                    optimalWindow=True)
    ax = roa.plotTimeDomain(result, highlightAt=None, legendLoc='lower left',
                   whichQubit=0, qubitNames=[r'$Q_2$'], bins=70)
    makePresentable(ax)
    return ax

# Quantum efficiency
# The functions here are copies from ejeffrey/labrad/dev/pyle/pyle/
#                                    dataking/devEvan
# on June 27 2014 

QUANTUM_EFFICIENCY_PATH = ['', 'Evan', 'Qubit', 'Purcell5', '130928']
# 37: dephasing
# 39: Lollipop vs. power

def uniq(it):
    '''
    A generator that compresses repeated values
    '''
    first = True
    for x in it:
        if first or last_val != x:
            first = False
            last_val = x
            yield x


def fit_gaussian_outliers(x):
    x0 = np.median(x)
    sigma = np.std(x)
    mask = np.logical_and(x > (x0-2*sigma), x<(x+2*sigma))
    hist, bin_edges = np.histogram(x[mask], bins=50)
    bin_centers = np.convolve(bin_edges, [.5, .5], mode='valid') # find bin centers
    def gaussian_fit(x, y0, x0, sigma):
        return 1/(np.sqrt(2*np.pi*sigma**2)) * y0*np.exp(-(x-x0)**2/(2 * sigma**2))
    x0_guess = x0
    sigma_guess = sigma
    y0_guess = np.max(hist)
    p_opt, pcov = scipy.optimize.curve_fit(gaussian_fit, bin_centers, hist, (y0_guess, x0_guess, sigma_guess))
    return p_opt


def analyze_SNR(ds, plot=True):
    '''
    dataset from lollipop(s, m, power=st.r[], dataformat='iqRaw')
    '''
    indep = ds[:,0]
    result = []
    for p in uniq(indep):
        mask = indep == p
        z0 = ds['I |0> pulse'][mask] + 1j * ds['Q |0> pulse'][mask]
        z1 = ds['I |1> pulse'][mask] + 1j * ds['Q |1> pulse'][mask]
        sep = np.mean(z0) - np.mean(z1)
        centroid = (np.mean(z0) + np.mean(z1))/2
        x0 = np.real((z0 - centroid) * np.exp(-1j*np.angle(sep)))
        x1 = np.real((z1 - centroid) * np.exp(-1j*np.angle(sep)))
        sep = np.mean(x0 - x1)
        std0 = fit_gaussian_outliers(x0)[2]
        std1 = fit_gaussian_outliers(x1)[2]
        snr = sep**2 / (std0**2 + std1**2)
        result.append([p, snr, sep**2, std0**2 + std1**2])
    result = np.array(result)
    mask = result[:,0] < -50
    if plot:
        plt.figure()
        plt.plot(result[:,0], 10*np.log10(result[:,1]))
        p = np.polyfit(result[mask,0], 10*np.log10(result[mask,1]), 1)
        plt.plot(result[:,0], np.polyval(p, result[:,0]), '-')
        plt.xlabel("%s (%s)" % (ds.indep[0][0], ds.indep[0][1]))
        plt.ylabel("log10 SNR")
        plt.figure()
        plt.plot(result[:,0], np.exp(-.25 * result[:,1]))
        plt.xlabel("%s (%s)" % (ds.indep[0][0], ds.indep[0][1]))
        plt.ylabel("maximum visibility")   
        plt.figure()
        plt.semilogy(result[:,0], result[:,2], label='Separation^2')
        plt.semilogy(result[:,0], result[:,3], label='variance')
        plt.legend()
    return result


def analyze_dephasing(ds, plot=True):
    indep = ds[:,0]
    phase = ds['Ramsey Phase']
    P1 = ds['P']
    
    def sin_fit(x, A0, V, phi0):
        y = A0*(1+V*np.sin(x+phi0))
        return y
    idx = 0
    rv = []
    for idx, p in enumerate(uniq(indep)):
        mask = indep==p
        x = phase[mask]
        y = P1[mask]
        p_opt, pcov = scipy.optimize.curve_fit(sin_fit, x, y, (np.mean(y), 0.5, np.pi/4))
        '''if (idx%5)==0:
            plt.figure()
            plt.plot(x, y)
            plt.plot(x, sin_fit(x, *p_opt), '-')
        '''
        rv.append((p, np.abs(p_opt[1])))
    data = np.array(rv)
    if plot:
        plt.figure()
        plt.plot(data[:,0], data[:,1]/max(data[:,1]))
        plt.xlabel('%s (%s)' % (ds.indep[0][0], ds.indep[0][1]))
        plt.ylabel('Relative Ramsey Visbility')
    return data


def find_threshold(x, y, threshold=0.5):
    '''
    Returns the x axis value where y=threshold using an interpolator
    '''
    f = scipy.interpolate.interp1d(x, y, bounds_error=False)
    def optim_func(z):
        return (f(z) - threshold)**2
    guess_idx = np.argmin(abs(y - threshold))
    rv = scipy.optimize.fmin(optim_func, x[guess_idx])
    return rv


def quantumEfficiency(cxn, threshold=0.5):
    
    dvw = dv.DataVaultWrapper(QUANTUM_EFFICIENCY_PATH, cxn)
    dephasing_data = dvw[37]
    snr_data = dvw[39]
    
    dephasing = analyze_dephasing(dephasing_data, False)
    snr = analyze_SNR(snr_data, False)
    dephasing[:,1] = dephasing[:,1]/np.max(dephasing[:,1])
    max_visibility = np.exp(-.25*snr[:,1])
    snr_thresh = find_threshold(snr[:,0], max_visibility, threshold)
    dephasing_thresh = find_threshold(dephasing[:,0], dephasing[:,1], threshold)
    qe_est = dephasing_thresh-snr_thresh

    print "Quantum efficiency estimate: %s" % qe_est
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.semilogy(dephasing[:,0], dephasing[:,1], 'bs', markersize=1.5*MARKER_SIZE,
                markeredgewidth=0, label='Measured Ramsey visibility')
    
    ax.semilogy(snr[:,0], np.exp(-.25*snr[:,1]), 'go',
                markersize=1.5*MARKER_SIZE,
                markeredgewidth=0, label='Quantum limit visibility: $\eta=1$')
    
    snr_fine = np.linspace(min(snr[:,0]), max(snr[:,0]), 200)
    
    ax.semilogy(snr[:,0], np.exp(-.25*snr[:,1]), '-',
                markersize=MARKER_SIZE, linewidth=LINE_WIDTH-1, color='g')
    
    ax.semilogy(snr[:,0]+qe_est, np.exp(-.25*snr[:,1]), '-',
                markersize=MARKER_SIZE, linewidth=LINE_WIDTH-1, color='r',
                label='Quantum limit visibility: $\eta=$%.1f dB' % qe_est)
    
    ax.legend(loc='lower left', numpoints=1, fontsize=LEGEND_FONT_SIZE)
    
    makePresentable(ax, xlabel='Measurement pulse power [dBm]',
                    ylabel='Relative Ramsey visibility',
                    labelFontSize=LABEL_FONT_SIZE,
                    tickWidth=2, tickLength=2, tickColor='k',
                    tickLabelFontSize=TICK_LABEL_FONT_SIZE,
                    yTicks=None, xTicks=None,
                    gridLineWidth=2, gridColor='k',
                    yLimits=[1E-3, 2], xLimits=[-71, -29])
    
#    ax.arrow(-43.4, 0.13, 8.5, 0, head_width=0.2, head_length=0.75,
#        fc='k', ec='k', linewidth=3)

    #ax.add_patch(matplotlib.patches.Arrow(-43.4, 0.13, 8.5, 0))
    
    ax.annotate('', xy=(-43.4, 0.13), xytext=(-43.4+8.5, 0.13),
        arrowprops=dict(facecolor='black', arrowstyle='<->', linewidth=4))
    ax.text(-39.5, 0.15, "9 dB", fontsize=24)

