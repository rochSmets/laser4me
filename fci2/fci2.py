
import sys
import os
from typing import Union

from numpy.core._multiarray_umath import ndarray

sys.path.append(os.path.join(os.environ['PYWI_DIR'], 'draws'))
import numpy as np
from scipy.interpolate import griddata
import colp
import matplotlib.pyplot as plt


# .. add the path
dataPath = os.path.join(os.environ['HOME'], 'shErpA/blAckDog/fcI2/run/lmj2019/')

# .. set the needed parameter to plot the 2d field
field     = 'Ne'                 # in ['B', 'Ne', 'P', 'Te', 'RO']
unit      = 'lab'                # in ['lab', 'hyb']
runId     = 'qa1cnnsr'           # in ['RLaaLr01', 'RLaoLr01', 'qa1cnnsr', 'L17CU001']
A         = 197                  # mass number
time      = 2.0                  # has to be in ns (like in fci2)
domain    = [[0, 200], [0, 1200]]  #in um ('lab') or in inertial length ('hyb')
Nsampl    = [50, 100]
bounds    = [0, 1e28]
colormap  = ['summer_r', 64]
flines    = 8
ticks     = None
subticks  = None
figsize   = [6, 2]
filetype  = 'pdf'

fileB = dataPath+runId+'_BTETA_'+'{:1.3f}ns.npy'.format(time)
fileN = dataPath+runId+'_NE_'   +'{:1.3f}ns.npy'.format(time)
fileR = dataPath+runId+'_RO_'   +'{:1.3f}ns.npy'.format(time)
fileP = dataPath+runId+'_P_'    +'{:1.3f}ns.npy'.format(time)
fileT = dataPath+runId+'_TE_'    +'{:1.3f}ns.npy'.format(time)
fileX = dataPath+runId+'_XMAIL_'+'{:1.3f}ns.npy'.format(time)
fileY = dataPath+runId+'_YMAIL_'+'{:1.3f}ns.npy'.format(time)

rawB = np.load(fileB)
rawN = np.load(fileN)
if os.path.isfile(fileR) :
    rawR = np.load(fileR)
if os.path.isfile(fileP) :
    rawP = np.load(fileP)
if os.path.isfile(fileT) :
    rawT = np.load(fileT)
rawX = np.load(fileX)
rawY = np.load(fileY)

mu0 = 1.26e-6
m0 = A * 1.67e-27
e0 = 1.6e-19
kB = 1.38e-23

if unit is 'hyb':
    b0Cgs = np.max(np.fabs(rawB)) ######################################
    # max de N pour r < 400 um, whatever z
    x = np.linspace(0, 0.01, 40)
    y = np.linspace(0, 0.04, 40)
    X, Y = np.meshgrid(x, y)
    data0 = np.nan_to_num(griddata((rawX, rawY), rawN, (X, Y)))
    n0Cgs = np.max(data0) ##############################################
    n0 = n0Cgs*1e6
    if os.path.isfile(fileR):
        roCgs = np.max(np.fabs(rawR)) ####################################
        ro = roCgs*1e3
    l0 = np.sqrt(m0/(mu0*n0*e0*e0))
    l0Cgs = l0*1e2
    v0 = np.sqrt(b0Cgs*b0Cgs*1e-8/(mu0*n0*m0))
    v0Cgs = v0*1e2
    p0 = n0*m0*v0*v0
    p0Cgs = p0*1e-6*1e3*1e4
    t0 = p0/(n0*kB)
    t0Cgs = t0*e0/kB

    if domain is None :
        domain = [[0, 20], [0, 200]]
    domainCgs =[[domain[0][0]*l0Cgs, domain[0][1]*l0Cgs], [domain[1][0]*l0Cgs, domain[1][1]*l0Cgs]]

elif unit is 'lab':
    if domain is None :
        domain = [[0, 200], [0, 1000]]
    domainCgs = [[domain[0][0]*1e-4, domain[0][1]*1e-4], [domain[1][0]*1e-4, domain[1][1]*1e-4]]

else :
    print('what the fuck is this unit ?')

if field is 'B' :
    rawF = rawB
elif field is 'Ne' :
    rawF = rawN
elif field is 'RO' :
    if os.path.isfile(fileR):
        rawF = rawR
elif field is 'P' :
    if os.path.isfile(fileP):
        rawF = rawP
    else :
        rawF = rawN*kB*rawT*1e7
elif field is 'Te' :
    if os.path.isfile(fileT) :
        rawF = rawT
    else :
        rawF = rawP/(rawN*kB*1e7)

# .. x, y are in cgs, the units in npy files
x = np.linspace(domainCgs[0][0], domainCgs[0][1], Nsampl[0])
y = np.linspace(domainCgs[1][0], domainCgs[1][1], Nsampl[1])
X,Y  = np.meshgrid(x, y)

# .. axis, data will also be in cgs
axis = [[domain[1][0], domain[1][1]], [domain[0][0], domain[0][1]]]
data = np.nan_to_num(griddata((rawX, rawY), rawF, (X, Y)))

if field is 'B' :
    #fstr = '_BTETA_'
    if unit is 'hyb' :
        text = ['$B/B_0$']
        labels = ['$r/l_0$', '$z/l_0$']
        fnorm = 1/np.min(data)
        xynorm = 1/l0Cgs
    elif unit is 'lab' :
        text = ['$B\mathrm{~(T)}$']
        labels = ['$r~\mathrm{(}\mu\mathrm{m)}$', '$z~\mathrm{(}\mu\mathrm{m)}$']
        fnorm = -1e-4
        xynorm = 1e-4

elif field is 'Ne' :
    #fstr = '_NE_'
    if unit is 'hyb' :
        text = ['$N/N_0$']
        labels = ['$r/l_0$', '$z/l_O$']
        fnorm = 1/np.max(data)
        xynorm = 1/l0Cgs
    elif unit is 'lab' :
        text = ['$N_e\mathrm{~(m}^{-3})$']
        labels = ['$r~\mathrm{(}\mu\mathrm{m)}$', '$z~\mathrm{(}\mu\mathrm{m)}$']
        fnorm = 1e6
        xynorm = 1e-4

elif field is 'RO' :
    #fstr = '_NE_'
    if unit is 'hyb' :
        text = ['$N/N_0$']
        labels = ['$r/l_0$', '$z/l_O$']
        fnorm = 1/np.max(data)
        xynorm = 1/l0Cgs
    elif unit is 'lab' :
        text = ['$\\rho_i \mathrm{~(Kg.m}^{-3})$']
        labels = ['$r~\mathrm{(}\mu\mathrm{m)}$', '$z~\mathrm{(}\mu\mathrm{m)}$']
        fnorm = 1e3
        xynorm = 1e-4

elif field is 'P' :
    #fstr = '_P_'
    if unit is 'hyb' :
        text = ['$P_{\mathrm{kin}}/P_0$']
        labels = ['$r/l_0$', '$z/l_O$']
        fnorm = 1/p0Cgs
        xynorm = 1e-4
    elif unit is 'lab' :
        text = ['$P_{\mathrm{kin}}\mathrm{~(Pa)}$']
        labels = ['$r~\mathrm{(}\mu\mathrm{m)}$', '$z~\mathrm{(}\mu\mathrm{m)}$']
        fnorm = 1e-1
        xynorm = 1e-4

elif field is 'Te' :
    if unit is 'hyb' :
        text = ['$T_e/T_0$']
        labels = ['$r/l_0$', '$z/l_O$']
        fnorm = 1/t0Cgs
        xynorm = 1e-4
    elif unit is 'lab' :
        text = ['$T_e\mathrm{~(eV)}$']
        labels = ['$r~\mathrm{(}\mu\mathrm{m)}$', '$z~\mathrm{(}\mu\mathrm{m)}$']
        fnorm = kB/e0
        xynorm = 1e-4

xytext = [(0.3*domain[1][0]+0.7*domain[1][1], 0.2*domain[0][0]+0.8*domain[0][1])]
data = data*fnorm

# .. draw the plot
plo = colp.Colp(coloraxis = axis,
                colordata = data,
                bounds = bounds,
                colormap = colormap,
                contouraxis = None,
                contourdata = None,
                flines = flines,
                arrowaxis = None,
                arrowdata = None,
                labels = labels,
                ticks = ticks,
                subticks = subticks,
                colorbar = True,
                text = text,
                xytext = xytext,
                figsize = figsize,
                filetype = filetype)
#
