#Some useful functions for plotting

import numpy as np
def tickfunc(bvals, varphivals, magvals, anglevals, width): #points to plot ticks
    xvals = []
    yvals = []
    for i in range(len(bvals)):
        cosangle = np.cos(anglevals[i])
        sinangle = np.sin(anglevals[i])
        xalpha = bvals[i] * np.cos(varphivals[i])
        ybeta = bvals[i] * np.sin(varphivals[i])
        xvals.append([xalpha - (width/2) * magvals[i] * cosangle, xalpha + (width/2) * magvals[i] * cosangle])
        yvals.append([ybeta - (width/2) * magvals[i] * sinangle, ybeta + (width/2) * magvals[i] * sinangle])
    return xvals, yvals