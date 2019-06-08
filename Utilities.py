from numba import njit, jit, prange
import numpy as np
import functools, time

def configurePlotSettings(lineCnt = 2, useTex = True, style = 'default', fontSize = 16, cmap = 'viridis', linewidth =
1):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from warnings import warn

    plt.style.use(style)
    lines = ("-", "--", "-.", ":")*5
    markers = ('o', 'v', '^', '<', '>', 's', '8', 'p')*3

    lineCnt += 1
    if cmap == 'jet':
        colors = plt.cm.jet(np.linspace(0, 1, lineCnt))
    elif cmap == 'coolwarm':
        colors = plt.cm.coolwarm(np.linspace(0, 1, lineCnt))
    elif cmap == 'hsv':
        colors = plt.cm.hsv(np.linspace(0, 1, lineCnt))
    else:
        if cmap != 'viridis':
            warn('\nInvalid color map! Using default [viridis] color map...\n', stacklevel = 2)
        colors = plt.cm.viridis(np.linspace(0, 1, lineCnt))

    mpl.rcParams.update({"legend.framealpha": 0.75,
                         'font.size': fontSize,
                         'text.usetex': useTex,
                         'font.family': 'serif',
                         'lines.linewidth': linewidth})

    return lines, markers, colors


#---------------------------------------------------------------
def convertDataTo2D(list):
    array = np.asarray(list)
    try:
        # If array is already 2D, skip this function
        array.shape(1)
        return
    except:
        array = np.reshape(list, (-1, 1))
        return array


# Apply Numba.njit speed up just for this numpy method. Can't think of better way to do it
# @timer
@njit
def takeClosest(array, val):
    """
    :param array: flattened ordered array
    :param val: value to compare, can be a list of values
    :return idx: index(s) where to plug in val
    :return np.array(list)[idx]: value(s) in list closest to val
    """
    idx = np.searchsorted(array, val)
    # No idea why np.array() doesn't work...
    # list = np.array(list)
    # closest = list[idx]
    return idx, array[idx]



def readData(dataNames, fileDir = './', delimiter = ',', skipRow = 0, skipCol = 0):
    import numpy as np
    import csv
    # import pandas as pd
    if isinstance(dataNames, str):
        dataNames = (dataNames,)

    data = {}
    for dataName in dataNames:
        dataTmp = []
        with open(fileDir + '/' + dataName) as csvFile:
            csv_reader = csv.reader(csvFile, delimiter = delimiter)
            for i, row in enumerate(csv_reader):
                if i >= skipRow:
                    try:
                        dataTmp.append(np.array(row, dtype = float))
                    except ValueError:
                        dataTmp.append(np.array(row))

            dataTmp = np.array(dataTmp, dtype = float)
            # dataTmp = dataTmp[skipRow:, skipCol:]
            data[dataName] = dataTmp

    if len(dataNames) == 1:
        data = data[dataName]

    return data

    # for dataName in dataNames:
    # dataFrame = pd.read_csv(fileDir + '/' + dataNames, sep = delimiter, squeeze = True, engine = 'c', memory_map =
    #     True, skiprows = skipRow, )
    # return dataFrame


def getArrayStepping(arr, section = (0, 1e9), which = 'min'):
    import numpy as np
    # section is if you manually define a section to get stepping
    arr = np.array(arr)
    # Left and right section index
    secL, secR = section[0], min(section[1], arr.size)

    i = secL
    if which is 'min':
        diff = 1000000
    else:
        diff = 0
    while i < secR - 1:
        diffNew = abs(arr[i + 1] - arr[i])
        if which is 'min' and diffNew < diff:
            diff = diffNew
        elif which is 'max' and diffNew > diff:
            diff = diffNew

        i += 1

    return diffNew


def convertAngleToNormalVector(cClockAngleXY, clockAngleZ, unit = 'deg'):
    # clockAngleZ is the clockwise angle from z axis when viewing from either xz plane or yz plane
    import numpy as np
    if unit is 'deg':
        # Counter clockwise angle in xy plane of the normal vector
        cClockAngleXYnorm = cClockAngleXY/180.*np.pi + 0.5*np.pi
        # Inclination of the normal vector from z axis into the xy plane
        clockAngleZnorm = 0.5*np.pi + clockAngleZ/180.*np.pi
    else:
        cClockAngleXYnorm = cClockAngleXY + 0.5*np.pi
        clockAngleZnorm = 0.5*np.pi + clockAngleZ

    dydx = np.tan(cClockAngleXYnorm)
    if dydx == np.inf:
        xNorm = 0
        if clockAngleZnorm == 0:
            yNorm = 1
            zNorm = 0
        else:
            dydz = np.tan(clockAngleZnorm)
            zNorm = np.sqrt(1/(1 + np.tan(clockAngleZnorm)))
            yNorm = zNorm*dydz

    else:
        xNorm = np.sqrt(1/(1 + dydx**2)/(1 + np.tan(0.5*np.pi - clockAngleZnorm)**2))
        yNorm = xNorm*dydx
        zNorm = np.tan(0.5*np.pi - clockAngleZnorm)*np.sqrt(xNorm**2 + yNorm**2)

    return (xNorm, yNorm, zNorm)


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
#        print(f"\nFinished {func.__name__!r} in {run_time:.4f} secs")
        print('\nFinished {!r} in {:.4f} s'.format(func.__name__, run_time))
        return value
    return wrapper_timer



@timer
# TODO: prange fails here
# @jit(parallel = True)
def sampleData(listData, sampleSize, replace=False):
    # Ensure list
    if isinstance(listData, np.ndarray):
        listData = [listData]
    elif isinstance(listData, tuple):
        listData = list(listData)

    # Get indices of the samples
    sampleIdx = np.random.choice(np.arange(len(listData[0])), sampleSize, replace = replace)
    # Go through all provided data
    listDataSamp = listData
    for i in range(len(listData)):
        listDataSamp[i] = listData[i][sampleIdx]

    print('\nData sampled to {0} with {1} replacement'.format(sampleSize, replace))
    return listDataSamp


def nDto2D_TensorField(nD_Tensor):
    """
    Convert a tensor field array of n (n >2) D to 2D as nPoint x nComponent,
    assuming the given tensor field has a structure of {mesh} x {tensor component}
    :param nD_Tensor:
    :type nD_Tensor:
    :return:
    :rtype:
    """
    # Ensure Numpy array
    data = np.array(nD_Tensor)
    # If data is 5D, then assume 3D tensor field, nX x nY x nZ x 3 x 3
    if len(data.shape) == 5:
        data = data.reshape((data.shape[0]*data.shape[1]*data.shape[2], 9))
    # Else if data is 4D
    elif len(data.shape) == 4:
        # If last D has 9/6 (symmetric tensor) component, then assume 3D tensor field, nX x nY x nZ x nComponent
        if data.shape[3] in (6, 9):
            data = data.reshape((data.shape[0]*data.shape[1]*data.shape[2], data.shape[3]))
        # Else if 3 in 4th D
        elif data.shape[3] == 3:
            # If 3 in 3rd D, then assume 2D tensor field, nX x nY x 3 x 3
            if data.shape[2] == 3:
                data = data.reshape((data.shape[0]*data.shape[1], 9))
            # Otherwise assume 3D vector field, nX x nY x nZ x 3
            else:
                data = data.reshape((data.shape[0]*data.shape[1]*data.shape[2], 3))

    # Else if 3D data
    elif len(data.shape) == 3:
        # If 3rd D is 6/9, then assume 2D (symmetric) tensor field, nX x nY x nComponent
        if data.shape[2] in (6, 9):
            data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        # Else if 3rd D is 3
        elif data.shape[2] == 3:
            # If 2nd D is 3, then assume 1D tensor, nPoint x 3 x 3
            if data.shape[1] == 3:
                data = data.reshape((data.shape[0], 9))
            # Otherwise assume 2D vector field, nX x nY x 3
            else:
                data = data.reshape((data.shape[0]*data.shape[1], 3))

    return data













