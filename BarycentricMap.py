import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
import PostProcess_AnisotropyTensor as PPAT
import time as t
from Utilities import timer

@timer
# @njit(parallel = True, fastmath = True)
def getBarycentricMapCoordinates(eigValsGrid, c_offset = 0.65, c_exp = 5.):
    # Coordinates of the anisotropy tensor in the tensor basis {a1c, a2c, a3c}. From Banerjee (2007),
    # C1c = lambda1 - lambda2,
    # C2c = 2(lambda2 - lambda3),
    # C3c = 3lambda3 + 1
    c1 = eigValsGrid[:, :, 0] - eigValsGrid[:, :, 1]
    # Not used for coordinates, only for color maps
    c2 = 2*(eigValsGrid[:, :, 1] - eigValsGrid[:, :, 2])
    c3 = 3*eigValsGrid[:, :, 2] + 1
    # Corners of the barycentric triangle
    # Can be random coordinates?
    x1c, x2c, x3c = 1., 0., 1/2.
    y1c, y2c, y3c = 0, 0, np.sqrt(3)/2
    # xBary, yBary = c1*x1c + c2*x2c + c3*x3c, c1*y1c + c2*y2c + c3*y3c
    xBary, yBary = c1 + 0.5*c3, y3c*c3
    # Origin RGB values
    rgbVals = np.dstack((c1, c2, c3))
    # For better barycentric map, use transformation on c1, c2, c3, as in Emory et al. (2014)
    # ci_star = (ci + c_offset)^c_exp,
    # Improved RGB = [c1_star, c2_star, c3_star]
    rgbValsNew = np.empty((c1.shape[0], c1.shape[1], 3))
    # Each 3rd dim is an RGB array of the 2D grid
    for i in range(3):
        rgbValsNew[:, :, i] = (rgbVals[:, :, i] + c_offset)**c_exp

    return xBary, yBary, rgbValsNew

# vals2D = y_pred
#
# t0 = t.time()
# # In each eigenvector 3 x 3 matrix, 1 col is a vector
# vals2D, tensors, eigValsGrid, eigVecsGrid = PPAT.processAnisotropyTensor_Uninterpolated(vals2D, realizeIter = 0)
# t1 = t.time()
# print('\nFinished processAnisotropyTensor_Uninterpolated in {:.4f} s'.format(t1 - t0))
