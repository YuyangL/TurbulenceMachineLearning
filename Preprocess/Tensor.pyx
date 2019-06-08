# cython: language_level = 3str
# cython: embedsignature = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport sqrt
from ..Utility import reverseOldGridShape

cpdef tuple convertTensorTo2D(np.ndarray tensor, bint infer_stress=True):
    """
    Convert a tensor array from nD to 2D. The first (n - 1)D are always collapsed to 1 when infer_stress is disabled.
    When infer_stress is enabled, if last 2D shape is (3, 3), the first (n - 2)D are collapsed to 1 and last 2D collapsed to 9.
    If 5D tensor, assume non-stress shape (n_x, n_y, n_z, n_extra, n_features), and stress shape (n_x, n_y, n_z, 3, 3).
    If 4D tensor, assume non-stress shape (n_x, n_y, n_z, n_features], and stress shape (n_x, n_y, 3, 3).
    If 3D tensor, assume non-stress shape (n_x, n_y, n_features), and stress shape (n_points, 3, 3).

    :param tensor: Tensor to convert to 2D
    :type tensor: np.ndarray[grid x n_features]
    :param infer_stress: Whether to infer if given tensor is stress. If True and if last 2D shape is (3, 3), then tensor is assumed stress.
    :type infer_stress: bool, optional (default=True)

    :return: 2D tensor of shape (n_points, n_features) and its original shape.
    :rtype: (np.ndarray[n_points x n_features], tuple)
    """
    cdef tuple shape_old
    cdef np.ndarray[np.float_t, ndim=2] tensor_2d

    shape_old = np.shape(tensor)

    if len(shape_old) == 5:
        if infer_stress and shape_old[3:5] == (3, 3):
            tensor_2d = tensor.reshape((shape_old[0]*shape_old[1]*shape_old[2], 9))
        else:
            tensor_2d = tensor.reshape((shape_old[0]*shape_old[1]*shape_old[2]*shape_old[3], shape_old[4]))

    elif len(shape_old) == 4:
        if infer_stress and shape_old[2:4] == (3, 3):
            tensor_2d = tensor.reshape((shape_old[0]*shape_old[1], 9))
        else:
            tensor_2d = tensor.reshape((shape_old[0]*shape_old[1]*shape_old[2], shape_old[3]))

    elif len(shape_old) == 3:
        if infer_stress and shape_old[1:3] == (3, 3):
            tensor_2d = tensor.reshape((shape_old[0], 9))
        else:
            tensor_2d = tensor.reshape((shape_old[0]*shape_old[1], shape_old[2]))

    else:
        tensor_2d = tensor

    print('\nTensor collapsed from shape ' + str(shape_old) + ' to ' + str(np.shape(tensor_2d)))
    return tensor_2d, shape_old


cpdef tuple processReynoldsStress(np.ndarray stress_tensor, bint make_anisotropic=True, int realization_iter=0, bint to_old_grid_shape=True):
    """
    Calculate anisotropy tensor bij, its eigenvalues and eigenvectors from given nD Reynolds stress of any shape.
    If make_anisotropic is disabled, then given Reynolds stress is assumed anisotropic.
    If realization_iter > 0, then bij is made realizable by realization_iter iterations.
    If to_old_grid_shape is enabled, then bij, eigenvalues, and eigenvectors are converted to old grid shape.

    :param stress_tensor: Reynolds stress u_i'u_j' or anisotropy stress tensor bij.
    :type stress_tensor: 2D or more np.ndarray of dimension (1/2/3D grid) x (3, 3)/6/9 components
    :param make_anisotropic: Whether to convert to bij from given Reynolds stress.
    :type make_anisotropic: bool, optional (default=True)
    :param realization_iter: How many iterations to make bij realizable. If 0, then no iteration is done.
    :type realization_iter: int, optional (default=0)
    :param to_old_grid_shape: Whether to convert bij, eigenvalues, eigenvectors to old grid shape.
    :type to_old_grid_shape: bool, optional (default=True)

    :return: Anisotropy tensor bij, eigenvalues, eigenvectors
    :rtype: (np.ndarray, np.ndarry, np.ndarray).
    Dimensions are either (n_points x 3 x 3, n_points x 3, n_points x 3 x 3)
    or (old grid x 3 x 3, old grid x 3, old grid x 3 x 3)
    """

    # If ndim is not provided but np.float_t is provided, 1D is assumed
    cdef np.ndarray[np.float_t] k, eigval_i
    cdef np.ndarray[np.float_t, ndim=2] eigvec_i
    cdef np.ndarray bij, eigval, eigvec
    cdef tuple shape_old
    cdef list shape_old_grid, shape_old_eigval, shape_old_matrix
    cdef int i, milestone
    cdef double progress

    print('\nProcessing Reynolds stress... ')
    # Ensure stress tensor is 2D
    stress_tensor, shape_old = convertTensorTo2D(stress_tensor)
    # Original shape without last D which is 9, or last 2D which is 3 x 3, representing the grid shape
    if shape_old[:(len(shape_old) - 2)] == (3, 3):
        shape_old_grid = list(shape_old[:(len(shape_old) - 2)])
    else:
        shape_old_grid = list(shape_old[:(len(shape_old) - 1)])

    # If stress_tensor is not anisotropic
    if make_anisotropic:
        # TKE
        if stress_tensor.shape[1] == 6:
            # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
            k = 0.5*(stress_tensor[:, 0] + stress_tensor[:, 3] + stress_tensor[:, 5])
        else:
            # xx, xy, xz | 0, 1, 2
            # yx, yy, yz | 3, 4, 5
            # zx, zy, zz | 6, 7, 8
            k = 0.5*(stress_tensor[:, 0] + stress_tensor[:, 4] + stress_tensor[:, 8])

        # Avoid FPE
        k[k < 1e-8] = 1e-8

        if stress_tensor.shape[1] == 6:
            # Convert Rij to bij
            for i in range(6):
                stress_tensor[:, i] = stress_tensor[:, i]/(2.*k) - 1/3. if i in (0, 3, 5) else stress_tensor[:, i]/(2.*k)

            # Add each anisotropy tensor to each mesh grid location, in depth
            # bij is 3D with z being b11, b12, b13, b21, b22, b23...
            bij = np.hstack((stress_tensor[:, 0], stress_tensor[:, 1], stress_tensor[:, 2],
                             stress_tensor[:, 1], stress_tensor[:, 3], stress_tensor[:, 4],
                             stress_tensor[:, 2], stress_tensor[:, 4], stress_tensor[:, 5]))
        else:
            for i in range(9):
                stress_tensor[:, i] = stress_tensor[:, i]/(2.*k) - 1/3. if i in (0, 4, 8) else stress_tensor[:, i]/(2.*k)

    # Else if stress tensor is already anisotropic
    else:
        if stress_tensor.shape[1] == 6:
            bij = np.hstack((stress_tensor[:, 0], stress_tensor[:, 1], stress_tensor[:, 2],
                             stress_tensor[:, 1], stress_tensor[:, 3], stress_tensor[:, 4],
                             stress_tensor[:, 2], stress_tensor[:, 4], stress_tensor[:, 5]))
        else:
            bij = stress_tensor

    for i in range(realization_iter):
        print('\nApplying realizability filter ' + str(i + 1))
        bij = _makeRealizable(bij)

    # Reshape the 3rd D to 3x3 instead of 9
    # Now bij is 3D, with shape (n_points, 3, 3)
    bij = bij.reshape((bij.shape[0], 3, 3))
    # Evaluate eigenvalues and eigenvectors of the symmetric tensor
    # eigval is n_points x 3
    # eigvec is n_points x 9, where 9 is the flattened eigenvector matrix from np.linalg.eigh()
    eigval, eigvec = np.empty((bij.shape[0], 3)), np.empty((bij.shape[0], 9))
    # For gauging progress
    milestone = 25
    # Go through each grid point
    # prange requires nogil that doesn't support python array slicing, and tuple, and numpy
    for i in range(bij.shape[0]):
        # eigval is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
        # Each col of eigvec is a vector, thus 3 x 3
        eigval_i, eigvec_i = np.linalg.eigh(bij[i, :, :])
        eigval_i, eigvec_i = np.flipud(eigval_i), np.fliplr(eigvec_i)
        eigval[i, :] = eigval_i
        # Each eigvec_i is a 3 x 3 matrix, flatten them add them to eigvec for point i
        eigvec[i, :] = eigvec_i.ravel()

        # Gauge progress
        progress = float(i)/(bij.shape[0] + 1.)*100.
        if progress >= milestone:
            printf(' %d %%... ', milestone)
            milestone += 25

    # Reshape eigval to old grid x 3, if requested
    # Also reshape eigvec from n_points x 9 to old grid x 3 x 3 if requested
    # so that each col of the 3 x 3 matrix is an eigenvector corresponding to an eigenvalue
    eigvec = eigvec.reshape((bij.shape[0], 3, 3))
    if to_old_grid_shape:
        shape_old_eigval = shape_old_grid.copy()
        # [old grid, 3]
        shape_old_eigval.append(3)
        shape_old_matrix = shape_old_eigval.copy()
        # [old grid, 3, 3]
        shape_old_matrix.append(3)
        eigval = eigval.reshape(tuple(shape_old_eigval))
        eigvec = eigvec.reshape(tuple(shape_old_matrix))
        bij = bij.reshape(tuple(shape_old_matrix))

    print('\nObtained bij with shape ' + str(np.shape(bij)) + ', ')
    print(' eigenvalues with shape ' + str(np.shape(eigval)) + ', ')
    print(' eigenvectors with shape ' + str(np.shape(eigvec)))
    return bij, eigval, eigvec


cpdef tuple getBarycentricMapData(np.ndarray eigval, bint optimize_cmap=True, double c_offset=0.65, double c_exp=5., bint to_old_grid_shape=False):
    """
    Get the Barycentric map coordinates and RGB values to visualize turbulent states based on given eigenvalues of the anisotropy tensor bij.
    Method from Banerjee (2007).
    If optimize_cmap is enabled, then the original Barycentric RGB is optimized to put more focus on turbulence inter-states.
    Additionally, if optimize_cmap is enabled, c_offset and c_exp are used to transform the old RGB to the optimized one.
    
    :param eigval: Eigenvalues of anisotropy tensor of a number of points. 
    The last dimension is assumed to store the three eigenvalues of a 3 x 3 bij. 
    :type eigval: np.ndarray[:, ..., 3] 
    :param optimize_cmap: Whether to optimize original Barycentric RGB map to a better one with more focus on turbulence inter-states.
    :type optimize_cmap: bool, optional (default=True)
    :param c_offset: If optimize_cmap is True, offset to shift the original RGB values. Recommended by Banerjee (2007) to be 0.65.
    :type c_offset: float, optional (default=0.65)
    :param c_exp: If optimize_cmap is True, exponent to power the shifted RGB values. Recommended by Banerjee (2007) to be 5.
    :type c_exp: float, optional (default=5.)
    :param to_old_grid_shape: Whether to convert RGB value array to old grid shape.
    :type to_old_grid_shape: bool, optional (default=False)
    
    :return: Barycentric triangle x, y coordinates and map original/optimized RGB values.
    :rtype: (np.ndarray[n_points, 2], np.ndarray[n_points, 3])
    """
    cdef np.ndarray[np.float_t] c1, c2, c3, x_bary, y_bary
    cdef np.ndarray[np.float_t, ndim=2] xy_bary, rgb_bary_orig, rgb_bary
    cdef int i
    cdef double x1c, x2c, x3c, y1c, y2c, y3c
    cdef tuple shape_old

    eigval, shape_old = convertTensorTo2D(eigval, infer_stress=False)
    # Coordinates of the anisotropy tensor in the tensor basis {a1c, a2c, a3c}. From Banerjee (2007),
    # C1c = lambda1 - lambda2,
    # C2c = 2(lambda2 - lambda3),
    # C3c = 3lambda3 + 1,
    # shape (n_points,)
    c1 = eigval[:, 0] - eigval[:, 1]
    # Not used for coordinates, only for color maps
    c2 = 2.*(eigval[:, 1] - eigval[:, 2])
    c3 = 3.*eigval[:, 2] + 1.
    # Corners of the barycentric triangle
    # Can be random coordinates?
    x1c, x2c, x3c = 1., 0., 1/2.
    y1c, y2c, y3c = 0., 0., sqrt(3.)/2.
    # x_bary, y_bary = c1*x1c + c2*x2c + c3*x3c, c1*y1c + c2*y2c + c3*y3c
    x_bary, y_bary = c1 + 0.5*c3, y3c*c3
    # Coordinates of the Barycentric triangle, 2 x n_points transposed to n_points x 2
    xy_bary = np.transpose(np.vstack((x_bary, y_bary)))
    # Original RGB values, 3 x n_points transposed to n_points x 3
    rgb_bary_orig = np.transpose(np.vstack((c1, c2, c3)))
    # For better barycentric map, use transformation on c1, c2, c3, as in Emory et al. (2014),
    # ci_star = (ci + c_offset)^c_exp
    if optimize_cmap:
        # Improved RGB = [c1_star, c2_star, c3_star]
        rgb_bary = np.empty_like(rgb_bary_orig)
        # Each 3rd dim is an RGB array of the 2D grid
        for i in range(3):
            rgb_bary[:, i] = (rgb_bary_orig[:, i] + c_offset)**c_exp

    else:
        rgb_bary = rgb_bary_orig

    # If reverse RGB from n_points x 3 to grid shape x 3
    if to_old_grid_shape:
        rgb_bary = reverseOldGridShape(rgb_bary, shape_old)

    print('\nBarycentric map coordinates and RGB values obtained for ' + str(xy_bary.shape[0]) + ' points')
    return xy_bary, rgb_bary




# -----------------------------------------------------
# Supporting Functions, Not Intended to Be Called From Python
# -----------------------------------------------------
cdef np.ndarray[np.float_t, ndim=2] _makeRealizable(np.ndarray[np.float_t, ndim=2] labels):
    """
    From Ling et al. (2016), see https://github.com/tbnn/tbnn.
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability
    by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
    Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called iteratively to get convergence
    to a realizable state.

    :param labels: The predicted anisotropy tensor.
    :type labels: np.ndarray[n_points, 9]

    :return: The predicted realizable anisotropy tensor.
    :type: np.ndarray[n_points, 9]
    """
    cdef int numPoints, i, j
    cdef np.ndarray[np.float_t, ndim=2] A, evectors
    cdef np.ndarray[np.float_t] evalues

    numPoints = labels.shape[0]
    A = np.zeros((3, 3))
    for i in range(numPoints):
        # Scales all on-diags to retain zero trace
        if np.min(labels[i, [0, 4, 8]]) < -1./3.:
            labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
        if 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
            labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
            labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
        if 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
            labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
            labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
        if 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
            labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
            labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])

        # Enforce positive semidefinite by pushing evalues to non-negative
        A[0, 0] = labels[i, 0]
        A[1, 1] = labels[i, 4]
        A[2, 2] = labels[i, 8]
        A[0, 1] = labels[i, 1]
        A[1, 0] = labels[i, 1]
        A[1, 2] = labels[i, 5]
        A[2, 1] = labels[i, 5]
        A[0, 2] = labels[i, 2]
        A[2, 0] = labels[i, 2]
        evalues, evectors = np.linalg.eig(A)
        if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])/2.:
            evalues = evalues*(3.*np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])/(2.*np.max(evalues))
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                labels[i, j] = A[j, j]
            labels[i, 1] = A[0, 1]
            labels[i, 5] = A[1, 2]
            labels[i, 2] = A[0, 2]
            labels[i, 3] = A[0, 1]
            labels[i, 7] = A[1, 2]
            labels[i, 6] = A[0, 2]
        if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
            evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                labels[i, j] = A[j, j]
            labels[i, 1] = A[0, 1]
            labels[i, 5] = A[1, 2]
            labels[i, 2] = A[0, 2]
            labels[i, 3] = A[0, 1]
            labels[i, 7] = A[1, 2]
            labels[i, 6] = A[0, 2]

    return labels
