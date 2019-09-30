# cython: language_level = 3str
# cython: embedsignature = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
import numpy as np
cimport numpy as np
from libc.math cimport fmax, ceil, sqrt, cbrt, sin, cos
from scipy import ndimage
from Preprocess.Tensor import contractSymmetricTensor
import functools, time
import warnings

cpdef tuple interpolateGridData(np.ndarray[np.float_t] x, np.ndarray[np.float_t] y, np.ndarray val, np.ndarray z=None,
                                tuple xlim=(None, None), tuple ylim=(None, None), tuple zlim=(None, None),
                                double mesh_target=1e4, str interp="linear", double fill_val=np.nan):
    """
    Interpolate given coordinates and field properties to satisfy given summed mesh size, given x, y (and z) limits, with given interpolation method.
    If z is not given, then the interpolation is 2D. 
    If any of xlim, ylim, zlim is given, the corresponding axis is limited.
    The number of cells nx, ny, (nz) in each dimension is automatically determined to scale with physical dimension length lx, ly, (lz),
    so that nx = lx*nbase; ny = ly*nbase; (nz = lz*nbase), 
    and nx*ny = mesh_target or nx*ny*nz = mesh_target. 
    
    :param x: X coordinates.
    :type x: ndarray[n_points]
    :param y: Y coordinates.
    :type y: ndarray[n_points]
    :param val: n_features number of field properties to interpolate with mesh.
    :type val: ndarray[n_points, n_features] if n_features > 1 or ndarray[n_points]
    :param z: Z coordinates in case of 3D.
    :type z: ndarray[n_points] or None, optional (default=None)
    :param xlim: X limit of the target mesh, in a tuple of minimum x and maximum x.
    If minimum x is None, min() of the known x is used. The same for maximum x.
    :type xlim: tuple, optional (default=(None, None))
    :param ylim: Y limit of the target mesh, in a tuple of minimum y and maximum y.
    If minimum y is None, min() of the known x is used. The same for maximum y.
    :type ylim: tuple, optional (default=(None, None))
    :param zlim: Z limit of the target mesh, in a tuple of minimum z and maximum z.
    If z is not provided, zlim has no effect.
    If z is provided and minimum z is None, min() of the known z is used. The same for maximum z.
    :type zlim: tuple, optional (default=(None, None))
    :param mesh_target: Summed size of the target mesh. 
    The function takes this value and assign cells to each dimension based on physical dimension length.
    :type mesh_target: float, optional (default=1e4)
    :param interp: Interpolation method used by scipy.interpolate.griddata.
    :type interp: "nearest" or "linear" or "cubic", optional (default="linear")
    :param fill_val: Value to replace NaN.
    :type fill_val: float, optional (default=nan)

    :return: X, Y, Z mesh grid and field properties mesh grid.
    If Z was not given, then Z is returned as dummy array.
    Field properties values are stacked as the last dimension, either 3rd (2D grid) or 4th (3D grid) dimension.
    :rtype: (ndarray[nx, ny], ndarray[nx, ny], empty(1), ndarray[nx, ny, n_features])
    or (ndarray[nx, ny, nz], ndarray[nx, ny, nz], ndarray[nx, ny, nz], ndarray[nx, ny, nz, n_features])
    """
    from scipy.interpolate import griddata

    cdef tuple shape_val = np.shape(val)
    cdef double lx, ly, lz, nbase
    cdef int nx, ny, nz, i, n_features
    cdef np.ndarray[np.float_t, ndim=2] coor_known
    cdef np.ndarray xmesh, ymesh, zmesh, val_mesh
    cdef tuple coor_request
    cdef double complex precision_x, precision_y, precision_z

    print('\nInterpolating data to target mesh size ' + str(mesh_target) + ' with ' + str(interp) + ' method...')
    # Ensure val is at least 2D with shape (n_points, 1) if it was 1D
    if len(shape_val) == 1:
        val = np.transpose(np.atleast_2d(val))

    n_features = val.shape[1]
    # Limit new mesh if requested
    xmin = x.min() if xlim[0] is None else xlim[0]
    xmax = x.max() if xlim[1] is None else xlim[1]
    ymin = y.min() if ylim[0] is None else ylim[0]
    ymax = y.max() if ylim[1] is None else ylim[1]
    if z is not None:
        zmin = z.min() if zlim[0] is None else zlim[0]
        zmax = z.max() if zlim[1] is None else zlim[1]
    else:
        zmin = zmax = None

    # Get x, y, z's length, and prevent 0 since they will be divided later
    lx, ly = fmax(xmax - xmin, 0.0001), fmax(ymax - ymin, 0.0001)
    lz = fmax(zmax - zmin, 0.0001) if z is not None else 0.

    # Since we want number of cells in x, y, z to scale with lx, ly, lz, create a base number of cells nbase and
    # let (lx*nbase)*(ly*nbase)*(lz*nbase) = mesh_target,
    # then nbase = [mesh_target/(lx*ly*lz)]^(1/3)
    nbase = cbrt(mesh_target/(lx*ly*lz)) if z is not None else sqrt(mesh_target/(lx*ly))

    nx, ny = <int>ceil(lx*nbase), <int>ceil(ly*nbase)
    nz = <int>ceil(lz*nbase) if z is not None else 1
    print("\nTarget resolution is {} x {} (x {})".format(nx, ny, nz))
    precision_x = nx*1j
    precision_y = ny*1j
    if z is not None: precision_z = nz*1j

    if z is not None:
        # Known coordinates with shape (3, n_points) trasposed to (n_points, 3)
        coor_known = np.transpose(np.vstack((x, y, z)))
        xmesh, ymesh, zmesh = np.mgrid[xmin:xmax:precision_x,
                              ymin:ymax:precision_y,
                              zmin:zmax:precision_z]
        coor_request = (xmesh, ymesh, zmesh)
        val_mesh = np.empty((xmesh.shape[0], xmesh.shape[1], xmesh.shape[2], n_features))
    else:
        coor_known = np.transpose(np.vstack((x, y)))
        xmesh, ymesh = np.mgrid[xmin:xmax:precision_x,
                       ymin:ymax:precision_y]
        # Dummy array for zmesh in 2D
        zmesh = np.empty(1)
        coor_request = (xmesh, ymesh)
        val_mesh = np.empty((xmesh.shape[0], xmesh.shape[1], n_features))

    # Interpolate for each value column
    for i in range(n_features):
        print('\n Interpolating value ' + str(i + 1) + '...')
        if z is not None:
            val_mesh[:, :, :, i] = griddata(coor_known, val[:, i], coor_request, method=interp, fill_value=fill_val)
        else:
            val_mesh[:, :, i] = griddata(coor_known, val[:, i], coor_request, method=interp, fill_value=fill_val)

    # In case provided value only has 1 feature, compress from shape (grid mesh, 1) to (grid mesh)
    if n_features == 1:
        if z is None:
            val_mesh = val_mesh.reshape((val_mesh.shape[0], val_mesh.shape[1]))
        else:
            val_mesh = val_mesh.reshape((val_mesh.shape[0], val_mesh.shape[1], val_mesh.shape[2]))

    print('\nValues interpolated to mesh ' + str(np.shape(xmesh)))
    return xmesh, ymesh, zmesh, val_mesh


cpdef tuple collapseMeshGridFeatures(np.ndarray meshgrid, bint infer_matrix_form=True, tuple matrix_shape=(3, 3), bint collapse_matrix=True):
    """
    Collapse a given value meshgrid of [grid shape x feature shape] to array of [n_points x feature shape] or [n_points x n_features].
    At least the last D has to be feature(s).
    When infer_matrix_form is disabled, 
        If meshgrid >= 3D, the 1st (n - 1)D are always collapsed to 1, with last D (features) unchanged. 
        Else, meshgrid is assumed (n_points, n_features) and the original meshgrid is returned.
    When infer_matrix_form is enabled, 
        if last len(matrix_shape)D shape is matrix_shape, the first (n - len(matrix_shape)D are collapsed to 1D, 
            and if collapse_matrix is enabled, collapse the matrix shape to 1D too;
        else if last len(matrix_shape)D shape is not matrix_shape, assume last D is features and collapse the 1st (n - 1)D to 1D.
        
    Typical examples:
        Meshgrid is (nx, ny, nz, 3, 3). 
            If infer_matrix_form and matrix_shape is (3, 3):
                Array is (nx*ny*nz, 3*3) if collapse_matrix is True else (nx*ny*nz, 3, 3).
            Else, array is (nx*ny*nz*3, 3).
        Meshgrid is (n_points, 10, 3, 3).
            If infer_matrix_form and matrix_shape is (10, 3, 3):
                Array is (n_points, 10*3*3) if collapse_matrix else (n_points, 10, 3, 3).
            Else, array is (n_points*10*3, 3).
        Meshgrid is (nx, ny, n_features).
            nz is assumed feature D instead of spatial D and array is (nx*ny, n_features).
            In case of (nx, ny, nz) 3D meshgrid, please use meshgrid.ravel(). 
        Meshgrid is (n_points, n_features).
            Nothing is done.

    :param meshgrid: Meshgrid of multiple features to collapse. At least last D should be feature(s).
    :type meshgrid: ndarray[grid_shape x feature shape]
    :param infer_matrix_form: Whether to infer if given meshgrid has features in the form matrix of shape matrix_shape. 
    If True and if last len(matrix_shape)D shape is matrix_shape, then grid shape of meshgrid excludes matrix_shape.
    :type infer_matrix_form: bool, optional (default=True)
    :param matrix_shape: If infer_matrix_form is True, shape of matrix to infer.
    :type matrix_shape: tuple, optional (default=(3, 3))
    :param collapse_matrix: If infer_matrix_form is True, whether to collapse matrix_shape to 1D, 
    once such matrix_shape has found in given meshgrid.
    :type collapse_matrix: bool, optional (default=True)

    :return: Meshgrid of multiple features collapsed to either (n_points, n_features) or (n_points, feature shape); and its original shape.
    :rtype: (ndarray[n_points x n_features] or ndarray[n_points x feature shape], tuple)
    """
    cdef tuple shape_old, shape_grid
    cdef np.ndarray arr
    cdef int matrix_nelem = 1
    cdef int grid_nelem = 1
    cdef int i
    cdef list shape_new

    shape_old = np.shape(meshgrid)
    # Go throuhg each matrix D to calculate number of matrix elements
    for i in range(len(matrix_shape)):
        matrix_nelem *= matrix_shape[i]

    # If infer_matrix_form
    # and at least one more D than len(matrix_shape)
    # and the last len(matrix_shape) D is given matrix_shape
    if infer_matrix_form and \
            len(shape_old) >= len(matrix_shape) + 1 and \
            shape_old[(len(shape_old) - len(matrix_shape)):len(shape_old)] == matrix_shape:
        shape_grid = shape_old[:(len(shape_old) - len(matrix_shape))]
        # Go through each grid D and find out number of grid elements
        for i in range(len(shape_grid)):
            grid_nelem *= shape_grid[i]

        # If collapse_matrix is True, then new array is collapsed both in grid D and matrix form D
        if collapse_matrix:
            arr = meshgrid.reshape((grid_nelem, matrix_nelem))
        # Else only collapsed in grid D and keep matrix form
        else:
            shape_new = [grid_nelem] + list(matrix_shape)
            arr = meshgrid.reshape(tuple(shape_new))

    # Else if either infer_matrix_form is False
    # and/or old array D <= len(matrix_shape) but >= 3
    # and/or old array's last len(matrix_shape) D is not given matrix_shape.
    # Note this inherently assumes the 3rd D is feature instead of z if given meshgrid is 3D
    elif len(shape_old) >= 3:
        shape_grid = shape_old[:(len(shape_old) - 1)]
        # Go through each grid dimension
        for i in range(len(shape_grid)):
            grid_nelem *= shape_grid[i]

        # New array is collapsed in grid and last D of original shape is kept
        arr = meshgrid.reshape((grid_nelem, shape_old[len(shape_old) - 1]))
    # Else if meshgrid is 2D, assume (n_points, n_features) and don't change anything
    else:
        arr = meshgrid

    print('\nMesh grid of multiple features collapsed from shape ' + str(shape_old) + ' to ' + str(np.shape(arr)))
    return arr, shape_old


cpdef np.ndarray reverseOldGridShape(np.ndarray arr, tuple shape_old, bint infer_matrix_form=True, tuple matrix_shape=(3, 3)):
    """
    Reverse the given array to the old mesh grid (but not value) shape, given the old array (both grid and value) shape.
    If infer_matrix_form is enabled, then the last 2D of given array will be checked whether it matches given matrix_shape. 
    If so, then grid shape is inferred to exclude the last 2D. Otherwise, grid shape is inferred to exclude the last D.
    If infer_matrix_form is disabled, then the given array is simply reversed/reshaped to the given shape (both grid and value shape).
    
    :param arr: Array to reverse shape. The last D or last 2D are considered non-grid related values and will not be touched.
    :type arr: ndarray[:, ..., :, 3, 3] or ndarray[:, ..., :]
    :param shape_old: Old full shape (first grid dimensions then value dimensions) used to reverse to old grid.
    If infer_matrix_form is False, then old shape's value dimensions are also used for array reshape.
    :type shape_old: tuple
    :param infer_matrix_form: Whether to detect for matrix form of matrix_shape in the last 2D of arr and shape_old,
    to infer value shape of arr and grid shape of shape_old.
    :type infer_matrix_form: bool optional (default=True)
    :param matrix_shape: If in_matrix_form is True, the matrix shape to detect.
    :type matrix_shape: tuple[:, :], optional (default=(3, 3))
     
    :return: Array reversed/reshaped to detected old grid or shape_old if infer_matrix_form is disabled.
    :rtype: ndarray[old grid x matrix_shape] or ndarray[old grid x values] or ndarray[shape_old]
    """
    cdef tuple shape_arr
    cdef list shape_old_grid

    shape_arr = np.shape(arr)
    # If inferring matrix form, then check if last 2D is a matrix_shape
    if infer_matrix_form:
        # Deduce the old grid shape
        # If the last 2D is matrix_shape then shape_old is a matrix_shape tensor
        if shape_old[len(shape_old) - 2:] == matrix_shape:
            shape_old_grid = list(shape_old[:len(shape_old) - 2])
        # Otherwise, only the last D is not grid related
        else:
            shape_old_grid = list(shape_old[:len(shape_old) - 1])

        # Likewise, if the last 2D is matrix_shape, then given arr is a matrix_shape tensor
        if shape_arr[len(shape_arr) - 2:] == matrix_shape:
            shape_old_grid += list(matrix_shape)
        # Otherwise, the last D is untouched
        else:
            shape_old_grid += [shape_arr[len(shape_arr) - 1]]

        arr = arr.reshape(tuple(shape_old_grid))
    # If not inferring tensor, then arr is simply reversed to shape_old
    else:
        arr = arr.reshape(shape_old)

    print('\nArray reversed from shape ' + str(shape_arr) + ' to ' + str(np.shape(arr)))
    return arr


cpdef np.ndarray rotateData(np.ndarray ndarr, double anglex=0., double angley=0., double anglez=0., tuple matrix_shape=(3, 3)):
    """
    Rotate nD array of matrices or vectors. Matrices must not be flattened.
    If infer_matrix_form, the last few D will be checked it matches provided matrix_shape. 
    The first remaining D are collapsed to 1 so that ndarr has shape (n_samples, vector size) or (n_samples, matrix_shape).
    If matrix_shape is 3D, then only the last 2D are considered for rotation. 
    E.g. Tensor basis Tij of shape (n_samples, n_bases, 3, 3) should have matrix_shape = (n_bases, 3, 3) to rotate properly.
    
    :param ndarr: nD array to rotate.
    :type ndarr: ndarray[mesh grid / n_samples x vector size / matrix_shape]
    :param anglex: Angle to rotate around x axis in degree or radian.
    :type anglex: double, optional (default=0.)
    :param angley: Angle to rotate around y axis in degree or radian.
    :type angley: double, optional (default=0.)
    :param anglez: Angle to rotate around z axis in degree or radian.
    :type anglez: double, optional (default=0.)
    :param matrix_shape: The shape to infer as matrix for rotation. 
    If matrix_shape is 3D, only the last 2D are rotated while the 1st D of matrix_shape is looped.
    E.g. to rotate tensor basis Tij of shape (n_samples / mesh grid, n_bases, 3, 3), matrix_shape should be (n_bases, 3, 3). 
    :type matrix_shape: 2/3D tuple, optional (default=(3, 3)) 
    
    :return: Rotated 2/3/4D array of vectors or matrices
    :rtype: ndarray[n_samples x vector size / matrix_shape]
    """
    cdef np.ndarray[np.float_t, ndim=2] rotij_x, rotij_y, rotij_z, qij
    cdef unsigned int i, j

    # Automatically detect wheter angles are in radian or degree
    if anglex > 2.*np.pi: anglex /= 180./np.pi
    if angley > 2.*np.pi: angley /= 180./np.pi
    if anglez > 2.*np.pi: anglez /= 180./np.pi
    # Collapse mesh grid and infer whether it's a matrix or vector
    # It'll have shape (n_samples, vector size) or (n_samples, matrix_shape)
    ndarr, shape_old = collapseMeshGridFeatures(ndarr, True, matrix_shape, collapse_matrix=False)
    # Rotation matrices in x, y, z
    rotij_x, rotij_y, rotij_z = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    # Qx = |1 0    0  |
    #      |0 cos -sin|
    #      |0 sin  cos|
    rotij_x[0, 0] = 1.
    rotij_x[1, 1] = rotij_x[2, 2] = cos(anglex)
    rotij_x[1, 2] = -sin(anglex)
    rotij_x[2, 1] = -rotij_x[1, 2]
    # Qy = | cos 0 sin|
    #      | 0   1 0  |
    #      |-sin 0 cos|
    rotij_y[0, 0] = rotij_y[2, 2] = cos(angley)
    rotij_y[0, 2] = sin(angley)
    rotij_y[1, 1] = 1.
    rotij_y[2, 0] = -rotij_y[0, 2]
    # Qz = |cos -sin 0|
    #      |sin  cos 0|
    #      |0    0   1|
    rotij_z[0, 0] = rotij_z[1, 1] = cos(anglez)
    rotij_z[0, 1] = -sin(anglez)
    rotij_z[1, 0] = -rotij_z[0, 1]
    rotij_z[2, 2] = 1.
    # Combined Qij
    qij = rotij_z @ (rotij_y @ rotij_x)
    # Go through each sample
    for i in range(ndarr.shape[0]):
        # For matrices, do Qij*matrix*Qij^T
        if len(np.shape(ndarr)) > 2:
            # If matrix_shape was 3D, e.g. (n_bases, 3, 3) for Tij, then loop through matrix_shape[0] (aka ndarr[i, j]) too
            if len(np.shape(ndarr)) == 4:
                for j in range(ndarr.shape[1]):
                    ndarr[i, j] = qij @ (ndarr[i, j] @ qij.T)

            else:
                ndarr[i] = qij @ (ndarr[i] @ qij.T)

        # Else for vectors, do Qij*vector
        else:
            ndarr[i] = np.dot(qij, ndarr[i])

    return ndarr


cpdef tuple fieldSpatialSmoothing(np.ndarray[np.float_t, ndim=2] val,
                                        np.ndarray[np.float_t] x, np.ndarray[np.float_t] y, np.ndarray z=None,
                                        tuple val_bnd=(-np.inf, np.inf), bint is_bij=False, double bij_bnd_multiplier=2.,
                                        tuple xlim=(None, None), tuple ylim=(None, None), tuple zlim=(None, None),
                                        double mesh_target=1e4):
    """
    Spatially smooth a field of shape (n_points, n_outputs). Therefore, if the field is a 2/3D mesh grid, it has to be flattened beforehand.
    The workflow is:
        1. Remove any component outside bound and set to NaN
        2. Interpolate to 2/3D slice/volumetric mesh grid with nearest method
        3. Use 2/3D Gaussian filter to smooth the mesh grid while ignoring NaN, for every component.
    The output will be a spatially smoothed field mesh grid of mesh_target number of points.
    The targeted mesh grid has to be at least 2D with x and y coordinates provided.
    If is_bij, the diagonal and off-diagonal components of anisotropy tensor bij will be treated separately such that 
    whatever is out of [-1/3, 2/3]*bij_bnd_multiplier diagonally is set to NaN;
    whatever is out of [-1/2, 1/2]*bij_bnd_multiplier off-diagonally is set to NaN.
    Otherwise, whatever is out of val_bnd is set to NaN.
    If one or more of xlim, ylim, zlim is provided, the target mesh is limited to xlim, and/or ylim, and/or zlim range.
    
    :param val: Field value to be made mesh grid and smoothed. If is anisotropy tensor bij, is_bij should be True.  
    :type val: ndarray[n_points x n_outputs]
    :param x: X / 1st axis coordinate of val. 
    :type x: ndarray[n_points]
    :param y: Y / 2nd axis coordinate of val.
    :type y: ndarray[n_points]
    :param z: Z / 3rd axis coordinate of val.
    :type z: ndarray[n_points] or None, optional (default=None)
    :param val_bnd: Bound of val, whatever outside val_bnd is treated as NaN.
    If is_bij, this has no effect.
    :type val_bnd: tuple, optional (default=(-np.inf, np.inf))
    :param is_bij: Whether the field value val is anisotropic tensor bij.
    If True, val_bnd has no effect and 
    diagonal bij outside bound [-1/3, 2/3]*bij_bnd_multiplier is set to NaN;
    off-diagonal bij outside bound [-1/2, 1/2]*bij_bnd_multiplier is set to NaN.
    :type is_bij: bool, optional (default=False)
    :param bij_bnd_multiplier: When is_bij, multiplier to define bij's bound such that whatever outside bound is set to NaN.
    If is_bij is False, this has no effect.
    :type bij_bnd_multiplier: float, optional (default=2.)
    :param xlim: Limit of X / 1st axis coordinate in the target mesh.
    :type xlim: tuple, optional (default=(None, None))
    :param ylim: Limit of Y / 2nd axis coordinate in the target mesh.
    :type ylim: tuple, optional (default=(None, None))
    :param zlim: Limit of Z / 3rd axis coordinate in the target mesh.
    :type zlim: tuple, optional (default=(None, None))
    :param mesh_target: Number of points for the target mesh. 
    Then, number of points in each axis is determined automatically depending on length in each axis. 
    :type mesh_target: float, optional (default=1e4)
    
    :return: Mesh grid coordinates of x, y, z, and spatially smoothed field mesh grid.
    :rtype: (ndarray[3D mesh grid], ndarray[3D mesh grid], ndarray[3D mesh grid], ndarray[3D mesh grid x n_outputs])
    or (ndarray[2D mesh grid], ndarray[2D mesh grid], ndarray[2D mesh grid], empty(1),  ndarray[2D mesh grid x n_outputs])
    """
    cdef unsigned int i
    cdef tuple valshape = np.shape(val)
    cdef unsigned int n_outputs = valshape[1]

    # Step 1
    if is_bij:
        val = contractSymmetricTensor(val)
        for i in range(n_outputs):
            if i in (0, 3, 5):
                val[:, i][val[:, i] > 2/3.*bij_bnd_multiplier] = np.nan
                val[:, i][val[:, i] < -1/3.*bij_bnd_multiplier] = np.nan
            else:
                val[:, i][val[:, i] > 1/2.*bij_bnd_multiplier] = np.nan
                val[:, i][val[:, i] < -1/2.*bij_bnd_multiplier] = np.nan

    else:
        for i in range(n_outputs):
            val[:, i][val[:, i] > val_bnd[1]] = np.nan
            val[:, i][val[:, i] < val_bnd[0]] = np.nan

    # Step 2
    xmesh, ymesh, zmesh, val_mesh = interpolateGridData(x, y, val, z=z, xlim=xlim, ylim=ylim, zlim=zlim,
                                       mesh_target=mesh_target, interp="nearest", fill_val=np.nan)
    # Step 3
    for i in range(n_outputs):
        val_mesh[..., i] = gaussianFilter(val_mesh[..., i])

    return xmesh, ymesh, zmesh, val_mesh


cpdef np.ndarray gaussianFilter(np.ndarray array, double sigma=2.):
    """
    Perform Gaussian filter to smooth an nD field. NaNs are ignored automatically. 
    
        array U         1   2   NaN 1   2    
    auxiliary V     1   2   0   1   2    
    auxiliary W     1   1   0   1   1
    position        a   b   c   d   e
    
    filtered VV_b   = 0.25*V_a  + 0.50*V_b  + 0.25*V_c
                    = 0.25*1    + 0.50*2    + 0
                    = 1.25
    
    filtered WW_b   = 0.25*W_a  + 0.50*W_b  + 0.25*W_c
                    = 0.25*1    + 0.50*1    + 0
                    = 0.75
    
    ratio Z         = VV_b / WW_b  
                    = (0.25*1 + 0.50*2) / (0.25*1    + 0.50*1)
                    = 0.333*1 + 0.666*2
                    = 1.666
    
    :param array: Array or mesh grid to perform Gaussian filtering.
    :type array: ndarray of any shape
    :param sigma: Standard deviation for Gaussian kernel. 
    :type sigma: double, optional (default=2.)
    
    :return: Filtered array or mesh grid of the same shape as input
    :rtype: ndarray
    """
    cdef np.ndarray v, vv, w, ww, arr_filtered
    
    v = array.copy()
    v[np.isnan(array)] = 0.
    vv = ndimage.gaussian_filter(v, sigma=sigma)
    w = np.ones_like(array)
    w[np.isnan(array)] = 0.
    ww = ndimage.gaussian_filter(w, sigma=sigma)
    arr_filtered = vv/ww

    return arr_filtered
    

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


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc










