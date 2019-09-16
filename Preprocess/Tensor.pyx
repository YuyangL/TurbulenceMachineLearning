# cython: language_level = 3str
# cython: embedsignature = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport sqrt
from Utility import collapseMeshGridFeatures, reverseOldGridShape
cimport cython

cpdef tuple processReynoldsStress(np.ndarray stress_tensor, bint make_anisotropic=True, int realization_iter=0, bint to_old_grid_shape=True):
    """
    Calculate anisotropy tensor bij, its eigenvalues and eigenvectors from given nD Reynolds stress of any shape.
    If make_anisotropic is disabled, then given Reynolds stress is assumed anisotropic.
    If realization_iter > 0, then bij is made realizable by realization_iter iterations.
    If to_old_grid_shape is enabled, then bij, eigenvalues, and eigenvectors are converted to old grid shape.

    :param stress_tensor: Reynolds stress u_i'u_j' or anisotropy stress tensor bij.
    :type stress_tensor: ND array of dimension (n_points or 2/3D grid) x (3 x 3 or 6 or 9 components)
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
    # Ensure stress tensor is 2D, (n_points, 9 or 6)
    # [DEPRECATED]
    # stress_tensor, shape_old = convertTensorTo2D(stress_tensor)
    stress_tensor, shape_old = collapseMeshGridFeatures(stress_tensor, matrix_shape=(3,3), collapse_matrix=True)
    # Original shape without last D which is 9, or last 2D which is 3 x 3, representing the grid shape
    if shape_old[(len(shape_old) - 2):] == (3, 3):
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
        k[k < 1e-12] = 1e-12

        if stress_tensor.shape[1] == 6:
            # Convert Rij to bij
            for i in range(6):
                stress_tensor[:, i] = stress_tensor[:, i]/(2.*k) - 1/3. if i in (0, 3, 5) else stress_tensor[:, i]/(2.*k)

            # Add each anisotropy tensor to each mesh grid location, in depth
            # bij is 3D with z being b11, b12, b13, b21, b22, b23...
            bij = stress_tensor[:, (0, 1, 2, 1, 3, 4, 2, 4, 5)]
            # bij = np.hstack((stress_tensor[:, 0], stress_tensor[:, 1], stress_tensor[:, 2],
            #                  stress_tensor[:, 1], stress_tensor[:, 3], stress_tensor[:, 4],
            #                  stress_tensor[:, 2], stress_tensor[:, 4], stress_tensor[:, 5]))
        else:
            bij = np.empty((stress_tensor.shape[0], 9))
            for i in range(9):
                bij[:, i] = stress_tensor[:, i]/(2.*k) - 1/3. if i in (0, 4, 8) else stress_tensor[:, i]/(2.*k)

    # Else if stress tensor is already anisotropic
    else:
        if stress_tensor.shape[1] == 6:
            bij = stress_tensor[:, (0, 1, 2, 1, 3, 4, 2, 4, 5)]
            # bij = np.hstack((stress_tensor[:, 0], stress_tensor[:, 1], stress_tensor[:, 2],
            #                  stress_tensor[:, 1], stress_tensor[:, 3], stress_tensor[:, 4],
            #                  stress_tensor[:, 2], stress_tensor[:, 4], stress_tensor[:, 5]))
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
        # eigval = eigval.reshape(tuple(shape_old_eigval))
        # eigvec = eigvec.reshape(tuple(shape_old_matrix))
        # bij = bij.reshape(tuple(shape_old_matrix))
        eigval = reverseOldGridShape(eigval, shape_old_eigval, infer_matrix_form=False)
        eigvec = reverseOldGridShape(eigvec, shape_old_matrix)
        bij = reverseOldGridShape(bij, shape_old_matrix)

    print('\nObtained bij with shape ' + str(np.shape(bij)) + ', ')
    print(' eigenvalues with shape ' + str(np.shape(eigval)) + ', ')
    print(' eigenvectors with shape ' + str(np.shape(eigvec)))
    return bij, eigval, eigvec


cpdef tuple getBarycentricMapData(np.ndarray eigval, bint optimize_cmap=True, double c_offset=0.65, double c_exp=5., bint to_old_grid_shape=True):
    """
    Get the Barycentric map coordinates and RGB values to visualize turbulent states based on given eigenvalues of the anisotropy tensor bij.
    Method from Banerjee (2007) and Emory et al. (2014).
    If optimize_cmap is enabled, then the original Barycentric RGB is optimized to put more focus on turbulence inter-states.
    Additionally, if optimize_cmap is enabled, c_offset and c_exp are used to transform the old RGB to the optimized one.
    
    :param eigval: Eigenvalues of anisotropy tensor of a number of points. 
    The last dimension is assumed to store the three eigenvalues of a 3 x 3 bij. 
    :type eigval: np.ndarray[mesh grid / n_points, 3] 
    :param optimize_cmap: Whether to optimize original Barycentric RGB map to a better one with more focus on turbulence inter-states.
    :type optimize_cmap: bool, optional (default=True)
    :param c_offset: If optimize_cmap is True, offset to shift the original RGB values. Recommended by Banerjee (2007) to be 0.65.
    :type c_offset: float, optional (default=0.65)
    :param c_exp: If optimize_cmap is True, exponent to power the shifted RGB values. Recommended by Banerjee (2007) to be 5.
    :type c_exp: float, optional (default=5.)
    :param to_old_grid_shape: Whether to convert RGB value array as well as barycentric map x, y coordinates array to old grid shape.
    :type to_old_grid_shape: bool, optional (default=False)
    
    :return: Barycentric triangle x, y coordinates and map original/optimized RGB values.
    :rtype: (ndarray[n_points x 2], ndarray[n_points x 3])
    or (ndarray[mesh grid x 2], ndarray[mesh grid x 3])
    """
    cdef np.ndarray[np.float_t] c1, c2, c3, x_bary, y_bary
    cdef np.ndarray[np.float_t, ndim=2] rgb_bary_orig
    cdef np.ndarray xy_bary, rgb_bary
    cdef unsigned int i, n_points
    cdef double x1c, x2c, x3c, y1c, y2c, y3c
    cdef tuple shape_old

    eigval, shape_old = collapseMeshGridFeatures(eigval, infer_matrix_form=False)
    n_points = eigval.shape[0]
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
    # and barycentric map x, y coordinates from n_points x 2 to grid shape x 2
    if to_old_grid_shape:
        rgb_bary = reverseOldGridShape(rgb_bary, shape_old)
        xy_bary = reverseOldGridShape(xy_bary, shape_old)

    print('\nBarycentric map coordinates and RGB values obtained for ' + str(n_points) + ' points')
    return xy_bary, rgb_bary


cpdef np.ndarray expandSymmetricTensor(np.ndarray tensor):
    """
    Expand an nD compact symmetric tensor of 6 components to 9 components.
    The given nD compact symmetric tensor can have any shape but the last D must be 6 components.
    The returned full nD symmetric tensor will have the same shape of given symmetric tensor except last D being 9 components.
    
    :param tensor: nD symmetric tensor of 6 components. n can be any number.
    :type tensor: ndarray[..., 6] 
    :return: nD symmetric tensor of 9 components with same nD as tensor.
    If given tensor doesn't have 6 components in the last D, the original tensor is returned. 
    :rtype: ndarray[..., 9] or ndarray
    """
    cdef tuple shape_old = np.shape(tensor)
    cdef np.ndarray tensor_full
    cdef unsigned int last_ax = len(shape_old) - 1
    cdef tuple idx_1to9

    # If the symmetric tensor nD array's last D isn't 6 components, return old tensor
    if tensor.shape[last_ax] != 6:

        return tensor

    # Indices to concatenate
    idx_1to9 = (0, 1, 2, 1, 3, 4, 2, 4, 5)
    tensor_full = tensor[..., idx_1to9]

    return tensor_full


cpdef np.ndarray contractSymmetricTensor(np.ndarray tensor):
    """
    Contract an nD full symmetric tensor of 9 components to 6 components.
    The given nD full symmetric tensor can have any shape but the last D must be 9 components.
    The returned compact nD symmetric tensor will have the same shape of given symmetric tensor except last D being 6 components.
    
    :param tensor: nD full symmetric tensor of 9 or (3, 3) components. n can be any number.
    :type tensor: ndarray[..., 9] or ndarray[..., 3, 3]
    
    :return: nD compact symmetric tensor of 6 components with same nD as tensor.
    If given tensor doesn't have 9 components in the last D, the original tensor is returned. 
    :rtype: ndarray[..., 6] or ndarray
    """
    cdef tuple shape_old = np.shape(tensor)
    cdef np.ndarray tensor_compact
    cdef unsigned int last_ax = len(shape_old) - 1
    cdef tuple idx_1to6
    cdef list shape_new

    # If the symmetric tensor 2D array's last D isn't 9 components, return old tensor
    if len(shape_old) == 2:
        if tensor.shape[last_ax] != 9: return tensor
    # Else if tensor is 3D or more neither last D isn't 9 nor last 2D aren't 3 x 3, return old tensor
    elif len(shape_old) > 2:
        if tensor.shape[last_ax] != 9 and shape_old[len(shape_old) - 2:] != (3, 3): return tensor
    # Else if 1D tensor, return old tensor
    else:
        return tensor

    # If tensor is given in the shape of (..., 3, 3), collapse matrix to 9
    if len(shape_old) > 2 and shape_old[len(shape_old) - 2:] == (3, 3):
        # Get rid of (3, 3) and append 9 to shape
        shape_new = list(shape_old[:len(shape_old) - 2])
        shape_new.append(9)
        tensor = tensor.reshape(shape_new)

    # Indices to concatenate
    idx_1to6 = (0, 1, 2, 4, 5, 8)
    tensor_compact = tensor[..., idx_1to6]

    return tensor_compact


cpdef tuple getStrainAndRotationRateTensor(np.ndarray grad_u, np.ndarray tke=None,  np.ndarray eps=None, double cap=1e9):
    """
    Calculate strain rate tensor sij as well as rotation rate tensor rij, given velocity gradient grad_u.
    If TKE tke and energy dissipaton rate eps are both provided, sij and rij are non-dimensionalized.
    If cap is provided, sij and rij magnitudes are capped to cap.
    
    :param grad_u: Velocity gradient
    :type grad_u: ndarray[mesh grid / n_samples x 3 x 3] or ndarray[mesh grid / n_samples x 9]
    :param tke: TKE, to non-dimensionalize Sij and Rij, along with epsilon.
    If None, no non-dimensionalization is done.
    :type tke: ndarray[mesh grid / n_samples x 0/1] or None, optional (default=None)
    :param eps: Turbulent energy dissipation rate, to non-dimensionalize Sij and Rij, along with TKE.
    If None, no non-dimensionalization is done.
    :type eps: ndarray[mesh grid / n_samples x 0/1] or None, optional (default=None)
    :param cap: Sij and Rij magnitude cap.
    :type cap: float, optional (default=1e9)
    
    :return: Strain and rotation rate tensor Sij and Rij. Only the 6 unique components of symmetric tensor Sij is returned.
    :rtype: ndarray[n_samples x 6], ndarray[n_samples x 9]
    """
    cdef list ij_uniq, ii6, ii9, ij_6to9
    cdef np.ndarray[np.float_t] tke_eps, sij_i, rij_i
    cdef np.ndarray[np.float_t, ndim=2] grad_u_i
    cdef double maxsij, maxrij, minsij, minrij, progress
    cdef unsigned int i
    cdef unsigned int milestone = 25

    print('\nCalculating strain and rotation rate tensor Sij and Rij...')
    # Collapse mesh grid but don't collapse matrix form of (3, 3) to 9
    grad_u, _ = collapseMeshGridFeatures(grad_u, collapse_matrix=False)
    # Indices
    ij_uniq = [0, 1, 2, 4, 5, 8]
    ii6, ii9 = [0, 3, 5], [0, 4, 8]
    ij_6to9 = [0, 1, 2, 1, 3, 4, 2, 4, 5]
    # If either TKE or epsilon is None, no non-dimensionalization is done
    if tke is None or eps is None:
        tke = np.ones(grad_u.shape[0])
        eps = np.ones(grad_u.shape[0])

    # Cap epsilon to 1e-10 to avoid FPE, also assuming no back-scattering
    eps[eps == 0.] = 1e-10
    # Non-dimensionalization coefficient for strain and rotation rate tensor
    tke_eps = tke.ravel()/eps.ravel()
    # Sij is strain rate tensor, Rij is rotation rate tensor
    # Sij is symmetric tensor, thus 6 unique components, while Rij is anti-symmetric and 9 unique components
    sij = np.empty((grad_u.shape[0], 6))
    rij = np.empty((grad_u.shape[0], 9))
    # Go through each point
    for i in range(grad_u.shape[0]):
        grad_u_i = grad_u[i].reshape((3, 3)) if len(np.shape(grad_u)) == 2 else grad_u[i]
        # Basically Sij = 0.5TKE/epsilon*(grad_u_i + grad_u_j) that has 0 trace
        sij_i = (tke_eps[i]*0.5*(grad_u_i + grad_u_i.T)).ravel()

        # Basically Rij = 0.5TKE/epsilon*(grad_u_i - grad_u_j) that has 0 in the diagonal
        rij_i = (tke_eps[i]*0.5*(grad_u_i - grad_u_i.T)).ravel()
        sij[i] = sij_i[ij_uniq]
        rij[i] = rij_i

        # Gauge progress
        progress = float(i)/(grad_u.shape[0] + 1)*100.
        if progress >= milestone:
            printf(' %d %%... ', milestone)
            milestone += 25

    # Maximum and minimum
    maxsij, maxrij = np.max(sij.ravel()), np.max(rij.ravel())
    minsij, minrij = np.min(sij.ravel()), np.min(rij.ravel())
    print(' Max of Sij is ' + str(maxsij) + ', and of Rij is ' + str(maxrij) + ' capped to ' + str(cap))
    print(' Min of Sij is ' + str(minsij) + ', and of Rij is ' + str(minrij)  + ' capped to ' + str(-cap))
    sij[sij > cap], rij[rij > cap] = cap, cap
    sij[sij < -cap], rij[rij < -cap] = -cap, -cap
    # Because we enforced limits on Sij, we need to re-enforce trace of 0.
    # Go through each point
    if any((maxsij > cap, minsij < cap)):
        for i in range(grad_u.shape[0]):
            # Recall Sij is symmetric and has 6 unique components
            sij[i, ii6] -=  ((1/3.*np.eye(3)*np.trace(sij[i, ij_6to9].reshape((3, 3)))).ravel()[ii9])

    return sij, rij


cpdef np.ndarray[np.float_t, ndim=3] getInvariantBases(np.ndarray sij, np.ndarray rij, 
                                                       bint quadratic_only=False, bint is_scale=True, bint zero_trace=False):
    """
    Calculate 4 or 10 invariant bases of shape (n_samples, n_outputs, n_bases) given strain rate tensor sij and rotation rate tensor rij.
    If quadratic_only is True, only 4 bases will be calculated.
    If is_scale is True, the bases will be divided by [1, 10, 10, 10, 100, 100, 1000, 1000, 1000, 1000].
    If zero_trace is True, check for 0 trace of tb and enforce it.
    
    :param sij: strain rate tensor
    :type sij: ndarray[grid shape / n_samples x 6/9] or ndarray[grid shape / n_samples x 3 x 3]
    :param rij: rotation rate tensor
    :type rij: ndarray[grid shape / n_samples x 9] or ndarray[grid shape / n_samples x 3 x 3]
    :param quadratic_only: True if only linear and quadratic terms are desired, n_bases = 4. False if full basis is desired, n_bases = 10.
    :type quadratic_only: bool, optional (default=False)
    :param is_scale: Whether scale Tij with [1, 10, 10, 10, 100, 100, 1000, 1000, 1000, 1000].
    :type is_scale: bool, optional (default=True)
    :param zero_trace: Whether enforce 0 trace of Tij.
    :type zero_trace: bool, optional (default=False)
    
    :return: Tij of shape (n_samples, 6, n_bases). 6 means taking the unique components of the symmetric tensor only.
    :rtype: ndarray[n_samples x 6 x n_bases]
    """
    cdef list ij_uniq, ij_6to9, scale_factor
    cdef unsigned int n_bases, i, j
    cdef np.ndarray[np.float_t, ndim=3] tb
    cdef np.ndarray[np.float_t, ndim=2] sij_i, rij_i, sijrij, rijsij, sijsij, rijrij
    cdef unsigned int milestone = 10
    cdef double progress

    print('\nCalculating invariant bases Tij...')
    # Ensure n_samples x 6 for Sij and n_samples x 9 for Rij
    sij, _ = collapseMeshGridFeatures(sij)
    rij, _ = collapseMeshGridFeatures(rij)
    if sij.shape[1] == 9: sij = contractSymmetricTensor(sij)
    # Indices
    ij_uniq = [0, 1, 2, 4, 5, 8]
    ij_6to9 = [0, 1, 2, 1, 3, 4, 2, 4, 5]
    # If 3D flow, then 10 tensor bases; else if 2D flow, then 4 tensor bases
    n_bases = 10 if not quadratic_only else 4
    # Tensor bases is nPoint x nBasis x 3 x 3
    # tb = np.zeros((Sij.shape[0], n_bases, 3, 3))
    tb = np.empty((sij.shape[0], 6, n_bases))
    # Go through each point
    for i in range(sij.shape[0]):
        # Sij only has 6 unique components, convert it to 9 using ij_6to9
        sij_i = sij[i, ij_6to9].reshape((3, 3))
        # Rij has 9 unique components already
        rij_i = rij[i].reshape((3, 3))
        # Convenient pre-computations
        sijrij = sij_i @ rij_i
        rijsij = rij_i @ sij_i
        sijsij = sij_i @ sij_i
        rijrij = rij_i @ rij_i
        # 10 tensor bases for each point and each (unique) bij component
        # 1: Sij
        tb[i, :, 0] = sij_i.ravel()[ij_uniq]
        # 2: SijRij - RijSij
        tb[i, :, 1] = (sijrij - rijsij).ravel()[ij_uniq]
        # 3: Sij^2 - 1/3I*tr(Sij^2)
        tb[i, :, 2] = (sijsij - 1./3.*np.eye(3)*np.trace(sijsij)).ravel()[ij_uniq]
        # 4: Rij^2 - 1/3I*tr(Rij^2)
        tb[i, :, 3] = (rijrij - 1./3.*np.eye(3)*np.trace(rijrij)).ravel()[ij_uniq]
        # If more than 4 bases
        if not quadratic_only:
            # 5: RijSij^2 - Sij^2Rij
            tb[i, :, 4] = (rij_i @ sijsij - sij_i @ sijrij).ravel()[ij_uniq]
            # 6: Rij^2Sij + SijRij^2 - 2/3I*tr(SijRij^2)
            tb[i, :, 5] = (rij_i @ rijsij
                           + sij_i @ rijrij
                           - 2./3.*np.eye(3)*np.trace(sij_i @ rijrij)).ravel()[ij_uniq]
            # 7: RijSijRij^2 - Rij^2SijRij
            tb[i, :, 6] = (rijsij @ rijrij - rijrij @ sijrij).ravel()[ij_uniq]
            # 8: SijRijSij^2 - Sij^2RijSij
            tb[i, :, 7] = (sijrij @ sijsij - sijsij @ rijsij).ravel()[ij_uniq]
            # 9: Rij^2Sij^2 + Sij^2Rij^2 - 2/3I*tr(Sij^2Rij^2)
            tb[i, :, 8] = (rijrij @ sijsij
                           + sijsij @ rijrij
                           - 2./3.*np.eye(3)*np.trace(sijsij @ rijrij)).ravel()[ij_uniq]
            # 10: RijSij^2Rij^2 - Rij^2Sij^2Rij
            tb[i, :, 9] = ((rij_i @ sijsij) @ rijrij
                           - (rij_i @ rijsij) @ sijrij).ravel()[ij_uniq]

        # If enforce zero trace for anisotropy for each basis
        if zero_trace:
            for j in range(n_bases):
                # Recall tb is shape (n_samples, 6, n_bases)
                tb[i, :, j] -= (1./3.*np.eye(3)*np.trace(tb[i, ij_6to9, j].reshape((3, 3)))).ravel()[ij_uniq]

        # Gauge progress
        progress = float(i)/(sij.shape[0] + 1)*100.
        if progress >= milestone:
            printf(' %d %%... ', milestone)
            milestone += 10

    # Scale down to promote convergence
    if is_scale:
        # Using tuple gives Numba error
        scale_factor = [1, 10, 10, 10, 100, 100, 1000, 1000, 1000, 1000]
        # Go through each basis
        for j in range(1, n_bases):
            tb[:, :, j] /= scale_factor[j]

    return tb


cpdef np.ndarray makeRealizable(np.ndarray bij):
    """
    From Ling et al. (2016), see https://github.com/tbnn/tbnn.
    
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability
    by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
    Then, if eigenvalues negative, shift them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called iteratively to get convergence
    to a realizable state.

    :param bij: The predicted anisotropy tensor.
    :type bij: np.ndarray[n_points or mesh grid, 9 or 6 or 3 x 3]

    :return: The predicted realizable anisotropy tensor, same shape as the input array.
    :type: np.ndarray[n_points or mesh grid, 6 or 9 or 3 x 3]
    """
    cdef unsigned int n_points, i, j, old_lastdim
    cdef np.ndarray[np.float_t, ndim=2] A, evectors
    cdef np.ndarray[np.float_t] evalues
    cdef list bii
    cdef tuple oldshape

    # Collapse mesh grid and if 3 x 3 form, collapse matrix form too to 9
    bij, oldshape = collapseMeshGridFeatures(bij, collapse_matrix=True)
    old_lastdim = len(oldshape) - 1
    # If bij is n_points x 6, expand it to full form of n_points x 9
    if bij.shape[1] == 6: bij = expandSymmetricTensor(bij)

    n_points = bij.shape[0]
    bii = [0, 4, 8]
    A = np.zeros((3, 3))
    for i in range(n_points):
        # Scales all on-diags to retain zero trace
        if np.min(bij[i, [0, 4, 8]]) < -1./3.:
            bij[i, [0, 4, 8]] *= -1./(3.*np.min(bij[i, [0, 4, 8]]))

        if 2.*np.abs(bij[i, 1]) > bij[i, 0] + bij[i, 4] + 2./3.:
            bij[i, 1] = (bij[i, 0] + bij[i, 4] + 2./3.)*.5*np.sign(bij[i, 1])
            bij[i, 3] = (bij[i, 0] + bij[i, 4] + 2./3.)*.5*np.sign(bij[i, 1])
        if 2.*np.abs(bij[i, 5]) > bij[i, 4] + bij[i, 8] + 2./3.:
            bij[i, 5] = (bij[i, 4] + bij[i, 8] + 2./3.)*.5*np.sign(bij[i, 5])
            bij[i, 7] = (bij[i, 4] + bij[i, 8] + 2./3.)*.5*np.sign(bij[i, 5])
        if 2.*np.abs(bij[i, 2]) > bij[i, 0] + bij[i, 8] + 2./3.:
            bij[i, 2] = (bij[i, 0] + bij[i, 8] + 2./3.)*.5*np.sign(bij[i, 2])
            bij[i, 6] = (bij[i, 0] + bij[i, 8] + 2./3.)*.5*np.sign(bij[i, 2])

        # Enforce positive semidefinite by pushing evalues to non-negative
        A[0, 0] = bij[i, 0]
        A[1, 1] = bij[i, 4]
        A[2, 2] = bij[i, 8]
        A[0, 1] = bij[i, 1]
        A[1, 0] = bij[i, 1]
        A[1, 2] = bij[i, 5]
        A[2, 1] = bij[i, 5]
        A[0, 2] = bij[i, 2]
        A[2, 0] = bij[i, 2]
        evalues, evectors = np.linalg.eig(A)
        if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])/2.:
            evalues = evalues*(3.*np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])/(2.*np.max(evalues))
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                bij[i, bii[j]] = A[j, j]

            bij[i, 1] = A[0, 1]
            bij[i, 5] = A[1, 2]
            bij[i, 2] = A[0, 2]
            bij[i, 3] = A[0, 1]
            bij[i, 7] = A[1, 2]
            bij[i, 6] = A[0, 2]

        if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
            evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                bij[i, bii[j]] = A[j, j]

            bij[i, 1] = A[0, 1]
            bij[i, 5] = A[1, 2]
            bij[i, 2] = A[0, 2]
            bij[i, 3] = A[0, 1]
            bij[i, 7] = A[1, 2]
            bij[i, 6] = A[0, 2]

    # Preparing to reverse to old mesh grid incase old shape is at least 3D and last dimension is n_components
    if old_lastdim > 1:
        if oldshape[old_lastdim] == 6:
            bij = contractSymmetricTensor(bij)
        elif oldshape[old_lastdim - 1:] == (3, 3):
            bij = bij.reshape((bij[0], 3, 3))

        bij = reverseOldGridShape(bij, oldshape)

    return bij




# -----------------------------------------------------
# Supporting Functions, Not Intended to Be Called From Python
# -----------------------------------------------------
cdef np.ndarray[np.float_t, ndim=2] _makeRealizable(np.ndarray[np.float_t, ndim=2] bij):
    """
    From Ling et al. (2016), see https://github.com/tbnn/tbnn.
    
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability
    by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
    Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called iteratively to get convergence
    to a realizable state.

    :param bij: The predicted anisotropy tensor.
    :type bij: np.ndarray[n_points, 9]

    :return: The predicted realizable anisotropy tensor.
    :type: np.ndarray[n_points, 9]
    """
    cdef unsigned int n_points, i, j
    cdef np.ndarray[np.float_t, ndim=2] A, evectors
    cdef np.ndarray[np.float_t] evalues
    cdef list bii

    n_points = bij.shape[0]
    bii = [0, 4, 8]
    A = np.zeros((3, 3))
    for i in range(n_points):
        # Scales all on-diags to retain zero trace
        if np.min(bij[i, [0, 4, 8]]) < -1./3.:
            bij[i, [0, 4, 8]] *= -1./(3.*np.min(bij[i, [0, 4, 8]]))

        if 2.*np.abs(bij[i, 1]) > bij[i, 0] + bij[i, 4] + 2./3.:
            bij[i, 1] = (bij[i, 0] + bij[i, 4] + 2./3.)*.5*np.sign(bij[i, 1])
            bij[i, 3] = (bij[i, 0] + bij[i, 4] + 2./3.)*.5*np.sign(bij[i, 1])
        if 2.*np.abs(bij[i, 5]) > bij[i, 4] + bij[i, 8] + 2./3.:
            bij[i, 5] = (bij[i, 4] + bij[i, 8] + 2./3.)*.5*np.sign(bij[i, 5])
            bij[i, 7] = (bij[i, 4] + bij[i, 8] + 2./3.)*.5*np.sign(bij[i, 5])
        if 2.*np.abs(bij[i, 2]) > bij[i, 0] + bij[i, 8] + 2./3.:
            bij[i, 2] = (bij[i, 0] + bij[i, 8] + 2./3.)*.5*np.sign(bij[i, 2])
            bij[i, 6] = (bij[i, 0] + bij[i, 8] + 2./3.)*.5*np.sign(bij[i, 2])

        # Enforce positive semidefinite by pushing evalues to non-negative
        A[0, 0] = bij[i, 0]
        A[1, 1] = bij[i, 4]
        A[2, 2] = bij[i, 8]
        A[0, 1] = bij[i, 1]
        A[1, 0] = bij[i, 1]
        A[1, 2] = bij[i, 5]
        A[2, 1] = bij[i, 5]
        A[0, 2] = bij[i, 2]
        A[2, 0] = bij[i, 2]
        evalues, evectors = np.linalg.eig(A)
        if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])/2.:
            evalues = evalues*(3.*np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])/(2.*np.max(evalues))
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                bij[i, bii[j]] = A[j, j]

            bij[i, 1] = A[0, 1]
            bij[i, 5] = A[1, 2]
            bij[i, 2] = A[0, 2]
            bij[i, 3] = A[0, 1]
            bij[i, 7] = A[1, 2]
            bij[i, 6] = A[0, 2]

        if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
            evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                bij[i, bii[j]] = A[j, j]

            bij[i, 1] = A[0, 1]
            bij[i, 5] = A[1, 2]
            bij[i, 2] = A[0, 2]
            bij[i, 3] = A[0, 1]
            bij[i, 7] = A[1, 2]
            bij[i, 6] = A[0, 2]

    return bij


cdef np.ndarray[np.float_t, ndim=2] _mapVectorToAntisymmetricTensor(np.ndarray[np.float_t, ndim=2] vec, np.ndarray scaler=None):
    """
    Map a vector to the anti-symmetric tensor A by
    A = -I x vector where I is the 2nd order identity matrix,
    and scale with a scaler to non-dimensionalize it if provided.
    
    :param vec: The vector array to map to an anti-symmetric tensor.
    :type vec: ndarray[n_points, vector size]
    :param scaler: Scaler scalar or vector to non-dimensionalized and/or normalize the given vector. 
    If None, then no scaling is applied.
    :type scaler: ndarray[n_points] or ndarray[n_points, vector size] or None, optional (default=None)
    
    :return: (Non-dimensionalized and/or normalized) anti-symmetric tensor from given vector array (and scaler).
    :rtype: ndarray[n_points, vector size^2]
    """
    cdef np.ndarray[np.float_t, ndim=2] asymm_tensor
    cdef int i

    # Antisymmetric tensor is n_points x vector size x vector size. E.g. grad(TKE) is n_points x 3 x 3
    asymm_tensor = np.empty((vec.shape[0], (vec.shape[1])**2))
    # Go through every point and calculate the cross product: -I x vec, e.g. -I x grad(TKE)
    for i in range(vec.shape[0]):
        # Scale the vector so that it's dimensionless/normalized.
        # E.g. grad(k) vector with sqrt(k)/epsilon scalar, or grad(p) vector with 1/(rho*|DU/Dt|) scalar
        if scaler is not None:
            vec[i] *= scaler[i]

        asymm_tensor[i] = -np.cross(np.eye(vec.shape[1]), vec[i]).ravel()

    print("\nVector mapped to an anti-asymmetric tensor of shape " + str(np.shape(asymm_tensor)))
    return asymm_tensor




# --------------------------------------------------------
# [DEPRECATED] But Still Usable
# --------------------------------------------------------
cpdef tuple convertTensorTo2D(np.ndarray tensor, bint infer_stress=True):
    """
    [DEPRECATED] See Utility.collapseMeshGridFeatures().
    
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
