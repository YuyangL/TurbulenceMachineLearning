# cython: language_level = 3str
# cython: embedsignature = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from warnings import warn
from Tensor cimport _mapVectorToAntisymmetricTensor
from Tensor import contractSymmetricTensor, expandSymmetricTensor
from Utility import collapseMeshGridFeatures


cpdef tuple getInvariantFeatureSet(np.ndarray sij, np.ndarray rij,
                     np.ndarray grad_k=None, np.ndarray grad_p=None,
                     np.ndarray k=None, np.ndarray eps=None,
                     np.ndarray u=None, np.ndarray grad_u=None, double rho=1.225):
    """
    Get invariant features based on at least dimensionless strain and rotation rate tensor Sij, Rij of shape (n_samples / mesh grid, 3, 3); 
    and possibly grad(TKE) and/or grad(p) of shape (n_samples / mesh grid, 3).
    If neither grad(TKE) nor grad(p) is provided, then 6 invariant features of shape (n_samples, 6) will be calculated based on Sij, Rij.
    If grad(TKE) is provided, it is mapped to an anti-symmetric tensor as -I x grad(TKE) 
    before calculating 19 invariant features of shape (n_samples, 19). 
        Moreover, if TKE and epsilon are provided, 
        then grad(TKE) is non-dimensionalized as sqrt(k)/epsilon*grad(TKE).
    If grad(p) is provided, it is mapped to an anti-symmetric tensor as -I x grad(p)
    before calculating 19 invariant features of shape (n_samples, 19).
         Moreover, if U, grad(U) and rho are provided, then grad(p) is non-dimensionalized as 1/(rho*|DU/Dt|)*grad(p).
    If both grad(TKE) and grad(p) are provided, 
    then 47 invariant features of shape (n_samples, 47) incl. interaction between grad(TKE) and grad(p) will be calculated.
    
    From Wu et al., Physics-Informed Machine Learning Approach for Augmenting Turbulence Models: A Comprehensive Framework.
    
    :param sij: Dimensionless strain rate tensor Sij of shape (n_samples / mesh grid, 3, 3) or (n_samples / mesh grid, 6/9).
    :type sij: ndarray[n_samples / mesh grid, 3, 3] or ndarray[n_samples / mesh grid, 6/9]
    :param rij: Dimensionless rotation rate tensor Rij of shape (n_samples / mesh grid, 3, 3) or (n_samples / mesh grid, 6/9).
    :type rij: ndarray[n_samples / mesh grid, 3, 3] or ndarray[n_samples / mesh grid, 6/9]
    :param grad_k: Spatial gradient of TKE of shape (n_samples / mesh grid, 3).
    If None, no invariant features are calculated based on grad(TKE).
    :type grad_k: ndarray[n_samples / mesh grid, 3] or None, optional (default=None)
    :param grad_p: Spatial gradient of pressure of shape (n_samples / mesh grid, 3).
    If None, no invariant features are calculated based on grad(p).
    :type grad_p: ndarray[n_samples / mesh grid, 3] or None, optional (default=None)
    :param k: TKE of shape (n_samples, / mesh grid) used to non-dimensionalize grad(TKE).
    If None, then no non-dimensionalization of grad(TKE) is done.
    If grad(TKE) is None, then k has no effect.
    :type k: ndarray[n_samples / mesh grid] or None, optional (default=None)
    :param eps: Turbulent dissipation rate of shape (n_samples,) or (mesh grid) used to non-dimensionalize grad(TKE).
    If None, then no non-dimensionalization of grad(TKE) is done.
    If grad(TKE) is None, then eps has no effect.
    :type eps: ndarray[n_samples / mesh grid] or None, optional (default=None)
    :param u: Velocity of shape (n_samples / mesh grid, 3) used to non-dimensionalize grad(p).
    If None, then no non-dimensionalization of grad(p) is done.
    If grad(p) is None, then u has no effect.
    :type u: ndarray[n_samples / mesh grid, 3] or None, optional (default=None)
    :param grad_u: Velocity spatial gradient of shape (n_samples / mesh grid, 3, 3) or (n_samples / mesh grid, 9),
    used to non-dimensionalize grad(p).
    If None, then no non-dimensionalization of grad(p) is done.
    If grad(p) is None, then grad_u has no effect.
    :type grad_u: ndarray[n_samples / mesh grid, 3, 3] or ndarray[n_samples / mesh grid, 9] or None, 
    optional (default=None)
    :param rho: Fluid density used for non-dimensionalizing grad(p).
    If grad(p) is None, then rho has no effect.
    :type rho: float, optional (default=1.225)
    
    :return: 6/19/47 invariant features of shape (n_samples, n_features) if none, grad(p) or grad(TKE) or both gradients
    are provided on top of Sij, Rij; and its corresponding string labels.
    :rtype: (ndarray[n_samples, n_features], tuple)
    """
    cdef np.ndarray scaler_k = None
    cdef np.ndarray scaler_p = None
    cdef np.ndarray[np.float_t, ndim=2] inv_set
    # U*grad(U) is (3,) for each sample
    cdef np.ndarray[np.float_t] ugrad_u = np.empty(3)
    cdef int i
    cdef tuple labels

    # n_samples x 6
    sij, _ = collapseMeshGridFeatures(sij, collapse_matrix=True)
    if sij.shape[1] == 9: sij = contractSymmetricTensor(sij)
    # n_samples x 9
    rij, _ = collapseMeshGridFeatures(rij, collapse_matrix=True)
    # Warn if non-dimensionalization couldn't be done due to lack of scaler inputs for TKE and/or p
    if grad_k is not None:
        grad_k, _ = collapseMeshGridFeatures(grad_k, infer_matrix_form=False)
        if k is None or eps is None:
            warn("\nFeatures related to grad(TKE) are not non-dimensionalized as TKE and epsilon inputs are missing!\n", stacklevel=2)
        else:
            # Ensure 1D if previously 2D meshgrid
            k, eps = k.ravel(), eps.ravel()
            scaler_k = np.sqrt(k)/eps

    if grad_p is not None:
        grad_p, _ = collapseMeshGridFeatures(grad_p, infer_matrix_form=False)
        # Calculate grad(p)'s non-dimensionalization scaler,
        # scaler = 1/(rho*|DU/Dt|)
        # Since DU/Dt = dU/dt + U*grad(U) and assume steady-state,
        # DU/Dt = U*grad(U),
        # and scaler = 1/(rho*|U*grad(U)|), taking Frobenius norm to reduce vector to scalar
        if u is None or grad_u is None:
            warn("\nFeatures related to grad(p) are not non-dimensionalized as grad(U), U, and rho inputs are missing!\n", stacklevel=2)
        else:
            u, _ = collapseMeshGridFeatures(u, infer_matrix_form=False)
            # Mesh grid collapsed to 1D and grad(U) matrix collapsed to 1D, if grad(U) were provided in matrix form
            grad_u, _ = collapseMeshGridFeatures(grad_u, collapse_matrix=True)
            # scaler_p is (n_samples,)
            scaler_p = np.empty(grad_p.shape[0])
            # Go through each sample, calculate 3 U*grad(U) and take its Frobenius norm for scaler_p scalar array
            for i in range(grad_u.shape[0]):
                ugrad_u[0] = u[i, 0]*grad_u[i, 0] + u[i, 1]*grad_u[i, 1] + u[i, 2]*grad_u[i, 2]
                ugrad_u[1] = u[i, 0]*grad_u[i, 3] + u[i, 1]*grad_u[i, 4] + u[i, 2]*grad_u[i, 5]
                ugrad_u[2] = u[i, 0]*grad_u[i, 6] + u[i, 1]*grad_u[i, 7] + u[i, 2]*grad_u[i, 8]
                scaler_p[i] = 1./(rho*np.linalg.norm(ugrad_u))

    # Calculate invariant features based on Sij, Rij (mandatory), grad(TKE) (optional), grad(p) (optional).
    # grad(TKE), grad(p) will receive anti-symmetric tensor mapping (and non-dimensionalization) in _getInvariantFeatureSet()
    inv_set, labels = _getInvaraintFeatureSet(sij, rij, grad1=grad_k, grad2=grad_p, grad1_scaler=scaler_k, grad2_scaler=scaler_p)

    return inv_set, labels


cpdef tuple getSupplementaryInvariantFeatures(np.ndarray k, np.ndarray d, np.ndarray epsilon, np.ndarray nu, np.ndarray sij=None, np.ndarray r=None):
    """
    Calculate 3 or 4 supplementary mean flow features as done in 
    Wu et al., Physics-Informed Machine Learning Approach for Augmenting Turbulence Models: A Comprehensive Framework.
    1st feature is wall-distance based Re number;
    2nd feature is turbulence intensity;
    3rd feature is ratio of turbulent time-scale to mean strain time-scale.
    If radial distance to (closest) turbine center, 4th feature is radial distance to (closest) turbine center based Re number.
    If strain rate tensor Sij is provided, then normalization as well as non-dimensionalization is done.
    
    :param k: TKE array of any shape. Will be flattened.
    :type k: ndarray
    :param d: Wall distance array of any shape. Will be flattened.
    :type d: ndarray
    :param epsilon: Turbulent energy dissipation rate array of any shape. Will be flattened.
    :type epsilon: ndarray
    :param nu: Kinematic viscosity array of any shape. Will be flattened.
    :type nu: ndarray
    :param sij: Strain rate tensor Sij. If None, no non-dimensionalization is performed.
    :type sij: ndarray[..., 6] or ndarray[..., 9] or ndarray[..., 3, 3] or None, optional (default=None)
    :param r: Radial distant to (closest) turbine center.
    :type r: ndarray or None, optional (default=None)
    
    :return: 3 or 4 supplementary features in a 2D array and labels
    :rtype: (ndarray[n_samples x 3/4], tuple)
    """
    cdef np.ndarray[np.float_t] sijnorm
    cdef np.ndarray[np.float_t, ndim=2] features
    cdef tuple labels

    # 1D array treatment
    k = k.ravel()
    d = d.ravel()
    epsilon = epsilon.ravel()
    nu = nu.ravel()
    if r is not None: r = r.ravel()
    # Calculate ||Sij|| for normalization if provided
    if isinstance(sij, np.ndarray):
        sij, _ = collapseMeshGridFeatures(sij, collapse_matrix=True)
        # Use full form of Sij for Frobenius norm
        sij = expandSymmetricTensor(sij)
        sijnorm = np.linalg.norm(sij, axis=1)
    else:
        sij = None

    # Features array has shape (n_samples, n_features), 4 if radial distance to turbine center is given
    features = np.empty((k.shape[0], 3)) if r is None else np.empty((k.shape[0], 4))
    # Feature 1: Wall-distance based Re number
    features[:, 0] = np.minimum(np.sqrt(k)*d/(50.*nu), 2.)
    # Feature 2: Turbulence intensity
    features[:, 1] = k if sij is None else k/(nu*sijnorm)
    # Feature 3: Ratio of turbulent time-scale to mean strain time-scale
    features[:, 2] = k/epsilon if sij is None else k/epsilon/(1/sijnorm)
    if r is not None:
        # Feature 4: Radial distance to turbine center based Re number
        features[:, 3] = np.sqrt(k)*r/nu

    # Labels depending on whether normalization is done and whether radial to distance turbine center is provided
    if sij is None:
        if r is None:
            labels = ('min[sqrt(k)d/(50nu), 2]', 'k', 'k/epsilon')
        else:
            labels = ('min[sqrt(k)d/(50nu), 2]', 'k', 'k/epsilon', 'sqrt(k)r/nu')

    else:
        if r is None:
            labels = ('min[sqrt(k)d/(50nu), 2]', 'k/(nu||Sij||)', 'k/epsilon/(1/||Sij||)')
        else:
            labels = ('min[sqrt(k)d/(50nu), 2]', 'k/(nu||Sij||)', 'k/epsilon/(1/||Sij||)', 'sqrt(k)r/nu')

    return features, labels


cpdef np.ndarray getRadialTurbineDistance(np.ndarray x, np.ndarray y, np.ndarray z=None, list turblocs=None):
    """
    Given x, y, (z) coordinates, calculate radial distance to (multiple) given turbine centers.
    If z is not provided, only horizontal distance to (closest) turbine center is computed.
    If multiple turbine locations are provided, the radial distance is to the closest turbine for each point.
    
    :param x: Cell center coordinates in x axis.
    :type x: ndarray of any shape
    :param y: Cell center coordinates in y axis.
    :type y: ndarray of same shape as x
    :param z: Cell center coordinates in z axis. If None, only horizontal distance to turbine center is considered.
    :type z: ndarray of same shape as x or None, optional (default=None)
    :param turblocs: 1/2D list of all turbine centers, each row representing a turbine's x, y, z coordinate. 
    If None, then default is a turbine at [0., 0., 0.].
    :type turblocs: list or None, optional (default=None)
    
    :return: Radial or horizontal radial distance to the given turbine center
    :rtype: ndarray of same shape as x
    """
    if turblocs is None: turblocs = [0., 0., 0.]
    cdef tuple old_shape = np.shape(x)
    cdef np.ndarray r, ri
    cdef np.ndarray turbarr = np.atleast_2d(turblocs)
    cdef unsigned int n_turbs = turbarr.shape[0]
    cdef unsigned int i

    # If z is None, only horizontal distance to turbine center is considered
    if z is None: z = 0.
    # r is the radial distance to the closest given turbine locations
    # Initial radial distance to the first given turbine center
    r = np.sqrt((x - turbarr[0, 0])**2 + (y - turbarr[0, 1])**2 + (z - turbarr[0, 2])**2)
    # For the other turbines
    for i in range(n_turbs - 1):
        ri = np.sqrt((x - turbarr[i, 0])**2 + (y - turbarr[i, 1])**2 + (z - turbarr[i, 2])**2)
        # Find the minima comparing the old radial distance array with the new one calculated based on new turbine
        r = np.minimum(r, ri)

    return r



# -----------------------------------------------------
# Supporting Functions, Not Intended to Be Called From Python
# -----------------------------------------------------
cdef tuple _getInvaraintFeatureSet(np.ndarray[np.float_t, ndim=2] sij, np.ndarray[np.float_t, ndim=2] rij,
                                                np.ndarray grad1=None, np.ndarray grad2=None, np.ndarray grad1_scaler=None, np.ndarray grad2_scaler=None):
    """
    Calculate invariant features for samples, given at least non-dimensionalized strain rate and rotation rate tensor Sij, Rij.
    If only non-dimensionalized Sij of shape (n_samples, 6) and Rij of shape (n_samples, 9) are provided, 
    then 6 invariant features are calculated.
    Else if an extra scalar gradient of shape (n_samples, 3) is provided, 
    then 19 invariant features will be calculated.
    Else if extra 2 scalar gradients of shape (n_samples, 3) are provided, 
    then 47 invariant features will be calculated.
    Scalar gradient(s) is first mapped to an anti-symmetric tensor and non-dimensionalized if corresponding scaler is provided, 
    before calculating invariant features.
    
    From Appendix C of Wu et al., Physics-Informed Machine Learning Approach for Augmenting Turbulence Models: A Comprehensive Framework.
    
    :param sij: Non-dimensionalized strain rate symmetric tensor Sij of shape (n_samples, 6). 
    :type sij: ndarray[n_samples, 6]
    :param rij: Non-dimensionalized rotation rate tensor Rij of shape (n_samples, 9).
    :type rij: ndarray[n_samples, 9]
    :param grad1: A scalar gradient e.g. grad(TKE) of shape (n_samples, 3).
    If None, then no invariant feature is calculated based on this information.
    :type grad1: ndarray[n_samples, 3] or None, optional (default=None)
    :param grad2: A 2nd scalar gradient e.g. grad(p) of shape (n_samples, 3).
    If None, then no invariant feature is calculated based on this information.
    If grad1 is also provided, then 15 invariant features based on grad1 and grad2 interactions are also calculated.
    :type grad2: ndarray[n_samples, 3] or None, optional (default=None)
    :param grad1_scaler: Scaler for grad1 to non-dimensionalize/normalize it.
    If grad1 is None, then grad1_scaler has no effect.
    :type grad1_scaler: ndarray[n_samples] or None, optional (default=None)
    :param grad2_scaler: Scaler for grad2 to non-dimensionalize/normalize it.
    If grad2 is None, then grad2_scaler has no effect. 
    :type grad2_scaler: ndarray[n_samples] or None, optional (default=None)
    
    :return: 6/19/47 invariant features of shape (n_samples, n_features) if 0/1/2 scalar gradients are provided on top of Sij, Rij; 
    and its corresponding string labels.
    :rtype: (ndarray[n_samples, n_features], tuple)
    """
    cdef np.ndarray[np.float_t, ndim=2] asymm_tensor1, asymm_tensor2
    cdef tuple labels
    cdef int milestone = 10
    cdef unsigned int i, j, n_inv, n_grad
    cdef double progress
    cdef list ij_6to9 = [0, 1, 2, 1, 3, 4, 2, 4, 5]
    cdef list ij_uniq = [0, 1, 2, 4, 5, 8]
    cdef np.ndarray[np.float_t, ndim=2] sij_i, rij_i, asymm_tensor1_i, asymm_tensor2_i
    cdef np.ndarray[np.float_t, ndim=2] inv_set, sijsij, rijrij, rijsij, rijsijsij, sijrijsijsij
    cdef np.ndarray[np.float_t, ndim=2] a1sij, rija1, a1sijsij, a1rijrij, sija1sijsij, a2sij, rija2, a2sijsij, a2rijrij, sija2sijsij
    cdef np.ndarray[np.float_t, ndim=2] a, asij, rija, asijsij, arijrij, sijasijsij

    # First map given scalar gradient to anti-symmetric tensor,
    # with A_scalar = [-I x grad(scalar)]*scaler. Shape (n_samples, 9)
    if grad1 is not None:
        asymm_tensor1 = _mapVectorToAntisymmetricTensor(grad1, grad1_scaler)
        
    # If two gradient fields provided, interactions between them are also considered
    if grad2 is not None:
        asymm_tensor2 = _mapVectorToAntisymmetricTensor(grad2, grad2_scaler)

    # Determine number of invariants and corresponding string labels
    if grad1 is None and grad2 is None:
        n_inv, n_grad = 6, 0
        labels = ('S^2', 'S^3', 'R^2', 'R^2*S', 'R^2*S^2', 'R^2*S*R*S^2')
        print("\nCalculating 6 invariant features with only Sij and Rij... ")
    elif any((grad1 is None, grad2 is None)):
        n_inv, n_grad = 19, 1
        labels = ('S^2', 'S^3', 'R^2', 'R^2*S', 'R^2*S^2', 'R^2*S*R*S^2',
                  'A^2', 'A^2*S', 'A^2*S^2', 'A^2*S*A*S^2', 'R*A', 'R*A*S', 'R*A*S^2', 'R^2*A*S', 'A^2*R*S', 'R^2*A*S^2', 'A^2*R*S^2', 'R^2*S*A*S^2', 'A^2*S*R*S^2')
        print("\nCalculating 19 invariant features with Sij, Rij, and a scalar gradient... ")
    else:
        n_inv, n_grad = 47, 2
        labels = ('S^2', 'S^3', 'R^2', 'R^2*S', 'R^2*S^2', 'R^2*S*R*S^2',
                  'A1^2', 'A1^2*S', 'A1^2*S^2', 'A1^2*S*A1*S^2', 'R*A1', 'R*A1*S', 'R*A1*S^2',
                  'R^2*A1*S', 'A1^2*R*S', 'R^2*A1*S^2', 'A1^2*R*S^2', 'R^2*S*A1*S^2', 'A1^2*S*R*S^2',
                  'A2^2', 'A2^2*S', 'A2^2*S^2', 'A2^2*S*A2*S^2', 'R*A2', 'R*A2*S', 'R*A2*S^2',
                  'R^2*A2*S', 'A2^2*R*S', 'R^2*A2*S^2', 'A2^2*R*S^2', 'R^2*S*A2*S^2', 'A2^2*S*R*S^2',
                  'A1*A2', 'A1*A2*S', 'A1*A2*S^2', 'A1^2*A2*S', 'A2^2*A1*S', 'A1^2*A2*S^2', 'A2^2*A1*S^2', 'A1^2*S*A2*S^2', 'A2^2*S*A1*S^2',
                  'R*A1*A2', 'R*A1*A2*S', 'R*A2*A1*S', 'R*A1*A2*S^2', 'R*A2*A1*S^2', 'R*A1*S*A2*S^2')
        print("\nCalculating 47 invariant features with Sij, Rij, and 2 scalar gradients... ")

    # Invariants feature set is n_samples x n_invariants
    inv_set = np.empty((sij.shape[0], n_inv))
    # Go through every sample and calculate invariants
    for i in range(sij.shape[0]):
        # Recall Sij has 6 unique components, expand it to 9
        sij_i = sij[i, ij_6to9].reshape((3, 3))
        rij_i = rij[i].reshape((3, 3))
        asymm_tensor1_i, asymm_tensor2_i = asymm_tensor1[i].reshape((3, 3)), asymm_tensor2[i].reshape((3, 3))
        # Common shortcuts
        sijsij = sij_i @ sij_i
        rijrij = rij_i @ rij_i
        rijsij = rij_i @ sij_i
        rijsijsij = rij_i @ sijsij
        sijrijsijsij = sij_i @ rijsijsij
        # Extra shortcuts if at least one of grad1/2 is provided
        if grad1 is not None:
            a1sij = asymm_tensor1_i @ sij_i
            rija1 = rij_i @ asymm_tensor1_i
            a1sijsij = asymm_tensor1_i @ sijsij
            a1rijrij = asymm_tensor1_i @ rijrij
            sija1sijsij = sij_i @ a1sijsij
        
        if grad2 is not None:
            a2sij = asymm_tensor2_i @ sij_i
            rija2 = rij_i @ asymm_tensor2_i
            a2sijsij = asymm_tensor2_i @ sijsij
            a2rijrij = asymm_tensor2_i @ rijrij
            sija2sijsij = sij_i @ a2sijsij

        # Invariant features involving only Sij and Rij, 6 in total
        # S^2
        inv_set[i, 0] = np.trace(sijsij)
        # S^3
        inv_set[i, 1] = np.trace(sij_i @ sijsij)
        # R^2
        inv_set[i, 2] = np.trace(rijrij)
        # R^2*S
        inv_set[i, 3] = np.trace(rij_i @ rijsij)
        # R^2*S^2
        inv_set[i, 4] = np.trace(rij_i @ rijsijsij)
        # R^2*S*R*S^2
        inv_set[i, 5] = np.trace(rij_i @ (rij_i @ sijrijsijsij))
        # For each sample, go through number of gradients, either 1 or 2.
        # If n_grad = 1 or 2, then one or both of grad1 and grad2 will be included
        for j in range(n_grad):
            # If only grad1 is provided
            if grad2 is None:
                a = asymm_tensor1_i
                asij, rija = a1sij, rija1
                asijsij, arijrij = a1sijsij, a1rijrij
                sijasijsij = sija1sijsij
            # Else if only grad 2 is provided
            elif grad1 is None:
                a = asymm_tensor2_i
                asij, rija = a2sij, rija2
                asijsij, arijrij = a2sijsij, a2rijrij
                sijasijsij = sija2sijsij
            # Else if both grad1 and grad2 are provided
            else:
                # Then go through grad1 then grad2
                if j == 0:
                    a = asymm_tensor1_i
                    asij, rija = a1sij, rija1
                    asijsij, arijrij = a1sijsij, a1rijrij
                    sijasijsij = sija1sijsij
                else:
                    a = asymm_tensor2_i
                    asij, rija = a2sij, rija2
                    asijsij, arijrij = a2sijsij, a2rijrij
                    sijasijsij = sija2sijsij

            # Invariant features involving a gradient, total 13 for a gradient
            # A^2
            inv_set[i, 6 + j*13] = np.trace(a @ a)
            # A^2*S
            inv_set[i, 7 + j*13] = np.trace(a @ asij)
            # A^2*S^2
            inv_set[i, 8 + j*13] = np.trace(a @ asijsij)
            # A^2*S*A*S^2
            inv_set[i, 9 + j*13] = np.trace(a @ (a @ sijasijsij))
            # R*A
            inv_set[i, 10 + j*13] = np.trace(rija)
            # R*A*S
            inv_set[i, 11 + j*13] = np.trace(rij_i @ asij)
            # R*A*S^2
            inv_set[i, 12 + j*13] = np.trace(rij_i @ asijsij)
            # R^2*A*S
            inv_set[i, 13 + j*13] = np.trace(rij_i @ (rij_i @ asij))
            # A^2*R*S (cyclic-permutation of anti-symmetric tensor labels of feature 13)
            inv_set[i, 14 + j*13] = np.trace(a @ (a @ rijsij))
            # R^2*A*S^2
            inv_set[i, 15 + j*13] = np.trace(rij_i @ (rij_i @ asijsij))
            # A^2*R*S^2 (cyclic-permutation of anti-symmetric tensor labels of feature 15)
            inv_set[i, 16 + j*13] = np.trace(a @ (a @ rijsijsij))
            # R^2*S*A*S^2
            inv_set[i, 17 + j*13] = np.trace(rij_i @ (rij_i @ sijasijsij))
            # A^2*S*R*S^2 (cyclic-permutation of anti-symmetric tensor labels of feature 17)
            inv_set[i, 18 + j*13] = np.trace(a @ (a @ sijrijsijsij))

        # If both grad1 and grad2 provided, then calculate their interaction invariant features, 15 in total
        if grad1 is not None and grad2 is not None:
            # A1*A2
            inv_set[i, 32] = np.trace(asymm_tensor1_i @ asymm_tensor2_i)
            # A1*A2*S
            inv_set[i, 33] = np.trace(asymm_tensor1_i @ a2sij)
            # A1*A2*S^2
            inv_set[i, 34] = np.trace(asymm_tensor1_i @ a2sijsij)
            # A1^2*A2*S
            inv_set[i, 35] = np.trace(asymm_tensor1_i @ (asymm_tensor1_i @ a2sij))
            # A2^2*A1*S (cyclic-permutation of anti-symmetric tensor labels of feature 35)
            inv_set[i, 36] = np.trace(asymm_tensor2_i @ (asymm_tensor2_i @ a1sij))
            # A1^2*A2*S^2
            inv_set[i, 37] = np.trace(asymm_tensor1_i @ (asymm_tensor1_i @ a2sijsij))
            # A2^2*A1*S^2 (cyclic-permutation of anti-symmetric tensor labels of feature 37)
            inv_set[i, 38] = np.trace(asymm_tensor2_i @ (asymm_tensor2_i @ a1sijsij))
            # A1^2*S*A2*S^2
            inv_set[i, 39] = np.trace(asymm_tensor1_i @ (asymm_tensor1_i @ sija2sijsij))
            # A2^2*S*A1*S^2 (cyclic-permutation of anti-symmetric tensor labels of feature 39)
            inv_set[i, 40] = np.trace(asymm_tensor2_i @ (asymm_tensor2_i @ sija1sijsij))
            # R*A1*A2
            inv_set[i, 41] = np.trace(rij_i @ (asymm_tensor1_i @ asymm_tensor2_i))
            # R*A1*A2*S
            inv_set[i, 42] = np.trace(rij_i @ (asymm_tensor1_i @ a2sij))
            # R*A2*A1*S
            inv_set[i, 43] = np.trace(rij_i @ (asymm_tensor2_i @ a1sij))
            # R*A1*A2*S^2
            inv_set[i, 44] = np.trace(rij_i @ (asymm_tensor1_i @ a2sijsij))
            # R*A2*A1*S^2
            inv_set[i, 45] = np.trace(rij_i @ (asymm_tensor2_i @ a1sijsij))
            # R*A1*S*A2*S^2
            inv_set[i, 46] = np.trace(rij_i @ (asymm_tensor1_i @ sija2sijsij))

        # Gauge progress
        progress = float(i)/(sij.shape[0] + 1)*100.
        if progress >= milestone:
            printf(' %d %%... ', milestone)
            milestone += 10

    print("\n" + str(n_inv) + " invariant features calculated and stored column-wise ")
    return inv_set, labels





