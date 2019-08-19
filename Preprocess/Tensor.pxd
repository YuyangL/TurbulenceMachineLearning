# cython: language_level = 3str
# cython: embedsignature = True
cimport numpy as np

cpdef tuple processReynoldsStress(np.ndarray stress_tensor, bint make_anisotropic=*, int realization_iter=*, bint to_old_grid_shape=*)

cpdef tuple getBarycentricMapData(np.ndarray eigval, bint optimize_cmap=*, double c_offset=*, double c_exp=*, bint to_old_grid_shape=*)

cpdef np.ndarray expandSymmetricTensor(np.ndarray tensor)

cpdef np.ndarray contractSymmetricTensor(np.ndarray tensor)

cpdef tuple getStrainAndRotationRateTensor(np.ndarray grad_u, np.ndarray tke=*,  np.ndarray eps=*, double cap=*)

cpdef np.ndarray[np.float_t, ndim=3] getInvariantBases(np.ndarray sij, np.ndarray rij, bint quadratic_only=*, bint is_scale=*, bint zero_trace=*)

cpdef np.ndarray makeRealizable(np.ndarray bij)


# -----------------------------------------------------
# Supporting Functions, Not Intended to Be Called From Python
# -----------------------------------------------------
cdef np.ndarray[np.float_t, ndim=2] _makeRealizable(np.ndarray[np.float_t, ndim=2] bij)

cdef np.ndarray[np.float_t, ndim=2] _mapVectorToAntisymmetricTensor(np.ndarray[np.float_t, ndim=2] vec, np.ndarray scaler=*)




# --------------------------------------------------------
# [DEPRECATED] But Still Usable
# --------------------------------------------------------
cpdef tuple convertTensorTo2D(np.ndarray tensor, bint infer_stress=*)
