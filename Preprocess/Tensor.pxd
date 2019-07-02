# cython: language_level = 3str
# cython: embedsignature = True
cimport numpy as np

cpdef tuple processReynoldsStress(np.ndarray stress_tensor, bint make_anisotropic=*, int realization_iter=*, bint to_old_grid_shape=*)


cpdef tuple getBarycentricMapData(np.ndarray eigval, bint optimize_cmap=*, double c_offset=*, double c_exp=*, bint to_old_grid_shape=*)


# -----------------------------------------------------
# Supporting Functions, Not Intended to Be Called From Python
# -----------------------------------------------------
cdef np.ndarray[np.float_t, ndim=2] _makeRealizable(np.ndarray[np.float_t, ndim=2] labels)

cdef np.ndarray[np.float_t, ndim=3] _mapVectorToAntisymmetricTensor(np.ndarray[np.float_t, ndim=2] vec, np.ndarray scaler=*)




# --------------------------------------------------------
# [DEPRECATED] But Still Usable
# --------------------------------------------------------
cpdef tuple convertTensorTo2D(np.ndarray tensor, bint infer_stress=*)
