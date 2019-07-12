# cython: language_level = 3str
# cython: embedsignature = True
cimport numpy as np

cpdef tuple getInvariantFeatureSet(np.ndarray sij, np.ndarray rij,
                                   np.ndarray grad_k=*, np.ndarray grad_p=*,
                                   np.ndarray k=*, np.ndarray eps=*,
                                   np.ndarray u=*, np.ndarray grad_u=*, double rho=*)




# -----------------------------------------------------
# Supporting Functions, Not Intended to Be Called From Python
# -----------------------------------------------------
cdef tuple _getInvaraintFeatureSet(np.ndarray[np.float_t, ndim=2] sij, np.ndarray[np.float_t, ndim=2] rij,
                                        np.ndarray grad1=*, np.ndarray grad2=*, np.ndarray grad1_scaler=*, np.ndarray grad2_scaler=*)


