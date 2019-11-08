# cython: language_level = 3str
# cython: embedsignature = True
cimport numpy as np

# Type aliases
ctypedef np.ndarray nparr
ctypedef np.float_t flt
ctypedef unsigned int unsignint

cpdef nparr[flt, ndim=3] computeDivDevR_2D(nparr bij_mesh, nparr[flt, ndim=2] tke_mesh, nparr[flt, ndim=3] ddev_rij_dz_mesh, double dx=*, double dy=*)
