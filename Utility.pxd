# cython: language_level = 3str
# cython: embedsignature = True
cimport numpy as np

# Type aliases
ctypedef np.ndarray nparr
ctypedef np.float_t flt
ctypedef unsigned int unsignint
ctypedef np.uint8_t uint8

cpdef tuple interpolateGridData(np.ndarray[np.float_t] x, np.ndarray[np.float_t] y, np.ndarray val, np.ndarray z=*,
                                tuple xlim=*, tuple ylim=*, tuple zlim=*,
                                double mesh_target=*, str interp=*, double fill_val=*)

cpdef tuple collapseMeshGridFeatures(np.ndarray meshgrid, bint infer_matrix_form=*, tuple matrix_shape=*, bint collapse_matrix=*)

cpdef np.ndarray reverseOldGridShape(np.ndarray arr, tuple shape_old, bint infer_matrix_form=*, tuple matrix_shape=*)

cpdef np.ndarray rotateData(np.ndarray ndarr, double anglex=*, double angley=*, double anglez=*, tuple matrix_shape=*)

cpdef tuple fieldSpatialSmoothing(np.ndarray[np.float_t, ndim=2] val,
                                       np.ndarray[np.float_t] x, np.ndarray[np.float_t] y, np.ndarray z=*,
                                       tuple val_bnd=*, bint is_bij=*, double bij_bnd_multiplier=*,
                                       tuple xlim=*, tuple ylim=*, tuple zlim=*,
                                       double mesh_target=*)

cpdef np.ndarray gaussianFilter(np.ndarray array, double sigma=*)

cdef uint8[:] _confineFieldDomain3D(nparr[flt, ndim=2] cc,
                                double box_l, double box_w, double box_h, tuple box_orig=*, double rot_z=*)

# def timer(func)
