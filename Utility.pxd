# cython: language_level = 3str
# cython: embedsignature = True
cimport numpy as np

cpdef tuple interpolateGridData(np.ndarray[np.float_t] x, np.ndarray[np.float_t] y, np.ndarray val, np.ndarray z=*,
                                double mesh_target=*, str interp=*, double fill_val=*)


cpdef tuple collapseMeshGridFeatures(np.ndarray meshgrid, bint infer_matrix_form=*, tuple matrix_shape=*, bint collapse_matrix=*)


cpdef np.ndarray reverseOldGridShape(np.ndarray arr, tuple shape_old, bint infer_matrix_form=*, tuple matrix_shape=*)


cpdef np.ndarray rotateData(np.ndarray ndarr, double anglex=*, double angley=*, double anglez=*, tuple matrix_shape=*)
