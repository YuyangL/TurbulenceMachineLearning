from scipy.optimize.cython_optimize cimport brentq, brenth
from libc cimport math

myargs = {'C0': 1.0, 'C1': 0.7}
