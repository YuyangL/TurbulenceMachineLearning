from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

system = 'linux'  # 'windows', 'linux'

# file_name = 'PostProcess_EnergySpectrum'
# file_name = 'PostProcess_AnisotropyTensor'
file_name = 'Tensor'

"""
python3 SetupCython.py build_ext --inplace
"""

if system == 'linux':
    ext_modules = [Extension(file_name,
                             [file_name + '.pyx'],
                             libraries=["m"],  # Unix-like specific
                             extra_compile_args=['-ffast-math', '-O3', '-fopenmp'],
                             extra_link_args=['-fopenmp'])]
else:
    ext_modules = [Extension(file_name,
                             [file_name + '.pyx'],
                             extra_compile_args=['/openmp'],
                             extra_link_args=['/openmp'])]

setup(name=file_name,
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
      include_dirs=[numpy.get_include()])
