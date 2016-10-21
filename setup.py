from Cython.Distutils import build_ext
from numpy.distutils.misc_util import Configuration
from setuptools import setup, Extension

import numpy

setup(
    maintainer='Jay Wang',
    name='SFlibFM',
    packages=['SFlibFM'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("pyfm_fast", ["pyfm_fast.pyx"],
                             libraries=["m"],
                             include_dirs=[numpy.get_include()])]
)