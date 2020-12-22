from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("swipes.lib.swipes_utils",["swipes/lib/swipes_utils.py"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),\
            Extension("swipes.isosurface",["swipes/isosurface.py"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),\
            Extension("swipes.lib.kdtree",["swipes/lib/kdtree.py"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]

setup(name='swipes',ext_modules = extensions)
