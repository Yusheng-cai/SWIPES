from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("swipes.lib.swipes_utils",["swipes/lib/swipes_utils.pyx"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])] 

setup(name='swipes',ext_modules = cythonize(extensions,compiler_directives={'language_level' : "3"}))
