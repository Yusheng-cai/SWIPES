from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


setup(name='swipes',\
        version='0.0.1',\
        author='Yusheng Cai',\
        packages=['swipes','swipes.lib'])
