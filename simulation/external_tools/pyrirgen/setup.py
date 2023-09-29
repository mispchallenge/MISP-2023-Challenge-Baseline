import os

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

# python3 setup.py build_ext --inplace

pyrirgen_ext = Extension('pyrirgen', ['pyrirgen/pyx/pyrirgen.pyx'], language='c++')
compiler_directives = dict(language_level=3, enbedsignature=True, c_string_encoding='default', c_string_type='str')

setup(name = 'pyrirgen',
      author = 'ehabets, Marvin182, ty274, and yoshipon', 
      description = 'Cython-based image method implementation for room impulse response generation',
      license='GPL',
      packages = setuptools.find_packages(exclude=('examples',)),
      ext_modules=cythonize([pyrirgen_ext], compiler_directives=compiler_directives)
