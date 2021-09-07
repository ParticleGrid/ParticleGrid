from setuptools import setup 

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"
min_python_version = '.'.join(map(str, (3, 6, 2)))

pybind_11_extension = Pybind11Extension("ParticleGrid",
                                        ["src/main.cpp"],
                                        cxx_std=17,
                                        extra_compile_args=["-mavx", "-fopenmp"],
                                        extra_link_args=['-lgomp'],
                                        define_macros=[('VERSION_INFO', __version__)])

ext_modules = [pybind_11_extension]

package_name = "ParticleGrid"
long_description = "ParticleGrid is an accelerated grid generation package for generating \
                   3D voxels of coordinates using the error function to integrate over \
                   grid blocks"

if __name__ == '__main__':
  setup(
      name=package_name,
      version=__version__, 
      description=(""), 
      long_description=long_description, 
      ext_modules=ext_modules,
      cmdclass={"build_ext": build_ext},
      url="",
      author="ParticleGrid Team",
      python_requires=f'>={min_python_version}',
      license='BSD-3',
      keywords='particlegrid voxels machine learning molecules')