from setuptools import setup, find_packages 

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"
min_python_version = '.'.join(map(str, (3, 6, 2)))

pybind_11_extension = Pybind11Extension("GridGenerator",
                                        ["src/cgridgen.cpp"],
                                        cxx_std=17,
                                        extra_compile_args=["-mavx", "-fopenmp"],
                                        extra_link_args=['-lgomp'],
                                        define_macros=[('VERSION_INFO', __version__),
                                                       ('DEBUG_MODE', 1)])
discretizer_extension = Pybind11Extension("Discretizer",
                                          ["src/discretizer.cpp"],
                                          cxx_std=17)

ext_modules = [ pybind_11_extension, discretizer_extension]

package_name = "ParticleGrid"
long_description = "ParticleGrid is an accelerated grid generation package for generating \
                   3D voxels of coordinates using the error function to integrate over \
                   grid blocks"

if __name__ == '__main__':
  setup(
      name=package_name,
      version=__version__,
      platforms='Linux', 
      description=(""), 
      long_description=long_description,
      ext_modules=ext_modules,
      packages=find_packages(exclude=('tests', 'docs')),
      url="https://github.com/ParticleGrid/ParticleGrid",
      author="ParticleGrid Team",
      python_requires=f'>={min_python_version}',
      tests_require=['pytest'],
      license='BSD-3',
      install_requires=['numpy'],
      keywords='particlegrid voxels machine learning molecules',
      include_package_data=True)
