import sys
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

# OpenMP flags depend on the compiler
c_args = []
l_args = []

if sys.platform == "win32":
    c_args = ["/openmp", "/O2"]
else:
    c_args = ["-fopenmp", "-O3", "-march=native"]
    l_args = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "hide_and_seek_engine._core",
        ["src/batched_env.cpp"],
        extra_compile_args=c_args,
        extra_link_args=l_args,
    ),
    Pybind11Extension(
        "hide_and_seek_engine._core_global",
        ["src/batched_env_global.cpp"],
        extra_compile_args=c_args,
        extra_link_args=l_args,
    ),
]

setuptools.setup(
    name="hide_and_seek_engine",
    version="0.2.6",
    author="Your Name",
    description="High-performance batched multi-agent environment",
    packages=["hide_and_seek_engine"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
