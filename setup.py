import sys
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages

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
        "hide_and_seek_engine.cpp_engine",
        ["src/batched_env.cpp"],
        extra_compile_args=c_args,
        extra_link_args=l_args,
    ),
]

setuptools.setup(
    name="hide_and_seek_engine",
    version="0.2.8",
    author="Your Name",
    description="High-performance batched multi-agent environment",
    packages=find_packages("src", exclude=("build", "dist", "tmp")),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    long_description=(
        open("README.md", encoding="utf-8").read()
        if "README.md" in __import__("os").listdir(".")
        else ""
    ),
    long_description_content_type="text/markdown",
)
