# -*- coding: utf-8 -*-
"""Setup file for layg package

Developer install
-----------------
Run following command in prompt/terminal:
    pip install -e .
"""
import setuptools  # type: ignore

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="layg",  # name of package on import
    version="0.0.1",
    description="Learn-as-you-go emulator with error estimation",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/auckland-cosmo/LearnAsYouGoEmulator",
    author="Nathan Musoke, Layne Price",
    author_email="n.musoke@aucland.ac.nz, layne.c.price@gmail.com",
    license="Apache Licence (2.0)",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "numpydoc",
        "torch",
        "scipy",
        "sphinx",
        "sphinx_rtd_theme",
        "emcee>=2,<3",
        "gif==1.0.3",
    ],  # dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.5",
)
