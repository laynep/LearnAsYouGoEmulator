# -*- coding: utf-8 -*-
"""Setup file for learn_as_you_go package

Developer install
-----------------
Run following command in prompt/terminal:
    pip install -e .
"""
import setuptools  # type: ignore

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="learn_as_you_go",  # name of package on import
    version="0.0.1.dev",
    description="Learn-as-you-go emulator with error estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laynep/LearnAsYouGoEmulator",
    author="Layne Price",  # author(s)
    author_email="layne.c.price@gmail.com",  # email
    # license='MIT',  # licensing
    packages=setuptools.find_packages(),
    install_requires=["matplotlib", "numpy", "scipy", "emcee>=2,<3"],  # dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: ",  # TODO: Licence
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.5",
)
