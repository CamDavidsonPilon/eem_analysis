# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages


def filepath(fname):
    return os.path.join(os.path.dirname(__file__), fname)


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="eem_conv_autoencoder",
    version="0.0.1",
    author="Cameron Davidson-Pilon",
    author_email="cam.davidson.pilon@gmail.com",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=requirements,
    package_data={"eem_conv_autoencoder": ["../README.md", "../LICENSE"]},
)