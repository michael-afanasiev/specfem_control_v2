#!/usr/bin/env python

from setuptools import find_packages, setup

setup_config = dict(
    name='oval_office',
    version='0.1',
    description='A SPECFEM3D_Globe control package',
    author='Michael Afanasiev',
    author_email='michael.afanasiev@erdw.ethz.ch',
    platforms='OS Independent',
    packages=find_packages(),
    entry_points={
        "console_scripts": "oval_office = specfem_control.cli:main"})

if __name__ == "__main__":
    setup(**setup_config)
