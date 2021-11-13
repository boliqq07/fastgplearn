#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/8/2 15:47
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='fastgplearn',
    version='0.0.14',
    keywords=['genetic programming","symbolic learning'],
    description='Fast fitting formula.',
    install_requires=['pandas', 'numpy', 'scipy', 'scikit-learn', 'joblib', 'deprecated',
                      "mgetool"],
    include_package_data=True,
    author='wangchangxin',
    author_email='986798607@qq.com',
    python_requires='>=3.6',
    url='https://github.com/boliqq07/fastgplearn',
    maintainer='wangchangxin',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],

    packages=find_packages(
        exclude=["test", "*.test", "*.*.test", "*.*.*.test","*script.py",
                "*/*/*script.py"
                "*/*script.py",
                "test*", "*.test*", "*.*.test*", "*.*.*.test*", "Instances", "Instance*"],
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={'console_scripts': ['fastgplearn = fastgplearn.cli.main:main']}
)
