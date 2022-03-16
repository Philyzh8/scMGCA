#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/6 15:41
# @Author  : Zhuohan Yu
# @Site    : 
# @File    : setup.py
# @Software: PyCharm
# @Description:

from setuptools import setup

setup(
    name='scMGCA',
    version='1.0.0',
    description='Topological Identification and Interpretation for High-throughput Single-cell Gene Regulation Elucidation across Multiple Platforms using scMGCA',
    author='Zhuohan Yu',
    author_email="zhuohan20@mails.jlu.edu.cn",
    packages=['scMGCA'],
    url='https://github.com/Philyzh8/scMGCA',
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Programming Language :: Python :: 3.5'],
)
