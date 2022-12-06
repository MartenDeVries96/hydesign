# -*- coding: utf-8 -*-
"""
Setup file for hydesign
"""
import os
from setuptools import setup, find_packages
import pkg_resources

repo = os.path.dirname(__file__)
try:
    from git_utils import write_vers
    version = write_vers(vers_file='hydesign/__init__.py', repo=repo, skip_chars=1)
except Exception:
    version = '999'


try:
    from pypandoc import convert_file

    def read_md(f): return convert_file(f, 'rst', format='md')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")

    def read_md(f): return open(f, 'r').read()


setup(name='hydesign',
      version=version,
      description='A tool for design and control of utility scale wind-solar-storage based hybrid power plant.',
      long_description=read_md('README.md'),
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign',
      project_urls={
          'Documentation': 'https://topfarm.pages.windenergy.dtu.dk/hydesign/',
          'Source': 'https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign',
          'Tracker': 'https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/-/issues',
      },
      author='DTU Wind Energy',
      author_email='jumu@dtu.dk',
      license='Proprietary',
      packages=find_packages(),
      package_data={
          'hydesign': [
            'tests/test_files/sm.pkl',
            'look_up_tables/*.nc'
            ],},
      include_package_data=True,
      zip_safe=True)
