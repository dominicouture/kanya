#!/usr/bin/env python

"""
setup.py: setup file for the kanya Python package. Inspired by:
https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
"""

import codecs
import os
import re
from setuptools import find_packages, setup

# Project specifics
name = 'kanya'
packages = find_packages(where='')
meta_path = os.path.join('kanya', '__init__.py')
project_urls = {'Source': 'https://github.com/dominicouture/kanya'}
classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python'
]
install_requires = [
    'numpy',
    'scipy',
    'matplotlib',
    'astropy',
    'galpy',
    'sklearn',
    'pandas'
]
setup_requires = [
    'setuptools>=40.6.0',
    'setuptools_scm',
    'wheel'
]
extra_require = {
    'extras': ['h5py'],
    'tests': ['pdb']
}

# Current file real location
here = os.path.dirname(os.path.realpath(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'rb', 'utf-8') as f:
        return f.read()

def find_meta(meta, meta_file=read(meta_path)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError('Unable to find __{meta}__ string.'.format(meta=meta))

if __name__ == '__main__':
    setup(
        name=name,
        use_scm_version={
            'write_to': os.path.join(name, '{0}_version.py'.format(name)),
            'write_to_template': '__version__ = "{version}"\n'
        },
        author=find_meta('author'),
        author_email=find_meta('email'),
        maintainer=find_meta('author'),
        maintainer_email=find_meta('email'),
        version=find_meta('version'),
        url=find_meta('uri'),
        project_urls=project_urls,
        license=find_meta('license'),
        description=find_meta('description'),
        long_description=read('README.rst'),
        long_description_content_type='text/x-rst',
        packages=packages,
        package_dir={'': ''},
        include_package_data=True,
        install_requires=install_requires,
        setup_requires=setup_requires,
        extras_require=extra_require,
        classifiers=classifiers,
        zip_safe=False,
        options={'bdist_wheel': {'universal': '1'}})
