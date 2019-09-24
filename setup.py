#! /usr/bin/env python
'''
brian2modelfitting setup script
'''
import os
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    raise RuntimeError('Only Python versions >= 3.6 are supported')

def readme():
    with open('README.md') as f:
        return f.read()

version = {}
with open(os.path.join('brian2modelfitting', 'version.py')) as fp:
    exec(fp.read(), version)

# Note that this does not set a version number explicitly, but automatically
# figures out a version based on git tags
setup(name='brian2modelfitting',
      url='https://github.com/brian-team/brian2modelfitting',
      version=version['version'],
      packages=find_packages(),
      # package_data={},
      install_requires=['numpy',
                        'brian2>=2.2',
                        'setuptools',
                        'nevergrad',
                        'scikit-optimize',
                        'efel',
                        'tqdm'
                        ],
      provides=['brian2modelfitting'],
      extras_require={'test': ['pytest'],
                      'docs': ['sphinx>=1.8']},
      use_2to3=False,
      zip_safe=False,
      description='Modelfitting Toolbox for the Brian 2 simulator',
      long_description=readme(),
      author='Aleksandra Teska, Marcel Stimberg, Romain Brette, Dan Goodman',
      author_email='team@briansimulator.org',
      license='CeCILL-2.1',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      keywords='model fitting computational neuroscience'
      )
