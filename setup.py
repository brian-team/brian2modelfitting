#! /usr/bin/env python
"""brian2modelfitting setup script"""

import os

from setuptools import setup, find_packages


version = {}
with open(os.path.join('brian2modelfitting', 'version.py')) as fp:
    exec(fp.read(), version)

with open('README.md') as f:
    long_description = f.read()

setup(name='brian2modelfitting',
      url='https://github.com/brian-team/brian2modelfitting',
      version=version['version'],
      packages=find_packages(),
      install_requires=['numpy',
                        'brian2>=2.2',
                        'setuptools',
                        'nevergrad>=0.2,<=0.3',
                        'scikit-optimize',
                        'tqdm',
                        ],
      provides=['brian2modelfitting'],
      extras_require={'test': ['pytest'],
                      'docs': ['sphinx>=1.8'],
                      'full': ['efel', 'lmfit', 'sbi'],
                      },
      python_requires='>=3.6',
      use_2to3=False,
      zip_safe=False,
      description='Modelfitting Toolbox for the Brian 2 simulator',
      long_description=long_description,
      author='Aleksandra Teska, Marcel Stimberg, Romain Brette, Dan Goodman',
      author_email='team@briansimulator.org',
      license='CeCILL-2.1',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, '
          'version 2.1 (CeCILL-2.1)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'],
      keywords='model fitting computational neuroscience',
      )
