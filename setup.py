#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy >= 1.11', 'scipy', 'pandas', 'scikit-learn',
                    'matplotlib', 'jupyter', 'lap', 'pillow', 'umap-learn',
                    'xarray', 'bokeh', 'tqdm']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='replay_explorer',
    version='0.1.1.dev0',
    license='MIT',
    description=('Explore latent state of replay'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/edeno/replay_explorer',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
