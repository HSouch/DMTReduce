# DMTReduce Pipeline

from distutils.core import setup
from setuptools import find_packages
import dmtreduce

setup(
    name='tbridge',
    packages=find_packages(),
    version=dmtreduce.__version__,
    license='MIT License',
    description = 'Spectra reduction pipeline for LRIS broad slit and DMT.',
    author="Harrison Souchereau",
    author_email='harrison.souchereau@yale.edu',
    keywords='',
    scripts=[],
    install_requires=[

    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

    ],
)