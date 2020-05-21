from setuptools import setup, find_packages

setup(
    name='idxhound',
    version='0.1',
    install_requires=[
        'bidict',
        'numpy',
    ],
    packages=find_packages(),
)
