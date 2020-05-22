from setuptools import setup, find_packages

with open('README.rst') as fp:
    long_description = fp.read()

setup(
    name='idxhound',
    version='0.1',
    install_requires=[
        'bidict',
        'numpy',
    ],
    packages=find_packages(),
    url='https://github.com/tillahoffmann/idxhound',
    long_description_content_type="text/x-rst",
    long_description=long_description,
)
