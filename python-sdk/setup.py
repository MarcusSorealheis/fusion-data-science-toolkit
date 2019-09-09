from distutils.core import setup
from setuptools import setup, find_packages


setup(
    name='lucidworks-data-science-integration-toolkit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'requests',
        'configparser',
    ],
    zip_safe=False
)
