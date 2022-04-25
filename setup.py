# install using 'pip install -e .'

from setuptools import setup, find_packages

setup(name='final_model',
      packages=find_packages(),
      install_requires=['torch',
                        'tqdm'],
      version='0.0.1')
