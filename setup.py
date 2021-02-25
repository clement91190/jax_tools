from setuptools import setup


long_description = '''
Python package for numerai
'''

setup(
    name='jax_tools',
    version='0.0.1',
    description='lib build on top of JAX to make it easier for code reuse and fast implementation',
    long_description=long_description,
    author='Clement Jambou',
    packages=["jax_tools"],
    install_requires=[]
)
