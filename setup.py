from setuptools import setup
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(here, 'README.md'), 'r') as mdFile:
    long_description = mdFile.read()

setup(
    name='nvsbenchmark', 
    version='0.0.1', 
    # package_dir={'': 'picutils'}, 
    # packages=setuptools.find_packages(), 
    packages=['nvsbenchmark', 'nvsbenchmark.data', 'nvsbenchmark.score'], 
    url='', 
    license='MIT', 
    author='NetVideoGroup', 
    author_email='', 
    description='a novel view synthesis benchmark utils', 
    long_description=long_description, 
    long_description_content_type="text/markdown",
    python_requires=">=3.8, <4", 
    install_requires=[
        'numpy>=1.22.3', 
        'opencv-python>=4.5.5.64', 
        'plyfile>=0.7.4', 
        'torch>=1.11.0', 
        'sklearn', 
        'open3d>=0.15.2'
    ]
)