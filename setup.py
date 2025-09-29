from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='osrs_utils',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    description='Funciones útiles para programar en OSRS',
    author='Cristobal Wiechert',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
