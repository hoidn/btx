from setuptools import setup, find_packages

setup(
    name="btx",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pytest",
        "matplotlib"
    ],
)
