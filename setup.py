from setuptools import setup, find_packages


setup(
    name="nn_magnetics",
    version="1.0",
    packages=find_packages(include=["src"]),
    install_requires=[
        "numpy==2.1.2",
        "matplotlib==3.9.2",
        "torch==2.4.1",
        "magpylib==5.1.0",
        "magpylib-material-response==0.3.0",
    ],
)
