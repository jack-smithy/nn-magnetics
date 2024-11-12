from setuptools import setup, find_packages


setup(
    name="nn_magnetics",
    version="0.0.1",
    packages=find_packages(include=["src"]),
    install_requires=[
        "numpy==2.1.2",
        "matplotlib==3.9.2",
        "torch==2.4.1",
        "magpylib==5.1.0",
        "magpylib-material-response==0.3.0",
        "wandb==0.18.5",
        "scikit-learn==1.5.2",
    ],
)
