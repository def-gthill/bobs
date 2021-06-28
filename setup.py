from setuptools import setup

setup(
    name="bobs",
    version="0.1.1",
    packages=["bobs"],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn==0.24.2",
    ],
)
