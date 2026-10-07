#i can build my ML Algorithms as package using pypi

from setuptools import setup, find_packages
from typing import List  # noqa: F401

def get_requirements(file_path: str) -> list[str]:
    # Opens the file for reading using UTF-8 encoding.
    # "f" refers to the opened file.
    # "with" automatically closes it when this block ends.
    with open(file_path, encoding="utf-8") as f:
        # read() reads the whole file as one string.
        # Example: "numpy\npandas\nscikit-learn\n"
        # splitlines() splits it into a list, removing line breaks.
        # Result: ["numpy", "pandas", "scikit-learn"]
        requirements = f.read().splitlines()

    return requirements
''' '''

setup(
    name='ml_algorithms_package',
    version='0.1.0',
    description='A package containing various machine learning algorithms',
    author='Mahim',
    packages=find_packages(),#find_packages() looks for package folders containing __init__.py. With your current structure, it finds
    install_requires=get_requirements('requirements.txt')
)