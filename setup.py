"""Setup script."""
from setuptools import find_packages, setup

setup(
    name="arranger",
    packages=find_packages(),
    install_requires=["muspy>=0.3", "imageio>=2.9"],
    python_requires=">=3.6",
)
