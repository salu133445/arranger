"""Setup script."""
from setuptools import find_packages, setup

setup(
    name="arranger",
    packages=find_packages(),
    install_requires=["muspy>=0.3", "imageio>=2.9", "tensorflow>=2.1,<2.3"],
    extras_require={
        "dev": [
            "black>=19.0",
            "flake8-docstrings>=1.5",
            "flake8>=3.8",
            "mypy>=0.770",
            "pylint>=2.5",
        ],
    },
    python_requires=">=3.6",
)
