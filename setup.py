"""Setup script for poker_chip package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="poker_chip",
    version="0.1.0",
    author="Corrado Maurini",
    description="Phase-field fracture simulation for poker chip test",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmaurini/poker-chip",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pyyaml>=5.4",
        "omegaconf>=2.3",
        "matplotlib>=3.5",
        "scienceplots>=2.0",
        "scifem>=0.16.1",
        # Note: dolfinx, petsc4py, mpi4py should be installed via conda
        # conda install -c conda-forge fenics-dolfinx mpich
    ],
    extras_require={
        "workflow": ["snakemake>=8.0"],
        "dev": ["pytest>=7.0", "black>=22.0", "ruff>=0.1"],
    },
    entry_points={
        "console_scripts": [
            "poker-chip=poker_chip.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
