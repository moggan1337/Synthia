"""
Synthia - Synthetic Biology Simulator
A comprehensive framework for computational biology and synthetic biology research.
"""

from setuptools import setup, find_packages

setup(
    name="synthia",
    version="1.0.0",
    author="Synthia Team",
    author_email="team@synthia.bio",
    description="Synthetic Biology Simulator - Computational Biology Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/moggan1337/Synthia",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "biopython>=1.79",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
