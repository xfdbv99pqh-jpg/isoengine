"""
Isomorphic Math Engine - Setup
"""

from setuptools import setup, find_packages

setup(
    name="isomorphic_math",
    version="33.0",
    description="Geometric embeddings for mathematical equations",
    author="Big J + Claude",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "neural": ["torch>=1.9.0"],
        "full": ["torch>=1.9.0", "matplotlib", "scikit-learn"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
    ],
)
