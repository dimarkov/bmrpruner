"""BMRPruner setup configuration."""
from setuptools import find_packages
from setuptools import setup

__version__ = 0.1

JAX_URL = "https://storage.googleapis.com/jax-releases/jax_releases.html"

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(
    name="bmrpruner",
    version=__version__,
    author="BMRPruner Contributors",
    author_email="bmrpruner-dev@example.com",
    description="BMRPruner: Bayesian Model Reduction for Neural Networks - A JaxPruner fork",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/bmrpruner",
    license="Apache 2.0",
    packages=find_packages(
        exclude=["*test.py", "algorithms/*.py"],
    ),
    zip_safe=False,
    install_requires=[
        "chex",
        "flax",
        "jax",
        "jaxlib",
        "optax",
        "numpy",
        "ml-collections",
        "equinox",
        "blrax @ git+https://github.com/dimarkov/blrax.git"
    ],
    dependency_links=[JAX_URL],
    python_requires=">=3.11",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.11",
    ],
)
