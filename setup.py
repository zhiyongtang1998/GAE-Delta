"""
GAE-Δ: Graph Autoencoder-Delta

Build instructions:
    1. Compile Cython extensions:  make cython
    2. Compile C++ extensions:     make cpp
    3. Install package:            pip install -e .
"""
import os
import sys
from setuptools import setup, Extension, find_packages

# ---------------------------------------------------------------------------
# Cython extensions
# ---------------------------------------------------------------------------
USE_CYTHON = True
try:
    from Cython.Build import cythonize
    import numpy as np
except ImportError:
    USE_CYTHON = False

ext_modules = []

if USE_CYTHON:
    cython_extensions = [
        Extension(
            "gae_delta.core.graph._correlation",
            sources=["gae_delta/core/graph/_correlation.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "gae_delta.core.graph._adjacency",
            sources=["gae_delta/core/graph/_adjacency.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]
    ext_modules = cythonize(
        cython_extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    )
else:
    # Fallback: try to use pre-compiled C files
    if os.path.exists("gae_delta/core/graph/_correlation.c"):
        ext_modules.append(
            Extension(
                "gae_delta.core.graph._correlation",
                sources=["gae_delta/core/graph/_correlation.c"],
                extra_compile_args=["-O3", "-ffast-math"],
            )
        )
    if os.path.exists("gae_delta/core/graph/_adjacency.c"):
        ext_modules.append(
            Extension(
                "gae_delta.core.graph._adjacency",
                sources=["gae_delta/core/graph/_adjacency.c"],
                extra_compile_args=["-O3"],
            )
        )

setup(
    ext_modules=ext_modules,
)
