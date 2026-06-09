"""
GAE-Δ: Graph Autoencoder-Delta

All native extensions (Cython PCC/adjacency + C++ KNN) are compiled as part of
``pip install .`` — no separate ``make`` step is required for the Python package.
``make all`` remains available for an in-place editable build during development.
"""
import os
import sys

from setuptools import setup, Extension

# Optimization flags differ between MSVC and GCC/Clang. Hardcoding GCC flags
# (e.g. -O3 / -ffast-math) breaks the build under MSVC on Windows.
#
# We intentionally avoid -march=native: it bakes the build host's instruction
# set into the binary and breaks portable wheels on other machines.
_IS_MSVC = sys.platform == "win32"
OPT_FLAGS = ["/O2"] if _IS_MSVC else ["-O3"]
FASTMATH_FLAGS = (["/O2", "/fp:fast"] if _IS_MSVC else ["-O3", "-ffast-math"])

# ---------------------------------------------------------------------------
# Cython extensions (with pre-generated .c fallback)
# ---------------------------------------------------------------------------
try:
    from Cython.Build import cythonize
    import numpy as np
    USE_CYTHON = True
except ImportError:
    cythonize = None
    np = None
    USE_CYTHON = False

ext_modules = []

if USE_CYTHON:
    cython_extensions = [
        Extension(
            "gae_delta.core.graph._correlation",
            sources=["gae_delta/core/graph/_correlation.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=FASTMATH_FLAGS,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "gae_delta.core.graph._adjacency",
            sources=["gae_delta/core/graph/_adjacency.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=OPT_FLAGS,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]
    ext_modules += cythonize(
        cython_extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    )
else:
    # Fallback: build from the checked-in .c files (no Cython at build time)
    inc = [np.get_include()] if np is not None else []
    if os.path.exists("gae_delta/core/graph/_correlation.c"):
        ext_modules.append(
            Extension(
                "gae_delta.core.graph._correlation",
                sources=["gae_delta/core/graph/_correlation.c"],
                include_dirs=inc,
                extra_compile_args=FASTMATH_FLAGS,
            )
        )
    if os.path.exists("gae_delta/core/graph/_adjacency.c"):
        ext_modules.append(
            Extension(
                "gae_delta.core.graph._adjacency",
                sources=["gae_delta/core/graph/_adjacency.c"],
                include_dirs=inc,
                extra_compile_args=OPT_FLAGS,
            )
        )

# ---------------------------------------------------------------------------
# C++ / pybind11 KNN extension
# ---------------------------------------------------------------------------
try:
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules.append(
        Pybind11Extension(
            "gae_delta.core.shift._knn_ext",
            sources=[
                "csrc/src/knn_regressor.cpp",
                "csrc/src/bindings.cpp",
            ],
            include_dirs=["csrc/include"],
            cxx_std=17,
            extra_compile_args=OPT_FLAGS,
        )
    )
except ImportError:
    # pybind11 not present at build time: skip the C++ extension. The Python
    # KNN path (scikit-learn) in knn_residual.py is used as a runtime fallback.
    pass

setup(
    ext_modules=ext_modules,
)
