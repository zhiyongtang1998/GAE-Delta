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
#
# IMPORTANT: these MUST return *fresh* lists per call. Pybind11Extension(cxx_std=)
# mutates the extension's extra_compile_args in place to add -std=c++17; if a
# single shared list object were passed to both the C++ extension and a Cython C
# extension, that C++ flag would leak onto the C build and fail under clang
# ("-std=c++17 not allowed with 'C'").
_IS_MSVC = sys.platform == "win32"


def opt_flags():
    return ["/O2"] if _IS_MSVC else ["-O3"]


def fastmath_flags():
    return ["/O2", "/fp:fast"] if _IS_MSVC else ["-O3", "-ffast-math"]

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
            extra_compile_args=fastmath_flags(),
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "gae_delta.core.graph._adjacency",
            sources=["gae_delta/core/graph/_adjacency.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=opt_flags(),
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
                extra_compile_args=fastmath_flags(),
            )
        )
    if os.path.exists("gae_delta/core/graph/_adjacency.c"):
        ext_modules.append(
            Extension(
                "gae_delta.core.graph._adjacency",
                sources=["gae_delta/core/graph/_adjacency.c"],
                include_dirs=inc,
                extra_compile_args=opt_flags(),
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
            extra_compile_args=opt_flags(),
        )
    )
except ImportError:
    # pybind11 not present at build time: skip the C++ extension. The Python
    # KNN path (scikit-learn) in knn_residual.py is used as a runtime fallback.
    pass

setup(
    ext_modules=ext_modules,
)
