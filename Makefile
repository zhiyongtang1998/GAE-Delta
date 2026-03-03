.PHONY: all cython cpp install clean test

PYTHON ?= python
CMAKE ?= cmake
PYBIND11_DIR := $(shell $(PYTHON) -m pybind11 --cmakedir 2>/dev/null)
BUILD_DIR := csrc/build

all: cython cpp install

# ---------- Cython extensions ----------
cython:
	@echo "==> Compiling Cython extensions..."
	$(PYTHON) setup.py build_ext --inplace
	@echo "==> Cython compilation done."

# ---------- C++ / pybind11 extensions ----------
cpp: $(BUILD_DIR)/Makefile
	@echo "==> Building C++ extensions..."
	$(CMAKE) --build $(BUILD_DIR) --config Release -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
	@cp $(BUILD_DIR)/_knn_ext*.so gae_delta/core/shift/ 2>/dev/null || \
	 cp $(BUILD_DIR)/_knn_ext*.dylib gae_delta/core/shift/ 2>/dev/null || \
	 cp $(BUILD_DIR)/Release/_knn_ext*.pyd gae_delta/core/shift/ 2>/dev/null || true
	@echo "==> C++ build done."

$(BUILD_DIR)/Makefile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CMAKE) .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DPYTHON_EXECUTABLE=$(shell which $(PYTHON)) \
		$(if $(PYBIND11_DIR),-Dpybind11_DIR=$(PYBIND11_DIR),)

# ---------- Package install ----------
install:
	@echo "==> Installing gae_delta package..."
	$(PYTHON) -m pip install -e . --no-build-isolation
	@echo "==> Installation done."

# ---------- Tests ----------
test:
	$(PYTHON) -m pytest tests/ -v

# ---------- Cleanup ----------
clean:
	rm -rf $(BUILD_DIR)
	rm -rf build/ dist/ *.egg-info
	rm -f gae_delta/core/graph/_correlation.c gae_delta/core/graph/_correlation*.so
	rm -f gae_delta/core/graph/_adjacency.c gae_delta/core/graph/_adjacency*.so
	rm -f gae_delta/core/shift/_knn_ext*.so gae_delta/core/shift/_knn_ext*.dylib
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
