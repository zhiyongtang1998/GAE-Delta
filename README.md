# GAE-Δ

Graph Autoencoder-based framework for characterizing outcome-specific gene role shifts in multi-omics cancer data.

## Requirements

- Python 3.9
- CUDA 11.7
- CMake ≥ 3.18
- C++17-compatible compiler

## Installation

```bash
conda env create -f environment.yml
conda activate gae-delta
make all
```

## Data

Input data must be provided as HDF5 files. A toy example is included under `data/example/`.

## Usage

```bash
python -m gae_delta.pipeline.runner data.hdf5_path=data/example/toy_demo.h5 data.fi_network_path=data/example/toy_fi_network.txt
```

## License

CC BY-NC-ND 4.0
