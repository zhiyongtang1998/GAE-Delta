<div align="center">

# 🧬 GAE-Δ

**Phenotype-Specific Gene Role Shifts in Multi-Omics Data via Graph Autoencoder Embedding Differences**

*When genes don't change expression — but change who they talk to*

[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG 2.3](https://img.shields.io/badge/PyG-2.3-3C2179?logo=pyg&logoColor=white)](https://pyg.org/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

</div>

---

## 💡 What is GAE-Δ?

Most multi-omics methods ask: *"Which genes are differentially expressed between phenotypes?"*

**GAE-Δ asks a different question:** *"Which genes change their network roles between phenotypic groups?"*

We train separate graph autoencoders on **group-specific** gene interaction networks, then compute **embedding differences** — capturing how each gene's functional context reorganizes across phenotypic conditions. This isn't about expression changes; it's about **network rewiring**.

GAE-Δ is a general framework applicable to any binary stratification of multi-omics cohorts — survival outcomes, treatment response, disease subtypes, or any clinically meaningful grouping.

<div align="center">

```
  ┌─────────────┐         ┌─────────────┐
  │   Group A    │         │   Group B    │
  │  Gene Graph  │         │  Gene Graph  │
  └──────┬──────┘         └──────┬──────┘
         │                       │
    ┌────▼────┐             ┌────▼────┐
    │  GAE    │             │  GAE    │
    │ Encoder │             │ Encoder │
    └────┬────┘             └────┬────┘
         │                       │
      z_A ∈ ℝ^d              z_B ∈ ℝ^d
         │                       │
         └──────────┬────────────┘
                    │
              Δz = z_B − z_A
                    │
              ┌─────▼─────┐
              │    KNN     │
              │  Residual  │
              └─────┬─────┘
                    │
         ε_g ∈ ℝ^d (per gene, per omics)
                    │
         ┌──────────┼──────────┐
         RNA      Meth       CNV
         │         │          │
         └─────────┼──────────┘
                   │
           s_g ∈ ℝ^(3d) (fused)
                   │
           ┌───────▼───────┐
           │ Isolation      │
           │ Forest (top-N) │
           └───────┬───────┘
                   │
           ┌───────▼───────┐
           │ Patient Embed  │
           │ + MLP → 0/1    │
           └───────────────┘
```

</div>

## 🔬 Key Features

- **Phenotype-specific graph learning** — separate GAEs for each group, capturing group-specific gene interaction topology
- **Embedding difference as biomarker** — gene-level network reorganization, not just expression fold-change
- **Multi-omics late fusion** — RNA-seq, DNA methylation, CNV integrated at the embedding-shift level
- **KNN residual correction** — removes globally smooth trends, highlights genes with atypical rewiring
- **Isolation Forest gene selection** — unsupervised anomaly detection on fused shift space
- **Cython + C++ accelerated** — performance-critical PCC computation and KNN in compiled extensions

## ⚡ Quick Start

### Prerequisites

| Dependency | Version |
|:-----------|:--------|
| Python | 3.9 |
| CUDA | 11.7 |
| CMake | ≥ 3.18 |
| C++ Compiler | C++17 support |

### Build & Install

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate gae-delta

# 2. Build native extensions + install
make all
```

> **Note:** `make all` compiles Cython extensions (PCC, adjacency), C++ KNN extension (pybind11), and installs the package. Requires CMake and a C++17 compiler.

### Run with toy data

```bash
python -m gae_delta.pipeline.runner \
    data.hdf5_path=data/example/toy_demo.h5 \
    data.fi_network_path=data/example/toy_fi_network.txt
```

### Run with real TCGA data

```bash
python -m gae_delta.pipeline.runner \
    data.hdf5_path=/path/to/your/cancer_data.h5 \
    data.fi_network_path=/path/to/FI_network.txt \
    n_folds=10 \
    device=auto
```

## 📂 Project Structure

```
GAE-Delta/
├── csrc/                    # C++ KNN extension (pybind11)
├── gae_delta/
│   ├── core/
│   │   ├── graph/           # Cython PCC + FI-constrained graph builder
│   │   ├── model/           # GCN Encoder → GAE → MLP
│   │   ├── shift/           # Δz computation + KNN residual + fusion
│   │   └── selection/       # Isolation Forest gene ranking
│   ├── data/                # HDF5 loader + omics preprocessing
│   ├── pipeline/            # 5-stage pipeline + Hydra runner
│   └── evaluation/          # 10-fold CV + metrics
├── configs/                 # Hydra YAML configs
├── data/example/            # Toy demo dataset
└── tests/
```

## 📊 Data Format

Input data must follow this HDF5 schema:

```
dataset.h5
├── rna/expression           float32  (n_patients × n_genes)
├── methylation/beta_values  float32  (n_patients × n_genes)
├── cnv/copy_ratios          float32  (n_patients × n_genes)
├── clinical/os_days         float32  (n_patients,)
├── clinical/os_status       int32    (n_patients,)    # 1=deceased
└── meta/gene_universe       str      (n_common_genes,)
```

Each group also contains `gene_symbols` and `patient_ids` arrays. See `gae_delta/data/tcga/loader.py` for the full specification.

## ⚙️ Configuration

GAE-Δ uses [Hydra](https://hydra.cc/) for configuration management. Override any parameter from the command line:

```bash
# Custom hyperparameters
python -m gae_delta.pipeline.runner \
    model.gae.encoder.out_channels=32 \
    model.mlp.architecture.hidden_dim=128 \
    experiment.graph.pcc_threshold=0.4 \
    n_top_genes=200

# Environment variables also work
export GAE_DELTA_DATA_PATH=/data/tcga/lihc.h5
python -m gae_delta.pipeline.runner
```

## 🧪 Testing

```bash
make test
```

## 📜 Citation

If you use GAE-Δ in your research, please cite:

```bibtex
@article{tang2026gaedelta,
  title={GAE-$\Delta$: Phenotype-Specific Gene Role Shifts in Multi-Omics
         Data via Graph Autoencoder Embedding Differences},
  author={Tang, Zhiyong and Chen, Zhe and Chen, Mengting and Ewing, Rob
          and Niranjan, Mahesan and Ennis, Sarah and Wang, Yihua},
  journal={Bioinformatics},
  year={2026},
  publisher={Oxford University Press}
}
```

## 📄 License

This project is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). You may share the material with attribution, but commercial use and derivative works are not permitted.
