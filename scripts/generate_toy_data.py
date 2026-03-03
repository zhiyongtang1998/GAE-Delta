#!/usr/bin/env python
"""Generate synthetic toy data for testing the GAE-Δ pipeline.

Creates a small HDF5 dataset and FI network file matching the expected schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def generate_toy_dataset(
    output_dir: str = "data/example",
    n_genes: int = 100,
    n_patients: int = 50,
    n_fi_edges: int = 200,
    seed: int = 42,
) -> None:
    """Generate synthetic multi-omics data and FI network."""
    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate gene symbols
    gene_symbols = np.array([f"GENE{i:04d}" for i in range(n_genes)])

    # Generate patient IDs
    patient_ids = np.array([f"TCGA-XX-{i:04d}" for i in range(n_patients)])

    # Generate OS days: good-outcome patients have longer OS
    os_days = np.concatenate([
        rng.exponential(800, size=n_patients // 2),   # good
        rng.exponential(300, size=n_patients - n_patients // 2),  # poor
    ]).astype(np.float32)
    os_status = rng.binomial(1, 0.7, size=n_patients).astype(np.int32)

    # Shuffle patients
    perm = rng.permutation(n_patients)
    os_days = os_days[perm]
    os_status = os_status[perm]

    # Generate omics data with subtle signal
    median_os = np.median(os_days)
    labels = (os_days <= median_os).astype(int)

    def _generate_omics(label_effect: float = 0.3) -> np.ndarray:
        base = rng.randn(n_patients, n_genes).astype(np.float32)
        # Add subtle outcome-dependent signal to first 20 genes
        for i in range(n_patients):
            if labels[i] == 1:  # poor outcome
                base[i, :20] += label_effect * rng.randn(20)
        return base

    rna_data = _generate_omics(0.3)
    meth_data = np.clip(_generate_omics(0.2) * 0.1 + 0.5, 0, 1).astype(np.float32)
    cnv_data = _generate_omics(0.15)

    # Generate FI network edges (random pairs)
    fi_edges = set()
    while len(fi_edges) < n_fi_edges:
        g1, g2 = sorted(rng.choice(n_genes, 2, replace=False))
        fi_edges.add((g1, g2))
    fi_edge_array = np.array(sorted(fi_edges), dtype=np.int64)

    # Write HDF5
    hdf5_path = output_dir / "toy_demo.h5"
    with h5py.File(hdf5_path, "w") as f:
        # RNA
        rna_grp = f.create_group("rna")
        rna_grp.create_dataset("expression", data=rna_data)
        rna_grp.create_dataset("gene_symbols", data=gene_symbols.astype("S"))
        rna_grp.create_dataset("patient_ids", data=patient_ids.astype("S"))

        # Methylation
        meth_grp = f.create_group("methylation")
        meth_grp.create_dataset("beta_values", data=meth_data)
        meth_grp.create_dataset("gene_symbols", data=gene_symbols.astype("S"))
        meth_grp.create_dataset("patient_ids", data=patient_ids.astype("S"))

        # CNV
        cnv_grp = f.create_group("cnv")
        cnv_grp.create_dataset("copy_ratios", data=cnv_data)
        cnv_grp.create_dataset("gene_symbols", data=gene_symbols.astype("S"))
        cnv_grp.create_dataset("patient_ids", data=patient_ids.astype("S"))

        # Clinical
        clin_grp = f.create_group("clinical")
        clin_grp.create_dataset("os_days", data=os_days)
        clin_grp.create_dataset("os_status", data=os_status)
        clin_grp.create_dataset("patient_ids", data=patient_ids.astype("S"))

        # Metadata
        meta_grp = f.create_group("meta")
        meta_grp.create_dataset("gene_universe", data=gene_symbols.astype("S"))
        meta_grp.create_dataset("fi_edge_list", data=fi_edge_array)

    print(f"HDF5 dataset saved to {hdf5_path}")
    print(f"  Patients: {n_patients}, Genes: {n_genes}")
    print(f"  Modalities: rna, methylation, cnv")

    # Write FI network as text file
    fi_path = output_dir / "toy_fi_network.txt"
    with open(fi_path, "w") as f:
        f.write("src\tdest\n")
        for g1, g2 in sorted(fi_edges):
            f.write(f"{gene_symbols[g1]}\t{gene_symbols[g2]}\n")

    print(f"FI network saved to {fi_path} ({n_fi_edges} edges)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate toy data for GAE-Δ")
    parser.add_argument("--output-dir", default="data/example")
    parser.add_argument("--n-genes", type=int, default=100)
    parser.add_argument("--n-patients", type=int, default=50)
    parser.add_argument("--n-fi-edges", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_toy_dataset(
        output_dir=args.output_dir,
        n_genes=args.n_genes,
        n_patients=args.n_patients,
        n_fi_edges=args.n_fi_edges,
        seed=args.seed,
    )
