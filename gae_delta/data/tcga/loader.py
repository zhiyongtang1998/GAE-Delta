"""HDF5-based multi-omics data loader.

Expected HDF5 schema:
    {dataset}.h5
    ├── rna/
    │   ├── expression    float32  (n_patients × n_genes)
    │   ├── gene_symbols  bytes    (n_genes,)
    │   └── patient_ids   bytes    (n_patients,)
    ├── methylation/
    │   ├── beta_values   float32  (n_patients × n_genes)
    │   ├── gene_symbols  bytes    (n_genes,)
    │   └── patient_ids   bytes    (n_patients,)
    ├── cnv/
    │   ├── copy_ratios   float32  (n_patients × n_genes)
    │   ├── gene_symbols  bytes    (n_genes,)
    │   └── patient_ids   bytes    (n_patients,)
    ├── clinical/
    │   ├── os_days       float32  (n_patients,)
    │   ├── os_status     int32    (n_patients,)  # 1=deceased, 0=censored
    │   └── patient_ids   bytes    (n_patients,)
    └── meta/
        ├── gene_universe bytes    (n_common_genes,)
        └── fi_edge_list  int32    (n_edges × 2)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_MODALITY_KEYS = {
    "rna": {"data": "expression", "genes": "gene_symbols", "patients": "patient_ids"},
    "methylation": {"data": "beta_values", "genes": "gene_symbols", "patients": "patient_ids"},
    "cnv": {"data": "copy_ratios", "genes": "gene_symbols", "patients": "patient_ids"},
}


@dataclass
class ModalityData:
    """Container for a single omics modality."""
    name: str
    data: np.ndarray  # (n_patients, n_genes)
    gene_symbols: np.ndarray  # (n_genes,)
    patient_ids: np.ndarray  # (n_patients,)

    @property
    def n_patients(self) -> int:
        return self.data.shape[0]

    @property
    def n_genes(self) -> int:
        return self.data.shape[1]


@dataclass
class ClinicalData:
    """Container for clinical outcome data."""
    patient_ids: np.ndarray
    os_days: np.ndarray
    os_status: np.ndarray  # 1=deceased, 0=censored


@dataclass
class MultiOmicsDataset:
    """Complete multi-omics dataset loaded from HDF5."""
    modalities: Dict[str, ModalityData]
    clinical: ClinicalData
    gene_universe: np.ndarray  # common genes across all modalities
    fi_edge_list: Optional[np.ndarray] = None  # (n_edges, 2) indices into gene_universe

    @property
    def n_genes(self) -> int:
        return len(self.gene_universe)

    def get_modality(self, name: str) -> ModalityData:
        if name not in self.modalities:
            raise KeyError(f"Modality '{name}' not found. Available: {list(self.modalities)}")
        return self.modalities[name]

    def get_patient_indices(self, patient_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """Map a subset of patient IDs to row indices in each modality."""
        result = {}
        target_set = set(patient_ids.tolist())
        for mod_name, mod_data in self.modalities.items():
            ids = [p if isinstance(p, str) else p.decode() for p in mod_data.patient_ids]
            indices = [i for i, pid in enumerate(ids) if pid in target_set]
            result[mod_name] = np.array(indices, dtype=np.int64)
        return result


def _decode_strings(arr: np.ndarray) -> np.ndarray:
    """Decode byte strings from HDF5 to Python strings."""
    if arr.dtype.kind == "S" or arr.dtype.kind == "O":
        return np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in arr])
    return arr


def load_hdf5_dataset(
    hdf5_path: str | Path,
    modalities: Sequence[str] = ("rna", "methylation", "cnv"),
) -> MultiOmicsDataset:
    """Load a multi-omics dataset from HDF5 file.

    Parameters
    ----------
    hdf5_path : path to the HDF5 file
    modalities : which omics layers to load

    Returns
    -------
    MultiOmicsDataset
    """
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    logger.info("Loading dataset from %s", hdf5_path)
    mod_dict: Dict[str, ModalityData] = {}

    with h5py.File(hdf5_path, "r") as f:
        # --- Load modalities ---
        for mod_name in modalities:
            if mod_name not in f:
                raise KeyError(f"Modality group '{mod_name}' not found in {hdf5_path}")
            keys = _MODALITY_KEYS[mod_name]
            grp = f[mod_name]
            data = grp[keys["data"]][:]
            genes = _decode_strings(grp[keys["genes"]][:])
            patients = _decode_strings(grp[keys["patients"]][:])
            mod_dict[mod_name] = ModalityData(
                name=mod_name,
                data=data.astype(np.float32),
                gene_symbols=genes,
                patient_ids=patients,
            )
            logger.info(
                "  %s: %d patients × %d genes", mod_name, data.shape[0], data.shape[1]
            )

        # --- Load clinical data ---
        clin_grp = f["clinical"]
        clinical = ClinicalData(
            patient_ids=_decode_strings(clin_grp["patient_ids"][:]),
            os_days=clin_grp["os_days"][:].astype(np.float32),
            os_status=clin_grp["os_status"][:].astype(np.int32),
        )
        logger.info("  clinical: %d patients", len(clinical.patient_ids))

        # --- Load metadata ---
        meta_grp = f["meta"]
        gene_universe = _decode_strings(meta_grp["gene_universe"][:])
        fi_edge_list = None
        if "fi_edge_list" in meta_grp:
            fi_edge_list = meta_grp["fi_edge_list"][:].astype(np.int64)

    logger.info("Gene universe: %d genes", len(gene_universe))
    return MultiOmicsDataset(
        modalities=mod_dict,
        clinical=clinical,
        gene_universe=gene_universe,
        fi_edge_list=fi_edge_list,
    )
