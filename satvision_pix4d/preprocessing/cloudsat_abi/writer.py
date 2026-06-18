"""Output models and writers for collocated chips."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CollocatedChip:
    """One serializable ABI and CloudSat collocation sample."""

    filename: str
    chip: np.ndarray
    offsets_minutes: np.ndarray
    valid_mask: np.ndarray
    scan_times: np.ndarray
    metadata: dict[str, Any]
    auxiliary_arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def arrays(self) -> dict[str, Any]:
        arrays: dict[str, Any] = {
            "chip": self.chip,
            "abi_offsets_minutes": self.offsets_minutes,
            "abi_valid_mask": self.valid_mask,
            "abi_scan_times": self.scan_times,
            "metadata_json": np.asarray(
                json.dumps(
                    self.metadata, sort_keys=True, separators=(",", ":")
                )
            ),
        }
        arrays.update(self.auxiliary_arrays)
        return arrays


class NPZChipWriter:
    """Write compressed NPZ samples without pickle-dependent object arrays."""

    def __init__(self, output_dir: Path, overwrite: bool = False):
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, sample: CollocatedChip) -> tuple[Path, bool]:
        output = self.output_dir / sample.filename
        if output.exists() and not self.overwrite:
            return output, False
        np.savez_compressed(output, **sample.arrays())
        return output, True
