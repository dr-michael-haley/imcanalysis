"""Opt-in five-million-cell fixture for browser scale checks; synthetic data only."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures"
    root.mkdir(exist_ok=True)
    count, per_roi = 5_000_000, 5_000
    row = np.arange(count, dtype=np.int32)
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical.from_codes(
                row // per_roi, [f"ROI_{i:04d}" for i in range(count // per_roi)]
            ),
            "ObjectNumber": row % per_roi + 1,
            "population": pd.Categorical.from_codes(
                row % 4, ["T cell", "B cell", "Myeloid", "Stroma"]
            ),
        },
        index=pd.Index(row.astype(str)),
    )
    data = ad.AnnData(
        X=np.column_stack((row, row % 100, np.zeros(count))).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=["row_number", "mod_100", "zero"]),
    )
    data.obsm["X_umap"] = np.column_stack((row % 100, row % per_roi // 100)).astype(np.float32)
    path = root / "scale.h5ad"
    data.write_h5ad(path, compression="gzip")
    print(f"Synthetic scale fixture: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
