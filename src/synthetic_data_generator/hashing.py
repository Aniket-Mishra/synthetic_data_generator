"""Content hashing for generated datasets.

Parquet bytes are not stable across pyarrow versions, so a byte-level hash would
flag environment upgrades as regressions. Hashing the dataframe values instead
keeps the check meaningful across versions.
"""

import hashlib
from pathlib import Path

import pandas as pd
from pandas.util import hash_pandas_object

DATA_ROOT = Path("generated_data")
BASELINE_PATH = Path("tests/baseline_hashes.txt")


def hash_dataframe(df: pd.DataFrame) -> str:
    values = hash_pandas_object(df, index=True).values.tobytes()
    columns = ",".join(df.columns).encode()
    return hashlib.sha256(values + columns).hexdigest()


def hash_parquet(path: Path) -> str:
    return hash_dataframe(pd.read_parquet(path))


def hash_all_datasets(root: Path = DATA_ROOT) -> dict[str, str]:
    return {
        str(path): hash_parquet(path)
        for path in sorted(root.rglob("*.parquet"))
    }


def read_baseline(path: Path = BASELINE_PATH) -> dict[str, str]:
    baseline = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, dataset = line.split(maxsplit=1)
        baseline[dataset] = digest
    return baseline


def write_baseline(hashes: dict[str, str], path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Content hashes of the datasets generated before restructure",
        "# Regenerate: uv run python scripts/freeze_baseline.py",
        "# Verify: uv run python scripts/check_baseline.py\n",
    ]
    rows = [
        f"{digest}  {dataset}" for dataset, digest in sorted(hashes.items())
    ]
    path.write_text("\n".join(header + rows) + "\n")
