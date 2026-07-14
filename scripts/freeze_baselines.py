"""Record the content hash of every generated dataset, as a pre-refactor reference."""

import sys
from synthetic_data_generator.hashing import (
    BASELINE_PATH,
    hash_all_datasets,
    write_baseline,
)


def main() -> int:
    hashes = hash_all_datasets()

    if not hashes:
        print("No parquet files found under generated_data/.")
        return 1

    write_baseline(hashes)

    for dataset, digest in sorted(hashes.items()):
        print(f"{digest[:12]}  {dataset}")

    print(f"Wrote {len(hashes)} hashes to {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
