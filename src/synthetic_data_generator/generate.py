import argparse
from pathlib import Path

import yaml

from synthetic_data_generator import fraud, pump, sales, solar
from synthetic_data_generator.wind import farm as wind

GENERATORS = {
    "wind": wind.run,
    "solar": solar.run,
    "pump": pump.run,
    "fraud": fraud.run,
    "sales": sales.run,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(parser.parse_args().config).read_text())
    print(f"""Simulating Dataset: {cfg["output_dir"].split("/")[-1]}""")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in GENERATORS[cfg["generator"]](cfg).items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    print(f"Dataset Generated. Saved at {cfg['output_dir']}")


if __name__ == "__main__":
    main()
