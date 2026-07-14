import argparse
from pathlib import Path

import yaml

from synthetic_data_generator.wind.farm import simulate_farm, write_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(parser.parse_args().config).read_text())
    print(f"""Simulating Dataset: {cfg["output_dir"].split("/")[-1]}""")
    df, events = simulate_farm(cfg["site"], cfg["devices"], seed=cfg["seed"])
    write_dataset(
        df, events, cfg["output_dir"], cfg["stream_name"], cfg["events_name"]
    )
    print(f"Dataset Generated. Saved at {cfg['output_dir']}")


if __name__ == "__main__":
    main()
