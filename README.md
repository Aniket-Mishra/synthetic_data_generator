# Synthetic Data Generator

A framework for generating synthetic datasets with realistic system behavior and configurable fault or anomaly injection. I built this to create labeled datasets for my ML research without relying on real operational data. It also provided the datasets for my TU/e Contest project.

The physical generators simulate the actual physics of the system they model. Faults follow degradation patterns you would see in the real world. The tabular generators focus on realistic statistical patterns, seasonal behavior, repeated customer behavior, and known ground truth labels.

The wind farm with repair is the one I have fully cleaned up. It runs from a YAML config through an installable package, comes with a Croissant 1.1 description of its output, and has a hash-based reproducibility check. The rest are still the original notebooks, and I am porting them into the same shape one at a time.

Motivation for the datasets:
The wind and solar ones are from interest. I spent 5 years with data like these, and I needed something similar for my Capita Selecta project in streaming data pattern mining. The pump dataset is here because I met a student team during TU/e contest who were very interested in what we were doing, and wanted to work with us. Their project was in pumps, so here we are. Sales is from a very old project and the imbalanced one is a test for a Hackathon, though we did not need to use the dataset there.

## What is in here

### Time series

**Wind farm** (`src/synthetic_data_generator/wind/`, config in `src/synthetic_data_generator/configs/wind.yaml`)

Simulates a 10-turbine wind farm over a full year at 10-minute intervals. Each turbine produces wind speed, air density, nacelle direction, rotor speed, generator speed, blade pitch angles, gearbox oil temperature, bearing temperature, and more. Power follows a standard turbine power curve. Yaw misalignment reduces power via cos(yaw_error)^3. Air density affects output.

Injected faults: gradual temperature drift, abrupt pitch misalignment, gradual yaw misalignment, each with its own physical effect on the downstream signals. The with repair variant adds repair events that restore faulted turbines back to normal.

This is the reference build. It generates from a config, emits a stream table and a fault-event table. This is extended with Croissant metadata.

This data is also used as a visualisation project when I was trying to make some pretty web apps for my website. I am kinda happy with what it looks like right now albeit some visual bugs: https://aniket-mishra.github.io/projects/windfield.html

**Solar farm** (still a notebook, `notebooks/initial_logic/solar_farm_with_downtime.ipynb`)

Simulates a 5-device PV plant over a full year at 5-minute intervals. Each device produces telemetry like active power, irradiance, module temperature, DC voltage, and more. Power follows a real irradiance-to-power curve. Sun position, cloud cover, storms, and seasonal temperature all vary realistically through the year.

Injected faults: soiling buildup, inverter overheating with intermittent trips, tracker actuator stuck, DC string outage. Each fault has configurable severity, ramp period, and timing. Maintenance outages are also included. The with repair variant (`notebooks/initial_logic/solar_farm_with_downtimes_with_repair.ipynb`) adds repair events that restore faulted devices back to normal.

### Tabular

These still live as notebooks under `notebooks/initial_logic/`, waiting their turn to be ported.

**Centrifugal pump** (`notebooks/initial_logic/centrifugal_pump.ipynb`)

Generates operating point snapshots for a 4-pump station. Each sample solves the pump-system intersection using affinity laws and a quadratic pump curve. Outputs include flow, head, efficiency, shaft power, motor current, vibration, bearing temperature, NPSH margin, and more.

Injected faults: cavitation, impeller wear, bearing friction. Each fault shifts the relevant physical quantities in the direction you would expect from the real failure mode.

**Imbalanced fraud detection** (`notebooks/initial_logic/imbalanced_fraud_detection.ipynb`)

Generates a transaction dataset for fraud detection. Customers have repeated transactions, account age is derived from signup date, and normal behavior includes point-of-sale payments, iDEAL payments, credit card online payments, cash withdrawals, and recurring subscriptions. I made it Dutch specific because I live in the Netherlands now.

Injected anomalies: card testing pings, high-value extrication, and velocity attacks. Fraud is rare and configurable, with labels for both fraud source and fraud scenario.

**Ecommerce sales** (`notebooks/initial_logic/ecommerce_sales_2019.ipynb`)

Generates monthly ecommerce sales data for 2019. Product demand varies by month, orders peak around noon and evening, and some products trigger realistic accessory purchases such as cables, headphones, keyboards, and cooling pads.

This code is from [one of my old projects](https://github.com/Aniket-Mishra/Sales-Analysis-and-Reporting). It is an updated version of a tutorial by [Keith Galli](https://github.com/KeithGalli).

## Repository structure

​```
.
├── src/synthetic_data_generator/
│   ├── wind/ # power curves, device, faults, farm
│   ├── configs/wind.yaml
│   ├── generate.py
│   ├── geometry.py
│   └── hashing.py
├── scripts/
│   ├── build_croissant.py
│   ├── check_baselines.py # reproducibility check
│   ├── freeze_baselines.py
│   └── dataset_explorer.py # Streamlit viewer - In progress, moved from earlier
├── notebooks/
│   ├── croissant_setup.ipynb
│   └── initial_logic/ # the original generators
├── tests/baseline_hashes.txt
├── generated_data/ # output, git-ignored
├── pyproject.toml
├── LICENSE
└── README.md
​```

## Accessing the data

The time series datasets are uploaded to OpenML:

1. Synthetic-Wind-Farm-Stream-No-Repair: https://www.openml.org/search?type=data&status=active&id=47241
2. Synthetic-Wind-Farm-Stream-With-Repair: https://www.openml.org/search?type=data&sort=runs&id=47242&status=active
3. Synthetic-Solar-Farm-Stream-No-Repair: https://www.openml.org/search?type=data&sort=runs&id=47243&status=active
4. Synthetic-Solar-Farm-Stream-With-Repair: https://www.openml.org/search?type=data&sort=runs&id=47244&status=active

## Setup

This is a uv project.

​```
uv sync
​```

## Usage

Generate the wind dataset from its config:

​```
uv run generate --config src/synthetic_data_generator/configs/wind.yaml
​```

Outputs in `generated_data/`. Edit the YAML to change turbine count, fault timing, repair windows, seed, or output path. No code change.

Check the output still matches the frozen baseline:

​```
uv run python scripts/check_baselines.py
​```

Poke around any generated dataset in the Streamlit viewer:

​```
uv run streamlit run scripts/dataset_explorer.py
​```

The other generators (solar, pump, fraud, sales) still run as notebooks under `notebooks/initial_logic/`.

## Croissant metadata

The wind farm with repair has a [Croissant 1.1](https://mlcommons.org/croissant/) description of its output, built with `scripts/build_croissant.py`. Two record sets, `measurements` (the sensor stream) and `fault_events` (one row per injected fault), joined by a foreign key on `device`. The file objects have a sha256 to verify the bytes.

​```
uv run python scripts/build_croissant.py --dataset-dir generated_data/time_series/wind_data_with_repair
​```

## What you can configure

Each generator exposes a configuration section. Depending on the dataset, you control:

- Number of devices, customers, transactions, or orders
- Simulation duration and sampling frequency
- Per-device physical parameters such as capacity, efficiency, thermal bias, and sensor noise
- Fault type, timing, ramp period, shape, and severity
- Whether faults are repaired and when
- Maintenance outage windows
- Fraud rate, fraud scenario mix, transaction channels, country mix, and subscription behavior
- Monthly sales volume, product weights, prices, and accessory bundle probabilities

For the wind farm this all lives in `configs/wind.yaml`:

​```yaml
seed: 42
site:
  start: "2025-01-01"
  n_days: 365
  freq: "10min"
devices:
  - device_id: WT002
    device_params: {power_efficiency: 0.99}
    faults:
      - {type: temperature, start_day: 100, ramp_days: 14, shape: linear, max_severity: 1.0}
​```

The other generators still keep their config as a dict inside the notebook.

Example from the solar farm notebook:

```python
device_configs = [
    {
        "device_id": "PV001",
        "device_params": {"dc_capacity_scale": 0.995},
        "faults": [
            {
                "type": "soiling",
                "start_day": 120,
                "end_day": 280,
                "ramp_days": 35,
                "shape": "linear",
                "max_severity": 1.0,
            }
        ],
        "outages": [],
    },
]
```


## Use cases

- Anomaly and drift detection, and fault classification model development
- Prototyping data pipelines before real data is available
- Benchmarking forecasting and condition monitoring algorithms
- Generating labeled datasets with known ground truth
- Testing fraud detection, imbalanced classification, and transaction monitoring systems
- Creating realistic tabular data for analytics and dashboard development

## Roadmap

- Port solar, pump, fraud, and sales into the same config-driven package (wind is done)
- Croissant metadata for all the other dataset
- Publish each dataset with a hosted Croissant file
- More modalities such as graph, text, and multimodal
- Streaming data simulation

## Notes

This project focuses on controllable, realistic synthetic data generation. It is not trying to be a set of domain-specific digital twins. The goal is to produce datasets that are structurally and statistically realistic enough to be useful for ML experimentation.

## License

MIT. See LICENSE.