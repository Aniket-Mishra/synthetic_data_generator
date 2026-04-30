# Synthetic Data Generator

A framework for generating synthetic datasets with realistic system behavior and configurable fault or anomaly injection. I built this to create labeled datasets for my ML research without relying on real operational data.

The physical generators simulate the actual physics of the system they model. Faults follow degradation patterns you would see in the real world. The tabular generators focus on realistic statistical patterns, seasonal behavior, repeated customer behavior, and known ground truth labels.

## What is in here

### Time series

**Solar farm** (time_series/solar_farm_with_downtime.ipynb)

Simulates a 5-device PV plant over a full year at 5-minute intervals. Each device produces telemetry like active power, irradiance, module temperature, DC voltage, and more. Power follows a real irradiance-to-power curve. Sun position, cloud cover, storms, and seasonal temperature all vary realistically through the year.

Injected faults: soiling buildup, inverter overheating with intermittent trips, tracker actuator stuck, DC string outage. Each fault has configurable severity, ramp period, and timing. Maintenance outages are also included.

The with repair variant (time_series/solar_farm_with_downtimes_with_repair.ipynb) extends this with repair events that restore faulted devices back to normal operation.

**Wind farm** (time_series/wind_farm_with_downtime.ipynb)

Simulates a 4-turbine wind farm over a full year at 10-minute intervals. Each turbine produces wind speed, air density, nacelle direction, rotor speed, generator speed, blade pitch angles, gearbox oil temperature, bearing temperature, and more. Power follows a standard turbine power curve. Yaw misalignment reduces power via cos(yaw_error)^3. Air density affects output.

Injected faults: gradual temperature drift, abrupt pitch misalignment, gradual yaw misalignment. Each fault has its own physical effect on the downstream signals.

The with repair variant (time_series/wind_farm_with_downtimes_with_repair.ipynb) extends this with repair events that restore faulted turbines back to normal operation.

### Tabular

**Centrifugal pump** (tabular/centrifugal_pump.ipynb)

Generates operating point snapshots for a 4-pump station. Each sample solves the pump-system intersection using affinity laws and a quadratic pump curve. Outputs include flow, head, efficiency, shaft power, motor current, vibration, bearing temperature, NPSH margin, and more.

Injected faults: cavitation, impeller wear, bearing friction. Each fault shifts the relevant physical quantities in the direction you would expect from the real failure mode.

**Imbalanced fraud detection** (tabular/imbalanced_fraud_detection.ipynb)

Generates a transaction dataset for fraud detection. Customers have repeated transactions, account age is derived from signup date, and normal behavior includes point-of-sale payments, iDEAL payments, credit card online payments, cash withdrawals, and recurring subscriptions. I made it Dutch specific because I live in the Netherlands now.

Injected anomalies: card testing pings, high-value extrication, and velocity attacks. Fraud is rare and configurable, with labels for both fraud source and fraud scenario.

**Ecommerce sales** (tabular/ecommerce_sales_2019.ipynb)

Generates monthly ecommerce sales data for 2019. Product demand varies by month, orders peak around noon and evening, and some products trigger realistic accessory purchases such as cables, headphones, keyboards, and cooling pads.

This code is from [one of my old projects](https://github.com/Aniket-Mishra/Sales-Analysis-and-Reporting). It is an updated version of a tutorial by [Keith Galli](https://github.com/KeithGalli)

## Repository structure

```
.
├── tabular/
│   ├── centrifugal_pump.ipynb
│   ├── ecommerce_sales_2019.ipynb
│   └── imbalanced_fraud_detection.ipynb
├── time_series/
│   ├── solar_farm_with_downtime.ipynb
│   ├── solar_farm_with_downtimes_with_repair.ipynb
│   ├── wind_farm_with_downtime.ipynb
│   └── wind_farm_with_downtimes_with_repair.ipynb
├── generated_data/
├── dataset_explorer.py
├── upload_to_openml.py
├── test_data.ipynb
├── utils.py
├── LICENSE
└── README.md
```

## Setup

```
pip install numpy pandas pyarrow jupyter
```

## Usage

Run any notebook to generate datasets. Outputs are written to generated_data/.

```
jupyter notebook time_series/solar_farm_with_downtime.ipynb
```

The dataset explorer is a Streamlit app for browsing and inspecting the generated datasets.

```
streamlit run dataset_explorer.py
```

To upload a generated dataset to OpenML, run:

```
python upload_to_openml.py
```

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

- More data modalities such as graph, text, and multimodal data
- Config-driven generation API
- CLI interface
- Streaming data simulation

## Notes

This project focuses on controllable, realistic synthetic data generation. It is not trying to be a set of domain-specific digital twins. The goal is to produce datasets that are structurally and statistically realistic enough to be useful for ML experimentation.

## License

MIT. See LICENSE.