# Synthetic Data Generator

A framework for generating synthetic datasets with realistic physics and configurable fault injection.

I built this to create labeled datasets for my ML research without relying on real operational data. Each generator simulates the actual physics of the system it models. Faults follow the degradation patterns you would see in the real world.

## What is in here

### Time series

**Solar farm** (`time_series/solar_farm_with_downtime.ipynb`)

Simulates a 5-device PV plant over a full year at 5-minute intervals. Each device produces telemetry like active power, irradiance, module temperature, DC voltage, and more. Power follows a real irradiance-to-power curve. Sun position, cloud cover, storms, and seasonal temperature all vary realistically through the year.

Injected faults: soiling buildup, inverter overheating with intermittent trips, tracker actuator stuck, DC string outage. Each fault has configurable severity, ramp period, and timing. Maintenance outages are also included.

**Wind farm** (`time_series/wind_farm_with_downtime.ipynb`)

Simulates a 4-turbine wind farm over a full year at 10-minute intervals. Each turbine produces wind speed, air density, nacelle direction, rotor speed, generator speed, blade pitch angles, gearbox oil temperature, bearing temperature, and more. Power follows a standard turbine power curve. Yaw misalignment reduces power via cos(yaw_error)^3. Air density affects output.

Injected faults: gradual temperature drift, abrupt pitch misalignment, gradual yaw misalignment. Each fault has its own physical effect on the downstream signals.

### Tabular

**Centrifugal pump** (`tabular/centrifugal_pump.ipynb`)

Generates operating point snapshots for a 4-pump station. Each sample solves the pump-system intersection using affinity laws and a quadratic pump curve. Outputs include flow, head, efficiency, shaft power, motor current, vibration, bearing temperature, NPSH margin, and more.

Injected faults: cavitation, impeller wear, bearing friction. Each fault shifts the relevant physical quantities in the direction you would expect from the real failure mode.

## Repository structure

```
.
├── tabular/
│   └── centrifugal_pump.ipynb
├── time_series/
│   ├── solar_farm_with_downtime.ipynb
│   └── wind_farm_with_downtime.ipynb
├── generated_data/
├── utils.py
├── LICENSE
└── README.md
```

## Setup

```
pip install numpy pandas pyarrow jupyter
```

## Usage

Run any notebook to generate datasets. Outputs are written to `generated_data/`.

```
jupyter notebook time_series/solar_farm_with_downtime.ipynb
```

## What you can configure

Each generator exposes a device config list. You control:

- Number of devices
- Simulation duration and sampling frequency
- Per-device physical parameters (capacity, efficiency, thermal bias, sensor noise, etc.)
- Fault type, timing, ramp period, shape, and severity
- Maintenance outage windows

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

## Roadmap

- More data modalities (graph, text, multimodal)
- Config-driven generation API
- CLI interface
- Streaming data simulation

## Notes

This project focuses on controllable, physics-grounded data generation. It is not trying to be a domain-specific digital twin. The goal is to produce datasets that are structurally and statistically realistic enough to be useful for ML experimentation.

## License

MIT. See `LICENSE`.