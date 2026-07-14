from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_data_generator.geometry import wrap_deg, angle_diff
from synthetic_data_generator.wind.device import (
    make_device_params,
    generate_healthy_device,
)
from synthetic_data_generator.wind.faults import (
    FAULT_BINARY_COLUMNS,
    apply_fault,
    finalize_fault_labels,
)

STEPS_PER_DAY = 24 * 6

OUTPUT_COLUMNS = [
    "time",
    "device",
    "active_power",
    "wind_speed",
    "air_density",
    "wind_direction",
    "nacelle_direction",
    "nacelle_position",
    "ambient_temp",
    "rotor_speed",
    "generator_speed",
    "gearbox_oil_temp",
    "generator_temp",
    "bearing_temp",
    "converter_temp",
    "pitch_blade_angle_1",
    "pitch_blade_angle_2",
    "pitch_blade_angle_3",
    "is_drifted",
    "fault_labels",
    "fault_severity",
    "drift_start_time",
]
ALL_COLUMNS = OUTPUT_COLUMNS + FAULT_BINARY_COLUMNS


def start_day_to_idx(start_day, steps_per_day=STEPS_PER_DAY):
    if start_day is None:
        return None
    return int(start_day * steps_per_day)


def write_dataset(df, events, out_dir, stream_name, events_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{stream_name}.parquet", index=False)
    events.to_parquet(out_dir / f"{events_name}.parquet", index=False)


def simulate_site_conditions(
    start="2025-01-01", n_days=365, freq="10min", *, seed
):
    rng = np.random.default_rng(seed)
    time = pd.date_range(start=start, periods=n_days * 24 * 6, freq=freq)
    n = len(time)
    idx = np.arange(n)

    hour_of_day = time.hour + time.minute / 60.0
    day_of_year = time.dayofyear

    temp_noise = np.zeros(n)
    ws_noise = np.zeros(n)

    for t in range(1, n):
        temp_noise[t] = 0.97 * temp_noise[t - 1] + rng.normal(0, 0.25)
        ws_noise[t] = 0.94 * ws_noise[t - 1] + rng.normal(0, 0.45)

    ambient_temp_site = (
        11.0
        + 8.0 * np.sin(2 * np.pi * (day_of_year / 365.25 - 0.15))
        + 4.0 * np.sin(2 * np.pi * (hour_of_day / 24.0 - 0.20))
        + temp_noise
    )

    pressure_pa = (
        101325.0
        + 1200.0 * np.sin(2 * np.pi * idx / (6 * 24 * 8))
        + rng.normal(0, 250, n)
    )

    air_density_site = pressure_pa / (287.05 * (ambient_temp_site + 273.15))

    wind_speed_site = (
        7.5
        + 1.2 * np.sin(2 * np.pi * idx / (6 * 24 * 5))
        + 0.5 * np.sin(2 * np.pi * (hour_of_day / 24.0 + 0.10))
        + ws_noise
    )

    storm_boost = np.zeros(n)
    storm_centers = rng.integers(0, n, size=max(6, n_days // 45))
    for center in storm_centers:
        width = rng.integers(10, 36)
        amplitude = rng.uniform(8.0, 18.0)
        storm_boost += amplitude * np.exp(-0.5 * ((idx - center) / width) ** 2)

    wind_speed_site = np.clip(wind_speed_site + storm_boost, 0.0, 30.0)

    wind_direction_site = np.zeros(n)
    wind_direction_site[0] = 220.0
    mean_direction = wrap_deg(
        220.0 + 35.0 * np.sin(2 * np.pi * idx / (6 * 24 * 14))
    )

    for t in range(1, n):
        wind_direction_site[t] = wrap_deg(
            wind_direction_site[t - 1]
            + 0.10 * angle_diff(mean_direction[t], wind_direction_site[t - 1])
            + rng.normal(0, 4.5)
        )

    return pd.DataFrame(
        {
            "time": time,
            "ambient_temp_site": ambient_temp_site,
            "air_density_site": air_density_site,
            "wind_speed_site": wind_speed_site,
            "wind_direction_site": wind_direction_site,
        }
    )


def round_and_clip(df):
    df["active_power"] = np.round(np.clip(df["active_power"], 0.0, None), 2)
    df["wind_speed"] = np.round(np.clip(df["wind_speed"], 0.0, 30.0), 2)
    df["air_density"] = np.round(np.clip(df["air_density"], 1.10, 1.35), 4)
    df["wind_direction"] = np.round(wrap_deg(df["wind_direction"]), 2)
    df["nacelle_direction"] = np.round(wrap_deg(df["nacelle_direction"]), 2)
    df["nacelle_position"] = np.round(wrap_deg(df["nacelle_position"]), 2)
    df["ambient_temp"] = np.round(df["ambient_temp"], 2)
    df["rotor_speed"] = np.round(np.clip(df["rotor_speed"], 0.0, 22.0), 2)
    df["generator_speed"] = np.round(
        np.clip(df["generator_speed"], 0.0, 2200.0), 2
    )
    df["gearbox_oil_temp"] = np.round(
        np.clip(df["gearbox_oil_temp"], -20.0, 120.0), 2
    )
    df["generator_temp"] = np.round(
        np.clip(df["generator_temp"], -20.0, 140.0), 2
    )
    df["bearing_temp"] = np.round(np.clip(df["bearing_temp"], -20.0, 120.0), 2)
    df["converter_temp"] = np.round(
        np.clip(df["converter_temp"], -20.0, 120.0), 2
    )
    df["pitch_blade_angle_1"] = np.round(
        np.clip(df["pitch_blade_angle_1"], 0.0, 90.0), 2
    )
    df["pitch_blade_angle_2"] = np.round(
        np.clip(df["pitch_blade_angle_2"], 0.0, 90.0), 2
    )
    df["pitch_blade_angle_3"] = np.round(
        np.clip(df["pitch_blade_angle_3"], 0.0, 90.0), 2
    )
    return df


def simulate_device(site_df, device_config, seed):
    rng = np.random.default_rng(seed)
    device_id = device_config["device_id"]
    device_params = make_device_params(rng, device_config.get("device_params"))

    df = generate_healthy_device(site_df, device_id, device_params, seed=seed)

    fault_states = {}
    fault_start_times = []
    steps_per_day = 24 * 6

    for fault in device_config.get("faults", []):
        fault_cfg = dict(fault)

        if "start_idx" not in fault_cfg:
            fault_cfg["start_idx"] = start_day_to_idx(
                fault_cfg.get("start_day"), steps_per_day
            )

        if "ramp_steps" not in fault_cfg:
            ramp_days = fault_cfg.get("ramp_days")
            fault_cfg["ramp_steps"] = (
                int(ramp_days * steps_per_day) if ramp_days is not None else 1
            )

        if "end_idx" not in fault_cfg and fault_cfg.get("end_day") is not None:
            fault_cfg["end_idx"] = start_day_to_idx(
                fault_cfg["end_day"], steps_per_day
            )

        if "ramp_down_steps" not in fault_cfg:
            ramp_down_days = fault_cfg.get("ramp_down_days")
            fault_cfg["ramp_down_steps"] = (
                int(ramp_down_days * steps_per_day)
                if ramp_down_days is not None
                else 0
            )

        df, profile = apply_fault(df, fault_cfg)
        if fault_cfg["type"] in fault_states:
            fault_states[fault_cfg["type"]] = np.maximum(
                fault_states[fault_cfg["type"]], profile
            )
        else:
            fault_states[fault_cfg["type"]] = profile

        if np.any(profile > 0):
            first_idx = int(np.where(profile > 0)[0][0])
            fault_start_times.append(site_df.iloc[first_idx]["time"])

    df = finalize_fault_labels(df, fault_states, fault_start_times)
    df = round_and_clip(df)
    return df[ALL_COLUMNS]


def build_fault_event_table(site_df, device_configs):
    rows = []
    steps_per_day = 24 * 6

    for cfg in device_configs:
        for fault in cfg.get("faults", []):
            start_idx = fault.get("start_idx")
            if start_idx is None:
                start_idx = start_day_to_idx(
                    fault.get("start_day"), steps_per_day
                )

            end_idx = fault.get("end_idx")
            if end_idx is None and fault.get("end_day") is not None:
                end_idx = start_day_to_idx(fault.get("end_day"), steps_per_day)

            start_time = pd.NaT
            end_time = pd.NaT

            if start_idx is not None and 0 <= start_idx < len(site_df):
                start_time = site_df.iloc[start_idx]["time"]

            if end_idx is not None and 0 <= end_idx < len(site_df):
                end_time = site_df.iloc[end_idx]["time"]

            rows.append(
                {
                    "device": cfg["device_id"],
                    "fault_type": fault["type"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "shape": fault.get("shape", "linear"),
                    "max_severity": fault.get("max_severity", 1.0),
                    "ramp_steps": fault.get("ramp_steps"),
                    "ramp_days": fault.get("ramp_days"),
                    "ramp_down_days": fault.get("ramp_down_days"),
                }
            )

    return pd.DataFrame(rows)


def simulate_farm(site_config, device_configs, seed):
    site_df = simulate_site_conditions(**site_config, seed=seed)
    all_devices = []

    for i, cfg in enumerate(device_configs):
        device_df = simulate_device(site_df, cfg, seed=seed + i + 1)
        all_devices.append(device_df)

    data = pd.concat(all_devices, ignore_index=True)
    data = data.sort_values(["time", "device"]).reset_index(drop=True)
    events = build_fault_event_table(site_df, device_configs)
    return data, events
