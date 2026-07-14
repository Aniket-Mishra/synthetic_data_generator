import numpy as np
import pandas as pd
from synthetic_data_generator.geometry import wrap_deg, angle_diff
from synthetic_data_generator.wind.curves import (
    power_curve_kw,
    commanded_pitch_angle,
)

from synthetic_data_generator.wind.faults import FAULT_BINARY_COLUMNS

random_seed = 42


def make_device_params(rng, overrides=None):
    params = {
        "rated_power_kw": float(
            np.clip(rng.normal(1790.0, 15.0), 1700.0, 1850.0)
        ),
        "gearbox_ratio": float(np.clip(rng.normal(95.0, 1.2), 90.0, 100.0)),
        "yaw_response_gain": float(
            np.clip(rng.normal(0.35, 0.03), 0.25, 0.45)
        ),
        "pitch_noise_std": float(np.clip(rng.normal(0.20, 0.03), 0.10, 0.35)),
        "power_efficiency": float(np.clip(rng.normal(1.0, 0.02), 0.95, 1.05)),
        "thermal_bias": float(np.clip(rng.normal(0.0, 0.8), -2.0, 2.0)),
        "sensor_noise_scale": float(
            np.clip(rng.normal(1.0, 0.08), 0.85, 1.15)
        ),
    }
    if overrides:
        params.update(overrides)
    return params


def generate_healthy_device(
    site_df, device_id, device_params, seed
):
    rng = np.random.default_rng(seed)
    n = len(site_df)
    noise_scale = device_params["sensor_noise_scale"]

    wind_speed = np.clip(
        site_df["wind_speed_site"].to_numpy()
        + rng.normal(0, 0.18 * noise_scale, n),
        0.0,
        30.0,
    )
    wind_direction = wrap_deg(
        site_df["wind_direction_site"].to_numpy()
        + rng.normal(0, 1.5 * noise_scale, n)
    )
    ambient_temp = site_df["ambient_temp_site"].to_numpy() + rng.normal(
        0, 0.25 * noise_scale, n
    )
    air_density = np.clip(
        site_df["air_density_site"].to_numpy()
        + rng.normal(0, 0.004 * noise_scale, n),
        1.10,
        1.35,
    )

    nacelle_direction = np.zeros(n)
    nacelle_direction[0] = wrap_deg(
        wind_direction[0] + rng.normal(0, 1.5 * noise_scale)
    )
    for t in range(1, n):
        target_heading = wind_direction[t]
        nacelle_direction[t] = wrap_deg(
            nacelle_direction[t - 1]
            + device_params["yaw_response_gain"]
            * angle_diff(target_heading, nacelle_direction[t - 1])
            + rng.normal(0, 1.0 * noise_scale)
        )

    nacelle_position = wrap_deg(
        nacelle_direction + rng.normal(0, 0.7 * noise_scale, n)
    )
    yaw_error = np.abs(angle_diff(wind_direction, nacelle_direction))

    pitch_base = commanded_pitch_angle(wind_speed)
    pitch_std = device_params["pitch_noise_std"]
    pitch_1 = np.clip(pitch_base + rng.normal(0, pitch_std, n), 0.0, 90.0)
    pitch_2 = np.clip(pitch_base + rng.normal(0, pitch_std, n), 0.0, 90.0)
    pitch_3 = np.clip(pitch_base + rng.normal(0, pitch_std, n), 0.0, 90.0)

    pitch_spread = np.maximum.reduce(
        [
            np.abs(pitch_1 - pitch_2),
            np.abs(pitch_1 - pitch_3),
            np.abs(pitch_2 - pitch_3),
        ]
    )

    raw_power = power_curve_kw(wind_speed)
    density_factor = np.clip(air_density / 1.225, 0.90, 1.10)
    yaw_efficiency = np.clip(np.cos(np.deg2rad(yaw_error)), 0.0, 1.0) ** 3
    pitch_efficiency = np.clip(1.0 - 0.012 * (pitch_spread**1.15), 0.75, 1.0)

    active_power = (
        raw_power
        * density_factor
        * yaw_efficiency
        * pitch_efficiency
        * device_params["power_efficiency"]
    )

    active_power += rng.normal(
        0, np.maximum(8.0, 0.015 * np.maximum(raw_power, 1.0)), n
    )
    active_power = np.clip(active_power, 0.0, device_params["rated_power_kw"])

    rotor_speed = np.where(
        wind_speed < 3.0,
        np.clip(
            0.3 + 0.7 * wind_speed + rng.normal(0, 0.25 * noise_scale, n),
            0.0,
            None,
        ),
        np.where(
            wind_speed < 10.0,
            4.0
            + 1.7 * (wind_speed - 3.0)
            + rng.normal(0, 0.35 * noise_scale, n),
            15.8
            + 0.08 * (wind_speed - 10.0)
            + rng.normal(0, 0.25 * noise_scale, n),
        ),
    )

    rotor_speed = rotor_speed * (
        0.96 + 0.04 * (active_power / np.maximum(raw_power, 1.0))
    )
    rotor_speed = np.where(
        active_power < 5.0,
        np.clip(rotor_speed - rng.uniform(1.5, 3.0, n), 0.0, None),
        rotor_speed,
    )
    rotor_speed = np.clip(rotor_speed, 0.0, 22.0)

    generator_speed = np.clip(
        rotor_speed * device_params["gearbox_ratio"]
        + rng.normal(0, 18 * noise_scale, n),
        0.0,
        2200.0,
    )

    thermal_bias = device_params["thermal_bias"]
    gearbox_oil_temp = (
        ambient_temp
        + 17.0
        + 0.010 * active_power
        + 0.10 * rotor_speed
        + thermal_bias
        + rng.normal(0, 1.2, n)
    )
    generator_temp = (
        ambient_temp
        + 20.0
        + 0.014 * active_power
        + 0.0008 * generator_speed
        + thermal_bias
        + rng.normal(0, 1.5, n)
    )
    bearing_temp = (
        ambient_temp
        + 12.0
        + 0.007 * active_power
        + 0.06 * rotor_speed
        + 0.015 * (yaw_error**1.7)
        + thermal_bias
        + rng.normal(0, 1.0, n)
    )
    converter_temp = (
        ambient_temp
        + 14.0
        + 0.011 * active_power
        + thermal_bias
        + rng.normal(0, 1.3, n)
    )

    df = pd.DataFrame(
        {
            "time": site_df["time"].to_numpy(),
            "device": device_id,
            "active_power": active_power,
            "wind_speed": wind_speed,
            "air_density": air_density,
            "wind_direction": wind_direction,
            "nacelle_direction": nacelle_direction,
            "nacelle_position": nacelle_position,
            "ambient_temp": ambient_temp,
            "rotor_speed": rotor_speed,
            "generator_speed": generator_speed,
            "gearbox_oil_temp": gearbox_oil_temp,
            "generator_temp": generator_temp,
            "bearing_temp": bearing_temp,
            "converter_temp": converter_temp,
            "pitch_blade_angle_1": pitch_1,
            "pitch_blade_angle_2": pitch_2,
            "pitch_blade_angle_3": pitch_3,
        }
    )

    df["is_drifted"] = 0
    df["fault_labels"] = "healthy"
    df["fault_severity"] = "none"
    df["drift_start_time"] = pd.NaT

    for col in FAULT_BINARY_COLUMNS:
        df[col] = 0

    return df
