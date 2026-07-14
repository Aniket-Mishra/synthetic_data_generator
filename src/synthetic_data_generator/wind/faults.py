import numpy as np
import pandas as pd
from synthetic_data_generator.geometry import wrap_deg, angle_diff
from synthetic_data_generator.wind.curves import power_curve_kw


FAULT_BINARY_COLUMNS = [
    "fault_temperature",
    "fault_pitch_misalignment",
    "fault_yaw_misalignment",
]

FAULT_COLUMN_MAP = {
    "temperature": "fault_temperature",
    "pitch_misalignment": "fault_pitch_misalignment",
    "yaw_misalignment": "fault_yaw_misalignment",
}


def severity_label(values):
    labels = np.full(len(values), "none", dtype=object)
    labels[(values > 0.0) & (values < 0.33)] = "low"
    labels[(values >= 0.33) & (values < 0.66)] = "medium"
    labels[values >= 0.66] = "high"
    return labels


def build_fault_profile(
    n,
    start_idx,
    ramp_steps=1,
    end_idx=None,
    shape="linear",
    max_severity=1.0,
    ramp_down_steps=0,
):
    profile = np.zeros(n, dtype=float)

    if start_idx is None or start_idx >= n:
        return profile

    if end_idx is None:
        end_idx = n

    start_idx = max(0, start_idx)
    end_idx = min(end_idx, n)

    if end_idx <= start_idx:
        return profile

    if shape == "abrupt":
        profile[start_idx:end_idx] = max_severity
        return profile

    if shape == "linear":
        ramp_steps = max(1, int(ramp_steps))
        ramp_end = min(start_idx + ramp_steps, end_idx)
        if ramp_end > start_idx:
            profile[start_idx:ramp_end] = np.linspace(
                0.0, max_severity, ramp_end - start_idx, endpoint=False
            )
        profile[ramp_end:end_idx] = max_severity

        if ramp_down_steps:
            ramp_down_steps = int(ramp_down_steps)
            ramp_down_start = max(ramp_end, end_idx - ramp_down_steps)
            if end_idx > ramp_down_start:
                profile[ramp_down_start:end_idx] = np.linspace(
                    max_severity,
                    0.0,
                    end_idx - ramp_down_start,
                    endpoint=False,
                )
        return profile

    if shape == "intermittent":
        ramp_steps = max(1, int(ramp_steps))
        ramp_end = min(start_idx + ramp_steps, end_idx)
        if ramp_end > start_idx:
            profile[start_idx:ramp_end] = np.linspace(
                0.0, max_severity, ramp_end - start_idx, endpoint=False
            )
        t = np.arange(ramp_end, end_idx)
        pattern = 0.5 * (
            1.0 + np.sin(2 * np.pi * (t - ramp_end) / (6 * 24 * 3))
        )
        profile[ramp_end:end_idx] = max_severity * (0.35 + 0.65 * pattern)
        return profile

    raise ValueError(f"Unsupported profile shape: {shape}")


def apply_temperature_fault(df, fault_config):
    profile = build_fault_profile(
        n=len(df),
        start_idx=fault_config["start_idx"],
        ramp_steps=fault_config.get("ramp_steps", 1),
        end_idx=fault_config.get("end_idx"),
        shape=fault_config.get("shape", "linear"),
        max_severity=fault_config.get("max_severity", 1.0),
        ramp_down_steps=fault_config.get("ramp_down_steps", 0),
    )

    if np.all(profile == 0):
        return df, profile

    late_derate = np.maximum(profile - 0.70, 0.0)

    df["gearbox_oil_temp"] += 10.0 * profile
    df["generator_temp"] += 16.0 * profile
    df["bearing_temp"] += 8.0 * profile
    df["converter_temp"] += 6.0 * profile
    df["active_power"] *= 1.0 - 0.05 * late_derate
    df["active_power"] = np.clip(df["active_power"], 0.0, None)

    return df, profile


def apply_pitch_misalignment_fault(df, fault_config):
    profile = build_fault_profile(
        n=len(df),
        start_idx=fault_config["start_idx"],
        ramp_steps=fault_config.get("ramp_steps", 1),
        end_idx=fault_config.get("end_idx"),
        shape=fault_config.get("shape", "linear"),
        max_severity=fault_config.get("max_severity", 10.0),
        ramp_down_steps=fault_config.get("ramp_down_steps", 0),
    )

    if np.all(profile == 0):
        return df, profile

    df["pitch_blade_angle_2"] += profile
    df["pitch_blade_angle_3"] -= 0.4 * profile

    for col in [
        "pitch_blade_angle_1",
        "pitch_blade_angle_2",
        "pitch_blade_angle_3",
    ]:
        df[col] = np.clip(df[col], 0.0, 90.0)

    pitch_spread = np.maximum.reduce(
        [
            np.abs(df["pitch_blade_angle_1"] - df["pitch_blade_angle_2"]),
            np.abs(df["pitch_blade_angle_1"] - df["pitch_blade_angle_3"]),
            np.abs(df["pitch_blade_angle_2"] - df["pitch_blade_angle_3"]),
        ]
    )

    pitch_efficiency = np.clip(1.0 - 0.012 * (pitch_spread**1.15), 0.75, 1.0)
    raw_curve_power = power_curve_kw(df["wind_speed"].to_numpy())
    density_factor = np.clip(df["air_density"].to_numpy() / 1.225, 0.90, 1.10)
    yaw_error = np.abs(
        angle_diff(
            df["wind_direction"].to_numpy(), df["nacelle_direction"].to_numpy()
        )
    )
    yaw_efficiency = np.clip(np.cos(np.deg2rad(yaw_error)), 0.0, 1.0) ** 3

    recomputed_power = (
        raw_curve_power * density_factor * yaw_efficiency * pitch_efficiency
    )
    df["active_power"] = np.minimum(
        df["active_power"].to_numpy(), recomputed_power
    )
    df["active_power"] = np.clip(df["active_power"], 0.0, None)

    df["bearing_temp"] += 0.12 * pitch_spread
    df["gearbox_oil_temp"] += 0.05 * pitch_spread

    return df, profile


def apply_yaw_misalignment_fault(df, fault_config):
    profile = build_fault_profile(
        n=len(df),
        start_idx=fault_config["start_idx"],
        ramp_steps=fault_config.get("ramp_steps", 1),
        end_idx=fault_config.get("end_idx"),
        shape=fault_config.get("shape", "linear"),
        max_severity=fault_config.get("max_severity", 18.0),
        ramp_down_steps=fault_config.get("ramp_down_steps", 0),
    )

    if np.all(profile == 0):
        return df, profile

    df["nacelle_direction"] = wrap_deg(
        df["nacelle_direction"].to_numpy() + profile
    )
    df["nacelle_position"] = wrap_deg(
        df["nacelle_position"].to_numpy() + profile
    )

    yaw_error = np.abs(
        angle_diff(
            df["wind_direction"].to_numpy(), df["nacelle_direction"].to_numpy()
        )
    )
    yaw_efficiency = np.clip(np.cos(np.deg2rad(yaw_error)), 0.0, 1.0) ** 3

    pitch_spread = np.maximum.reduce(
        [
            np.abs(df["pitch_blade_angle_1"] - df["pitch_blade_angle_2"]),
            np.abs(df["pitch_blade_angle_1"] - df["pitch_blade_angle_3"]),
            np.abs(df["pitch_blade_angle_2"] - df["pitch_blade_angle_3"]),
        ]
    )
    pitch_efficiency = np.clip(1.0 - 0.012 * (pitch_spread**1.15), 0.75, 1.0)

    raw_curve_power = power_curve_kw(df["wind_speed"].to_numpy())
    density_factor = np.clip(df["air_density"].to_numpy() / 1.225, 0.90, 1.10)

    recomputed_power = (
        raw_curve_power * density_factor * yaw_efficiency * pitch_efficiency
    )
    df["active_power"] = np.minimum(
        df["active_power"].to_numpy(), recomputed_power
    )
    df["active_power"] = np.clip(df["active_power"], 0.0, None)

    df["bearing_temp"] += 0.015 * (yaw_error**1.7)

    return df, profile


FAULT_REGISTRY = {
    "temperature": apply_temperature_fault,
    "pitch_misalignment": apply_pitch_misalignment_fault,
    "yaw_misalignment": apply_yaw_misalignment_fault,
}


def apply_fault(df, cfg):
    fault_type = cfg["type"]
    if fault_type not in FAULT_REGISTRY:
        raise ValueError(f"Unknown fault type: {fault_type}")
    return FAULT_REGISTRY[fault_type](df, cfg)


def finalize_fault_labels(df, fault_states, fault_start_times):
    n = len(df)
    if not fault_states:
        df["is_drifted"] = 0
        df["fault_labels"] = "healthy"
        df["fault_severity"] = "none"
        df["drift_start_time"] = pd.NaT
        return df

    max_profile = np.maximum.reduce(list(fault_states.values()))
    fault_labels = np.full(n, "healthy", dtype=object)

    for fault_type, profile in fault_states.items():
        active = profile > 0
        df[FAULT_COLUMN_MAP[fault_type]] = active.astype(int)
        for i in np.where(active)[0]:
            if fault_labels[i] == "healthy":
                fault_labels[i] = fault_type
            else:
                fault_labels[i] = f"{fault_labels[i]}|{fault_type}"

    drift_start_time = pd.Series(
        pd.NaT, index=df.index, dtype="datetime64[ns]"
    )
    if fault_start_times:
        drift_start_time[:] = min(fault_start_times)

    df["is_drifted"] = (max_profile > 0).astype(int)
    df["fault_labels"] = fault_labels
    df["fault_severity"] = severity_label(max_profile)
    df["drift_start_time"] = drift_start_time
    return df
