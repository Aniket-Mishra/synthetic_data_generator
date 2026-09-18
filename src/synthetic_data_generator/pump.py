import numpy as np
import pandas as pd

from synthetic_data_generator.wind.faults import severity_label

ALL_COLUMNS = [
    "sample_id", "device", "active_fault", "speed_rpm", "flow_m3h", "differential_head_m",
    "suction_abs_pressure_bar", "discharge_pressure_bar", "ambient_temp_c", "fluid_temp_c",
    "fluid_density_kgm3", "efficiency", "hydraulic_power_kw", "shaft_power_kw", "motor_power_kw",
    "motor_current_a", "vibration_mm_s", "bearing_temp_c", "npsha_m", "npshr_m",
    "cavitation_margin_m", "fault_cavitation", "fault_impeller_wear", "fault_bearing_friction",
    "is_faulted", "fault_labels", "fault_severity",
]


def water_density_kgm3(temperature_c):
    temperature = np.asarray(temperature_c, dtype=float)
    return 1000.0 * (
        1.0
        - ((temperature + 288.9414) / (508929.2 * (temperature + 68.12963)))
        * (temperature - 3.9863) ** 2
    )


def water_vapor_pressure_pa(temperature_c):
    log10_pressure_mmhg = 8.07131 - 1730.63 / (233.426 + np.asarray(temperature_c, dtype=float))
    return (10.0 ** log10_pressure_mmhg) * 133.322368


def make_device_params(rng, overrides=None):
    q_bep_m3h = float(np.clip(rng.normal(90.0, 4.0), 78.0, 100.0))
    h_bep_m = float(np.clip(rng.normal(45.0, 1.8), 40.0, 50.0))
    h0_m = float(np.clip(1.24 * h_bep_m + rng.normal(0.0, 0.8), 1.18 * h_bep_m, 1.30 * h_bep_m))

    params = {
        "nominal_speed_rpm": float(np.clip(rng.normal(1480.0, 18.0), 1440.0, 1510.0)),
        "q_bep_m3h": q_bep_m3h,
        "h_bep_m": h_bep_m,
        "h0_m": h0_m,
        "efficiency_peak": float(np.clip(rng.normal(0.82, 0.015), 0.76, 0.86)),
        "npshr_bep_m": float(np.clip(rng.normal(3.0, 0.25), 2.4, 3.8)),
        "motor_efficiency": float(np.clip(rng.normal(0.93, 0.01), 0.90, 0.96)),
        "motor_power_factor": float(np.clip(rng.normal(0.86, 0.02), 0.80, 0.90)),
        "line_voltage_v": 400.0,
        "sensor_noise_scale": float(np.clip(rng.normal(1.0, 0.06), 0.88, 1.12)),
        "thermal_bias_c": float(np.clip(rng.normal(0.0, 0.8), -2.0, 2.0)),
    }
    if overrides:
        params.update(overrides)

    q_bep_m3s = params["q_bep_m3h"] / 3600.0
    params["pump_curve_k"] = (params["h0_m"] - params["h_bep_m"]) / max(q_bep_m3s ** 2, 1e-9)
    return params


def efficiency_curve(flow_m3s, q_bep_m3s, efficiency_peak):
    ratio = np.divide(flow_m3s, np.maximum(q_bep_m3s, 1e-9))
    efficiency = efficiency_peak * (1.0 - 0.26 * (ratio - 1.0) ** 2)
    return np.clip(efficiency, 0.35, efficiency_peak)


def solve_operating_point(params, speed_rpm, static_head_m, system_k):
    speed_ratio = speed_rpm / params["nominal_speed_rpm"]
    shutoff_head_m = params["h0_m"] * speed_ratio ** 2
    numerator = np.maximum(shutoff_head_m - static_head_m, 0.0)
    denominator = params["pump_curve_k"] + system_k
    flow_m3s = np.sqrt(np.divide(numerator, np.maximum(denominator, 1e-9)))
    head_m = static_head_m + system_k * flow_m3s ** 2
    q_bep_m3s = (params["q_bep_m3h"] / 3600.0) * speed_ratio
    efficiency = efficiency_curve(flow_m3s, q_bep_m3s, params["efficiency_peak"])
    return flow_m3s, head_m, efficiency


def sample_conditions(rng, n, fault_type):
    speed_ratio = rng.uniform(0.78, 1.04, n)
    ambient_temp_c = rng.uniform(8.0, 38.0, n)
    fluid_temp_c = ambient_temp_c + rng.uniform(1.0, 10.0, n)

    if fault_type == "cavitation":
        fluid_temp_c = np.clip(rng.uniform(24.0, 55.0, n), 5.0, 60.0)
        ambient_temp_c = np.clip(fluid_temp_c - rng.uniform(2.0, 10.0, n), 5.0, 45.0)
        suction_abs_pressure_bar = rng.uniform(1.08, 1.28, n)
        suction_static_head_m = rng.uniform(0.2, 2.0, n)
        suction_loss_k = rng.uniform(2500.0, 7000.0, n)
    else:
        suction_abs_pressure_bar = rng.uniform(1.20, 1.70, n)
        suction_static_head_m = rng.uniform(1.0, 4.0, n)
        suction_loss_k = rng.uniform(800.0, 3500.0, n)

    return {
        "speed_ratio": speed_ratio,
        "ambient_temp_c": ambient_temp_c,
        "fluid_temp_c": fluid_temp_c,
        "suction_abs_pressure_bar": suction_abs_pressure_bar,
        "suction_static_head_m": suction_static_head_m,
        "suction_loss_k": suction_loss_k,
        "static_head_m": rng.uniform(10.0, 26.0, n),
        "system_k": rng.uniform(9000.0, 28000.0, n),
    }


def motor_current_a(motor_power_kw, params):
    return (motor_power_kw * 1000.0) / (np.sqrt(3.0) * params["line_voltage_v"] * params["motor_power_factor"])


def recompute_power_columns(df, params):
    hydraulic_power_kw = (
        df["fluid_density_kgm3"].to_numpy()
        * 9.81
        * (df["flow_m3h"].to_numpy() / 3600.0)
        * df["differential_head_m"].to_numpy()
        / 1000.0
    )
    shaft_power_kw = hydraulic_power_kw / np.maximum(df["efficiency"].to_numpy(), 0.35)
    motor_power_kw = shaft_power_kw / params["motor_efficiency"]

    df["hydraulic_power_kw"] = hydraulic_power_kw
    df["shaft_power_kw"] = shaft_power_kw
    df["motor_power_kw"] = motor_power_kw
    df["motor_current_a"] = motor_current_a(motor_power_kw, params)
    df["discharge_pressure_bar"] = df["suction_abs_pressure_bar"] + (
        df["fluid_density_kgm3"] * 9.81 * df["differential_head_m"] / 1e5
    )
    return df


def build_healthy_samples(device_id, params, conditions, rng, sample_offset):
    n = len(conditions["speed_ratio"])
    noise_scale = params["sensor_noise_scale"]

    speed_rpm = params["nominal_speed_rpm"] * conditions["speed_ratio"]
    flow_m3s, head_m, efficiency = solve_operating_point(
        params, speed_rpm, conditions["static_head_m"], conditions["system_k"]
    )

    fluid_density = water_density_kgm3(conditions["fluid_temp_c"])
    hydraulic_power_kw = fluid_density * 9.81 * flow_m3s * head_m / 1000.0
    shaft_power_kw = hydraulic_power_kw / efficiency
    motor_power_kw = shaft_power_kw / params["motor_efficiency"]

    vapor_pressure_pa = water_vapor_pressure_pa(conditions["fluid_temp_c"])
    suction_pressure_pa = conditions["suction_abs_pressure_bar"] * 1e5
    suction_loss_m = conditions["suction_loss_k"] * flow_m3s ** 2
    npsha_m = (
        (suction_pressure_pa - vapor_pressure_pa) / (fluid_density * 9.81)
        + conditions["suction_static_head_m"]
        - suction_loss_m
    )

    speed_ratio = speed_rpm / params["nominal_speed_rpm"]
    q_bep_m3s = (params["q_bep_m3h"] / 3600.0) * speed_ratio
    flow_ratio = np.divide(flow_m3s, np.maximum(q_bep_m3s, 1e-9))
    npshr_m = params["npshr_bep_m"] * np.clip(speed_ratio, 0.6, None) ** 2 * np.clip(flow_ratio, 0.55, 1.35) ** 1.8

    off_bep = np.abs(flow_ratio - 1.0)
    vibration_mm_s = 0.9 + 2.2 * off_bep ** 1.5 + 0.015 * shaft_power_kw
    bearing_temp_c = conditions["ambient_temp_c"] + 16.0 + 0.55 * shaft_power_kw + 5.0 * off_bep ** 1.6 + params["thermal_bias_c"]
    discharge_pressure_bar = conditions["suction_abs_pressure_bar"] + fluid_density * 9.81 * head_m / 1e5

    def noisy(values, sigma):
        return values + rng.normal(0.0, sigma * noise_scale, n)

    data = pd.DataFrame({
        "sample_id": np.arange(sample_offset, sample_offset + n),
        "device": device_id,
        "active_fault": "healthy",
        "speed_rpm": noisy(speed_rpm, 2.0),
        "flow_m3h": noisy(3600.0 * flow_m3s, 0.35),
        "differential_head_m": noisy(head_m, 0.12),
        "suction_abs_pressure_bar": noisy(conditions["suction_abs_pressure_bar"], 0.01),
        "discharge_pressure_bar": noisy(discharge_pressure_bar, 0.015),
        "ambient_temp_c": noisy(conditions["ambient_temp_c"], 0.15),
        "fluid_temp_c": noisy(conditions["fluid_temp_c"], 0.12),
        "fluid_density_kgm3": noisy(fluid_density, 0.4),
        "efficiency": noisy(efficiency, 0.003),
        "hydraulic_power_kw": noisy(hydraulic_power_kw, 0.04),
        "shaft_power_kw": noisy(shaft_power_kw, 0.05),
        "motor_power_kw": noisy(motor_power_kw, 0.05),
        "motor_current_a": noisy(motor_current_a(motor_power_kw, params), 0.08),
        "vibration_mm_s": noisy(vibration_mm_s, 0.05),
        "bearing_temp_c": noisy(bearing_temp_c, 0.18),
        "npsha_m": noisy(npsha_m, 0.04),
        "npshr_m": noisy(npshr_m, 0.03),
        "cavitation_margin_m": noisy(npsha_m - npshr_m, 0.04),
    })
    data["fault_cavitation"] = 0
    data["fault_impeller_wear"] = 0
    data["fault_bearing_friction"] = 0
    data["is_faulted"] = 0
    data["fault_labels"] = "healthy"
    data["fault_severity"] = "none"
    return data


def apply_cavitation_fault(df, severity, params):
    effective_npsha = df["npsha_m"].to_numpy() - (1.0 + 4.0 * severity)
    effective_margin = effective_npsha - df["npshr_m"].to_numpy()
    intensity = severity * np.clip((1.8 - effective_margin) / 2.2, 0.25, 1.0)

    df["flow_m3h"] *= 1.0 - 0.04 * intensity
    df["differential_head_m"] *= 1.0 - 0.10 * intensity
    df["efficiency"] *= 1.0 - 0.12 * intensity
    df["npsha_m"] = effective_npsha
    df["cavitation_margin_m"] = effective_margin

    df = recompute_power_columns(df, params)
    df["vibration_mm_s"] += 1.3 + 3.2 * intensity
    df["bearing_temp_c"] += 1.5 + 4.0 * intensity
    return df


def apply_impeller_wear_fault(df, severity, params):
    df["flow_m3h"] *= 1.0 - 0.03 * severity
    df["differential_head_m"] *= 1.0 - 0.12 * severity
    df["efficiency"] *= 1.0 - 0.08 * severity

    df = recompute_power_columns(df, params)
    df["vibration_mm_s"] += 0.4 + 0.9 * severity
    df["bearing_temp_c"] += 0.6 + 1.6 * severity
    return df


def apply_bearing_friction_fault(df, severity, params):
    df["shaft_power_kw"] *= 1.0 + 0.06 + 0.14 * severity
    df["motor_power_kw"] = df["shaft_power_kw"] / params["motor_efficiency"]
    df["motor_current_a"] = motor_current_a(df["motor_power_kw"].to_numpy(), params)
    df["vibration_mm_s"] += 0.7 + 1.8 * severity
    df["bearing_temp_c"] += 5.0 + 11.0 * severity
    return df


FAULT_REGISTRY = {
    "cavitation": apply_cavitation_fault,
    "impeller_wear": apply_impeller_wear_fault,
    "bearing_friction": apply_bearing_friction_fault,
}


def label_faults(df, fault_type, severity):
    df["active_fault"] = fault_type
    df[f"fault_{fault_type}"] = 1
    df["is_faulted"] = 1
    df["fault_labels"] = fault_type
    df["fault_severity"] = severity_label(severity)
    return df


def round_and_clip(df):
    df["speed_rpm"] = np.round(np.clip(df["speed_rpm"], 0.0, 1800.0), 1)
    df["flow_m3h"] = np.round(np.clip(df["flow_m3h"], 0.0, 240.0), 2)
    df["differential_head_m"] = np.round(np.clip(df["differential_head_m"], 0.0, 80.0), 2)
    df["suction_abs_pressure_bar"] = np.round(np.clip(df["suction_abs_pressure_bar"], 0.8, 3.0), 3)
    df["discharge_pressure_bar"] = np.round(np.clip(df["discharge_pressure_bar"], 0.8, 10.0), 3)
    df["ambient_temp_c"] = np.round(np.clip(df["ambient_temp_c"], 0.0, 50.0), 2)
    df["fluid_temp_c"] = np.round(np.clip(df["fluid_temp_c"], 0.0, 80.0), 2)
    df["fluid_density_kgm3"] = np.round(np.clip(df["fluid_density_kgm3"], 960.0, 1001.0), 2)
    df["efficiency"] = np.round(np.clip(df["efficiency"], 0.30, 0.90), 4)
    df["hydraulic_power_kw"] = np.round(np.clip(df["hydraulic_power_kw"], 0.0, 40.0), 3)
    df["shaft_power_kw"] = np.round(np.clip(df["shaft_power_kw"], 0.0, 55.0), 3)
    df["motor_power_kw"] = np.round(np.clip(df["motor_power_kw"], 0.0, 60.0), 3)
    df["motor_current_a"] = np.round(np.clip(df["motor_current_a"], 0.0, 120.0), 3)
    df["vibration_mm_s"] = np.round(np.clip(df["vibration_mm_s"], 0.0, 20.0), 3)
    df["bearing_temp_c"] = np.round(np.clip(df["bearing_temp_c"], 0.0, 120.0), 2)
    df["npsha_m"] = np.round(np.clip(df["npsha_m"], -10.0, 30.0), 3)
    df["npshr_m"] = np.round(np.clip(df["npshr_m"], 0.0, 15.0), 3)
    df["cavitation_margin_m"] = np.round(np.clip(df["cavitation_margin_m"], -15.0, 25.0), 3)
    return df


def simulate_device(device_config, seed, sample_offset):
    rng = np.random.default_rng(seed)
    params = make_device_params(rng, device_config.get("device_params"))

    fault_mix = device_config["fault_mix"]
    fault_probs = np.array(list(fault_mix.values()), dtype=float)
    counts = rng.multinomial(int(device_config["n_samples"]), fault_probs / fault_probs.sum())

    records = []
    for fault_type, count in zip(fault_mix, counts):
        if count == 0:
            continue
        conditions = sample_conditions(rng, count, fault_type)
        severity = np.zeros(count) if fault_type == "healthy" else rng.uniform(0.15, 1.0, count)
        df = build_healthy_samples(device_config["device_id"], params, conditions, rng, sample_offset)
        if fault_type != "healthy":
            df = FAULT_REGISTRY[fault_type](df, severity, params)
            df = label_faults(df, fault_type, severity)
        records.append(df)
        sample_offset += count

    data = round_and_clip(pd.concat(records, ignore_index=True))
    return data[ALL_COLUMNS], params


def simulate_station(device_configs, seed):
    all_data = []
    device_rows = []
    sample_offset = 0
    for i, cfg in enumerate(device_configs):
        df, params = simulate_device(cfg, seed + i + 1, sample_offset)
        all_data.append(df)
        sample_offset += len(df)
        device_rows.append({"device": cfg["device_id"], **params})

    data = pd.concat(all_data, ignore_index=True)
    data = data.sort_values(["device", "sample_id"]).reset_index(drop=True)
    return data, pd.DataFrame(device_rows)


def run(cfg):
    df, devices = simulate_station(cfg["devices"], seed=cfg["seed"])
    return {cfg["data_name"]: df, cfg["devices_name"]: devices}
