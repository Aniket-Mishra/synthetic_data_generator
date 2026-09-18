import numpy as np
import pandas as pd

from synthetic_data_generator.wind.faults import build_fault_profile, severity_label

# Inverter output in W for irradiance in W/m2, read off a 4.4 MW plant.
POWER_CURVE = np.array([
    (0, 4.491939776), (10, 50.3), (20, 92.425), (30, 136.525), (40, 183.075),
    (50, 227.775), (60, 270.95), (70, 316.84), (80, 361.025), (90, 407.675),
    (100, 451.425), (110, 494.175), (120, 536.575), (130, 575.675), (140, 616.6),
    (150, 658.925), (160, 700.375), (170, 741.425), (180, 782.66), (190, 825.625),
    (200, 869.505), (210, 910.325), (220, 953.05), (230, 997.69), (240, 1040.575),
    (250, 1080.95), (260, 1122.225), (270, 1164.075), (280, 1205.5), (290, 1247.45),
    (300, 1288.25), (310, 1329.715), (320, 1372.76), (330, 1417.265), (340, 1459.125),
    (350, 1500.935), (360, 1543.2), (370, 1582.975), (380, 1621.5), (390, 1663),
    (400, 1704.075), (410, 1744.75), (420, 1783.55), (430, 1824.76), (440, 1867.475),
    (450, 1908.075), (460, 1950.25), (470, 1991.78), (480, 2031.95), (490, 2073),
    (500, 2113.375), (510, 2155.16), (520, 2197.5), (530, 2237), (540, 2276.815),
    (550, 2318), (560, 2359.45), (570, 2402.225), (580, 2445.75), (590, 2488.55),
    (600, 2529), (610, 2572), (620, 2612.725), (630, 2656.25), (640, 2698.2),
    (650, 2742.55), (660, 2784.55), (670, 2825.33), (680, 2867.325), (690, 2905.89),
    (700, 2945.05), (710, 2984.575), (720, 3023), (730, 3062.03), (740, 3101.84),
    (750, 3139.8), (760, 3177.5), (770, 3214.875), (780, 3250.975), (790, 3286.54675),
    (800, 3322.25), (810, 3358.125), (820, 3394.05), (830, 3431.825), (840, 3471.65),
    (850, 3512.475), (860, 3551.125), (870, 3591.155), (880, 3629.935), (890, 3669.375),
    (900, 3707.5), (910, 3745.9), (920, 3786.625), (930, 3826.53), (940, 3867.3),
    (950, 3904.8), (960, 3942.05), (970, 3979), (980, 4015), (990, 4052.4),
    (1000, 4087), (1010, 4124.5), (1020, 4164.85), (1030, 4200), (1040, 4234),
    (1050, 4265.78), (1060, 4294.325), (1070, 4319.375), (1080, 4343), (1090, 4363.125),
    (1100, 4378.5), (1110, 4390), (1120, 4395.5), (1130, 4400), (1140, 4400),
    (1150, 4400), (1160, 4400), (1170, 4400), (1180, 4400), (1200, 4400),
    (1210, 4400), (1220, 4400), (1230, 4400), (1240, 4400), (1250, 4400),
], dtype=float)
IRRADIANCE_POINTS, POWER_POINTS = POWER_CURVE.T

ALL_COLUMNS = [
    "time", "device", "active_power", "irradiance", "clear_sky_irradiance",
    "ambient_temp", "module_temp", "inverter_temp", "dc_voltage", "dc_current",
    "ac_voltage", "ac_current", "performance_ratio", "cloud_cover", "sun_elevation",
    "sun_azimuth", "day_length_hours", "fault_soiling", "fault_inverter_overheat",
    "fault_tracker_stuck", "fault_dc_string_outage", "is_faulted", "fault_labels",
    "downtime_start_time", "fault_severity",
]
FAULT_COLUMN_MAP = {
    "soiling": "fault_soiling",
    "inverter_overheat": "fault_inverter_overheat",
    "tracker_stuck": "fault_tracker_stuck",
    "dc_string_outage": "fault_dc_string_outage",
}
ELECTRICAL_COLUMNS = ["active_power", "dc_current", "ac_current"]


def power_curve_w(irradiance):
    irradiance = np.clip(irradiance, IRRADIANCE_POINTS.min(), IRRADIANCE_POINTS.max())
    return np.interp(irradiance, IRRADIANCE_POINTS, POWER_POINTS)


def outage_steps(window):
    # 6 steps per hour is a 10 minute leftover at 5 minute data. The frozen
    # solar baselines depend on it, so it stays.
    return max(1, int(round(window["duration_hours"] * 6)))


def make_device_params(rng, overrides=None):
    params = {
        "dc_capacity_scale": float(np.clip(rng.normal(1.00, 0.02), 0.95, 1.05)),
        "voltage_bias": float(np.clip(rng.normal(0.0, 5.0), -15.0, 15.0)),
        "sensor_noise_scale": float(np.clip(rng.normal(1.0, 0.08), 0.85, 1.15)),
        "thermal_bias": float(np.clip(rng.normal(0.0, 0.8), -2.0, 2.0)),
        "dc_wiring_efficiency": float(np.clip(rng.normal(0.985, 0.005), 0.97, 0.995)),
        "tracker_gain": float(np.clip(rng.normal(1.0, 0.015), 0.96, 1.03)),
    }
    if overrides:
        params.update(overrides)
    return params


def solar_position(time_index, latitude_deg):
    latitude_rad = np.deg2rad(latitude_deg)
    day_of_year = time_index.dayofyear.to_numpy()
    hour = time_index.hour.to_numpy() + time_index.minute.to_numpy() / 60.0

    gamma = 2.0 * np.pi * (day_of_year - 1) / 365.0
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    solar_time = hour + equation_of_time / 60.0
    hour_angle = np.deg2rad(15.0 * (solar_time - 12.0))

    sin_elevation = np.sin(latitude_rad) * np.sin(declination) + np.cos(
        latitude_rad
    ) * np.cos(declination) * np.cos(hour_angle)
    elevation_rad = np.arcsin(np.clip(sin_elevation, -1.0, 1.0))

    azimuth_rad = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(latitude_rad) - np.tan(declination) * np.cos(latitude_rad),
    )
    azimuth_deg = (np.rad2deg(azimuth_rad) + 180.0) % 360.0

    cos_h0 = np.clip(-np.tan(latitude_rad) * np.tan(declination), -1.0, 1.0)
    day_length_hours = 2.0 * np.rad2deg(np.arccos(cos_h0)) / 15.0

    return np.rad2deg(elevation_rad), azimuth_deg, day_length_hours


def simulate_site_conditions(start, n_days, freq, latitude_deg, seed):
    rng = np.random.default_rng(seed)
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    time = pd.date_range(start=start, periods=n_days * steps_per_day, freq=freq)
    n = len(time)
    step_index = np.arange(n)

    sun_elevation, sun_azimuth, day_length_hours = solar_position(time, latitude_deg)
    daylight = sun_elevation > 0.0
    seasonal = np.sin(2.0 * np.pi * (time.dayofyear.to_numpy() / 365.25 - 0.28))
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60.0
    diurnal = np.sin(2.0 * np.pi * hour / 24.0 - np.pi / 2)

    temp_noise = np.zeros(n)
    cloud_state = np.zeros(n)
    for i in range(1, n):
        temp_noise[i] = 0.985 * temp_noise[i - 1] + rng.normal(0.0, 0.18)
        cloud_state[i] = 0.94 * cloud_state[i - 1] + rng.normal(0.0, 0.20)

    ambient_temp = 18.0 + 11.0 * seasonal + 5.0 * diurnal + temp_noise

    summer_boost = np.clip((seasonal + 1.0) / 2.0, 0.0, 1.0)
    convective_clouds = np.clip((diurnal + 0.2) * summer_boost, 0.0, None)
    cloud_cover = (
        0.32
        + 0.18 * np.sin(2.0 * np.pi * step_index / (steps_per_day * 6.0) + 0.4)
        + 0.10 * convective_clouds
        + 0.20 * cloud_state
    )
    cloud_cover = np.clip(cloud_cover, 0.02, 0.95)

    storm_centers = rng.integers(0, n, size=max(8, n_days // 30))
    storm_profile = np.zeros(n)
    for center in storm_centers:
        width = rng.integers(6, 36)
        depth = rng.uniform(0.10, 0.40)
        storm_profile += depth * np.exp(-0.5 * ((step_index - center) / width) ** 2)
    cloud_cover = np.clip(cloud_cover + storm_profile, 0.02, 0.98)

    elevation_rad = np.deg2rad(np.clip(sun_elevation, 0.0, None))
    extraterrestrial = 1361.0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * time.dayofyear.to_numpy() / 365.0))
    clear_sky_irradiance = extraterrestrial * np.power(np.sin(elevation_rad), 1.12)
    clear_sky_irradiance = np.clip(clear_sky_irradiance * 0.94, 0.0, 1250.0)

    transmittance = 1.0 - 0.72 * np.power(cloud_cover, 1.35)
    variability = 1.0 + rng.normal(0.0, 0.015, n)
    irradiance = clear_sky_irradiance * transmittance * variability
    irradiance = np.where(daylight, irradiance, 0.0)
    irradiance = np.clip(irradiance, 0.0, clear_sky_irradiance)

    return pd.DataFrame({
        "time": time,
        "ambient_temp_site": ambient_temp,
        "cloud_cover_site": cloud_cover,
        "clear_sky_irradiance_site": clear_sky_irradiance,
        "irradiance_site": irradiance,
        "sun_elevation_site": np.clip(sun_elevation, -90.0, 90.0),
        "sun_azimuth_site": sun_azimuth,
        "day_length_hours_site": day_length_hours,
    })


def generate_healthy_device(site_df, device_id, device_params, seed):
    rng = np.random.default_rng(seed)
    n = len(site_df)
    noise_scale = device_params["sensor_noise_scale"]

    ambient_temp = site_df["ambient_temp_site"].to_numpy() + rng.normal(0.0, 0.18 * noise_scale, n)
    cloud_cover = np.clip(site_df["cloud_cover_site"].to_numpy() + rng.normal(0.0, 0.015 * noise_scale, n), 0.0, 1.0)
    clear_sky_irradiance = np.clip(site_df["clear_sky_irradiance_site"].to_numpy() + rng.normal(0.0, 4.0 * noise_scale, n), 0.0, 1250.0)
    sun_elevation = site_df["sun_elevation_site"].to_numpy()
    daylight = sun_elevation > 0.0

    tracker_efficiency = np.where(daylight, 0.985 + 0.010 * device_params["tracker_gain"], 0.0)
    thermal_derate = np.clip(1.0 - 0.0016 * np.clip(ambient_temp - 32.0, 0.0, None), 0.88, 1.0)

    irradiance = site_df["irradiance_site"].to_numpy() * tracker_efficiency
    irradiance = irradiance * device_params["dc_capacity_scale"] * thermal_derate * device_params["dc_wiring_efficiency"]
    irradiance = np.where(daylight, irradiance, 0.0)
    irradiance = np.clip(irradiance + rng.normal(0.0, 5.0 * noise_scale, n), 0.0, 1250.0)

    active_power = np.where(daylight, power_curve_w(irradiance), 0.0)

    module_temp = ambient_temp + 0.028 * irradiance + 0.8 * cloud_cover + device_params["thermal_bias"]
    inverter_temp = ambient_temp + 0.0105 * active_power + 0.35 * cloud_cover + 0.5 * device_params["thermal_bias"]

    dc_voltage = 820.0 - 0.55 * np.clip(module_temp - 25.0, -25.0, 55.0) + device_params["voltage_bias"]
    dc_voltage = np.clip(np.where(active_power > 1.0, dc_voltage, 0.0), 0.0, 950.0)

    dc_current = np.divide(active_power * 1.03, np.maximum(dc_voltage, 1.0), out=np.zeros(n), where=dc_voltage > 0.0)
    ac_voltage = np.where(active_power > 1.0, 400.0 + rng.normal(0.0, 2.0 * noise_scale, n), 0.0)
    ac_current = np.divide(active_power, np.sqrt(3.0) * np.maximum(ac_voltage, 1.0), out=np.zeros(n), where=ac_voltage > 0.0)

    clear_power = power_curve_w(np.clip(clear_sky_irradiance, 0.0, 1250.0))
    performance_ratio = np.divide(active_power, np.maximum(clear_power, 1.0), out=np.zeros(n), where=clear_power > 1.0)
    performance_ratio = np.clip(performance_ratio, 0.0, 1.02)

    df = pd.DataFrame({
        "time": site_df["time"].to_numpy(),
        "device": device_id,
        "active_power": active_power,
        "irradiance": irradiance,
        "clear_sky_irradiance": clear_sky_irradiance,
        "ambient_temp": ambient_temp,
        "module_temp": module_temp,
        "inverter_temp": inverter_temp,
        "dc_voltage": dc_voltage,
        "dc_current": dc_current,
        "ac_voltage": ac_voltage,
        "ac_current": ac_current,
        "performance_ratio": performance_ratio,
        "cloud_cover": cloud_cover,
        "sun_elevation": sun_elevation,
        "sun_azimuth": site_df["sun_azimuth_site"].to_numpy(),
        "day_length_hours": site_df["day_length_hours_site"].to_numpy(),
    })
    for col in FAULT_COLUMN_MAP.values():
        df[col] = 0
    return df


def fault_profile(df, fault_cfg):
    return build_fault_profile(
        n=len(df),
        start_idx=fault_cfg["start_idx"],
        ramp_steps=fault_cfg["ramp_steps"],
        end_idx=fault_cfg.get("end_idx"),
        shape=fault_cfg["shape"],
        max_severity=fault_cfg["max_severity"],
    )


def apply_soiling_fault(df, fault_cfg):
    profile = fault_profile(df, fault_cfg)
    if not profile.any():
        return df, profile

    sunlight_factor = np.clip(df["irradiance"].to_numpy() / 900.0, 0.0, 1.0)
    power_factor = 1.0 - (0.03 + 0.14 * profile) * sunlight_factor

    df["irradiance"] *= power_factor
    df["active_power"] = power_curve_w(df["irradiance"].to_numpy())
    df["module_temp"] += 1.2 * profile * sunlight_factor
    df["performance_ratio"] *= np.clip(power_factor, 0.0, 1.0)
    return df, profile


def apply_inverter_overheat_fault(df, fault_cfg):
    n = len(df)
    step_index = np.arange(n)
    hour = df["time"].dt.hour.to_numpy() + df["time"].dt.minute.to_numpy() / 60.0
    ambient_temp = df["ambient_temp"].to_numpy()
    irradiance = df["irradiance"].to_numpy()
    hot_midday = (
        (step_index >= fault_cfg["start_idx"])
        & (step_index < fault_cfg["end_idx"])
        & (ambient_temp >= fault_cfg["temp_threshold"])
        & (irradiance >= fault_cfg["irradiance_threshold"])
        & (hour >= 11.0)
        & (hour <= 16.5)
    )
    if not hot_midday.any():
        return df, np.zeros(n)

    base = np.clip((ambient_temp - fault_cfg["temp_threshold"]) / 10.0, 0.0, 1.0)
    base += np.clip((irradiance - fault_cfg["irradiance_threshold"]) / 350.0, 0.0, 1.0)
    intermittent = (np.sin(step_index / 2.3) > 0.15).astype(float)
    profile = np.where(hot_midday, np.clip(0.45 * base + 0.55 * intermittent, 0.0, 1.0), 0.0)

    trip = profile >= fault_cfg["trip_level"]
    derate = (profile > 0.0) & ~trip
    df.loc[derate, "active_power"] *= 1.0 - 0.22 * profile[derate]
    df.loc[trip, ELECTRICAL_COLUMNS] = 0.0
    df.loc[trip, "ac_voltage"] = 0.0
    df.loc[trip, "performance_ratio"] = 0.0
    df["inverter_temp"] += 10.0 * profile
    return df, profile


def apply_tracker_stuck_fault(df, fault_cfg):
    profile = fault_profile(df, fault_cfg)
    if not profile.any():
        return df, profile

    hour = df["time"].dt.hour.to_numpy() + df["time"].dt.minute.to_numpy() / 60.0
    shoulder_factor = np.clip(np.abs(hour - 12.0) / 5.0, 0.0, 1.0)
    sun_factor = np.clip(df["sun_elevation"].to_numpy() / 45.0, 0.0, 1.0)
    tracking_loss = 0.05 + 0.32 * profile * shoulder_factor * (1.15 - 0.35 * sun_factor)
    power_factor = np.clip(1.0 - tracking_loss, 0.55, 1.0)

    df["irradiance"] *= power_factor
    df["active_power"] = power_curve_w(df["irradiance"].to_numpy())
    df["module_temp"] += 0.6 * profile * shoulder_factor
    df["performance_ratio"] *= power_factor
    return df, profile


def apply_dc_string_outage_fault(df, fault_cfg):
    profile = fault_profile(df, fault_cfg)
    if not profile.any():
        return df, profile

    power_factor = 1.0 - (0.10 + 0.22 * profile)
    df["active_power"] *= power_factor
    df["dc_current"] *= power_factor
    df["dc_voltage"] *= 1.0 - 0.03 * profile
    df["performance_ratio"] *= power_factor
    df["inverter_temp"] += 1.0 * profile
    return df, profile


FAULT_REGISTRY = {
    "soiling": apply_soiling_fault,
    "inverter_overheat": apply_inverter_overheat_fault,
    "tracker_stuck": apply_tracker_stuck_fault,
    "dc_string_outage": apply_dc_string_outage_fault,
}


def outage_profile(n, outages, steps_per_day):
    profile = np.zeros(n, dtype=float)
    for window in outages:
        start_idx = int(window["start_day"] * steps_per_day)
        end_idx = min(n, start_idx + outage_steps(window))
        profile[start_idx:end_idx] = np.maximum(profile[start_idx:end_idx], window["severity"])
    return profile


def finalize_fault_labels(df, fault_states, downtime, fault_start_times):
    n = len(df)
    if not fault_states and not downtime.any():
        df["is_faulted"] = 0
        df["fault_labels"] = "healthy"
        df["downtime_start_time"] = pd.NaT
        df["fault_severity"] = "none"
        return df

    profiles = list(fault_states.values()) if fault_states else [np.zeros(n)]
    max_profile = np.maximum.reduce(profiles + [downtime])
    df["is_faulted"] = (max_profile > 0).astype(int)
    df["fault_severity"] = severity_label(max_profile)

    labels = np.full(n, "healthy", dtype=object)
    for fault_type, profile in fault_states.items():
        active = profile > 0
        df[FAULT_COLUMN_MAP[fault_type]] = active.astype(int)
        labels = np.where(active & (labels == "healthy"), fault_type, labels)
        labels = np.where(active & (labels != "healthy") & (labels != fault_type), labels + "|" + fault_type, labels)

    downtime_active = downtime > 0
    labels = np.where(downtime_active & (labels == "healthy"), "downtime", labels)
    labels = np.where(downtime_active & (labels != "healthy") & (labels != "downtime"), labels + "|downtime", labels)

    downtime_start_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if fault_start_times:
        downtime_start_time[:] = min(fault_start_times)

    df["fault_labels"] = labels
    df["downtime_start_time"] = downtime_start_time
    return df


def round_and_clip(df):
    night = (df["sun_elevation"] <= 0.5) | (df["irradiance"] <= 0.5)
    df.loc[night, ELECTRICAL_COLUMNS + ["irradiance", "dc_voltage", "ac_voltage", "performance_ratio"]] = 0.0

    df["active_power"] = np.round(np.clip(df["active_power"], 0.0, 4400.0), 2)
    df["irradiance"] = np.round(np.clip(df["irradiance"], 0.0, 1250.0), 2)
    df["clear_sky_irradiance"] = np.round(np.clip(df["clear_sky_irradiance"], 0.0, 1250.0), 2)
    df["ambient_temp"] = np.round(np.clip(df["ambient_temp"], -20.0, 55.0), 2)
    df["module_temp"] = np.round(np.clip(df["module_temp"], -20.0, 95.0), 2)
    df["inverter_temp"] = np.round(np.clip(df["inverter_temp"], -20.0, 100.0), 2)
    df["dc_voltage"] = np.round(np.clip(df["dc_voltage"], 0.0, 950.0), 2)
    df["dc_current"] = np.round(np.clip(df["dc_current"], 0.0, 12.0), 3)
    df["ac_voltage"] = np.round(np.clip(df["ac_voltage"], 0.0, 460.0), 2)
    df["ac_current"] = np.round(np.clip(df["ac_current"], 0.0, 8.0), 3)
    df["performance_ratio"] = np.round(np.clip(df["performance_ratio"], 0.0, 1.05), 4)
    df["cloud_cover"] = np.round(np.clip(df["cloud_cover"], 0.0, 1.0), 3)
    df["sun_elevation"] = np.round(np.clip(df["sun_elevation"], -90.0, 90.0), 2)
    df["sun_azimuth"] = np.round(df["sun_azimuth"] % 360.0, 2)
    df["day_length_hours"] = np.round(np.clip(df["day_length_hours"], 0.0, 24.0), 2)
    return df


def simulate_device(site_df, device_config, steps_per_day, seed):
    rng = np.random.default_rng(seed)
    device_params = make_device_params(rng, device_config.get("device_params"))
    df = generate_healthy_device(site_df, device_config["device_id"], device_params, seed)

    fault_states = {}
    fault_start_times = []
    for fault in device_config["faults"]:
        fault_cfg = dict(fault)
        fault_cfg["start_idx"] = int(fault["start_day"] * steps_per_day)
        if "end_day" in fault:
            fault_cfg["end_idx"] = int(fault["end_day"] * steps_per_day)
        fault_cfg["ramp_steps"] = max(1, int(fault.get("ramp_days", 1) * steps_per_day))

        df, profile = FAULT_REGISTRY[fault["type"]](df, fault_cfg)
        fault_states[fault["type"]] = np.maximum(fault_states.get(fault["type"], np.zeros(len(df))), profile)
        if profile.any():
            fault_start_times.append(site_df["time"].iloc[int(np.argmax(profile > 0))])

    downtime = outage_profile(len(df), device_config["outages"], steps_per_day)
    if downtime.any():
        fault_start_times.append(site_df["time"].iloc[int(np.argmax(downtime > 0))])
        down = downtime > 0
        df.loc[down, ELECTRICAL_COLUMNS] = 0.0
        df.loc[down, "ac_voltage"] = 0.0
        df.loc[down, "performance_ratio"] = 0.0

    df = finalize_fault_labels(df, fault_states, downtime, fault_start_times)
    return round_and_clip(df)[ALL_COLUMNS]


def time_at(site_df, step):
    if step is None or step >= len(site_df):
        return pd.NaT
    return site_df["time"].iloc[step]


def build_fault_event_table(site_df, device_configs, steps_per_day):
    rows = []
    for cfg in device_configs:
        for fault in cfg["faults"]:
            end_idx = int(fault["end_day"] * steps_per_day) if "end_day" in fault else None
            rows.append({
                "device": cfg["device_id"],
                "event_type": fault["type"],
                "start_time": time_at(site_df, int(fault["start_day"] * steps_per_day)),
                "end_time": time_at(site_df, end_idx),
                "shape": fault.get("shape", "linear"),
                "max_severity": fault.get("max_severity", 1.0),
            })
        for outage in cfg["outages"]:
            start_idx = int(outage["start_day"] * steps_per_day)
            rows.append({
                "device": cfg["device_id"],
                "event_type": "downtime",
                "start_time": time_at(site_df, start_idx),
                "end_time": time_at(site_df, start_idx + outage_steps(outage)),
                "shape": "abrupt",
                "max_severity": outage["severity"],
            })
    return pd.DataFrame(rows)


def simulate_farm(site_config, device_configs, seed):
    site_df = simulate_site_conditions(**site_config, seed=seed)
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(site_config["freq"]))
    devices = [
        simulate_device(site_df, cfg, steps_per_day, seed + i + 1)
        for i, cfg in enumerate(device_configs)
    ]
    data = pd.concat(devices, ignore_index=True).sort_values(["time", "device"]).reset_index(drop=True)
    return data, build_fault_event_table(site_df, device_configs, steps_per_day)


def run(cfg):
    df, events = simulate_farm(cfg["site"], cfg["devices"], seed=cfg["seed"])
    return {cfg["stream_name"]: df, cfg["events_name"]: events}
