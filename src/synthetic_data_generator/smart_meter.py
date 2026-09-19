import numpy as np
import pandas as pd

from synthetic_data_generator.wind.curves import power_curve_kw

TIMEZONE = "Europe/Amsterdam"
LATITUDE_DEG = 52.1
LONGITUDE_DEG = 5.18

WIND_CAPACITY_GW = 11.8
SOLAR_CAPACITY_GW = 25.9
MEAN_LOAD_GW = 13.6
LOAD_SHAPE = np.array(
    [0.76, 0.73, 0.72, 0.72, 0.74, 0.80, 0.90, 1.02, 1.12, 1.22, 1.30, 1.35,
     1.37, 1.34, 1.28, 1.20, 1.12, 1.08, 1.05, 1.01, 0.96, 0.90, 0.84, 0.79]
)
QUARTER_HOUR_PRICES_FROM = pd.Timestamp("2025-10-01", tz=TIMEZONE)


def autoregressive_noise(rng, n, phi, sigma):
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + rng.normal(0, sigma)
    return noise


def simulate_time(site):
    steps_per_day = pd.Timedelta("1D") // pd.Timedelta(site["freq"])
    return pd.date_range(
        site["start"],
        periods=site["n_days"] * steps_per_day,
        freq=site["freq"],
        tz=TIMEZONE,
    )


def sun_elevation_deg(time):
    day_of_year = time.dayofyear.to_numpy()
    utc = time.tz_convert("UTC")
    hour_utc = utc.hour.to_numpy() + utc.minute.to_numpy() / 60.0

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
    solar_time = hour_utc + LONGITUDE_DEG / 15.0 + equation_of_time / 60.0
    hour_angle = np.deg2rad(15.0 * (solar_time - 12.0))

    latitude = np.deg2rad(LATITUDE_DEG)
    sin_elevation = np.sin(latitude) * np.sin(declination) + np.cos(
        latitude
    ) * np.cos(declination) * np.cos(hour_angle)
    return np.rad2deg(np.arcsin(np.clip(sin_elevation, -1.0, 1.0)))


def simulate_weather(time, rng):
    n = len(time)
    day_of_year = time.dayofyear.to_numpy()
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60.0

    temperature = (
        10.8
        + 7.8 * np.sin(2 * np.pi * (day_of_year / 365.25 - 0.31))
        + 3.5 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
        + autoregressive_noise(rng, n, 0.995, 0.35)
    )

    sin_elevation = np.clip(np.sin(np.deg2rad(sun_elevation_deg(time))), 0.0, None)
    clear_sky = (
        0.94 * 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365.0))
        * sin_elevation**1.12
    )
    cloud_cover = np.clip(
        0.68 + 0.25 * autoregressive_noise(rng, n, 0.97, 0.25), 0.02, 0.98
    )
    irradiance = clear_sky * (1.0 - 0.80 * cloud_cover**1.35)

    wind_speed = (
        6.0
        + 1.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
        + autoregressive_noise(rng, n, 0.99, 0.30)
    )
    steps_per_day = pd.Timedelta("1D") // (time[1] - time[0])
    storm_centers = rng.integers(0, n, size=max(6, n // (steps_per_day * 45)))
    step = np.arange(n)
    for center in storm_centers:
        width = rng.integers(16, 64)
        amplitude = rng.uniform(6.0, 14.0)
        wind_speed += amplitude * np.exp(-0.5 * ((step - center) / width) ** 2)

    return temperature, irradiance, np.clip(wind_speed, 0.0, 30.0)


def simulate_prices(time, is_holiday, irradiance, wind_speed, rng):
    n = len(time)
    day_of_year = time.dayofyear.to_numpy()
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60.0
    off_day = (time.dayofweek.to_numpy() >= 5) | is_holiday

    load_gw = (
        MEAN_LOAD_GW
        * np.interp(hour, np.arange(24), LOAD_SHAPE, period=24)
        * np.where(off_day, 0.94, 1.03)
        * (1 + 0.08 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25))
    )
    wind_gw = WIND_CAPACITY_GW * power_curve_kw(np.minimum(wind_speed, 20.0)) / 1790.0
    solar_gw = SOLAR_CAPACITY_GW * irradiance / 1000.0 * 0.95
    residual_gw = load_gw - wind_gw - solar_gw

    price = (
        12.0
        + 10.0 * residual_gw
        + 3.0 * np.maximum(residual_gw - 12.0, 0.0) ** 2
        + autoregressive_noise(rng, n, 0.9, 6.0)
    )
    price = np.where(price < 0, 0.15 * price, price)

    steps_per_hour = int(pd.Timedelta("1h") / (time[1] - time[0]))
    hourly_until = time.searchsorted(QUARTER_HOUR_PRICES_FROM)
    hourly_means = price[:hourly_until].reshape(-1, steps_per_hour).mean(axis=1)
    price[:hourly_until] = np.repeat(hourly_means, steps_per_hour)
    return price


def simulate_context(site, holidays, seed):
    rng = np.random.default_rng(seed)
    time = simulate_time(site)
    holiday_dates = pd.DatetimeIndex(holidays).tz_localize(TIMEZONE)
    is_holiday = time.normalize().isin(holiday_dates)
    temperature, irradiance, wind_speed = simulate_weather(time, rng)
    price = simulate_prices(time, is_holiday, irradiance, wind_speed, rng)
    return pd.DataFrame(
        {
            "time": time,
            "temperature_c": np.round(temperature, 1),
            "irradiance_w_m2": np.round(irradiance, 0),
            "wind_speed_m_s": np.round(wind_speed, 1),
            "day_ahead_price_eur_mwh": np.round(price, 2),
            "is_holiday": is_holiday,
        }
    )


SUMMER_HOLIDAY_DAYS = (193, 243)


def sample_events(rng, household, time, rates):
    events = []
    for _ in range(rng.poisson(rates["vacation_per_year"])):
        days = rng.integers(5, 17)
        events.append(sample_window(rng, household, "vacation", time, pd.Timedelta(days=days), 0.6))
    if household["pv_kwp"] > 0:
        for _ in range(rng.poisson(rates["pv_inverter_outage_per_year"])):
            days = rng.integers(2, 15)
            events.append(sample_window(rng, household, "pv_inverter_outage", time, pd.Timedelta(days=days)))
    for _ in range(rng.poisson(rates["meter_outage_per_year"])):
        hours = rng.integers(1, 37)
        events.append(sample_window(rng, household, "meter_outage", time, pd.Timedelta(hours=hours)))

    has_heat_pump = household["heating"] in ("heat_pump", "hybrid_heat_pump")
    if household["ev_charger_kw"] > 0 and rng.random() < rates["ev_acquired_share"]:
        events.append(sample_change(rng, household, "ev_acquired", time))
    if has_heat_pump and rng.random() < rates["heat_pump_installed_share"]:
        events.append(sample_change(rng, household, "heat_pump_installed", time))
    if household["contract"] == "dynamic" and rng.random() < rates["switched_to_dynamic_share"]:
        events.append(sample_change(rng, household, "switched_to_dynamic", time))
    return events


def sample_window(rng, household, event_type, time, duration, summer_share=0.0):
    steps_per_day = pd.Timedelta("1D") // (time[1] - time[0])
    if rng.random() < summer_share:
        start = time[rng.integers(*SUMMER_HOLIDAY_DAYS) * steps_per_day]
    else:
        start = time[rng.integers(0, len(time))]
    return {
        "household_id": household["household_id"],
        "event_type": event_type,
        "start_time": start,
        "end_time": start + duration,
    }


def sample_change(rng, household, event_type, time):
    return {
        "household_id": household["household_id"],
        "event_type": event_type,
        "start_time": time[rng.integers(0, len(time))],
        "end_time": pd.NaT,
    }


def active_mask(event, time):
    if pd.isna(event["end_time"]):
        return time >= event["start_time"]
    return (time >= event["start_time"]) & (time < event["end_time"])


def any_active(events, event_type, time):
    mask = np.zeros(len(time), dtype=bool)
    for event in events:
        if event["event_type"] == event_type:
            mask |= active_mask(event, time)
    return mask


def change_time(events, event_type, default):
    for event in events:
        if event["event_type"] == event_type:
            return event["start_time"]
    return default


def event_labels(events, time):
    labels = np.full(len(time), "none", dtype=object)
    for event in events:
        active = active_mask(event, time)
        labels[active] = np.where(
            labels[active] == "none",
            event["event_type"],
            labels[active] + "|" + event["event_type"],
        )
    return labels


YEARLY_KWH_BY_OCCUPANTS = {1: 1600, 2: 2410, 3: 2950, 4: 3510, 5: 3880}
TYPICAL_FLOOR_AREA_M2 = {
    "apartment": 70,
    "terraced": 110,
    "corner": 120,
    "semi_detached": 140,
    "detached": 180,
}
MEAN_OCCUPANTS = {
    "apartment": 1.6,
    "terraced": 2.4,
    "corner": 2.5,
    "semi_detached": 2.8,
    "detached": 3.0,
}
HEAT_KWH_PER_M2 = {"A": 35, "B": 50, "C": 65, "D": 80, "E": 95, "F": 110, "G": 125}
HOT_WATER_KWH_PER_OCCUPANT = 800
GAS_KWH_PER_M3 = 8.8
HOT_WATER_COP = 2.5
HYBRID_SWITCH_C = 4.0
WEEKDAY_SHAPE = np.array(
    [0.55, 0.55, 0.55, 0.55, 0.55, 0.65, 0.95, 1.20, 1.10, 0.85, 0.80, 0.80,
     0.80, 0.80, 0.80, 0.90, 1.20, 1.60, 1.65, 1.60, 1.50, 1.30, 1.10, 0.80]
)
WEEKEND_SHAPE = np.array(
    [0.60, 0.55, 0.55, 0.55, 0.55, 0.55, 0.65, 0.85, 1.10, 1.20, 1.15, 1.15,
     1.15, 1.10, 1.10, 1.10, 1.20, 1.55, 1.60, 1.55, 1.45, 1.30, 1.10, 0.80]
)


def choose(rng, shares):
    names = list(shares)
    weights = np.array(list(shares.values()), dtype=float)
    return names[rng.choice(len(names), p=weights / weights.sum())]


def sample_household(rng, index, population):
    dwelling = choose(rng, population["dwelling_type"])
    heating = choose(rng, population["heating"])
    if heating == "district_heating" and dwelling not in ("apartment", "terraced"):
        heating = "gas_boiler"
    has_pv = rng.random() < population["pv_share_by_dwelling"][dwelling]
    has_ev = rng.random() < population["ev_share"]
    dynamic_share = population["dynamic_contract_share"]["ev" if has_ev else "no_ev"]
    if rng.random() < dynamic_share:
        contract = "dynamic"
    else:
        contract = choose(rng, {"single": 0.3, "dual": 0.7})
    return {
        "household_id": f"HH{index + 1:03d}",
        "dwelling_type": dwelling,
        "floor_area_m2": int(TYPICAL_FLOOR_AREA_M2[dwelling] * rng.lognormal(0, 0.15)),
        "energy_label": choose(rng, population["energy_label"]),
        "occupants": int(np.clip(rng.poisson(MEAN_OCCUPANTS[dwelling] - 1) + 1, 1, 5)),
        "heating": heating,
        "pv_kwp": round(rng.uniform(2, 8) * 2) / 2 if has_pv else 0.0,
        "ev_charger_kw": (11.0 if rng.random() < 0.65 else 7.4) if has_ev else 0.0,
        "contract": contract,
    }


def base_load_kwh(rng, steps, occupants, step_hours):
    hour = steps["hour"].to_numpy()
    shape = np.where(
        steps["workday"],
        np.interp(hour, np.arange(24), WEEKDAY_SHAPE, period=24),
        np.interp(hour, np.arange(24), WEEKEND_SHAPE, period=24),
    )
    lighting_kw = 0.06 * np.sqrt(occupants) * steps["dark"].to_numpy() * (hour >= 6)
    spikes = rng.poisson(0.15 * occupants * ((hour >= 7) & (hour < 23))) * np.minimum(
        rng.lognormal(np.log(0.12), 0.6, len(hour)), 1.5
    )
    load = (0.22 * shape + lighting_kw) * step_hours + spikes
    yearly_kwh = YEARLY_KWH_BY_OCCUPANTS[occupants] * rng.lognormal(0, 0.15)
    load *= yearly_kwh / load.sum()
    return load * np.where(steps["vacation"], 0.3, 1.0)


def hot_water_weights(hour):
    morning = (hour >= 6.5) & (hour < 8.5)
    daytime = (hour >= 8.5) & (hour < 18)
    evening = (hour >= 18) & (hour < 22)
    return 0.40 * morning / 2.0 + 0.20 * daytime / 9.5 + 0.40 * evening / 4.0


def heat_demand_kwh(rng, steps, household, step_hours):
    hour = steps["hour"].to_numpy()
    vacation = steps["vacation"].to_numpy()
    setpoint = np.where((hour >= 7) & (hour < 23), 20.0, 16.0)
    setpoint = np.where(vacation, 15.0, setpoint)
    degree_hours = np.maximum(setpoint - 3.5 - steps["temperature_c"].to_numpy(), 0.0) * step_hours
    yearly_space_heat = household["floor_area_m2"] * HEAT_KWH_PER_M2[household["energy_label"]]
    space_heat = yearly_space_heat * degree_hours / degree_hours.sum()

    day_index = np.cumsum(hour == 0) - 1
    daily_factor = rng.lognormal(0, 0.3, day_index[-1] + 1)[day_index]
    daily_hot_water = HOT_WATER_KWH_PER_OCCUPANT * household["occupants"] / 365.0
    hot_water = daily_hot_water * hot_water_weights(hour) * step_hours * daily_factor * ~vacation
    return space_heat, hot_water


def heat_pump_cop(temperature):
    return np.clip(3.0 + 0.11 * temperature, 1.8, 4.5)


def route_heat(space_heat, hot_water, heating_type, temperature):
    gas_boiler = heating_type == "gas_boiler"
    hybrid = heating_type == "hybrid_heat_pump"
    heat_pump = heating_type == "heat_pump"
    district = heating_type == "district_heating"
    hybrid_on_pump = hybrid & (temperature >= HYBRID_SWITCH_C)

    pump_heat = np.where(heat_pump | hybrid_on_pump, space_heat, 0.0)
    boiler_heat = np.where(gas_boiler | (hybrid & ~hybrid_on_pump), space_heat, 0.0)
    boiler_heat += np.where(gas_boiler | hybrid, hot_water, 0.0)

    heating_kwh = pump_heat / heat_pump_cop(temperature)
    heating_kwh += np.where(heat_pump, hot_water / HOT_WATER_COP, 0.0)
    gas_m3 = np.where(gas_boiler | hybrid, boiler_heat / GAS_KWH_PER_M3, np.nan)
    heat_gj = np.where(district, (space_heat + hot_water) * 0.0036, np.nan)
    return heating_kwh, gas_m3, heat_gj


def pv_kwh(steps, kwp, orientation, step_hours):
    irradiance = steps["irradiance_w_m2"].to_numpy()
    module_temperature = steps["temperature_c"].to_numpy() + 0.03 * irradiance
    power_kw = kwp * irradiance / 1000.0 * orientation * 0.83 * (1 - 0.004 * (module_temperature - 25))
    return np.clip(power_kw, 0.0, 0.9 * kwp) * step_hours


def step_at_hour(time, day_start, hour):
    quarters = int(np.clip(round(float(hour) * 4), 0, 95))
    clock_hour, quarter = divmod(quarters, 4)
    return int(
        time.searchsorted(
            time[day_start].normalize().replace(hour=int(clock_hour), minute=int(quarter * 15))
        )
    )


def charging_window(rng, start, next_start, workday, time, charger_kw, step_hours):
    n = len(workday)
    workday_tomorrow = workday[next_start] if next_start < n else True
    arrival_hour = rng.normal(17.75, 1.0) if workday[start] else rng.normal(15.5, 2.5)
    departure_hour = rng.normal(7.75, 0.75) if workday_tomorrow else rng.normal(9.5, 1.5)
    arrival = step_at_hour(time, start, np.clip(arrival_hour, 12.0, 23.0))
    departure = n if next_start >= n else step_at_hour(time, next_start, np.clip(departure_hour, 5.0, 12.0))
    energy_kwh = np.clip(rng.lognormal(np.log(9.0), 0.45), 2.0, 40.0)
    steps_needed = int(np.ceil(energy_kwh / (charger_kw * step_hours)))
    return arrival, min(departure, n), steps_needed


def apply_charging(ev, energy_per_step, arrival, departure, steps_needed, strategy, night_tariff_step, price):
    if departure <= arrival:
        return
    n = min(steps_needed, departure - arrival)
    if strategy == "dynamic":
        ev[arrival + np.argpartition(price[arrival:departure], n - 1)[:n]] = energy_per_step
        return
    start = min(max(arrival, night_tariff_step), departure) if strategy == "dual" else arrival
    ev[start : min(start + n, departure)] = energy_per_step


def ev_kwh(rng, steps, household, ev_from, dynamic_from, step_hours):
    ev = np.zeros(len(steps))
    charger_kw = household["ev_charger_kw"]
    if charger_kw == 0:
        return ev

    time = pd.DatetimeIndex(steps["time"])
    workday = steps["workday"].to_numpy()
    vacation = steps["vacation"].to_numpy()
    has_ev = np.asarray(time >= ev_from)
    contract_before = "dual" if household["contract"] == "dynamic" else household["contract"]
    strategy = np.where(time >= dynamic_from, "dynamic", contract_before)
    price = steps["day_ahead_price_eur_mwh"].to_numpy()
    energy_per_step = charger_kw * step_hours

    day_starts = np.flatnonzero(steps["hour"].to_numpy() == 0)
    for start, next_start in zip(day_starts, np.append(day_starts[1:], len(steps))):
        session_chance = 0.65 if workday[start] else 0.40
        if not has_ev[start] or vacation[start] or rng.random() > session_chance:
            continue
        arrival, departure, steps_needed = charging_window(
            rng, start, next_start, workday, time, charger_kw, step_hours
        )
        night_tariff_step = step_at_hour(time, start, 23.0) if workday[start] else start
        apply_charging(
            ev,
            energy_per_step,
            arrival,
            departure,
            steps_needed,
            strategy[start],
            night_tariff_step,
            price,
        )
    return ev


def hourly_totals(values, steps_per_hour):
    hourly = np.full(len(values), np.nan)
    hourly[::steps_per_hour] = values.reshape(-1, steps_per_hour).sum(axis=1)
    return hourly


def simulate_household(context, household, events, rng):
    time = pd.DatetimeIndex(context["time"])
    step_hours = (time[1] - time[0]) / pd.Timedelta("1h")
    steps_per_hour = round(1 / step_hours)
    steps = pd.DataFrame(
        {
            "time": time,
            "hour": time.hour + time.minute / 60.0,
            "workday": ~((time.dayofweek >= 5) | context["is_holiday"].to_numpy()),
            "dark": context["irradiance_w_m2"].to_numpy() < 5,
            "vacation": any_active(events, "vacation", time),
            "temperature_c": context["temperature_c"].to_numpy(),
            "irradiance_w_m2": context["irradiance_w_m2"].to_numpy(),
            "day_ahead_price_eur_mwh": context["day_ahead_price_eur_mwh"].to_numpy(),
        }
    )
    never = time[-1] + pd.Timedelta(days=1)

    base = base_load_kwh(rng, steps, household["occupants"], step_hours)
    space_heat, hot_water = heat_demand_kwh(rng, steps, household, step_hours)
    heat_pump_from = change_time(events, "heat_pump_installed", time[0])
    heating_type = np.where(time < heat_pump_from, "gas_boiler", household["heating"])
    heating, gas, heat = route_heat(space_heat, hot_water, heating_type, steps["temperature_c"].to_numpy())
    pv = pv_kwh(steps, household["pv_kwp"], rng.uniform(0.95, 1.15), step_hours)
    pv *= ~any_active(events, "pv_inverter_outage", time)
    ev_from = change_time(events, "ev_acquired", time[0]) if household["ev_charger_kw"] else never
    dynamic_from = change_time(events, "switched_to_dynamic", time[0]) if household["contract"] == "dynamic" else never
    ev = ev_kwh(rng, steps, household, ev_from, dynamic_from, step_hours)

    net = base + heating + ev - pv
    df = pd.DataFrame(
        {
            "time": time,
            "household_id": household["household_id"],
            "import_kwh": np.round(np.maximum(net, 0.0), 3),
            "export_kwh": np.round(np.maximum(-net, 0.0), 3),
            "gas_m3": np.round(hourly_totals(gas, steps_per_hour), 3),
            "heat_gj": np.round(hourly_totals(heat, steps_per_hour), 4),
            "base_kwh": np.round(base, 3),
            "heating_kwh": np.round(heating, 3),
            "ev_kwh": np.round(ev, 3),
            "pv_kwh": np.round(pv, 3),
            "event_labels": event_labels(events, time),
        }
    )
    meter_outage = any_active(events, "meter_outage", time)
    df.loc[meter_outage, ["import_kwh", "export_kwh", "gas_m3", "heat_gj"]] = np.nan
    return df


def simulate_portfolio(cfg):
    context = simulate_context(cfg["site"], cfg["holidays"], seed=cfg["seed"])
    time = pd.DatetimeIndex(context["time"])
    households, events, readings = [], [], []

    for i in range(cfg["n_households"]):
        rng = np.random.default_rng(cfg["seed"] + i + 1)
        household = sample_household(rng, i, cfg["population"])
        household_events = sample_events(rng, household, time, cfg["events"])
        readings.append(simulate_household(context, household, household_events, rng))
        households.append(household)
        events.extend(household_events)

    readings = pd.concat(readings, ignore_index=True)
    readings = readings.sort_values(["time", "household_id"]).reset_index(drop=True)
    events = pd.DataFrame(
        events, columns=["household_id", "event_type", "start_time", "end_time"]
    ).sort_values(["start_time", "household_id"]).reset_index(drop=True)
    return {
        "smart_meter_households": pd.DataFrame(households),
        "smart_meter_context": context,
        "smart_meter_readings": readings,
        "smart_meter_events": events,
    }


def run(cfg):
    return simulate_portfolio(cfg)
