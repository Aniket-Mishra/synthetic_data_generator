import uuid

import numpy as np
import pandas as pd

COUNTRIES = ["NL", "BE", "DE", "GB", "FR", "ES", "IN", "SG", "CA", "AU"]
COUNTRY_PROBABILITIES = np.array([0.62, 0.12, 0.07, 0.05, 0.035, 0.025, 0.03, 0.015, 0.025, 0.01])
CROSS_BORDER_MERCHANT_PROBABILITIES = np.array([0.00, 0.22, 0.22, 0.14, 0.13, 0.09, 0.09, 0.05, 0.04, 0.02])

DEVICE_TYPES = ["android", "iphone", "desktop", "tablet", "laptop", "macbook"]
DEVICE_TYPE_PROBABILITIES = np.array([0.34, 0.30, 0.15, 0.07, 0.09, 0.05])
FRAUD_DEVICE_TYPE_PROBABILITIES = np.array([0.42, 0.35, 0.08, 0.04, 0.07, 0.04])

MERCHANT_CATEGORIES = [
    "Groceries", "Food & Dining", "Public Transport", "Bike & Mobility", "Fuel", "Retail",
    "Electronics", "Travel", "Entertainment", "Digital Goods", "Health", "Jewelry", "Luxury Retail",
]
MERCHANT_CATEGORY_PROBABILITIES = np.array(
    [0.22, 0.15, 0.07, 0.04, 0.035, 0.16, 0.07, 0.04, 0.08, 0.07, 0.045, 0.01, 0.01]
)
# Per category: amount multiplier, chance the purchase is online, chance an online purchase uses a credit card.
CATEGORY_PROFILES = {
    "Groceries": (0.9, 0.08, 0.06),
    "Food & Dining": (0.6, 0.12, 0.08),
    "Public Transport": (0.45, 0.60, 0.08),
    "Bike & Mobility": (0.85, 0.55, 0.10),
    "Fuel": (0.95, 0.01, 0.05),
    "Retail": (1.1, 0.35, 0.12),
    "Electronics": (3.2, 0.55, 0.18),
    "Travel": (3.8, 0.75, 0.35),
    "Entertainment": (0.9, 0.40, 0.16),
    "Digital Goods": (0.5, 0.96, 0.22),
    "Health": (1.4, 0.20, 0.10),
    "Jewelry": (5.0, 0.30, 0.28),
    "Luxury Retail": (4.5, 0.45, 0.32),
}
ONLINE_CHANNELS = ["ideal_online", "credit_card_online"]

SUBSCRIPTION_PRODUCTS = [
    {"subscription_name": "Netflix", "merchant_category": "Entertainment", "base_amount": 15.49},
    {"subscription_name": "Spotify", "merchant_category": "Entertainment", "base_amount": 10.99},
    {"subscription_name": "Videoland", "merchant_category": "Entertainment", "base_amount": 10.99},
    {"subscription_name": "NPO Plus", "merchant_category": "Entertainment", "base_amount": 2.95},
    {"subscription_name": "NS Flex", "merchant_category": "Public Transport", "base_amount": 5.60},
    {"subscription_name": "Swapfiets", "merchant_category": "Bike & Mobility", "base_amount": 19.90},
    {"subscription_name": "Basic-Fit", "merchant_category": "Health", "base_amount": 29.99},
    {"subscription_name": "Bol Select", "merchant_category": "Retail", "base_amount": 12.99},
    {"subscription_name": "Cloud Storage", "merchant_category": "Digital Goods", "base_amount": 9.99},
    {"subscription_name": "News Subscription", "merchant_category": "Digital Goods", "base_amount": 7.99},
    {"subscription_name": "Meal Kit", "merchant_category": "Food & Dining", "base_amount": 59.99},
]

OUTPUT_COLUMNS = [
    "transaction_id", "timestamp", "customer_id", "account_age_days", "home_country", "merchant_id",
    "merchant_country", "merchant_category", "device_type", "transaction_channel", "amount",
    "distance_from_home", "is_cross_border", "is_subscription", "subscription_name", "is_fraud",
    "fraud_source", "fraud_scenario",
]
STRING_COLUMNS = [
    "customer_id", "home_country", "merchant_id", "merchant_country", "merchant_category",
    "device_type", "transaction_channel", "subscription_name", "fraud_source", "fraud_scenario",
]


def choose_other_country(home_country, rng):
    probabilities = CROSS_BORDER_MERCHANT_PROBABILITIES.copy()
    probabilities[COUNTRIES.index(home_country)] = 0.0
    return rng.choice(COUNTRIES, p=probabilities / probabilities.sum())


def choose_merchant_country(home_country, is_cross_border, rng):
    if is_cross_border:
        return choose_other_country(home_country, rng)
    return home_country


def make_merchant_id(merchant_category, merchant_country, rng):
    clean_category = merchant_category.upper().replace("&", "AND").replace(" ", "_")
    return f"{merchant_country}_{clean_category}_{rng.integers(1, 200):03d}"


def get_account_age_days(timestamp, signup_date):
    age_days = (pd.Timestamp(timestamp).normalize() - pd.Timestamp(signup_date).normalize()).days
    return max(0, int(age_days))


def sample_normal_hours(n_samples, rng):
    hours = []
    while len(hours) < n_samples:
        n_needed = n_samples - len(hours)
        peak_names = rng.choice(["midday", "evening", "other"], size=n_needed * 2, p=[0.45, 0.45, 0.10])

        sampled_hours = np.empty(n_needed * 2)
        midday = peak_names == "midday"
        evening = peak_names == "evening"
        other = peak_names == "other"
        sampled_hours[midday] = rng.normal(12.5, 2.8, midday.sum())
        sampled_hours[evening] = rng.normal(18.5, 2.4, evening.sum())
        sampled_hours[other] = rng.uniform(6.0, 23.0, other.sum())

        sampled_hours = sampled_hours[(sampled_hours >= 0.0) & (sampled_hours < 24.0)]
        hours.extend(sampled_hours[:n_needed])
    return np.array(hours[:n_samples])


def sample_normal_timestamps(start, n_days, n_samples, rng):
    day_offsets = rng.integers(0, n_days, size=n_samples)
    hours = sample_normal_hours(n_samples, rng)
    minutes = rng.integers(0, 60, size=n_samples)
    seconds = rng.integers(0, 60, size=n_samples)
    return (
        pd.Timestamp(start)
        + pd.to_timedelta(day_offsets, unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )


def sample_late_night_timestamp(start, n_days, rng):
    day_offset = rng.integers(0, n_days)
    hour = rng.uniform(0.0, 4.0)
    minute = rng.integers(0, 60)
    second = rng.integers(0, 60)
    return (
        pd.Timestamp(start)
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(second, unit="s")
    )


def make_customers(n_customers, start, seed):
    rng = np.random.default_rng(seed)
    age_groups = rng.choice(["new", "regular", "mature"], size=n_customers, p=[0.08, 0.34, 0.58])

    account_age_at_start = np.empty(n_customers, dtype=int)
    account_age_at_start[age_groups == "new"] = rng.integers(1, 90, size=(age_groups == "new").sum())
    account_age_at_start[age_groups == "regular"] = rng.integers(90, 730, size=(age_groups == "regular").sum())
    account_age_at_start[age_groups == "mature"] = rng.integers(730, 3650, size=(age_groups == "mature").sum())

    return pd.DataFrame({
        "customer_id": [f"CUST_{number:05d}" for number in range(1, n_customers + 1)],
        "signup_date": pd.Timestamp(start) - pd.to_timedelta(account_age_at_start, unit="D"),
        "home_country": rng.choice(COUNTRIES, size=n_customers, p=COUNTRY_PROBABILITIES),
        "primary_device_type": rng.choice(DEVICE_TYPES, size=n_customers, p=DEVICE_TYPE_PROBABILITIES),
        "normal_spend_level": rng.lognormal(mean=0.0, sigma=0.45, size=n_customers),
        "normal_travel_radius": rng.gamma(shape=2.0, scale=5.0, size=n_customers) + 1.0,
        "activity_level": rng.gamma(shape=2.0, scale=1.0, size=n_customers),
    })


def choose_transaction_channels(merchant_categories, rng):
    channels = []
    for merchant_category in merchant_categories:
        _, online_probability, credit_card_probability = CATEGORY_PROFILES[merchant_category]
        if rng.random() < 0.02:
            channels.append("atm_cash")
        elif rng.random() >= online_probability:
            channels.append("pos_pin")
        elif rng.random() < credit_card_probability:
            channels.append("credit_card_online")
        else:
            channels.append("ideal_online")
    return np.array(channels)


def choose_device_types(primary_device_types, rng):
    device_types = []
    for primary_device_type in primary_device_types:
        if rng.random() < 0.88:
            device_types.append(primary_device_type)
        else:
            other_devices = [device for device in DEVICE_TYPES if device != primary_device_type]
            device_types.append(rng.choice(other_devices))
    return np.array(device_types)


def sample_normal_amounts(merchant_categories, spend_levels, rng):
    amounts = []
    for merchant_category, spend_level in zip(merchant_categories, spend_levels):
        median_amount = 28.0 * spend_level * CATEGORY_PROFILES[merchant_category][0]
        amounts.append(rng.lognormal(mean=np.log(median_amount), sigma=0.75))
    return np.round(np.clip(amounts, 1.0, 5000.0), 2)


def sample_normal_cross_border_flags(merchant_categories, transaction_channels, rng):
    flags = []
    for merchant_category, transaction_channel in zip(merchant_categories, transaction_channels):
        probability = 0.012
        if transaction_channel == "ideal_online":
            probability += 0.010
        if transaction_channel == "credit_card_online":
            probability += 0.055
        if merchant_category == "Travel":
            probability += 0.120
        if merchant_category in ["Digital Goods", "Electronics", "Luxury Retail", "Jewelry"]:
            probability += 0.025
        flags.append(rng.random() < probability)
    return np.array(flags, dtype=bool)


def sample_distances_from_home(normal_travel_radii, is_cross_border, transaction_channels, rng):
    distances = rng.exponential(scale=normal_travel_radii)

    online = np.isin(transaction_channels, ONLINE_CHANNELS)
    distances[online] = rng.exponential(scale=35.0, size=online.sum())
    atm = transaction_channels == "atm_cash"
    distances[atm] = rng.exponential(scale=2.5, size=atm.sum())
    distances[is_cross_border] = rng.lognormal(mean=np.log(1800.0), sigma=0.75, size=is_cross_border.sum())

    return np.round(np.clip(distances, 0.0, 12000.0), 2)


def generate_everyday_transactions(customers, n_transactions, start, n_days, seed):
    rng = np.random.default_rng(seed)

    customer_weights = customers["activity_level"].to_numpy()
    selected_positions = rng.choice(
        customers.index.to_numpy(), size=n_transactions, replace=True, p=customer_weights / customer_weights.sum()
    )
    selected = customers.loc[selected_positions].reset_index(drop=True)

    timestamps = sample_normal_timestamps(start, n_days, n_transactions, rng)
    merchant_categories = rng.choice(MERCHANT_CATEGORIES, size=n_transactions, p=MERCHANT_CATEGORY_PROBABILITIES)
    transaction_channels = choose_transaction_channels(merchant_categories, rng)
    device_types = choose_device_types(selected["primary_device_type"].to_numpy(), rng)
    amounts = sample_normal_amounts(merchant_categories, selected["normal_spend_level"].to_numpy(), rng)
    is_cross_border = sample_normal_cross_border_flags(merchant_categories, transaction_channels, rng)

    merchant_countries = []
    merchant_ids = []
    for home_country, merchant_category, cross_border in zip(selected["home_country"], merchant_categories, is_cross_border):
        merchant_country = choose_merchant_country(home_country, cross_border, rng)
        merchant_countries.append(merchant_country)
        merchant_ids.append(make_merchant_id(merchant_category, merchant_country, rng))

    distances = sample_distances_from_home(
        selected["normal_travel_radius"].to_numpy(), is_cross_border, transaction_channels, rng
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "customer_id": selected["customer_id"],
        "account_age_days": [
            get_account_age_days(timestamp, signup_date)
            for timestamp, signup_date in zip(timestamps, selected["signup_date"])
        ],
        "home_country": selected["home_country"],
        "merchant_id": merchant_ids,
        "merchant_country": merchant_countries,
        "merchant_category": merchant_categories,
        "device_type": device_types,
        "transaction_channel": transaction_channels,
        "amount": amounts,
        "distance_from_home": distances,
        "is_cross_border": is_cross_border,
        "is_subscription": False,
        "subscription_name": "none",
        "is_fraud": 0,
        "fraud_source": "none",
        "fraud_scenario": "none",
    })


def choose_subscription_channel(subscription_name, rng):
    credit_card_probability = 0.18
    if subscription_name in ["Netflix", "Spotify", "Videoland"]:
        credit_card_probability = 0.32
    if subscription_name in ["NS Flex", "Swapfiets", "Basic-Fit"]:
        credit_card_probability = 0.10
    if rng.random() < credit_card_probability:
        return "credit_card_online"
    return "ideal_online"


def generate_subscription_transactions(customers, start, n_days, seed):
    rng = np.random.default_rng(seed)
    rows = []
    start_timestamp = pd.Timestamp(start)
    end_timestamp = start_timestamp + pd.to_timedelta(n_days, unit="D")

    subscription_counts = rng.choice([0, 1, 2, 3], size=len(customers), p=[0.55, 0.30, 0.12, 0.03])
    for customer_position, subscription_count in enumerate(subscription_counts):
        if subscription_count == 0:
            continue
        customer = customers.iloc[customer_position]
        subscriptions = rng.choice(len(SUBSCRIPTION_PRODUCTS), size=subscription_count, replace=False)

        for subscription_index in subscriptions:
            subscription = SUBSCRIPTION_PRODUCTS[subscription_index]
            charge_day = rng.integers(1, 29)
            charge_date = start_timestamp + pd.DateOffset(months=int(rng.integers(0, 4)))
            charge_date = charge_date.replace(day=int(charge_day))

            while charge_date < end_timestamp:
                charge_time = (
                    charge_date
                    + pd.to_timedelta(rng.normal(9.0, 1.5), unit="h")
                    + pd.to_timedelta(rng.integers(0, 60), unit="m")
                    + pd.to_timedelta(rng.integers(0, 60), unit="s")
                )
                is_cross_border = rng.random() < 0.06
                merchant_country = choose_merchant_country(customer["home_country"], is_cross_border, rng)
                amount = round(max(1.0, subscription["base_amount"] * rng.normal(1.0, 0.03)), 2)
                distance_from_home = rng.lognormal(mean=np.log(1600.0), sigma=0.6) if is_cross_border else 0.0

                rows.append({
                    "timestamp": charge_time,
                    "customer_id": customer["customer_id"],
                    "account_age_days": get_account_age_days(charge_time, customer["signup_date"]),
                    "home_country": customer["home_country"],
                    "merchant_id": f"{merchant_country}_SUB_{subscription['subscription_name'].upper().replace(' ', '_')}",
                    "merchant_country": merchant_country,
                    "merchant_category": subscription["merchant_category"],
                    "device_type": customer["primary_device_type"],
                    "transaction_channel": choose_subscription_channel(subscription["subscription_name"], rng),
                    "amount": amount,
                    "distance_from_home": round(distance_from_home, 2),
                    "is_cross_border": bool(is_cross_border),
                    "is_subscription": True,
                    "subscription_name": subscription["subscription_name"],
                    "is_fraud": 0,
                    "fraud_source": "none",
                    "fraud_scenario": "none",
                })
                charge_date = charge_date + pd.DateOffset(months=1)

    return pd.DataFrame(rows)


def choose_fraud_source(rng):
    return rng.choice(["account_takeover", "stolen_card", "new_account_fraud"], p=[0.42, 0.38, 0.20])


def choose_fraud_device_type(primary_device_type, rng):
    if rng.random() < 0.72:
        return rng.choice([device for device in DEVICE_TYPES if device != primary_device_type])
    return primary_device_type


def get_fraud_customer(customers, fraud_source, timestamp, event_number, rng):
    if fraud_source == "new_account_fraud":
        return pd.Series({
            "customer_id": f"FRAUD_CUST_{event_number:05d}",
            "signup_date": pd.Timestamp(timestamp).normalize() - pd.to_timedelta(rng.integers(0, 15), unit="D"),
            "home_country": rng.choice(COUNTRIES, p=COUNTRY_PROBABILITIES),
            "primary_device_type": rng.choice(DEVICE_TYPES, p=FRAUD_DEVICE_TYPE_PROBABILITIES),
            "normal_spend_level": rng.lognormal(mean=0.0, sigma=0.45),
            "normal_travel_radius": rng.gamma(shape=2.0, scale=5.0) + 1.0,
            "activity_level": 1.0,
        })
    if fraud_source == "account_takeover":
        older_customers = customers[customers["signup_date"] < customers["signup_date"].quantile(0.65)]
        return older_customers.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

    customer_weights = customers["activity_level"].to_numpy()
    return customers.loc[rng.choice(customers.index.to_numpy(), p=customer_weights / customer_weights.sum())]


def sample_fraud_distance(is_cross_border, minimum_distance, rng):
    if is_cross_border:
        distance = rng.lognormal(mean=np.log(2200.0), sigma=0.7)
    else:
        distance = minimum_distance + rng.exponential(scale=120.0)
    return round(float(np.clip(distance, minimum_distance, 12000.0)), 2)


def fraud_row(event, timestamp, amount):
    customer = event["customer"]
    return {
        "timestamp": timestamp,
        "customer_id": customer["customer_id"],
        "account_age_days": get_account_age_days(timestamp, customer["signup_date"]),
        "home_country": customer["home_country"],
        "merchant_id": event["merchant_id"],
        "merchant_country": event["merchant_country"],
        "merchant_category": event["merchant_category"],
        "device_type": event["device_type"],
        "transaction_channel": event["transaction_channel"],
        "amount": round(float(amount), 2),
        "distance_from_home": event["distance_from_home"],
        "is_cross_border": bool(event["is_cross_border"]),
        "is_subscription": False,
        "subscription_name": "none",
        "is_fraud": 1,
        "fraud_source": event["fraud_source"],
        "fraud_scenario": event["fraud_scenario"],
    }


def generate_ping_fraud(customers, start, n_days, event_number, max_transactions, rng):
    n_transactions = int(min(max_transactions, rng.integers(3, 6)))
    base_timestamp = sample_late_night_timestamp(start, n_days, rng)
    fraud_source = choose_fraud_source(rng)
    customer = get_fraud_customer(customers, fraud_source, base_timestamp, event_number, rng)

    device_type = choose_fraud_device_type(customer["primary_device_type"], rng)
    is_cross_border = rng.random() < 0.68
    merchant_country = choose_merchant_country(customer["home_country"], is_cross_border, rng)
    event = {
        "customer": customer,
        "merchant_id": make_merchant_id("Digital Goods", merchant_country, rng),
        "merchant_country": merchant_country,
        "merchant_category": "Digital Goods",
        "device_type": device_type,
        "transaction_channel": "credit_card_online",
        "distance_from_home": sample_fraud_distance(is_cross_border, 5.0, rng),
        "is_cross_border": is_cross_border,
        "fraud_source": fraud_source,
        "fraud_scenario": "ping",
    }

    rows = []
    for transaction_number in range(n_transactions):
        timestamp = base_timestamp + pd.to_timedelta(rng.integers(10, 180) * transaction_number, unit="s")
        rows.append(fraud_row(event, timestamp, rng.uniform(0.10, 1.99)))
    return rows


def generate_high_value_fraud(customers, start, n_days, high_amount_threshold, event_number, rng):
    if rng.random() < 0.35:
        timestamp = sample_late_night_timestamp(start, n_days, rng)
    else:
        timestamp = sample_normal_timestamps(start, n_days, 1, rng)[0]
    fraud_source = choose_fraud_source(rng)
    customer = get_fraud_customer(customers, fraud_source, timestamp, event_number, rng)

    merchant_category = rng.choice(["Electronics", "Jewelry", "Luxury Retail", "Travel"], p=[0.38, 0.24, 0.20, 0.18])
    is_cross_border = rng.random() < 0.74
    merchant_country = choose_merchant_country(customer["home_country"], is_cross_border, rng)
    event = {
        "customer": customer,
        "merchant_id": make_merchant_id(merchant_category, merchant_country, rng),
        "merchant_country": merchant_country,
        "merchant_category": merchant_category,
        "device_type": choose_fraud_device_type(customer["primary_device_type"], rng),
        "transaction_channel": rng.choice(["credit_card_online", "ideal_online", "pos_pin"], p=[0.58, 0.12, 0.30]),
        "is_cross_border": is_cross_border,
        "fraud_source": fraud_source,
        "fraud_scenario": "high_value_extrication",
    }
    amount = rng.lognormal(mean=np.log(high_amount_threshold * 1.7), sigma=0.45)
    event["distance_from_home"] = sample_fraud_distance(is_cross_border, 80.0, rng)
    return [fraud_row(event, timestamp, amount)]


def generate_velocity_fraud(customers, start, n_days, event_number, max_transactions, rng):
    n_transactions = int(min(max_transactions, rng.integers(10, 19)))
    if n_transactions < 10:
        return []
    base_timestamp = sample_late_night_timestamp(start, n_days, rng)
    fraud_source = choose_fraud_source(rng)
    customer = get_fraud_customer(customers, fraud_source, base_timestamp, event_number, rng)

    merchant_category = rng.choice(["Retail", "Electronics", "Digital Goods", "Food & Dining"], p=[0.34, 0.26, 0.25, 0.15])
    is_cross_border = rng.random() < 0.57
    merchant_country = choose_merchant_country(customer["home_country"], is_cross_border, rng)
    event = {
        "customer": customer,
        "merchant_id": make_merchant_id(merchant_category, merchant_country, rng),
        "merchant_country": merchant_country,
        "merchant_category": merchant_category,
        "device_type": choose_fraud_device_type(customer["primary_device_type"], rng),
        "transaction_channel": rng.choice(["credit_card_online", "ideal_online", "pos_pin"], p=[0.55, 0.30, 0.15]),
        "distance_from_home": sample_fraud_distance(is_cross_border, 25.0, rng),
        "is_cross_border": is_cross_border,
        "fraud_source": fraud_source,
        "fraud_scenario": "velocity_attack",
    }

    rows = []
    for offset_seconds in np.sort(rng.integers(0, 300, size=n_transactions)):
        timestamp = base_timestamp + pd.to_timedelta(int(offset_seconds), unit="s")
        amount = np.clip(rng.lognormal(mean=np.log(65.0), sigma=0.55), 12.0, 250.0)
        rows.append(fraud_row(event, timestamp, amount))
    return rows


def generate_fraud_transactions(customers, normal_transactions, start, n_days, target_fraud_count, seed):
    rng = np.random.default_rng(seed)
    rows = []
    event_number = 1
    high_amount_threshold = normal_transactions["amount"].quantile(0.99)

    while len(rows) < target_fraud_count:
        remaining = target_fraud_count - len(rows)
        if remaining < 3:
            scenario = "high_value_extrication"
        elif remaining < 10:
            scenario = rng.choice(["ping", "high_value_extrication"], p=[0.55, 0.45])
        else:
            scenario = rng.choice(["ping", "high_value_extrication", "velocity_attack"], p=[0.34, 0.36, 0.30])

        if scenario == "ping":
            rows += generate_ping_fraud(customers, start, n_days, event_number, remaining, rng)
        elif scenario == "high_value_extrication":
            rows += generate_high_value_fraud(customers, start, n_days, high_amount_threshold, event_number, rng)
        else:
            rows += generate_velocity_fraud(customers, start, n_days, event_number, remaining, rng)
        event_number += 1

    return pd.DataFrame(rows[:target_fraud_count])


def make_transaction_ids(n_transactions, seed):
    return [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"beegbank_fraud_{seed}_{row_number}")) for row_number in range(n_transactions)]


def round_and_order_columns(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["account_age_days"] = df["account_age_days"].astype(int)
    df["amount"] = df["amount"].round(3)
    df["distance_from_home"] = df["distance_from_home"].round(3)
    df["is_cross_border"] = df["is_cross_border"].astype(bool)
    df["is_subscription"] = df["is_subscription"].astype(bool)
    df["is_fraud"] = df["is_fraud"].astype(int)
    for column in STRING_COLUMNS:
        df[column] = df[column].map(str)
    return df[OUTPUT_COLUMNS]


def simulate_fraud_dataset(dataset_config, seed):
    start, n_days = dataset_config["start"], dataset_config["n_days"]
    customers = make_customers(dataset_config["n_customers"], start, seed)
    everyday = generate_everyday_transactions(customers, dataset_config["n_everyday_transactions"], start, n_days, seed + 1)
    subscriptions = generate_subscription_transactions(customers, start, n_days, seed + 2)
    normal_transactions = pd.concat([everyday, subscriptions], ignore_index=True)

    fraud_rate = dataset_config["target_fraud_rate"]
    target_fraud_count = int(round(len(normal_transactions) * fraud_rate / (1.0 - fraud_rate)))
    fraud_transactions = generate_fraud_transactions(customers, normal_transactions, start, n_days, target_fraud_count, seed + 3)

    df = pd.concat([normal_transactions, fraud_transactions], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", make_transaction_ids(len(df), seed))
    return round_and_order_columns(df), customers


def run(cfg):
    df, customers = simulate_fraud_dataset(cfg["dataset"], seed=cfg["seed"])
    return {cfg["transactions_name"]: df, cfg["customers_name"]: customers}
