import calendar
import datetime

import numpy as np
import pandas as pd

# Product: (price in USD, demand weight).
PRODUCTS = {
    "iPhone": (700, 12),
    "Google Phone": (600, 9),
    "Samsung Phone": (650, 12),
    "20in Monitor": (109.99, 8),
    "34in Ultrawide Monitor": (379.99, 10),
    "27in 4K Gaming Monitor": (389.99, 8),
    "27in FHD Monitor": (149.99, 14),
    "Flatscreen TV": (300, 8),
    "Macbook Pro": (1699, 9),
    "Macbook Air": (1200, 6),
    "Dell Laptop": (999.99, 7),
    "Lenovo Laptop": (899.99, 4),
    "AA Batteries (4-pack)": (3.84, 40),
    "AAA Batteries (4-pack)": (2.99, 40),
    "USB-C Charging Cable": (11.95, 30),
    "Lightning Charging Cable": (14.95, 30),
    "Wired Headphones": (11.99, 30),
    "Bose SoundSport Headphones": (99.99, 25),
    "Apple Airpods Headphones": (150, 25),
    "Gaming Mouse": (100, 15),
    "Mechanical Keyboard": (250, 10),
    "Normal Keyboard": (99.99, 15),
    "Cooling Pad": (29.99, 10),
    "LG Washing Machine": (600.00, 1),
    "LG Dryer": (600.00, 1),
}
PRODUCT_NAMES = list(PRODUCTS)
PRODUCT_WEIGHTS = np.array([weight for _, weight in PRODUCTS.values()], dtype=float)

# Accessories that ride along with a main product: (accessory, chance).
ANDROID_ACCESSORIES = [("USB-C Charging Cable", 0.18), ("Bose SoundSport Headphones", 0.05), ("Wired Headphones", 0.07)]
MACBOOK_ACCESSORIES = [("Normal Keyboard", 0.16), ("Apple Airpods Headphones", 0.04)]
LAPTOP_ACCESSORIES = [("Cooling Pad", 0.13), ("Gaming Mouse", 0.05), ("Wired Headphones", 0.08)]
MONITOR_ACCESSORIES = [
    ("Mechanical Keyboard", 0.12), ("Wired Headphones", 0.04), ("Bose SoundSport Headphones", 0.05), ("Gaming Mouse", 0.08),
]
ACCESSORIES = {
    "iPhone": [("Lightning Charging Cable", 0.15), ("Apple Airpods Headphones", 0.07)],
    "Google Phone": ANDROID_ACCESSORIES,
    "Samsung Phone": ANDROID_ACCESSORIES,
    "Macbook Pro": MACBOOK_ACCESSORIES,
    "Macbook Air": MACBOOK_ACCESSORIES,
    "Dell Laptop": LAPTOP_ACCESSORIES,
    "Lenovo Laptop": LAPTOP_ACCESSORIES,
    "27in 4K Gaming Monitor": MONITOR_ACCESSORIES,
    "34in Ultrawide Monitor": MONITOR_ACCESSORIES,
}

# Orders per month: (mean, spread). Months not listed use the March to September level.
MONTHLY_ORDERS = {1: (10000, 3000), 2: (15000, 2000), 10: (15000, 3000), 11: (20000, 3000), 12: (26000, 2000)}
DEFAULT_MONTHLY_ORDERS = (12000, 2000)

STREET_NAMES = [
    "Main", "2nd", "1st", "4th", "5th", "Park", "6th", "7th", "Maple", "Pine", "Washington", "8th",
    "Cedar", "Elm", "Walnut", "9th", "10th", "Lake", "Sunset", "Lincoln", "Jackson", "Church", "River",
    "11th", "Willow", "Jefferson", "Center", "12th", "North", "Lakeview", "Ridge", "Hickory", "Adams",
    "Cherry", "Highland", "Johnson", "South", "Dogwood", "West", "Chestnut", "13th", "Spruce", "14th",
    "Wilson", "Meadow", "Forest", "Hill", "Madison",
]
# City, state, ZIP code, weight. Portland appears twice on purpose, Oregon and Maine.
CITIES = [
    ("San Francisco", "CA", "94016", 9),
    ("Boston", "MA", "02215", 4),
    ("New York City", "NY", "10001", 5),
    ("Austin", "TX", "73301", 2),
    ("Dallas", "TX", "75001", 3),
    ("Atlanta", "GA", "30301", 3),
    ("Portland", "OR", "97035", 2),
    ("Portland", "ME", "04101", 0.5),
    ("Los Angeles", "CA", "90001", 6),
    ("Seattle", "WA", "98101", 3),
]
CITY_WEIGHTS = np.array([weight for *_, weight in CITIES], dtype=float)

OUTPUT_COLUMNS = ["Order ID", "Product", "Quantity Ordered", "Price Each", "Order Date", "Purchase Address"]


def random_order_time(month, year, rng):
    day = int(rng.integers(1, calendar.monthrange(year, month)[1] + 1))
    peak_hour = 12 if rng.random() < 0.5 else 20
    base_date = datetime.datetime(year, month, day, peak_hour, 0)
    order_time = base_date + datetime.timedelta(minutes=float(rng.normal(loc=0.0, scale=180.0)))
    return order_time.strftime("%m/%d/%y %H:%M")


def random_address(rng):
    street_name = rng.choice(STREET_NAMES)
    city, state, zip_code, _ = CITIES[rng.choice(len(CITIES), p=CITY_WEIGHTS / CITY_WEIGHTS.sum())]
    street_number = int(rng.integers(1, 1000))
    return f"{street_number} {street_name} St, {city}, {state} {zip_code}"


def order_row(order_number, product_name, order_date, address, rng):
    price = PRODUCTS[product_name][0]
    quantity = int(rng.geometric(p=1.0 - (1.0 / price)))
    return [order_number, product_name, quantity, price, order_date, address]


def generate_month_sales(month, starting_order_number, year, rng):
    mean, spread = MONTHLY_ORDERS.get(month, DEFAULT_MONTHLY_ORDERS)
    order_count = max(0, int(rng.normal(loc=mean, scale=spread)))
    product_probabilities = PRODUCT_WEIGHTS / PRODUCT_WEIGHTS.sum()

    rows = []
    order_number = starting_order_number
    for _ in range(order_count):
        address = random_address(rng)
        order_date = random_order_time(month, year, rng)
        product_name = rng.choice(PRODUCT_NAMES, p=product_probabilities)
        rows.append(order_row(order_number, product_name, order_date, address, rng))

        for accessory, chance in ACCESSORIES.get(product_name, []):
            if rng.random() < chance:
                rows.append(order_row(order_number, accessory, order_date, address, rng))

        if rng.random() <= 0.02:
            extra_product = rng.choice(PRODUCT_NAMES, p=product_probabilities)
            rows.append(order_row(order_number, extra_product, order_date, address, rng))
        order_number += 1

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), order_number


def simulate_sales_year(config, seed):
    rng = np.random.default_rng(seed)
    order_number = config["starting_order_number"]
    months = []
    for month in range(1, 13):
        month_df, order_number = generate_month_sales(month, order_number, config["year"], rng)
        months.append(month_df)
    return pd.concat(months, ignore_index=True)


def run(cfg):
    return {cfg["data_name"]: simulate_sales_year(cfg["sales"], seed=cfg["seed"])}
