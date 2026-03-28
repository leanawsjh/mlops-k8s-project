import numpy as np
import pandas as pd


def generate_housing_data(n_samples=10000, drift=False, seed=42):
    np.random.seed(seed)

    # Income (log-normal distribution)
    if not drift:
        med_inc = np.random.lognormal(mean=1.5, sigma=0.5, size=n_samples)
    else:
        # Drift: income distribution shifted
        med_inc = np.random.lognormal(mean=1.2, sigma=0.7, size=n_samples)

    # House age
    house_age = np.random.randint(1, 50, size=n_samples)

    # Rooms
    ave_rooms = np.random.normal(loc=5, scale=1.5, size=n_samples)
    ave_rooms = np.clip(ave_rooms, 1, None)

    # Bedrooms correlated with rooms
    ave_bedrms = ave_rooms * np.random.uniform(0.15, 0.3, size=n_samples)

    # Population
    if not drift:
        population = np.random.randint(100, 5000, size=n_samples)
    else:
        population = np.random.randint(500, 8000, size=n_samples)

    # Occupancy
    ave_occup = np.random.normal(loc=3, scale=1, size=n_samples)
    ave_occup = np.clip(ave_occup, 1, None)

    # Location (California bounding box)
    if not drift:
        latitude = np.random.uniform(32, 42, size=n_samples)
        longitude = np.random.uniform(-124, -114, size=n_samples)
    else:
        # Drift: more inland shift
        latitude = np.random.uniform(34, 40, size=n_samples)
        longitude = np.random.uniform(-120, -114, size=n_samples)

    # Target variable (price)
    med_house_val = (
        50000
        + (med_inc * 50000)
        + (house_age * 1000)
        + (ave_rooms * 10000)
        - (population * 2)
        + np.random.normal(0, 20000, size=n_samples)
    )

    data = pd.DataFrame(
        {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
            "MedHouseVal": med_house_val,
        }
    )

    return data


if __name__ == "__main__":
    # Training dataset
    train_data = generate_housing_data(n_samples=10000, drift=False)

    # Drift dataset (production-like)
    drift_data = generate_housing_data(n_samples=5000, drift=True)

    # Save
    train_data.to_csv("data/train.csv", index=False)
    drift_data.to_csv("data/drift.csv", index=False)
