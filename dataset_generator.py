"""
dataset_generator.py
Generate a realistic synthetic hotel guest dataset for the Smart Contactless Hotel Experience demo.

Run:
    python dataset_generator.py
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from faker import Faker

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "hotel_dataset.csv"


def generate_dataset(rows: int = 750, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic hotel guest data with fraud labels."""
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    records = []

    for i in range(1, rows + 1):
        guest_id = f"GUEST-{i:05d}"
        name = fake.name()
        email = fake.email()
        phone = fake.phone_number()

        stay_probs = np.array([0.16, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.035, 0.025, 0.02, 0.015, 0.01])
        stay_probs = stay_probs / stay_probs.sum()
        stay_days = int(np.random.choice(range(1, 15), p=stay_probs))
        room_rate = round(float(np.random.choice([2500, 3000, 3500, 4500, 5500, 7000, 9000, 12000]) + np.random.normal(0, 250)), 2)
        room_rate = max(room_rate, 1500.0)
        total_bill = round(stay_days * room_rate, 2)

        biometric_score = round(float(np.random.uniform(0.70, 1.00)), 3)
        checked_in = int(biometric_score > 0.75 and random.random() > 0.08)

        # Fraud label is synthetic but rule-informed:
        # larger bills, longer stays, low biometric match, and failed check-in raise risk.
        risk_score = 0
        risk_score += 1 if total_bill > 60000 else 0
        risk_score += 1 if room_rate > 9000 else 0
        risk_score += 1 if stay_days >= 10 else 0
        risk_score += 1 if biometric_score < 0.78 else 0
        risk_score += 1 if checked_in == 0 else 0

        fraud_probability = 0.04 + (risk_score * 0.12)
        fraud_flag = int(random.random() < min(fraud_probability, 0.70))

        records.append({
            "guest_id": guest_id,
            "name": name,
            "email": email,
            "phone": phone,
            "stay_days": stay_days,
            "room_rate": room_rate,
            "total_bill": total_bill,
            "biometric_score": biometric_score,
            "checked_in": checked_in,
            "fraud_flag": fraud_flag,
        })

    return pd.DataFrame(records)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = generate_dataset(rows=750)
    df.to_csv(DATA_FILE, index=False)
    print(f"Dataset generated successfully: {DATA_FILE}")
    print(f"Rows: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()
