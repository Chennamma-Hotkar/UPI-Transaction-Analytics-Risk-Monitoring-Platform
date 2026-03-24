import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

NUM_TRANSACTIONS = 5000
NUM_USERS = 200
NUM_MERCHANTS = 80

DEVICES = ["Android", "iOS", "Web", "Unknown"]
LOCATIONS = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
             "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]

start_date = datetime(2024, 1, 1)

records = []
for i in range(NUM_TRANSACTIONS):
    user_id = f"USER_{random.randint(1, NUM_USERS):04d}"
    merchant_id = f"MER_{random.randint(1, NUM_MERCHANTS):04d}"
    timestamp = start_date + timedelta(
        days=random.randint(0, 365),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    is_fraud = random.random() < 0.08  # 8% fraud rate

    if is_fraud:
        amount = round(random.choice([
            random.uniform(1, 10),           # micro transaction
            random.uniform(50000, 200000)    # large spike
        ]), 2)
        device = random.choice(DEVICES)
        location = random.choice(LOCATIONS)
    else:
        amount = round(np.random.lognormal(mean=7, sigma=1.2), 2)
        amount = min(amount, 50000)
        device = random.choices(DEVICES, weights=[50, 30, 15, 5])[0]
        location = random.choices(LOCATIONS, weights=[20,18,15,10,10,8,7,5,4,3])[0]

    records.append({
        "transaction_id": f"TXN_{i+1:06d}",
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "device_info": device,
        "location": location,
        "is_fraud": int(is_fraud)
    })

df = pd.DataFrame(records)
df.to_csv("transactions.csv", index=False)
print(f"Dataset created: {len(df)} transactions, {df['is_fraud'].sum()} fraudulent")
print(df.head())
