import pandas as pd
import numpy as np

# Parameters
num_rows = 1000

# Generate synthetic data
np.random.seed(0)  # For reproducibility
data = {
    'Id': np.arange(1, num_rows + 1),
    'trans_hour': np.random.randint(0, 24, num_rows),
    'trans_day': np.random.randint(1, 32, num_rows),
    'trans_month': np.random.randint(1, 13, num_rows),
    'trans_year': np.random.randint(2020, 2025, num_rows),
    'category': np.random.choice(['Groceries', 'Electronics', 'Clothing', 'Entertainment'], num_rows),
    'UPI_number': np.random.randint(1000000000, 9999999999, num_rows).astype(str),
    'age': np.random.randint(18, 80, num_rows),
    'trans_amount': np.random.uniform(10, 5000, num_rows).round(2),
    'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], num_rows),
    'zip': np.random.randint(10000, 99999, num_rows),
    'fraud_risk': np.random.choice(['Low', 'Medium', 'High'], num_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file = 'synthetic_dataset.csv'
df.to_csv(csv_file, index=False)

print(f"Dataset created and saved as {csv_file}")
