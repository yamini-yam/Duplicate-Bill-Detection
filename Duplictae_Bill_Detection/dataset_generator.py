import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker and set the random seed for reproducibility
fake = Faker()
np.random.seed(42)

# Define the base data as a list of dictionaries
base_data = [
    {"Description": "Office Supplies - Printer Ink", "Amount": 426.42},
    {"Description": "Conference Room Booking", "Amount": 378.14},
    {"Description": "Office Furniture - Desk", "Amount": 113.22},
    {"Description": "Office Supplies - Printer Paper", "Amount": 443.65},
    {"Description": "Travel - Business Trip", "Amount": 407.36},
    {"Description": "Office Supplies - Stationery", "Amount": 80.01},
    {"Description": "Office Supplies - Notebooks", "Amount": 93.65},
]

# Generate 1500 entries
num_samples = 1500
data = []

for i in range(num_samples):
    expense_id = i + 1
    base_entry = base_data[np.random.randint(len(base_data))]
    description = base_entry["Description"]
    amount = base_entry["Amount"]
    date = fake.date_this_year().strftime("%Y-%m-%d")
    bill_number = fake.unique.random_number(digits=6, fix_len=True)
    
    # Add the entry to the dataset
    data.append([expense_id, description, amount, date, f"B{bill_number}"])

# Create DataFrame
df = pd.DataFrame(data, columns=['Expense_ID', 'Description', 'Amount', 'Date', 'Bill_Number'])

# Save to CSV
df.to_csv('expense_data_1500.csv', index=False)

print("CSV file 'expense_data_1500.csv' has been created successfully.")
