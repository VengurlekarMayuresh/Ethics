#!/usr/bin/env python3
"""List some valid car name options from the dataset"""

import pandas as pd

df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
unique_names = df['name'].unique()
print(f"Total unique car names: {len(unique_names)}")
print("\nFirst 20:")
for name in sorted(unique_names)[:20]:
    print(f"  {name}")

# Find one that contains "Maruti Alto"
maruti_alto = [n for n in unique_names if 'Maruti Alto' in n]
print("\nMaruti Alto variants:")
for name in maruti_alto[:10]:
    print(f"  {name}")
