#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

print("Unique values for categorical features:")
print("\nfuel:", df['fuel'].unique())
print("\nseller_type:", df['seller_type'].unique())
print("\ntransmission:", df['transmission'].unique())
print("\nowner:", df['owner'].unique())
