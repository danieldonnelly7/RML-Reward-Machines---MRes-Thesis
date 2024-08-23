import pandas as pd
import pickle

with open('results/machine_sizes_letterenv_RML.pkl', 'rb') as f:
    table = pickle.load(f)

print(table)

csv_filename = 'results/machine_sizes_letterenv_RML.csv'
table.to_csv(csv_filename, index=False)