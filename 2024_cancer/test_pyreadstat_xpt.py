import os
import pandas as pd

print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

file_path = "LLCP2024.XPT"

try:
    df = pd.read_sas(file_path, format='xport', encoding='utf-8')
    print("Successfully read the file with pandas. First 5 rows:")
    print(df.head())
except Exception as e:
    print(f"Error reading file with pandas: {e}")
