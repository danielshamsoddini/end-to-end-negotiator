import pandas as pd


c_data = pd.read_parquet("casino_data.parquet")


c_data.to_csv('output.csv', index=False)

