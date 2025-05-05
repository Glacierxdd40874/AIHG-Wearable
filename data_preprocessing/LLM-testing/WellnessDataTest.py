import pandas as pd
wellness_data = pd.read_csv("C:\\Uni Python\\wellness_data.csv")
wellness_synthetic = pd.read_csv("C:\\Uni Python\\synthetic_wellness_data.csv")
wellness_final = pd.concat([wellness_data, wellness_synthetic], ignore_index=True)
wellness_final = wellness_final.sample(frac=1, random_state=42).reset_index(drop=True)
print(wellness_final.describe())