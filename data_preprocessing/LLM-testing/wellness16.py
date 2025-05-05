import pandas as pd

# Load the original and synthetic datasets
wellness_data = pd.read_csv("C:\\Uni Python\\wellness_data.csv")
wellness_synthetic = pd.read_csv("C:\\Uni Python\\synthetic_wellness_data.csv")

# Combine and shuffle the datasets
wellness_final = pd.concat([wellness_data, wellness_synthetic], ignore_index=True)
wellness_final = wellness_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Define participant ID ranges
ranges = {
    "p01-p16": range(1, 17),
    "p01-p20": range(1, 21),
    "p01-p40": range(1, 41),
    "p01-p60": range(1, 61),
    "p01-p80": range(1, 81),
    "p01-p100": range(1, 101)
}

# Create a dictionary to store summary stats for each range
summaries = {}

for label, r in ranges.items():
    ids = [f"p{str(i).zfill(2)}" for i in r]
    subset = wellness_final[wellness_final['participant_id'].isin(ids)]
    summaries[label] = subset['sleep_quality'].describe()

# Combine all summaries into a single DataFrame
combined_summary = pd.concat(summaries.values(), axis=1)
combined_summary.columns = summaries.keys()

# Display the final result
print("Sleep Quality Summary for Participant Ranges:")
print(combined_summary)

