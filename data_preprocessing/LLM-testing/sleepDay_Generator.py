import openai
import pandas as pd
import time
import random
from tqdm import tqdm
import os


# Set your OpenAI API key here
#openai.api_key = "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"
client = openai.OpenAI(api_key="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")  # Replace with your actual API key

file_path = os.path.join(os.getcwd(), "data_preprocessing\LLM-testing\sleepDay_merged.csv")

# Configuration
model = "gpt-3.5-turbo-1106"
records_per_user = 100
batch_size = 20
user_ids = [f'p{str(i).zfill(2)}' for i in range(1, 117)]  # p01 to p116

# Columns expected in each row
columns = [
    'Id', 'SleepDay', 'TotalSleepRecords',
    'TotalMinutesAsleep', 'TotalTimeInBed'
]

# Container for all generated records
all_records = []

# Function to get synthetic data for one batch
def generate_batch(participant_id, batch_size=10, retries=5):

    # Read the existing data from the CSV file
    df = pd.read_csv(file_path)
    
    # Extract sample data from the file (e.g., latest records, or a subset of relevant data)
    # Example: get the last 5 records for that participant
    relevant_data = df[df['Id'] == participant_id].tail(5)

    # Create a summary of the recent data to guide the generation of new records
    recent_data_summary = ""
    for _, row in relevant_data.iterrows():
        recent_data_summary += f"Date: {row['SleepDay']}, Sleep Records: {row['TotalSleepRecords']}, Minutes Asleep: {row['TotalMinutesAsleep']}, Time in Bed: {row['TotalTimeInBed']}\n"


    prompt = (
        f"Based on the following recent sleep data for participant {participant_id}, generate {batch_size} new sleep records in CSV format:\n"
        f"Recent Data:\n{recent_data_summary}\n"
        f"New records should have the following fields:\n"
        "- Id: always {participant_id}\n"
        "- SleepDay: format YYYY-MM-DD\n"
        "- TotalSleepRecords: integer 1–3 ONLY\n"
        "- TotalMinutesAsleep: integer between 200 and 800\n"
        "- TotalTimeInBed: integer, >= TotalMinutesAsleep and up to 960\n"
        "Output ONLY raw CSV rows. No headers or explanations.\n"
        "Each row must have EXACTLY 5 comma-separated values."
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            csv_text = response.choices[0].message.content 
            lines = csv_text.strip().split('\n')
            data = [line.split(',') for line in lines]
            return data
        except Exception as e:
            print(f"⚠️ Error: {e} (Attempt {attempt + 1})")
            if attempt < retries - 1:
                time.sleep(2 ** attempt + random.random())
            else:
                print("❌ Skipping batch due to repeated errors.")
                return []

# Main data generation loop with progress bar
for participant_id in tqdm(user_ids, desc="Generating synthetic wellness data"):
    for _ in range(records_per_user // batch_size):
        batch = generate_batch(participant_id)
        all_records.extend(batch)
        time.sleep(random.uniform(0.8, 1.5))  # avoid rate limits

# Filter out malformed rows
filtered_records = [row for row in all_records if len(row) == len(columns)]

# Optional: print summary of bad rows
bad_rows = [row for row in all_records if len(row) != len(columns)]
if bad_rows:
    print(f"\n⚠️ Skipped {len(bad_rows)} malformed rows (wrong number of columns).")

# Create DataFrame and save
df = pd.DataFrame(filtered_records, columns=columns)
df.to_csv("synthetic_sleepDay.csv", index=False)

print(f"\n✅ Done! {len(df)} valid records saved to 'synthetic_sleepDay.csv'")


