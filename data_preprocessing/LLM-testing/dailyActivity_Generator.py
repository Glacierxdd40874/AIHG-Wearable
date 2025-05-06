import openai
import pandas as pd
import time
import random
from tqdm import tqdm
import os


# Set your OpenAI API key here
#openai.api_key = "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"
client = openai.OpenAI(api_key="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")  # Replace with your actual API key

# Configuration
model = "gpt-3.5-turbo-1106"
records_per_user = 100
batch_size = 20
user_ids = [f'p{str(i).zfill(2)}' for i in range(1, 117)]  # p01 to p116

file_path = os.path.join(os.getcwd(), "data_preprocessing\LLM-testing\dailyActivity_merged.csv")

# Columns expected in each row
columns = [
    'Id', 'ActivityDate', 'TotalSteps', 'TotalDistance', 'TrackerDistance',
    'LoggedActivitiesDistance', 'VeryActiveDistance', 'ModeratelyActiveDistance',
    'LightActiveDistance', 'SedentaryActiveDistance', 'VeryActiveMinutes',
    'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories'
]

# Container for all generated records
all_records = []

# Function to get synthetic data for one batch
def generate_batch(participant_id, batch_size=10, retries=5):

    # Read the existing data from the CSV file
    df = pd.read_csv(file_path)
    
    # Extract recent data for that participant
    relevant_data = df[df['Id'] == participant_id].tail(5)

    # Create a summary of the recent data to guide generation
    recent_data_summary = ""
    for _, row in relevant_data.iterrows():
        recent_data_summary += (
            f"Date: {row['ActivityDate']}, Steps: {row['TotalSteps']}, Distance: {row['TotalDistance']} mi, "
            f"Very Active Minutes: {row['VeryActiveMinutes']}, Sedentary Minutes: {row['SedentaryMinutes']}, "
            f"Calories: {row['Calories']}\n"
        )

    # Build prompt
    prompt = (
        f"Based on the recent activity data below for participant {participant_id}, generate {batch_size} new activity records in CSV format:\n\n"
        f"Recent Data:\n{recent_data_summary}\n"
        f"Each new record must include the following columns, in order:\n"
        "Id, ActivityDate (YYYY-MM-DD), TotalSteps, TotalDistance, TrackerDistance, "
        "LoggedActivitiesDistance, VeryActiveDistance, ModeratelyActiveDistance, "
        "LightActiveDistance, SedentaryActiveDistance, VeryActiveMinutes, FairlyActiveMinutes, "
        "LightlyActiveMinutes, SedentaryMinutes, Calories\n\n"
        f"Constraints:\n"
        f"- Id must always be {participant_id}\n"
        f"- Distances should be consistent with steps and activity level\n"
        f"- Minutes values should sum reasonably (~1440/day max)\n"
        f"- Output ONLY raw CSV rows (no headers, no explanation)\n"
        f"- Output exactly 15 comma-separated values per row"
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
df.to_csv("synthetic_dailyActivity.csv", index=False)

print(f"\n✅ Done! {len(df)} valid records saved to 'synthetic_dailyActivity.csv'")


