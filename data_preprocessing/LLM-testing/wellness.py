import openai
import pandas as pd
import time
import random
from tqdm import tqdm

# Set your OpenAI API key here
openai.api_key = "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"

# Configuration
model = "gpt-3.5-turbo-1106"
records_per_user = 100
batch_size = 20
user_ids = [f'p{str(i).zfill(2)}' for i in range(1, 117)]  # p01 to p116

# Columns expected in each row
columns = [
    'effective_time_frame', 'fatigue', 'mood', 'readiness',
    'sleep_duration_h', 'sleep_quality', 'soreness', 'soreness_area',
    'stress', 'participant_id', 'date'
]

# Container for all generated records
all_records = []

# Function to get synthetic data for one batch
def generate_batch(participant_id, retries=5):
    prompt = (
        f"Generate {batch_size} fake wellness records in CSV format with the following fields and constraints:\n"
        "- effective_time_frame: ISO 8601 timestamp like '2019-11-01T08:31:40.751000+00:00'\n"
        "- fatigue: integer 1–5 ONLY\n"
        "- mood: integer 1–5 ONLY\n"
        "- readiness: integer 1–10 ONLY\n"
        "- sleep_duration_h: integer 4–10 ONLY\n"
        "- sleep_quality: integer 1–5 ONLY (MAX 5)\n"
        "- soreness: integer 1–5 ONLY\n"
        "- soreness_area: a stringified list like \"[12921003]\" or \"[]\"\n"
        "- stress: integer 1–5 ONLY\n"
        f"- participant_id: always '{participant_id}'\n"
        "- date: format YYYY-MM-DD\n"
        "Output ONLY raw CSV rows. No headers or explanations.\n"
        "Each row must have EXACTLY 11 comma-separated values."
    )

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            csv_text = response['choices'][0]['message']['content']
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
df.to_csv("synthetic_wellness_data_4.csv", index=False)

print(f"\n✅ Done! {len(df)} valid records saved to 'synthetic_wellness_data_4.csv'")


