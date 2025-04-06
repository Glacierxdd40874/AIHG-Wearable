import pandas as pd
import re

# Load datasets
url = "https://huggingface.co/datasets/sarthak-wiz01/nutrition_dataset/resolve/main/nutrition_dataset.csv"
df = pd.read_csv(url, header=0)

df = df.drop([40, 208], axis=0)  # Delete useless data from lines 42 and 210 (note: index starts at 0)

# Define the cleaning function
def clean_and_split_suggestions(text):
    if isinstance(text, str):
        # First remove "with", "and", and "on" as conjunctions, but keep them as delimiters
        text = re.sub(r'\s(?:with|and|on)\s', ', ', text)
        
        # Replace extra whitespace characters with spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Separate food names by commas
        food_items = text.split(', ')
        
        # Returns the list of food names after segmentation
        return food_items
    return []

# Clean and split the recommendations in each column
df['Breakfast Suggestion'] = df['Breakfast Suggestion'].apply(clean_and_split_suggestions)
df['Lunch Suggestion'] = df['Lunch Suggestion'].apply(clean_and_split_suggestions)
df['Dinner Suggestion'] = df['Dinner Suggestion'].apply(clean_and_split_suggestions)
df['Snack Suggestion'] = df['Snack Suggestion'].apply(clean_and_split_suggestions)

# View the processed data (show the first few rows)
print(df[['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion']].head())

# Save the names of the cleaned food in a trainable format (with commas as separators)
df['Breakfast Suggestion'] = df['Breakfast Suggestion'].apply(lambda x: ', '.join(x))
df['Lunch Suggestion'] = df['Lunch Suggestion'].apply(lambda x: ', '.join(x))
df['Dinner Suggestion'] = df['Dinner Suggestion'].apply(lambda x: ', '.join(x))
df['Snack Suggestion'] = df['Snack Suggestion'].apply(lambda x: ', '.join(x))

# Save the cleaned data
df.to_csv('cleaned_and_split_nutrition_dataset.csv', index=False)
