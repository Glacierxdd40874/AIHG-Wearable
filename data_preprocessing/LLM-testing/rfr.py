import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# ----------- Load and Combine Data -----------
wellness_data = pd.read_csv("C:\\Uni Python\\wellness_data.csv")
wellness_synthetic = pd.read_csv("C:\\Uni Python\\synthetic_wellness_data.csv")
wellness_final = pd.concat([wellness_data, wellness_synthetic], ignore_index=True)
wellness_final = wellness_final.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------- Prepare Features and Target -----------
wellness_numeric = wellness_final.select_dtypes(include='number').dropna()
X_all = wellness_numeric.drop(columns=['sleep_quality'])
y_all = wellness_numeric['sleep_quality']

# Define participant groups
participant_ranges = {
    "p01-p16": range(1, 17),
    "p01-p20": range(1, 21),
    "p01-p40": range(1, 41),
    "p01-p60": range(1, 61),
    "p01-p80": range(1, 81),
    "p01-p100": range(1, 101),
}

# Initialize results table
results = []

# ----------- Loop over each participant group -----------
for label, r in participant_ranges.items():
    # Generate participant IDs (e.g., p01, p02, ..., pNN)
    ids = [f"p{str(i).zfill(2)}" for i in r]
    
    # Filter data for that range of participants
    subset = wellness_final[wellness_final['participant_id'].isin(ids)]
    X = subset.select_dtypes(include='number').drop(columns=['sleep_quality'])
    y = subset['sleep_quality']
    
    if X.shape[0] < 5:
        print(f"Skipping {label}: too few rows for t-SNE or Random Forest.")
        continue
    
    # ----------- Cross-Validated Random Forest -----------
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Perform 5-fold cross-validation using R² score
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Calculate mean and std of R²
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Append results to the table
    results.append([label, f"{mean_score:.3f} ± {std_score:.3f}"])

# ----------- Convert Results to DataFrame and Display -----------
results_df = pd.DataFrame(results, columns=['Participant Range', 'Random Forest R² Score'])
print(results_df)
