import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Load and combine datasets
wellness_data = pd.read_csv("C:\\Uni Python\\wellness_data.csv")
wellness_synthetic = pd.read_csv("C:\\Uni Python\\synthetic_wellness_data.csv")
wellness_final = pd.concat([wellness_data, wellness_synthetic], ignore_index=True)
wellness_final = wellness_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Output folder (change if needed)
output_folder = "C:\\Uni Python\\tsne_plots"
os.makedirs(output_folder, exist_ok=True)

# Define participant ranges
participant_ranges = {
    "p01-p16": range(1, 17),
    "p01-p20": range(1, 21),
    "p01-p40": range(1, 41),
    "p01-p60": range(1, 61),
    "p01-p80": range(1, 81),
    "p01-p100": range(1, 101),
}

# Generate t-SNE plots and save them
for label, r in participant_ranges.items():
    ids = [f"p{str(i).zfill(2)}" for i in r]
    subset = wellness_final[wellness_final['participant_id'].isin(ids)]
    numeric_data = subset.select_dtypes(include='number')

    if numeric_data.shape[0] < 5:
        print(f"Skipping {label}: too few rows.")
        continue

    scaled_data = StandardScaler().fit_transform(numeric_data)

    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    tsne_results = tsne.fit_transform(scaled_data)

    # Create and save the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7, c='skyblue')
    plt.title(f"t-SNE Plot for Participants {label}")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True)
    plt.tight_layout()
    
    file_path = os.path.join(output_folder, f"tsne_{label}.png")
    plt.savefig(file_path)
    plt.close()

    print(f"Saved: {file_path}")
