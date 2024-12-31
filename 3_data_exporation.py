import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


## Data Exploration
df = pd.read_csv("dataset_with_descriptors.csv")

descriptors = ["MolecularWeight", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]

# Histograms for selected descriptors
def plot_histograms(data, columns, bins=30):
    for column in columns:
        plt.figure(figsize=(8, 6))
        plt.hist(data[column].dropna(), bins=bins, edgecolor="k", alpha=0.7)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

plot_histograms(df, descriptors)

# Correlation heatmap
def plot_correlation_heatmap(data, columns):
    correlation_matrix = data[columns].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.title("Correlation Heatmap")
    
    # Annotate the heatmap
    for i in range(len(columns)):
        for j in range(len(columns)):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()

plot_correlation_heatmap(df, descriptors)
