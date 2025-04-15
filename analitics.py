import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("datasets/dataset_new.csv")

# Compute correlation matrix
corr_matrix = dataset[["Matching Amount", "# of Contributors", "Contribution Amount"]].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

print(dataset[["Matching Amount", "# of Contributors", "Contribution Amount"]].corr())

