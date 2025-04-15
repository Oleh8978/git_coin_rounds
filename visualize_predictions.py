import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load datasets
dataset_path = "datasets/dataset_new.csv"
submission_path = "submission.csv"

dataset = pd.read_csv(dataset_path)
submission = pd.read_csv(submission_path)

# Ensure column names are consistent
dataset.rename(columns={"Gitcoin Project Id": "PROJECT", "Matching Amount": "ACTUAL_AMOUNT"}, inplace=True)

# Get project name from command line argument
if len(sys.argv) < 2:
    print("Usage: python visualize_predictions.py <PROJECT_NAME>")
    sys.exit(1)

project_name = sys.argv[1]

# Filter actual and predicted values
actual = dataset[dataset["PROJECT"] == project_name]
predicted = submission[submission["PROJECT"] == project_name]

if actual.empty or predicted.empty:
    print(f"No data found for project: {project_name}")
    sys.exit(1)

# Merge data
merged = actual.merge(predicted, on="PROJECT", how="inner")
merged = merged[["PROJECT", "ACTUAL_AMOUNT", "AMOUNT"]]  # Predicted column is "AMOUNT"

# Plot
plt.figure(figsize=(8, 5))
plt.bar(["Actual", "Predicted"], [merged["ACTUAL_AMOUNT"].values[0], merged["AMOUNT"].values[0]], color=["blue", "orange"])
plt.xlabel("Type")
plt.ylabel("Funding Amount")
plt.title(f"Actual vs Predicted Funding for {project_name}")
plt.show()
