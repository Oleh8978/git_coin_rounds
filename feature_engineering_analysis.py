import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
import shap

# Load dataset
dataset_path = "datasets/dataset_new.csv"  # Change this to your file path
df = pd.read_csv(dataset_path)

# Step 1: Log Transformations
# Log transformation helps normalize skewed distributions
df['LOG_MATCHING_AMOUNT'] = np.log1p(df['Matching Amount'])  # log(1 + x)
df['LOG_CONTRIBUTION_AMOUNT'] = np.log1p(df['Contribution Amount'])

# Step 2: Contribution Ratios
# These ratios help understand project funding dynamics
df['CONTRIBUTION_PER_CONTRIBUTOR'] = df['Contribution Amount'] / df['# of Contributors'].replace(0, np.nan)
df['MATCH_TO_CONTRIBUTION_RATIO'] = df['Matching Amount'] / df['Contribution Amount'].replace(0, np.nan)

# Step 3: Aggregate Features at Round Level
# Summing up all contributions and matchings in each round
df['ROUND_TOTAL_CONTRIBUTIONS'] = df.groupby('Round Name')['Contribution Amount'].transform('sum')
df['ROUND_TOTAL_MATCHING'] = df.groupby('Round Name')['Matching Amount'].transform('sum')

# Step 4: Project Frequency
# Count how many times a project appears in different rounds
df['PROJECT_FREQUENCY'] = df.groupby('Gitcoin Project Id')['Gitcoin Project Id'].transform('count')

# Step 5: Feature Correlation Analysis

# Select only numeric features
numeric_df = df.select_dtypes(include=[np.number])

# Compute and plot the correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 6: Clustering Projects Based on Contributions
features = ['LOG_CONTRIBUTION_AMOUNT', 'LOG_MATCHING_AMOUNT', 'CONTRIBUTION_PER_CONTRIBUTOR']
df.dropna(subset=features, inplace=True)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=3, random_state=42)
df['CLUSTER'] = kmeans.fit_predict(scaled_features)

# Visualizing Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['LOG_CONTRIBUTION_AMOUNT'], y=df['LOG_MATCHING_AMOUNT'], hue=df['CLUSTER'], palette='viridis')
plt.title('Project Clustering')
plt.xlabel('Log Contribution Amount')
plt.ylabel('Log Matching Amount')
plt.show()

# Step 7: Outlier Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['OUTLIER_SCORE'] = iso_forest.fit_predict(df[features])

# Highlighting Outliers
outliers = df[df['OUTLIER_SCORE'] == -1]
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['LOG_CONTRIBUTION_AMOUNT'], y=df['LOG_MATCHING_AMOUNT'], hue=df['OUTLIER_SCORE'], palette='coolwarm')
plt.title('Outlier Detection')
plt.xlabel('Log Contribution Amount')
plt.ylabel('Log Matching Amount')
plt.show()

# Step 8: Feature Importance Analysis with Random Forest
X = df[['LOG_CONTRIBUTION_AMOUNT', 'LOG_MATCHING_AMOUNT', 'CONTRIBUTION_PER_CONTRIBUTOR', 'PROJECT_FREQUENCY']]
y = df['Matching Amount']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot feature importance
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()

# Step 9: SHAP Value Analysis
explainer = shap.Explainer(rf, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)

# Save the enhanced dataset
df.to_csv("datasets/dataset_with_features.csv", index=False)
