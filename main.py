import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Load historical dataset
dataset_path = 'datasets/dataset_new.csv'
df = pd.read_csv(dataset_path)

# Rename columns
df.rename(columns={'Round Name': 'ROUND', 'Contribution Amount': 'AMOUNT', 'Application Title': 'PROJECT', 'Matching Amount': 'MATCHING', '# of Contributors': 'CONTRIBUTORS'}, inplace=True)

# Load project list for predictions
projects_path = 'datasets/projects_Apr_1.csv'
projects = pd.read_csv(projects_path)
projects.rename(columns={'Application Title': 'PROJECT'}, inplace=True)

# Preserve PROJECT and ROUND columns
projects_original = projects[['PROJECT_ID', 'PROJECT', 'ROUND']].copy()

# Feature Engineering
df['project_length'] = df['PROJECT'].apply(lambda x: len(str(x)))

# Calculate historical mean and count of funding per project
df['project_mean_funding'] = df.groupby('PROJECT')['AMOUNT'].transform('mean')
df['project_count'] = df.groupby('PROJECT')['AMOUNT'].transform('count')

# Fill missing values for new projects
df['project_mean_funding'] = df['project_mean_funding'].fillna(0)
df['project_count'] = df['project_count'].fillna(0)

# One-hot encode ROUND feature
encoder = OneHotEncoder(handle_unknown='ignore')
round_encoded = encoder.fit_transform(df[['ROUND']]).toarray()
round_encoded_df = pd.DataFrame(round_encoded, columns=encoder.get_feature_names_out(['ROUND']))
df = pd.concat([df, round_encoded_df], axis=1)

# Text features from Application Title
vectorizer = TfidfVectorizer(max_features=100)
df_title_features = vectorizer.fit_transform(df['PROJECT']).toarray()
df_title_df = pd.DataFrame(df_title_features, columns=[f'title_{i}' for i in range(df_title_features.shape[1])])
df = pd.concat([df, df_title_df], axis=1)

# Include Matching Amount and Contributors
features = ['project_length', 'project_mean_funding', 'project_count', 'MATCHING', 'CONTRIBUTORS'] + list(round_encoded_df.columns) + list(df_title_df.columns)
target = 'AMOUNT'

# Apply Robust Scaling
scaler = RobustScaler()
df[features] = scaler.fit_transform(df[features])

# Split data for training and validation with cross-validation
X_train, X_val, y_train, y_val = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Hyperparameter Tuning using GridSearchCV for RandomForest and XGBoost
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['auto', 'sqrt']
}
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)

# Grid Search for RandomForest
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# Grid Search for XGBoost
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_

# Stacking Regressor with additional base models (GradientBoosting)
estimators = [
    ('rf', best_rf),
    ('xgb', best_xgb),
    ('gb', GradientBoostingRegressor(random_state=42)),
]

model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
model.fit(X_train, y_train)

# K-fold Cross-Validation
cv = KFold(n_splits=5, random_state=42, shuffle=True)
cross_val_scores = cross_val_score(model, df[features], df[target], cv=cv, scoring='neg_root_mean_squared_error')
print(f'5-Fold Cross-Validation RMSE: {-cross_val_scores.mean()}')

# Validate model
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f'Validation RMSE: {rmse}')
print(f'Validation R²: {r2}')

# Prepare test data
projects['project_length'] = projects['PROJECT'].apply(lambda x: len(str(x)))
projects['project_mean_funding'] = projects.groupby('PROJECT')['AMOUNT'].transform('mean').fillna(0)
projects['project_count'] = projects.groupby('PROJECT')['AMOUNT'].transform('count').fillna(0)
projects['MATCHING'] = 0
projects['CONTRIBUTORS'] = 0

round_encoded_test = encoder.transform(projects[['ROUND']]).toarray()
round_encoded_test_df = pd.DataFrame(round_encoded_test, columns=encoder.get_feature_names_out(['ROUND']))
projects = pd.concat([projects, round_encoded_test_df], axis=1)

projects_title_features = vectorizer.transform(projects['PROJECT']).toarray()
projects_title_df = pd.DataFrame(projects_title_features, columns=[f'title_{i}' for i in range(projects_title_features.shape[1])])
projects = pd.concat([projects, projects_title_df], axis=1)

projects_scaled = scaler.transform(projects[features])
projects_scaled_df = pd.DataFrame(projects_scaled, columns = features)

# Make predictions
projects['AMOUNT'] = model.predict(projects_scaled_df)

# Restore PROJECT and ROUND columns
projects = pd.concat([projects_original, projects[['AMOUNT']]], axis=1)

# Save submission file
submission_path = 'submission.csv'
projects.to_csv(submission_path, index=False)
print(f'Submission file saved: {submission_path}')

# Save model
joblib.dump(model, 'model.pkl')