import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------
# 1. Data Loading and Cleaning
# -----------------

file_path = 'LLCP2024.XPT'

 # Use pandas to read the .xpt file
print("Loading data, please wait...")
df = pd.read_sas(file_path, format='xport')
print("Data loading complete.")
print("\nColumn names in the data:")
print(list(df.columns))
# Save column names to a text file
with open("column_names.txt", "w", encoding="utf-8") as f:
    for col in df.columns:
        f.write(col + "\n")
print("\nAll column names have been saved to column_names.txt. Please check the folder.")

selected_features = ['CHCSCNC1', '_SMOKER3', '_AGE_G', '_SEX', '_BMI5', '_STATE']
df_filtered = df[selected_features].copy()

# Remove rows with missing values
df_filtered = df_filtered.dropna()
print(f"Data cleaning complete, {len(df_filtered)} rows remaining.")

# Count and print the number of records per state
state_counts = df_filtered['_STATE'].value_counts().sort_index()
print("\nNumber of records per state:")
for state, count in state_counts.items():
    print(f"State {int(state)}: {count} records")

# Save the processed data as CSV for Tableau use
output_csv_path = 'brfss_cancer_data.csv'
df_filtered.to_csv(output_csv_path, index=False)
print(f"Processed data saved to: {output_csv_path}")

# -----------------
# 2. Machine Learning Model Training
# -----------------

# Define features (X) and target (y)
# Target variable 'CHCSCNC1': 1=Yes, 2=No
# Convert 2 to 0 (No) for classification model
df_filtered['CHCSCNC1'] = df_filtered['CHCSCNC1'].replace({2: 0})

X = df_filtered.drop('CHCSCNC1', axis=1) # Remove target column
y = df_filtered['CHCSCNC1']              # Set target column

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['_SMOKER3', '_AGE_G', '_SEX'], drop_first=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
print("\nTraining machine learning model...")
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate model performance
y_pred = model.predict(X_test)
print("\nModel classification report:")
print(classification_report(y_test, y_pred))

# -----------------
# 3. Save Model
# -----------------

# Save the trained model as a .pkl file
model_filename = 'cancer_risk_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"\nTrained model saved to: {model_filename}")

# Save feature list for use in web applications
features_filename = 'model_features.pkl'
with open(features_filename, 'wb') as file:
    pickle.dump(X.columns.tolist(), file)
print(f"Model feature list saved to: {features_filename}")

print("\nScript execution complete. Your data and model are ready!")