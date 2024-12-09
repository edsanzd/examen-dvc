import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the previously split data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

# Drop the 'date' column before scaling
X_train = X_train.drop(columns=['date'])
X_test = X_test.drop(columns=['date'])

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Save the scaled data to processed directory
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed/X_test_scaled.csv', index=False)

# Save the scaler for later use
joblib.dump(scaler, 'models/scaler.pkl')

print("Data normalization complete and scaler saved.")
