import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the best parameters from GridSearch
best_params = joblib.load('models/best_params.pkl')

# Load the data
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()  # Convert to 1D array
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()    # Convert to 1D array

# Initialize the model with the best parameters
model = RandomForestRegressor(**best_params, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'models/gbr_model.pkl')

print("Model trained and saved successfully.")

