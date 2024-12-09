from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib
import json

# Load the trained model
model = joblib.load('models/gbr_model.pkl')

# Load the test data
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Make predictions
y_pred = model.predict(X_test)

# Save predictions to a CSV file
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv('data/prediction.csv', index=False)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Prepare the metrics dictionary
metrics = {
    'mean_squared_error': mse,
    'r2_score': r2
}

# Save the metrics to a JSON file
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

print("Model evaluation completed and metrics saved.")
