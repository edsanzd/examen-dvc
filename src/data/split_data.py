import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset
data_path = os.path.join(os.getcwd(), 'data', 'raw', 'raw.csv')
df = pd.read_csv(data_path)

# Target variable is 'silica_concentrate' (last column)
X = df.drop(columns=['silica_concentrate'])
y = df['silica_concentrate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the datasets into the processed folder
processed_path = os.path.join(os.getcwd(), 'data', 'processed')
os.makedirs(processed_path, exist_ok=True)

X_train.to_csv(os.path.join(processed_path, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(processed_path, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(processed_path, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(processed_path, 'y_test.csv'), index=False)

print("Data splitting completed and saved in data/processed.")
