from neural import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_dataset(file):
    df = pd.read_excel(file)
    # Remove unneeded id column
    df.drop('No', axis=1, inplace=True)
    # Extract month from transaction date and drop original column
    df['month'] = pd.to_datetime(df['X1 transaction date']).dt.month
    df.drop('X1 transaction date', axis=1, inplace=True)
    # Min-max normalization for specified columns with safe division check
    for col in ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'month', 'Y house price of unit area']:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val != 0:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0
    # Normalize latitude and longitude using known Taiwan coordinate bounds
    df['latitude'] = (df['X5 latitude'] - 21.5) / (25.5 - 21.5)
    df['longitude'] = (df['X6 longitude'] - 119) / (122 - 119)
    # Build NN input-output pairs
    # Inputs are [month, house age, distance to MRT, convenience stores, latitude, longitude]
    data = []
    for _, row in df.iterrows():
        inputs = [row['month'], row['X2 house age'], row['X3 distance to the nearest MRT station'], row['X4 number of convenience stores'], row['latitude'], row['longitude']]
        output = [row['Y house price of unit area']]
        data.append((inputs, output))
    return data

print("Processing data...", end="")
data = preprocess_dataset("data.xlsx")
print("done!")
# 80% training, 20% testing
train_data, test_data = train_test_split(data, test_size=0.2)

print(f"Training network (train: {len(train_data)}, test: {len(test_data)})")
real_estate_net = NeuralNet(6, 3, 1) # 6 inputs, 3 hidden neurons, 1 output
real_estate_net.train(train_data, iters=10000, print_interval=500)

# Evaluate model
predictions = [real_estate_net.evaluate(inputs) for (inputs, output) in test_data]
print("\nTrained! Predictions:")
for (inputs, output), pred in zip(test_data, predictions):
    print(f"Predicted: {round(pred[0], 5)} | Actual: {round(float(output[0]), 5)}")
