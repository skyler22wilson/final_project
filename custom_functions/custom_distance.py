import numpy as np
import pandas as pd

parts_data = pd.read_csv('/Users/skylerwilson/Desktop/Lighthouse_Labs/Projects/final_project/data/Project_Data/final_parts_data.csv')

feature_mask = (parts_data['Months No Sale'] < 3) & (parts_data['Sales Last 3 Months'] > 0)

# Create a numpy array with the features you want to use for clustering
X_features = parts_data.loc[feature_mask, ['Turnover', 'EOQ', 'Demand', 'Months No Sale']].values

# Define the custom distance function using vectorized operations
def custom_distance_function(x, y):
    weights = np.array([0.35, 0.15, 0.20, 0.30])
    diffs = x - y
    squared_diffs = diffs**2
    weighted_squared_diffs = weights * squared_diffs
    return np.sum(weighted_squared_diffs)