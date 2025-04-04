import json
import sys
import os
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_json, save_json  # Assuming you have these utility functions.

# bootsratp made hardly any difference
param_grid_rf = {
    "n_estimators": [5, 100, 300, 500],        
    "max_depth": [5, None],
    "max_features": [None, 0.5,  "sqrt"],
    "min_samples_split": [2, 8],
}

param_grid_xgb = {
    "n_estimators": [5,  100, 300, 500],          
    "max_depth": [5, None],
    "max_features": [None, 0.5,  "sqrt"],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8], # tested one before wasn'ta s good
}

output_path = os.path.join(os.path.dirname(__file__), "model_parameters.json")

model_names = ["XGBoost", "RandomForest"]
param_grids = [param_grid_rf, param_grid_xgb]

grids_dict = {
    "XGBoost": param_grid_xgb,
    "RandomForest": param_grid_rf
}
output_dict = {}

output_dict["parameter_names"] = list(set(list(param_grid_rf.keys()) + list(param_grid_xgb.keys())))

for model_name in model_names:
    param_grid = grids_dict[model_name]
    parameter_list = []
    
    # Collect all parameter names and their possible values
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    # Generate all combinations using itertools.product
    for combination in product(*param_values):
        # Build a dictionary for each combination
        combo_dict = dict(zip(param_names, combination))
        parameter_list.append(combo_dict)
    
    output_dict[model_name] = parameter_list

# Save to JSON
save_json(output_dict, output_path)

print(f"Parameters saved to {output_path}")