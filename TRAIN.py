# Import required libraries
import pandas as pd
import numpy as np
from pycaret.regression import setup, compare_models, tune_model, save_model
import warnings

warnings.filterwarnings('ignore')  # Suppress common warnings

# Load the master dataset
master_file_path ='main_file_updated.csv'  # Use your path
try:
    master_df = pd.read_csv(master_file_path)
except FileNotFoundError:
    print(f"ERROR: Master file not found at {master_file_path}")
    exit()
except Exception as e:
    print(f"ERROR: Could not read master file: {e}")
    exit()

# --- Clean Column Names FIRST ---
print("Cleaning column names...")
original_cols = master_df.columns
master_df.columns = master_df.columns.str.replace('  ', ' ', regex=False)  # Replace double space with single
master_df.columns = master_df.columns.str.strip()  # Remove leading/trailing whitespace
new_cols = master_df.columns
changed_cols = {o: n for o, n in zip(original_cols, new_cols) if o != n}
if changed_cols:
    print("Renamed columns:", changed_cols)
else:
    print("No column names needed renaming.")

# --- Feature Engineering ---
# Select features and target
features = [
    'Credits', 'Player Type', 'Team', 'Innings', 'Runs', 'Balls Faced', 'Fours', 'Sixes',
    'Batting Average', 'Strike Rate', 'Boundary_Percentage', 'Innings_X', 'Balls Bowled',
    'Runs_Conceded', 'Wickets', 'Bowling Average', 'Economy Rate', 'Bowling Strike_rate',
]
target = 'Weighted_Score'

# Filter the dataset to include only the selected features and target
data = master_df[features + [target]].dropna()

# --- PyCaret Setup ---
print("\nSetting up PyCaret experiment...")
try:
    exp_reg = setup(data=data, target=target, session_id=123, normalize=True)
    print("PyCaret setup complete.")

    # --- Model Training and Tuning ---
    print("\nComparing models...")
    best_model = compare_models()
    print(f"\nBest model found: {best_model}")

    print("\nTuning the best model...")
    tuned_model = tune_model(best_model, optimize='R2')
    print("\nModel tuning complete.")
    print(tuned_model)

    # --- Save the Model ---
    save_path = r'C:\Users\Radhakrishna\Desktop\GAMEATHON\ml_avg_predictor_model'  # Use your path
    print(f"\nSaving the final tuned model to {save_path}...")
    save_model(tuned_model, save_path)  # Saves the best (original or tuned)

    print("✅ Model training complete. Model saved as 'ml_avg_predictor_model'.")

except Exception as e:
    # Generic exception handling
    print("\n❌❌❌ An error occurred during PyCaret setup or training! ❌❌❌")
    import traceback
    print(traceback.format_exc())
    print("\nData Info just before error:")
    print(data.info())
    print("\nFeatures List:")
    print(features)