# run_model.py
import pandas as pd
from preprocess import preprocess_data
from model import train_model

# Load the raw data
df_raw = pd.read_csv("kc_house_data.csv")

# Preprocess the data
df_cleaned = preprocess_data(df_raw)

# Save the cleaned data for use in the app
df_cleaned.to_csv("property_data.csv", index=False)

# Train and save the model
train_model(df_cleaned)
