import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("🚀 Running Nassau Candy Data Pipeline...")

# 1. Load Dataset
df = pd.read_csv("Nassau Candy Distributor.csv")

# 2. Date Cleansing and Target Logic
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')
df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df = df.dropna(subset=['Lead Time'])

# Remove Extreme Outliers
df = df[df['Lead Time'] >= 0]

print(f"Dataset Parsed successfully. Total Rows: {len(df)}")

# 3. Features & Target Definition
# Simulating a simple model matrix block
df['Is_Standard_Class'] = np.where(df['Ship Mode'] == 'Standard Class', 1, 0)
df['Unit_Cost'] = df['Cost'] / df['Units']

X = df[['Units', 'Sales', 'Is_Standard_Class']]
y = df['Lead Time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)

# 4. Fit Baseline Model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Performance
preds = model.predict(X_test)
print(f"📊 Baseline Random Forest Performance Evaluation:")
print(f" -> MAE:  {mean_absolute_error(y_test, preds):.2f} days")
print(f" -> RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f} days")
print(f" -> R² Score: {r2_score(y_test, preds):.2f}")
print("🏁 Pipeline run complete.")