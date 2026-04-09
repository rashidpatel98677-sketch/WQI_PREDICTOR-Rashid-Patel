import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 1️⃣ Load dataset
data = pd.read_csv("data.csv")

# 2️⃣ Clean column names (strip whitespace)
data.columns = data.columns.str.strip()

# Confirm columns:
# print(data.columns.tolist())
# Expect: ['sample', 'pH', 'TDS', 'Cl', 'SO4', 'Na', 'K', 'Ca', 'Mg', 'Total Hardness', 'WQI']

# 3️⃣ Separate features and target
X = data.drop(["sample", "WQI"], axis=1)
y = data["WQI"]

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5️⃣ Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 6️⃣ Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7️⃣ Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"Test RMSE: {rmse:.3f}")

# 8️⃣ Save model & scaler
joblib.dump(model, "wqi_regressor.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully.")
