import os
import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 1. Create cache directory if missing
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# 2. Load session data
session_2024 = fastf1.get_session(2024, "Japan", "R")
session_2024.load()

# 3. Get laps data with CORRECT column names
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# 4. Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# 5. Aggregate sector times
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 6. Qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "L极", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [86.983, 86.995, 87.027, 87.299, 87.318, 87.610, 87.822, 87.897, 88.000, 87.836, 88.570, 88.696, 89.271]
})

# 7. Performance metrics
driver_wet_performance = {
    "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179, "ALO": 0.972655,
    "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338, "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

# 8. Robust weather API handling
API_KEY = "YOURAPIKEY"  # Replace with actual key
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=34.8823&lon=136.5845&appid={API_KEY}&units=metric"
try:
    response = requests.get(weather_url)
    response.raise_for_status()  # Raise error for bad status
    weather_data = response.json()
except (requests.exceptions.RequestException, ValueError) as e:
    print(f"Weather API error: {e}")
    weather_data = {}

# 9. Weather data extraction
forecast_time = "2025-04-05 14:00:00"
forecast_data = None

if weather_data.get("list"):
    for forecast in weather_data["list"]:
        if forecast.get("dt_txt") == forecast_time:
            forecast_data = forecast
            break

# Safe data extraction
rain_probability = forecast_data.get("pop", 0) if forecast_data else 0
temperature = forecast_data.get("main", {}).get("temp", 20) if forecast_data else 20

# 10. Merge datasets
merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# 11. Prepare features
features = [
    "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "WetPerformanceFactor", "RainProbability", "Temperature"
]
X = merged_data[features].fillna(0)

# 12. Prepare target with aligned index
mean_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean()
y = merged_data["Driver"].map(mean_lap_times)

# 13. Handle missing values
print("\nChecking for drivers missing in 2024 data...")
missing_indices = y[y.isna()].index
if not missing_indices.empty:
    missing_drivers = merged_data.loc[missing_indices, "Driver"].tolist()
    print(f"Removing drivers with missing lap times: {missing_drivers}")
else:
    print("No missing drivers found")

# Create mask of valid drivers
valid_mask = y.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]
print(f"Training data: {len(X_clean)} drivers")

# 14. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=38)

# 15. Train model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# 16. Add predictions
merged_data.loc[valid_mask, "PredictedRaceTime (s)"] = model.predict(X_clean)

# 17. Results
final_results = merged_data[valid_mask].sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Japanese GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

# 18. Podium results
podium = final_results.head(3)[["Driver", "PredictedRaceTime (s)"]]
print("\n🏆 Predicted Podium 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")

# 19. Model evaluation
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
