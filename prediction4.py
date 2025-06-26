import os
import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# 1. Create cache directory if missing
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# 2. Load session data
session_2024 = fastf1.get_session(2024, "Bahrain", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# 3. Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()


# 4. Aggregate sector times
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 5. Qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.886, 92.283]
})

# 6. Performance metrics
driver_wet_performance = {
    "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179, "ALO": 0.972655,
    "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338, "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

season_points = {
    "VER": 61, "NOR": 62, "PIA": 80, "LEC": 20, "RUS": 20,
    "HAM": 20, "GAS": 20, "ALO": 20, "T极": 20, "SAI": 20,
    "HUL": 2, "OCO": 8, "STR": 11
}
qualifying_2025["SeasonPoints"] = qualifying_2025["Driver"].map(season_points)

# 7. Robust weather API handling
API_KEY = "YOURAPIKEY"  # Replace with actual key
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=26.0325&lon=50.5106&appid={API_KEY}&units=metric"
response = requests.get(weather_url)

forecast_data = None
if response.status_code == 200:
    weather_data = response.json()
    if "list" in weather_data:
        forecast_time = "2025-04-30 15:00:00"
        forecast_data = next((f for f in weather_data["list"] if f.get("dt_txt") == forecast_time), None)

rain_probability = forecast_data.get("pop", 0) if forecast_data else 0
temperature = forecast_data.get("main", {}).get("temp", 20) if forecast_data else 20

# 8. Merge datasets
merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# 9. Prepare features/target
features = [
    "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "WetPerformanceFactor", "RainProbability", "Temperature", "SeasonPoints"
]
X = merged_data[features].fillna(0)

# Create target with aligned index
mean_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean()
y = merged_data["Driver"].map(mean_lap_times)

# 10. Handle missing values
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

# 11. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=38)

# 12. Train model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# 13. Add predictions only for valid drivers
merged_data.loc[valid_mask, "PredictedRaceTime (s)"] = model.predict(X_clean)

# 14. Results
final_results = merged_data[valid_mask].sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Bahrain GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

# 15. Podium results
podium = final_results.head(3)[["Driver", "PredictedRaceTime (s)"]]
print("\n🏆 Predicted Podium 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")

# 16. Model evaluation
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# 17. Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(features, model.feature_importances_)
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()

# 18. Clean air pace visualization (optional)
plt.figure(figsize=(12, 8))
plt.scatter(
    final_results["QualifyingTime (s)"],
    final_results["PredictedRaceTime (s)"],
    c=final_results["SeasonPoints"],
    cmap="viridis",
    alpha=0.7
)
plt.colorbar(label="Season Points")
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, 
                 (final_results["QualifyingTime (s)"].iloc[i], 
                  final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), 
                 textcoords='offset points')
plt.xlabel("Qualifying Time (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Qualifying vs Predicted Race Times (Bahrain GP 2025)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('bahrain_performance.png')
plt.show()
