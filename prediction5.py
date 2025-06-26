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
session_2024 = fastf1.get_session(2024, "Saudi Arabia", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# 3. Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# 4. Aggregate sector times
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()
sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# 5. Qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR", "NOR"],
    "QualifyingTime (s)": [87.294, 87.304, 87.670, 87.407, 88.201, 88.367, 88.303, 88.204, 88.164, 88.782, 89.092, 88.645, 87.489]
})

# 6. Performance metrics
average_2025 = {
    "VER": 88.0, "PIA": 89.1, "LEC": 89.2, "RUS": 89.3, "HAM": 89.4, 
    "GAS": 89.5, "ALO": 89.6, "TSU": 89.7, "SAI": 89.8, "HUL": 89.9, 
    "OCO": 90.0, "STR": 90.1, "NOR": 90.2
}

driver_wet_performance = {
    "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179, "ALO": 0.972655,
    "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338, "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)


# 7. Robust weather API handling
API_KEY = "yourkey"  # Replace with actual key
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=21.4225&lon=39.1818&appid={API_KEY}&units=metric"
response = requests.get(weather_url)

forecast_data = None
if response.status_code == 200:
    weather_data = response.json()
    if "list" in weather_data:
        forecast_time = "2025-04-20 18:00:00"
        forecast_data = next((f for f in weather_data["list"] if f.get("dt_txt") == forecast_time), None)

rain_probability = forecast_data.get("pop", 0) if forecast_data else 0
temperature = forecast_data.get("main", {}).get("temp", 20) if forecast_data else 20

# 8. Weather adjustment
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# 9. Team performance data
team_points = {
    "McLaren": 78, "Mercedes": 53, "Red Bull": 36, "Williams": 17, "Ferrari": 17,
    "Haas": 14, "Aston Martin": 10, "Kick Sauber": 6, "Racing Bulls": 3, "Alpine": 0
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "极rari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)
qualifying_2025["Average2025Performance"] = qualifying_2025["Driver"].map(average_2025)

# 10. Merge datasets
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
last_year_winner = "VER" 
merged_data["LastYearWinner"] = (merged_data["Driver"] == last_year_winner).astype(int)

# 11. Prepare features/target
features = [
    "QualifyingTime", "RainProbability", "Temperature", 
    "TeamPerformanceScore", "TotalSectorTime (s)", "Average2025Performance"
]
X = merged_data[features]

# Create target variable with aligned index
mean_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean()
y = merged_data["Driver"].map(mean_lap_times)

# 12. Handle missing values
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

# 13. Handle missing values in features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_clean)

# 14. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_clean, test_size=0.2, random_state=39)

# 15. Train model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=39)
model.fit(X_train, y_train)

# 16. Add predictions only for valid drivers
merged_data.loc[valid_mask, "PredictedRaceTime (s)"] = model.predict(X_imputed)

# 17. Results (only for drivers with predictions)
final_results = merged_data[valid_mask].sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Saudi Arabian GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

# 18. Podium results
podium = final_results.head(3)[["Driver", "PredictedRaceTime (s)"]]
print("\n🏆 Predicted Podium 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")

# 19. Model evaluation
y_pred = model.predict(X_test)
print(f"\nModel Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# 20. Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(features, model.feature_importances_)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# 21. Team performance visualization
plt.figure(figsize=(12, 8))
plt.scatter(
    final_results["TeamPerformanceScore"],
    final_results["PredictedRaceTime (s)"],
    c=final_results["QualifyingTime"],
    cmap="viridis",
    alpha=0.7
)
plt.colorbar(label="Qualifying Time (s)")
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, 
                 (final_results["TeamPerformanceScore"].iloc[i], 
                  final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), 
                 textcoords='offset points')
plt.xlabel("Team Performance Score")
plt.ylabel("Predicted Race Time (s)")
plt.title("Team Performance vs Predicted Results (Saudi Arabian GP 2025)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('team_performance.png')
plt.show()

# 22. Correlation analysis - FIXED FOR NON-NUMERIC COLUMNS
print("\nCorrelation Analysis:")
numeric_data = merged_data.select_dtypes(include=[np.number])
if not numeric_data.empty:
    corr_matrix = numeric_data.corr()
    print(corr_matrix)
else:
    print("No numeric columns for correlation analysis")
