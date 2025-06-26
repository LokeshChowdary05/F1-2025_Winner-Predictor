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
session_2024 = fastf1.get_session(2024, 7, "Q")
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

# 5. Define clean air race pace
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128
}

# 6. Qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
               "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [  
        74.704, 74.962, 74.670, 74.807, 75.432, 75.473, 75.604,
        76.613, 75.765, 75.581, 75.787, 75.431, 76.518
    ]
})
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# 7. Robust weather API handling
API_KEY = "YOURAPIKEY"  # Replace with actual key
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=44.3439&lon=11.7167&appid={API_KEY}&units=metric"
response = requests.get(weather_url)

forecast_data = None
if response.status_code == 200:
    weather_data = response.json()
    if "list" in weather_data:
        forecast_time = "2025-05-18 06:00:00"
        forecast_data = next((f for f in weather_data["list"] if f.get("dt_txt") == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20

# 8. Simplified weather adjustment (remove wet factor)
qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# 9. Team performance data
team_points = {
    "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37, "Ferrari": 94,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# 10. Merge datasets
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# 11. Prepare features/target
features = [
    "QualifyingTime", "RainProbability", "Temperature", 
    "TeamPerformanceScore", "CleanAirRacePace (s)"
]
X = merged_data[features]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# 12. Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 13. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=34)

# 14. Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=34)
model.fit(X_train, y_train)
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# 15. Results
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Emilia Romagna GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# 16. Podium results
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n🏆 Predicted Podium 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")

# 17. Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(features, model.feature_importances_)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# 18. Clean air pace visualization
plt.figure(figsize=(12, 8))
plt.scatter(merged_data["CleanAirRacePace (s)"], merged_data["PredictedRaceTime (s)"])
for i, driver in enumerate(merged_data["Driver"]):
    plt.annotate(driver, 
                 (merged_data["CleanAirRacePace (s)"].iloc[i], 
                  merged_data["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), 
                 textcoords='offset points')
plt.xlabel("Clean Air Race Pace (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Clean Air Race Pace on Predicted Results")
plt.tight_layout()
plt.show()
