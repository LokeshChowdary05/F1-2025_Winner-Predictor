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

# Initialize cache
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

def predict_gp(gp_name):
    """Unified prediction function for all GPs"""
    # Common data loading and processing
    if gp_name == "Miami":
        session_2024 = fastf1.get_session(2024, "Miami", "R")
    else:
        session_2024 = fastf1.get_session(2024, gp_name, "R")
    
    session_2024.load()
    laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps_2024.dropna(inplace=True)
    
    # Convert times to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()
    
    # Aggregate sector times
    sector_times_2024 = laps_2024.groupby("Driver").agg({
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean"
    }).reset_index()
    
    # GP-specific data
    qualifying_2025, weather_data = get_gp_specific_data(gp_name)
    
    # Merge datasets
    merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")
    merged_data["RainProbability"] = weather_data["rain_probability"]
    merged_data["Temperature"] = weather_data["temperature"]
    
    # Prepare features/target
    features = get_gp_features(gp_name)
    X = merged_data[features]
    
    # Create target with aligned index
    mean_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean()
    y = merged_data["Driver"].map(mean_lap_times)
    
    # Handle missing values
    valid_mask = y.notna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_clean)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_clean, test_size=0.2, random_state=39)
    
    # Train model
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=39)
    model.fit(X_train, y_train)
    
    # Add predictions
    merged_data.loc[valid_mask, "PredictedRaceTime (s)"] = model.predict(X_imputed)
    
    # Results
    final_results = merged_data[valid_mask].sort_values("PredictedRaceTime (s)")
    podium = final_results.head(3)[["Driver", "PredictedRaceTime (s)"]]
    
    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Feature importance plot
    feature_importance = model.feature_importances_
    fig1 = plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importance)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    
    # Team performance visualization
    fig2 = plt.figure(figsize=(12, 8))
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
    plt.title(f"Team Performance vs Predicted Results ({gp_name} GP 2025)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return {
        'podium': podium,
        'full_results': final_results,
        'mae': mae,
        'feature_importance_fig': fig1,
        'team_performance_fig': fig2
    }

# Helper functions for GP-specific data
def get_gp_specific_data(gp_name):
    """Returns GP-specific qualifying data and weather"""
    # Sample implementation for Bahrain GP
    if gp_name == "Bahrain":
        qualifying_2025 = pd.DataFrame({
            "Driver": ["VER", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR", "NOR"],
            "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 92.283, 87.489]
        })
        # Add other Bahrain-specific data
    # Add similar blocks for other GPs
    
    # Weather data (simplified)
    weather_data = {
        "rain_probability": 0.1,
        "temperature": 25
    }
    return qualifying_2025, weather_data

def get_gp_features(gp_name):
    """Returns GP-specific feature set"""
    # Common features
    base_features = [
        "QualifyingTime", "RainProbability", "Temperature", 
        "TeamPerformanceScore", "TotalSectorTime (s)"
    ]
    
    # GP-specific additions
    if gp_name == "Monaco":
        base_features.append("AveragePositionChange")
    elif gp_name == "Miami":
        base_features.append("LastYearWinner")
    
    return base_features
