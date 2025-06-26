import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import fastf1
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Page setup
st.set_page_config(
    page_title="F1 Predictor 2025",
    layout="wide",
    page_icon="🏎️"
)
st.title("🏎️ F1 Race Predictor 2025")
st.markdown("Predicting race outcomes using machine learning models")

# Create cache directory
cache_dir = "f极_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Grand Prix selection
gp_options = {
    "Bahrain": "Bahrain",
    "Saudi Arabia": "Saudi Arabia",
    "Australia": "Australia",
    "Japan": "Japan",
    "Miami": "Miami",
    "Emilia Romagna": "Emilia Romagna",
    "Monaco": "Monaco",
    "Spain": "Spain"
}
selected_gp = st.sidebar.selectbox("Select Grand Prix", list(gp_options.keys()))

# Weather customization
st.sidebar.subheader("Weather Settings")
override_weather = st.sidebar.checkbox("Override Default Weather", False)
if override_weather:
    rain_probability = st.sidebar.slider("Rain Probability", 0.0, 1.0, 0.1)
    temperature = st.sidebar.slider("Temperature (°C)", 0, 40, 20)
else:
    rain_probability = None
    temperature = None

show_details = st.sidebar.checkbox("Show Model Details", True)

def get_gp_specific_data(gp_name):
    """Returns GP-specific qualifying data and weather"""
    if gp_name == "Bahrain":
        qualifying = pd.DataFrame({
            "Driver": ["VER", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR", "NOR"],
            "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 92.283, 87.489]
        })
    elif gp_name == "Saudi Arabia":
        qualifying = pd.DataFrame({
            "Driver": ["VER", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR", "NOR"],
            "QualifyingTime (s)": [87.294, 87.304, 87.670, 87.407, 88.201, 88.367, 88.303, 88.204, 88.164, 88.782, 89.092, 88.645, 87.489]
        })
    elif gp_name == "Australia":
        qualifying = pd.DataFrame({
            "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO", "HAM", "STR", "GAS", "ALO", "HUL"],
            "QualifyingTime (s)": [70.669, 69.954, 70.129, None, 71.362, 71.213, 70.063, 70.942, 70.382, 72.563, 71.994, 70.924, 71.596]
        })
    elif gp_name == "Japan":
        qualifying = pd.DataFrame({
            "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
            "QualifyingTime (s)": [86.983, 86.995, 87.027, 87.299, 87.318, 87.610, 87.822, 87.897, 88.000, 87.836, 88.570, 88.696, 89.271]
        })
    elif gp_name == "Miami":
        qualifying = pd.DataFrame({
            "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO", "TSU", "HAM", "STR", "GAS", "ALO", "HUL"],
            "QualifyingTime (s)": [86.204, 86.269, 86.375, 86.385, 86.569, 86.682, 86.754, 86.824, 86.943, 87.006, 87.830, 87.710, 87.604, 87.473],
            # FIXED: Added LastYearWinner feature
            "LastYearWinner": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 for VER, 0 for others
        })
    elif gp_name == "Emilia Romagna":
        qualifying = pd.DataFrame({
            "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO", "HAM", "STR", "GAS", "ALO", "HUL"],
            "QualifyingTime (s)": [74.704, 74.962, 74.670, 74.807, 75.432, 75.473, 75.604, 76.613, 75.765, 75.581, 75.787, 75.431, 76.518]
        })
    elif gp_name == "Monaco":
        qualifying = pd.DataFrame({
            "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO", "HAM", "STR", "GAS", "ALO", "HUL"],
            "QualifyingTime (s)": [70.669, 69.954, 70.129, 70.063, 71.362, 71.213, 70.942, 70.382, 72.563, 71.994, 70.924, 71.596, 72.000],
            "OvertakingDifficulty": [9.8] * 13  # Monaco-specific feature
        })
    elif gp_name == "Spain":
        qualifying = pd.DataFrame({
            "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO", "HAM", "STR", "GAS", "ALO", "HUL"],
            "QualifyingTime (s)": [72.123, 72.456, 72.789, 73.012, 73.456, 73.789, 74.012, 74.456, 74.789, 75.012, 75.456, 75.789, 76.012]
        })
    else:
        qualifying = pd.DataFrame()

    # Default weather data
    weather_data = {
        "rain_probability": 0.1,
        "temperature": 25
    }
    return qualifying, weather_data

def get_features_for_gp(gp_name):
    """Returns GP-specific feature set"""
    base_features = [
        "QualifyingTime (s)", "RainProbability", "Temperature",
        "TeamPerformanceScore", "TotalSectorTime (s)"
    ]
    if gp_name == "Monaco":
        base_features.append("OvertakingDifficulty")
    elif gp_name == "Miami":
        base_features.append("LastYearWinner")
    return base_features

def predict_gp(gp_name, rain_prob_override=None, temp_override=None):
    """Predict race times for the selected GP with weather override"""
    try:
        session = fastf1.get_session(2024, gp_name, "R")
        session.load()
        laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
        laps.dropna(inplace=True)
    except Exception as e:
        st.error(f"Error loading session data: {str(e)}")
        return None
    
    # Convert times to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()
    
    # Aggregate sector times
    sector_times = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    sector_times["TotalSectorTime (s)"] = (
        sector_times["Sector1Time (s)"] +
        sector_times["Sector2Time (s)"] +
        sector_times["Sector3Time (s)"]
    )

    # Get GP-specific data
    qualifying, weather_data = get_gp_specific_data(gp_name)
    if qualifying.empty:
        st.error(f"No qualifying data available for {gp_name} GP")
        return None

    # Apply weather override if provided
    if rain_prob_override is not None:
        weather_data["rain_probability"] = rain_prob_override
    if temp_override is not None:
        weather_data["temperature"] = temp_override

    # Team performance setup
    team_points = {
        "Red Bull": 100, "McLaren": 90, "Ferrari": 80, "Mercedes": 70,
        "Aston Martin": 60, "Alpine": 50, "Williams": 40, "Kick Sauber": 30,
        "Racing Bulls": 20, "Haas": 10
    }
    driver_to_team = {
        "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", 
        "RUS": "Mercedes", "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", 
        "TSU": "Racing Bulls", "SAI": "Ferrari", "HUL": "Kick Sauber", 
        "OCO": "Alpine", "STR": "Aston Martin", "ALB": "Williams"
    }
    
    # Add team data
    qualifying["Team"] = qualifying["Driver"].map(driver_to_team)
    qualifying["TeamPerformanceScore"] = qualifying["Team"].map(team_points) / max(team_points.values())
    
    # Add weather data
    qualifying["RainProbability"] = weather_data["rain_probability"]
    qualifying["Temperature"] = weather_data["temperature"]
    
    # Merge with sector times
    merged_data = qualifying.merge(sector_times, on="Driver", how="left")
    
    # Monaco-specific features
    if gp_name == "Monaco":
        position_change = {
            "VER": -1.2, "NOR": 0.5, "PIA": 0.7, "LEC": -1.5, "RUS": 0.3,
            "SAI": -0.8, "ALB": 1.2, "OCO": 0.4, "HAM": 0.6, "STR": 1.8,
            "GAS": 0.9, "ALO": -0.5, "HUL": 1.0
        }
        merged_data["PositionChange"] = merged_data["Driver"].map(position_change)
        merged_data["QualifyingTime"] = merged_data["QualifyingTime (s)"].fillna(0) * merged_data["PositionChange"]

    # Prepare features and target
    features = get_features_for_gp(gp_name)
    X = merged_data[features].fillna(0)
    mean_lap_times = laps.groupby("Driver")["LapTime (s)"].mean()
    y = merged_data["Driver"].map(mean_lap_times)

    # Handle missing values
    valid_mask = y.notna()
    if not valid_mask.any():
        st.error(f"No valid drivers found for {gp_name} GP")
        return None
        
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    # Impute missing features
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_clean)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_clean, test_size=0.2, random_state=39)

    # Train model
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=39)
    model.fit(X_train, y_train)

    # Predict
    merged_data.loc[valid_mask, "PredictedRaceTime (s)"] = model.predict(X_imputed)
    final_results = merged_data[valid_mask].sort_values("PredictedRaceTime (s)")
    podium = final_results.head(3)[["Driver", "PredictedRaceTime (s)"]]

    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Feature importance plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(features, model.feature_importances_)
    ax1.set_xlabel("Importance")
    ax1.set_title("Feature Importance")
    plt.tight_layout()

    # Team performance visualization
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    scatter = ax2.scatter(
        final_results["TeamPerformanceScore"],
        final_results["PredictedRaceTime (s)"],
        c=final_results["QualifyingTime (s)"],
        cmap="viridis",
        alpha=0.7
    )
    plt.colorbar(scatter, label="Qualifying Time (s)")
    for i, driver in enumerate(final_results["Driver"]):
        ax2.annotate(driver, 
                     (final_results["TeamPerformanceScore"].iloc[i],
                      final_results["PredictedRaceTime (s)"].iloc[i]),
                     xytext=(5, 5), 
                     textcoords='offset points')
    ax2.set_xlabel("Team Performance Score")
    ax2.set_ylabel("Predicted Race Time (s)")
    ax2.set_title(f"Team Performance vs Predicted Results ({gp_name} GP 2025)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    return {
        'podium': podium,
        'full_results': final_results,
        'mae': mae,
        'feature_importance_fig': fig1,
        'team_performance_fig': fig2
    }

# Prediction section
if st.sidebar.button("Run Prediction"):
    with st.spinner(f"Predicting {selected_gp} GP..."):
        try:
            # Pass weather override if enabled
            rain_override = rain_probability if override_weather else None
            temp_override = temperature if override_weather else None
            
            results = predict_gp(
                gp_options[selected_gp],
                rain_prob_override=rain_override,
                temp_override=temp_override
            )
            
            if results is None:
                st.error("Prediction failed - no valid data available")
            else:
                # Display results
                st.header(f"🏁 {selected_gp} GP 2025 Prediction")
                
                # Podium results
                st.subheader("🏆 Podium Prediction")
                cols = st.columns(3)
                podium = results['podium']
                for i in range(min(3, len(podium))):
                    with cols[i]:
                        driver = podium.iloc[i]['Driver']
                        time = podium.iloc[i]['PredictedRaceTime (s)']
                        st.metric(
                            label=f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} P{i+1}",
                            value=driver,
                            delta=f"{time:.3f}s"
                        )
                
                # Full results table
                st.subheader("Full Prediction Results")
                st.dataframe(
                    results['full_results'][['Driver', 'PredictedRaceTime (s)']].reset_index(drop=True),
                    use_container_width=True,
                    height=400
                )
                
                # Model metrics
                st.subheader("Model Performance")
                st.metric("Mean Absolute Error", f"{results['mae']:.3f} seconds")
                
                # Visualizations
                if show_details:
                    st.subheader("Model Insights")
                    with st.expander("Feature Importance"):
                        st.pyplot(results['feature_importance_fig'])
                    with st.expander("Team Performance Analysis"):
                        st.pyplot(results['team_performance_fig'])
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)
else:
    st.info("Select a Grand Prix and click 'Run Prediction' to see forecasts")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with FastF1, Scikit-learn & Streamlit")
