# 🌊 AI-Based Reservoir Evaporation & Water Level Prediction System

An end-to-end physics-informed Machine Learning and Web Application platform for predicting daily water reservoir evaporation loss, forecasting water storage depletion, and analyzing meteorological drivers across any dam location worldwide.

Developed as a **Semester Main Project**, this system defaults to **Khadakwasla Dam (Pune, India)** while maintaining complete location independence to support any global reservoir via coordinates or local hardware pan sensor data.

---

## 📌 Project Highlights & Key Features

1. **Location-Agnostic & Live Weather Integration**:
   - Built-in REST client for **Open-Meteo & NASA POWER APIs**.
   - Presets for **Khadakwasla Dam** (18.4419°N, 73.7628°E), **Koyna Dam**, **Tehri Dam**, or any custom latitude/longitude worldwide.
   - Real-time 7-day weather forecasting and daily volumetric water loss prediction.

2. **Physics-Informed Machine Learning Architecture**:
   - Integrates the **WMO FAO-56 Penman-Monteith Thermodynamic Benchmark Equation** alongside 6 Machine Learning & Neural Network models:
     - **FAO-56 Penman-Monteith Equation** (Physics Baseline)
     - **Multi-Layer Perceptron (ANN)**
     - **Support Vector Regressor (SVR)**
     - **Gradient Boosting Regressor (GBDT)**
     - **Extra Trees Regressor**
     - **Random Forest Regressor**
     - **Ridge Regression**

3. **Key Meteorological Parameters Considered**:
   - **Temperature**: Max, Min, and Mean Daily Temperature (°C)
   - **Relative Humidity**: Mean daily relative humidity (%)
   - **Wind Speed**: Wind speed converted to 2m height ($u_2$ in m/s)
   - **Solar Radiation**: Shortwave solar radiation sum ($\text{MJ/m}^2/\text{day}$) & sunshine hours
   - **Vapor Pressure Deficit (VPD)**: Atmospheric drying power ($\text{kPa}$)
   - **Atmospheric Pressure**: Surface pressure ($\text{hPa}$)
   - **Reservoir Surface Area**: Surface area ($\text{km}^2$) for volume loss calculations ($\text{Liters}$ / $\text{MCM}$)
   - **Hardware Evaporation Pan Readings**: Pan sensor baseline ($E_{pan}$ in mm/day)

4. **Model Performance Evaluation Leaderboard**:
   - Evaluated on test split using standard hydrologic and statistical metrics:
     - **$R^2$ Score** (Up to 0.9983)
     - **RMSE** (Root Mean Squared Error mm/day)
     - **MAE** (Mean Absolute Error mm/day)
     - **NSE** (Nash-Sutcliffe Efficiency)
     - **MAPE** (Mean Absolute Percentage Error %)

5. **Interactive Streamlit Web Dashboard**:
   - **Tab 1: 🌐 Live 7-Day Forecast & Evaporation Predictor**: Live weather API fetch and 7-day predicted evaporation loss depth + volumetric loss in liters.
   - **Tab 2: 📊 Exploratory Data Analysis (EDA)**: Interactive diagnostic heatmaps, scatter plots, and seasonal trends.
   - **Tab 3: 🏆 Multi-Model Leaderboard**: Model evaluation charts and test set predictions comparison.
   - **Tab 4: 💧 Volumetric Loss & Storage Gauge Calculator**: Calculates storage depletion over $N$ days and converts lost water volume to equivalent days of drinking water supply for Pune City (1.4 Billion Liters/day demand).
   - **Tab 5: 🧪 "What-If" Climate Shift Simulator**: Simulates temperature rise (+0.5°C to +5.0°C), wind variations, and humidity shifts.

---

## 🛠️ Project Structure

```
EDI_Sem5/
├── app.py                      # Interactive Streamlit Web Application Dashboard
├── src/
│   ├── data_loader.py          # Data ingestion engine (Open-Meteo REST API & Fallback generator)
│   ├── physics_penman.py       # FAO-56 Penman-Monteith thermodynamic physics model
│   ├── eda_analysis.py         # Diagnostic EDA plot generator & statistical summary
│   └── model_trainer.py        # Multi-model training, evaluation & leaderboard builder
├── data/
│   ├── khadakwasla_evaporation_dataset.csv   # Cleaned 4-year daily meteorological dataset
│   └── eda_plots/              # High-resolution diagnostic figures (correlation, trends)
├── models/
│   ├── model_performance_leaderboard.csv     # Model metric comparison table
│   └── trained_models.pkl      # Serialized ML model artifacts & features
└── README.md                   # Project documentation
```

---

## 🚀 How to Run the Application

### 1. Execute Data Ingestion & Model Training Pipeline
```bash
# Fetch weather data & generate physics benchmark
python src/physics_penman.py

# Run Exploratory Data Analysis (EDA)
python src/eda_analysis.py

# Train & evaluate all 6 ML/DL models
python src/model_trainer.py
```

### 2. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model Leaderboard Summary

| Model | $R^2$ Score | RMSE (mm/day) | MAE (mm/day) | NSE | MAPE (%) |
|---|---|---|---|---|---|
| **FAO-56 Penman Monteith (Physics Baseline)** | **0.9983** | **0.1460** | **0.1159** | **0.9983** | **2.14%** |
| **Multi-Layer Perceptron (ANN)** | **0.9959** | **0.2235** | **0.1769** | **0.9959** | **3.25%** |
| **Support Vector Regressor (SVR)** | **0.9957** | **0.2312** | **0.1637** | **0.9957** | **2.90%** |
| **Gradient Boosting Regressor** | **0.9923** | **0.3072** | **0.2088** | **0.9923** | **3.27%** |
| **Extra Trees Regressor** | **0.9917** | **0.3202** | **0.2066** | **0.9917** | **3.05%** |
| **Random Forest Regressor** | **0.9887** | **0.3734** | **0.2465** | **0.9887** | **3.63%** |
| **Ridge Regression** | **0.9873** | **0.3960** | **0.2957** | **0.9873** | **4.85%** |
