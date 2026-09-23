# 📚 Project Prerequisites & Comprehensive Study Guide

This document serves as a complete study guide and reference manual detailing all foundational domain knowledge, mathematical concepts, machine learning algorithms, and software development prerequisites required to understand, build, and explain **Hydra: AI-Based Reservoir Evaporation & Water Level Prediction System**.

---

## 1. Hydrology & Environmental Physics Prerequisites

To understand how reservoir water evaporates and how physical models operate, you must study the following concepts:

### **1.1 The Evaporation Process & Energy Balance**
- **Latent Heat of Vaporization ($\lambda$)**: The amount of energy required to convert $1\text{ kg}$ of liquid water into water vapor without changing its temperature ($\approx 2.45\text{ MJ/kg}$ at $20^\circ\text{C}$).
- **Energy Balance Equation**:
  $$R_n = G + H + \lambda E$$
  Where $R_n$ is Net Solar Radiation, $G$ is heat flux into the water body, $H$ is sensible heat flux to the air, and $\lambda E$ is latent heat flux driving evaporation.

### **1.2 Atmospheric Vapor Pressure & Drying Power**
- **Saturation Vapor Pressure ($e_s$)**: The maximum pressure exerted by water vapor when air is fully saturated at temperature $T$ (°C):
  $$e^0(T) = 0.6108 \cdot \exp\left(\frac{17.27 \cdot T}{T + 237.3}\right) \text{ (kPa)}$$
- **Actual Vapor Pressure ($e_a$)**: The actual partial pressure exerted by water vapor present in the air:
  $$e_a = e_s \cdot \left(\frac{RH}{100}\right)$$
- **Vapor Pressure Deficit ($VPD$)**: The difference between $e_s$ and $e_a$ ($VPD = e_s - e_a$). $VPD$ measures the "drying power" of the atmosphere. Larger $VPD$ leads to faster evaporation.

### **1.3 Aerodynamic & Wind Dynamics**
- **Boundary Layer Removal**: Wind sweeps away saturated air immediately above the reservoir surface, preventing local humidity buildup.
- **Logarithmic Wind Speed Profile**: Weather stations measure wind at $10\text{ m}$ height ($u_{10}$), but evaporation physics requires wind speed at $2\text{ m}$ ($u_2$):
  $$u_2 = u_{10} \cdot \frac{4.87}{\ln(67.8 \cdot 10 - 5.42)} \approx 0.748 \cdot u_{10}$$

### **1.4 Class A Evaporation Pan & Pan Coefficient ($K_p$)**
- **Class A Pan**: A standard galvanized iron pan ($122\text{ cm}$ diameter, $25\text{ cm}$ depth) placed near reservoirs to measure daily water level drop.
- **Pan Coefficient ($K_p$)**: Because pan walls heat up faster than deep reservoir water, pan evaporation exceeds actual reservoir evaporation. $K_p \approx 0.70 - 0.80$ is used as a conversion factor:
  $$E_{reservoir} = K_p \cdot E_{pan}$$

### **1.5 The FAO-56 Penman-Monteith Standard Equation**
Combines radiative thermal energy and aerodynamic mass transfer:
$$E_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T + 273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}$$
- $\Delta$: Slope of saturation vapor pressure curve ($\text{kPa/°C}$).
- $\gamma$: Psychrometric constant ($\approx 0.066\text{ kPa/°C}$ near sea level).
- $R_n$: Net radiation at the surface ($\text{MJ/m}^2/\text{day}$).

---

## 2. Mathematical & Statistical Prerequisites

### **2.1 Evaluation Metrics in Hydrological Regression**
- **Coefficient of Determination ($R^2$)**: Measures the proportion of variance in evaporation explained by the model ($1.0$ is perfect fit):
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- **Root Mean Squared Error (RMSE)**: Penalizes larger prediction errors in $\text{mm/day}$:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- **Mean Absolute Error (MAE)**: Average absolute magnitude of daily errors ($\text{mm/day}$).
- **Nash-Sutcliffe Efficiency (NSE)**: Standard hydrological model assessment metric ($1.0$ = perfect, $0.0$ = baseline mean model, $<0$ = unacceptable):
  $$\text{NSE} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

### **2.2 Data Splitting for Time-Series Datasets**
- **Temporal Splitting vs. Random K-Fold**: In weather and evaporation data, random K-Fold cross-validation causes **data leakage** because consecutive days are correlated. Time-based splitting (e.g. 80% past train, 20% future test) must be used.

---

## 3. Machine Learning Algorithms to Master

### **3.1 Regularized Linear Models (Ridge Regression)**
- Prevents overfitting on correlated weather variables by adding an $L_2$ regularization penalty $\alpha \sum \beta_j^2$ to the loss function.

### **3.2 Support Vector Regressor (SVR)**
- Maps non-linear weather relationships into higher-dimensional feature space using the **Radial Basis Function (RBF) Kernel**:
  $$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
- Robust against noise and small sample sizes.

### **3.3 Decision Tree Ensembles (Random Forest & Extra Trees)**
- **Random Forest**: Constructs an ensemble of decision trees trained on bootstrap samples with random feature subsets. Reduces variance through averaging.
- **Extra Trees (Extremely Randomized Trees)**: Randomly chooses split thresholds for features, making training faster and further reducing model variance.

### **3.4 Gradient Boosting Regressors (GBDT)**
- Builds decision trees sequentially, where each new tree fits the residual errors of the prior ensemble.

### **3.5 Neural Networks & Multi-Layer Perceptron (MLP)**
- Architecture: Input layer $\rightarrow$ Hidden Dense Layer 1 (64 neurons, ReLU) $\rightarrow$ Hidden Dense Layer 2 (32 neurons, ReLU) $\rightarrow$ Output Linear Neuron (Daily Evaporation).
- Trained using the **Adam Optimizer** and Backpropagation.

---

## 4. Software Engineering & Python Tools Prerequisites

| Tool / Library | Topics / Modules to Study | Purpose in Project |
|---|---|---|
| **Python 3.x** | Object-Oriented Programming, Modules, Exception Handling. | Structuring `src/` modular pipeline. |
| **REST APIs & `requests`** | HTTP GET requests, JSON parsing, URL parameters. | Ingesting live & historical weather data from Open-Meteo. |
| **Pandas & NumPy** | DataFrames, Datetime indexing, vectorized arrays, missing value handling. | Data cleaning, feature engineering, and volumetric math. |
| **Scikit-Learn** | Pipelines, `StandardScaler`, Regressors, `r2_score`, `mean_squared_error`. | Training, cross-validating, and evaluating all ML models. |
| **Joblib** | `joblib.dump()`, `joblib.load()`. | Saving trained model artifacts (`.pkl`) for quick dashboard inference. |
| **Plotly** | `plotly.express`, `plotly.graph_objects`, subplots, dual Y-axes. | Rendering interactive charts in web UI. |
| **Streamlit** | `st.sidebar`, `st.tabs`, `st.metric`, `@st.cache_data`, `@st.cache_resource`. | Building the interactive user web application dashboard (`app.py`). |

---

## 5. Suggested Study Order & Roadmap

```
1. Study Atmospheric Thermodynamics (Temp, RH, Wind u2, Solar Radiation, VPD)
                          │
                          ▼
2. Master FAO-56 Penman-Monteith Physics Formula
                          │
                          ▼
3. Practice Data Manipulation & REST APIs in Python (Pandas, Requests)
                          │
                          ▼
4. Learn Regression Algorithms & Hydrologic Metrics (SVR, GBDT, MLP, R², NSE)
                          │
                          ▼
5. Build Interactive Web Interfaces using Streamlit & Plotly
```
