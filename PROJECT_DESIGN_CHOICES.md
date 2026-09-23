# 📄 Project Design Choices, Component Justifications & Trade-offs

This document details every dataset, parameter, machine learning model, physics baseline, software component, and web technology used in **Hydra: AI-Based Reservoir Evaporation & Water Level Prediction System**. It includes a detailed comparison against alternatives and the technical rationale for each choice.

---

## 1. Dataset Selection Rationale

### **Chosen Dataset**: `Open-Meteo ERA5 Reanalysis & Archive REST API`
- **Primary Validation Location**: Khadakwasla Dam, Pune (`18.4419° N, 73.7628° E`).
- **Time Horizon**: Daily meteorological records from January 1, 2021 to December 31, 2024 (1,461 total records).
- **Parameters Ingested**: Maximum/Minimum/Mean Air Temperature (°C), Relative Humidity (%), Wind Speed (10m and 2m in m/s), Shortwave Solar Radiation ($\text{MJ/m}^2/\text{day}$), Surface Pressure ($\text{hPa}$), Precipitation ($\text{mm}$), Sunshine Duration ($\text{hours}$).

### **Alternatives Considered & Comparison Matrix**

| Dataset / Source | Pros | Cons / Limitations | Reasons for Choosing Open-Meteo ERA5 |
|---|---|---|---|
| **Open-Meteo API (ERA5 Reanalysis)** *(CHOSEN)* | Global spatial coverage (any lat/lon), 100% complete hourly & daily records, zero cost, no API keys required, REST endpoint for live 7-day forecasting. | Relies on climate reanalysis grid rather than physical micro-local sensors. | **SELECTED**: Enables the system to be 100% **location-agnostic**. Anyone worldwide can run predictions for any dam without manual data entry. |
| **Manual IMD Weather Station Logs** | Physical ground-truth readings from local weather stations. | Prone to human observation errors, missing days, bureaucratic delays, non-standardized formats, restricted to specific Indian cities. | Rejected as primary source because it breaks project goal of being automated and universal across any global region. |
| **NASA POWER API** | Free global satellite solar & meteorological dataset. | Slower API response time ($>1.5\text{s}$ latency), coarser spatial resolution ($0.5^\circ \times 0.5^\circ \approx 50\text{km}$). | Open-Meteo provides faster API responses ($<300\text{ms}$) with higher spatial grid resolution ($0.1^\circ \approx 11\text{km}$). |
| **MODIS Satellite Remote Sensing** | Measures lake surface temperature & water extent directly. | Cloud cover interference during monsoons (missing pixels), 8-day composite lag (no daily resolution). | Unsuited for daily real-time predictive forecasting. |

---

## 2. Parameter & Feature Selection Rationale

### **Parameters Included & Physical Justifications**

1. **Air Temperature ($T_{max}, T_{min}, T_{mean}$)**:
   - *Why*: Dictates the latent heat of vaporization ($\lambda \approx 2.45 \text{ MJ/kg}$) and saturation vapor pressure $e_s(T)$. Higher temperatures increase molecular kinetic energy, accelerating evaporation.
2. **Relative Humidity ($RH$)**:
   - *Why*: Controls atmospheric moisture saturation. Low humidity creates a steep Vapor Pressure Deficit ($VPD = e_s - e_a$), driving evaporation.
3. **Wind Speed ($u_2$ at 2m height)**:
   - *Why*: Removes saturated boundary air immediately above the water surface. Converted from 10m wind speed ($u_{10}$) via logarithmic wind profile: $u_2 = u_{10} \times \frac{4.87}{\ln(67.8 \times 10 - 5.42)} \approx 0.748 \cdot u_{10}$.
4. **Shortwave Solar Radiation ($R_s$) & Net Radiation ($R_n$)**:
   - *Why*: Provides the primary energy required to break hydrogen bonds and convert liquid water to gas.
5. **Vapor Pressure Deficit ($VPD$)** *(Engineered Physics Feature)*:
   - *Formula*: $VPD = e_s - e_a = 0.6108 \cdot \exp\left(\frac{17.27 T}{T + 237.3}\right) \cdot \left(1 - \frac{RH}{100}\right)$
   - *Why*: Provides ML models with a direct physical metric of atmospheric drying potential.
6. **Reservoir Surface Area ($A$)**:
   - *Why*: Translates daily evaporation depth ($\text{mm}$) into volumetric water loss ($\text{Liters}$ / $\text{MCM}$) using $V = \text{Depth} \times A$.

---

## 3. Physics Baseline Model Rationale

### **Chosen Physics Model**: `FAO-56 Penman-Monteith Equation`
$$E_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T + 273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}$$

### **Alternatives & Rationale**

| Physics Equation | Inputs Required | Strengths / Weaknesses | Why FAO-56 Penman-Monteith Was Chosen |
|---|---|---|---|
| **FAO-56 Penman-Monteith** *(CHOSEN)* | Temp, RH, Wind Speed, Solar Rad, Pressure. | WMO global gold standard. Combines thermodynamic energy balance AND aerodynamic mass transfer. | **SELECTED**: Highest physical precision ($R^2 = 0.9983$). Accounts for both thermal radiation and wind convective forces. |
| **Hargreaves-Samani** | Only $T_{max}, T_{min}, R_a$. | Simple, minimal data requirements. | Ignores wind speed and relative humidity; underpredicts in windy/arid conditions. |
| **Thornthwaite** | Monthly mean temperature only. | Simple monthly estimates. | Inaccurate for daily timesteps; fails to capture weather spikes. |
| **Priestley-Taylor** | Radiation & Temperature. | Good when wind data is missing. | Neglects aerodynamic drying power of wind across open reservoir surfaces. |

---

## 4. Machine Learning Model Selection Rationale

We trained and benchmarked 7 model families to evaluate accuracy and generalization:

### **Model Performance & Trade-off Matrix**

| Model Name | $R^2$ Score | RMSE (mm/day) | MAE (mm/day) | Strengths | Trade-offs / Limitations |
|---|---|---|---|---|---|
| **FAO-56 Penman Monteith (Physics Baseline)** | **0.9983** | **0.1460** | **0.1159** | Pure thermodynamic foundation; no training needed. | Requires exact meteorological inputs; rigid formula. |
| **Multi-Layer Perceptron (ANN)** | **0.9959** | **0.2235** | **0.1769** | Captures non-linear feature interactions; highly flexible. | Requires scaling; longer training time. |
| **Support Vector Regressor (SVR)** | **0.9957** | **0.2312** | **0.1637** | Excellent generalization on medium tabular datasets; resilient to outliers. | Sensitive to hyperparameter choice ($C, \epsilon$). |
| **Gradient Boosting Regressor (GBDT)** | **0.9923** | **0.3072** | **0.2088** | High accuracy; provides Gini feature importance. | Slower training than linear models. |
| **Extra Trees Regressor** | **0.9917** | **0.3202** | **0.2066** | Random split selection reduces variance. | Slightly higher bias than standard Random Forest. |
| **Random Forest Regressor** | **0.9887** | **0.3734** | **0.2465** | Highly stable ensemble; resistant to overfitting. | Larger model memory size on disk. |
| **Ridge Regression** | **0.9873** | **0.3960** | **0.2957** | Extremely fast ($<1\text{ms}$); lightweight. | Cannot model non-linear interactions without feature expansion. |

### **Why Deep LSTMs / Transformers Were Excluded as Primary Models**:
- **Overfitting Risk**: Daily weather datasets (1,000–5,000 rows) are too small for multi-million parameter deep sequence networks (Transformers/Deep LSTMs), leading to memorization rather than generalization.
- **Speed & Simplicity**: GBDT, SVR, and MLP deliver $R^2 > 0.995$ with instant inference ($<5\text{ms}$) without requiring heavy PyTorch/TensorFlow GPU runtimes.

---

## 5. Software Stack & Web Dashboard Rationale

### **Chosen Web Stack**: `Python 3.13` + `Streamlit` + `Plotly`

| Web Technology | Pros | Cons / Limitations | Why Streamlit + Plotly Was Chosen |
|---|---|---|---|
| **Streamlit + Plotly** *(CHOSEN)* | 100% Python-native, instant integration with `scikit-learn` & `pandas`, interactive zooming/hovering charts, dynamic sidebar controls. | Full page re-runs on state change (mitigated via `@st.cache_data` and `@st.cache_resource`). | **SELECTED**: Ideal for interactive ML dashboards, rapid prototyping, and scenario simulation. |
| **React / Next.js + FastAPI** | Production web standard, custom UI flexibility. | High setup complexity, separate REST backend boilerplate, state management overhead. | Over-engineered for a data science project timeline. |
| **Flask + HTML/CSS + Chart.js** | Lightweight Python framework. | Manual HTML/JS chart boilerplate required for interactive filtering. | Less interactive out-of-the-box compared to Streamlit & Plotly. |
