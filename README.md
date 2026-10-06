# 🚖 Urban Mobility: Predicting Ride Demand
### Zone-Level Short-Term Ride-Demand Forecasting & Fair Driver Positioning System
**Academic Capstone Project | B.Sc. Computer Science with Artificial Intelligence**  
**Domain:** Transportation / Smart City / Spatio-Temporal Forecasting  

---

## 📌 1. Problem Statement
In modern metropolitan smart cities, ride-hailing platforms experience acute spatial and temporal demand imbalances every 15 minutes. During weekday rush hours, torrential rainstorms, flight arrival waves, or stadium concerts, passenger requests surge drastically in specific neighborhoods while drivers remain stranded or idle in low-demand zones. 

This spatial mismatch causes:
- **Excessive passenger waiting times** (15 to 25+ minutes).
- **High ride cancellation rates** and customer dissatisfaction.
- **Unnecessary driver idle time**, wasted fuel, and high greenhouse emissions from cruising empty ("dead-heading").

A reactive dispatch system fails because drivers take 10–20 minutes to navigate cross-city traffic. Fleet managers need a **proactive 15-minute forecasting system** that anticipates demand before it occurs and provides balanced, fair repositioning guidance.

---

## 🎯 2. Project Objective
1. **Forecast Demand:** Predict the exact number of ride requests for the upcoming 15-minute interval across 8 heterogeneous city zones.
2. **Prevent Data Leakage:** Ensure all engineered lag and rolling features represent information available strictly before the target interval.
3. **Compare Models:** Benchmark a simple baseline against tree-based ensemble models (Random Forest and Gradient Boosting Regressors) using MAE, RMSE, and $R^2$.
4. **Driver Positioning System:** Translate raw numerical predictions into actionable, constrained driver allocation recommendations:
   - `Deploy more drivers`
   - `Deploy drivers`
   - `Maintain/increase coverage`
   - `Normal coverage`
   - `Reduce excess drivers`
5. **Enforce Safety & Fairness:** Maintain guaranteed minimum coverage in every zone (preventing neighborhood abandonment) and enforce anti-gridlock caps.
6. **Real-Time Interactive Dashboard:** Deliver an executive Streamlit dashboard with error tracking, anomaly spike detection, and distribution drift alerts.

---

## 📊 3. Dataset Description
The system utilizes a realistic synthetic ride-demand dataset of **42,240 records** covering **8 city zones** over **55 continuous days** at **15-minute intervals** (96 intervals/day).

### City Zones Represented:
1. **Downtown:** Commercial heart, high dining and nightlife peaks, massive public event sensitivity.
2. **Airport:** Continuous high baseline, morning departure banks (05:00–08:00) & late-night arrival surges (21:00–24:00).
3. **Railway Station:** Periodic arrival waves aligned with regional express trains.
4. **IT Park:** Strict bi-modal weekday commuter peaks (08:30–10:30 and 17:30–20:00); quiet on weekends.
5. **Residential Area:** Morning outbound commute waves and evening inbound returns.
6. **Shopping District:** Weekend afternoon and evening retail peaks (13:00–22:00).
7. **University Area:** Active daytime campus movements (10:00–16:00); quiet late at night.
8. **Industrial Area:** Shift change surges (06:00, 14:00, 22:00).

### Key Attributes:
| Attribute | Data Type | Description |
| :--- | :--- | :--- |
| `timestamp` | Datetime | Exact 15-minute interval start (e.g. `2026-09-15 08:30:00`) |
| `zone` | Categorical | One of the 8 urban zones |
| `historical_demand` | Float | Zone-specific diurnal expected demand |
| `day_of_week` | Integer | Day index ($0 = \text{Monday}, \dots, 6 = \text{Sunday}$) |
| `day_type` | Categorical | `Weekday` vs `Weekend` |
| `hour` / `minute` | Integer | Hour ($0–23$) and minute ($0, 15, 30, 45$) |
| `rush_hour` | Binary | Weekday morning (8–10 AM) & evening (17–20 PM) commute peak |
| `weekend` | Binary | 1 for Saturday/Sunday, 0 for weekdays |
| `event_indicator` | Binary | 1 if a stadium concert/game/expo is active in the zone |
| `weather` | Categorical | `Clear`, `Cloudy`, `Rainy`, `Stormy` |
| `temperature` | Float | Ambient temperature in Celsius (realistic diurnal curve) |
| `rainfall` | Float | Precipitation in millimeters |
| `recent_cancellation_rate`| Float | Rolling passenger cancellation fraction ($0.01–0.35$) |
| `previous_15min_demand` | Integer | **Lag-1:** Target ride requests from interval $t-1$ |
| `previous_30min_demand` | Integer | **Lag-2:** Target ride requests from interval $t-2$ |
| `previous_1hr_demand` | Integer | **Lag-4:** Target ride requests from interval $t-4$ |
| `ride_requests` | Integer | **Target Variable ($y$):** Actual passenger ride requests |

---

## 🛠️ 4. Feature Engineering & Strict Leakage Prevention
To guarantee zero data leakage:
1. **Temporal Lags:** Computed using `df.groupby('zone')['ride_requests'].shift(k)`.
2. **Rolling Moving Averages:** 1-hour rolling mean (`rolling_mean_4`) and volatility (`rolling_std_4`) are calculated strictly over past intervals (shift $\ge 1$).
3. **Cyclical Transformations:** Trigonometric continuous variables $\sin(\frac{2\pi \cdot t}{24})$ and $\cos(\frac{2\pi \cdot t}{24})$ eliminate midnight boundary discontinuities.
4. **Interaction Terms:** `event_in_downtown = event_indicator * is_downtown`, `rain_x_rush_hour = is_rainy * rush_hour`.
5. **One-Hot Encoding:** Scikit-Learn `OneHotEncoder(handle_unknown='ignore')` safely encodes categorical zones and weather conditions.
6. **Chronological Splitting:** Enforces an 80/20 train/test split along the timeline (**Aug 10 to Sep 22 for training, Sep 23 to Oct 03 for testing**). Random shuffling is strictly prohibited.

---

## 🤖 5. Machine Learning Models & Real Evaluation Results

The models were trained on **33,792 historical training rows** and evaluated on **8,448 unseen chronological test rows**.

### Performance Benchmark Table:

| Model | MAE (Rides) | RMSE (Rides) | $R^2$ Score | Overprediction % | Underprediction % | Mean Surplus | Mean Deficit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline (Lag-1)** | 4.132 | 5.821 | 0.9096 | 45.6% | 45.7% | 4.53 | 4.52 |
| **Historical Mean Baseline** | 2.973 | 6.449 | 0.8891 | 47.7% | 49.0% | 2.65 | 3.49 |
| **Random Forest Regressor** | 2.781 | 3.920 | 0.9590 | 50.5% | 49.5% | 2.81 | 2.75 |
| **Gradient Boosting Regressor** | **2.685** | **3.564** | **0.9661** | **51.0%** | **49.0%** | **2.67** | **2.70** |

### Key Takeaways:
- **Gradient Boosting Regressor** achieved the best performance with an **RMSE of 3.56 rides**, an **MAE of 2.68 rides**, and an **$R^2$ of 96.6%**.
- **Error Reduction:** A **38.8% reduction in RMSE** and **35.0% reduction in MAE** compared to the naive persistence baseline.
- **Balanced Residuals:** The model exhibits virtually zero structural bias, evenly balancing overprediction (51.0%) and underprediction (49.0%).

---

## 📈 6. Generated Visualizations
The system automatically generates and saves 9 presentation-grade visualizations in `reports/figures/`:
1. `1_actual_vs_predicted.png`: Continuous 48-hour trajectory tracking actual vs predicted demand in Downtown.
2. `2_demand_by_hour.png`: Diurnal hourly distribution matching commuter and nightlife patterns.
3. `3_demand_by_zone.png`: Zone-wise mean demand ranking across all 8 zones.
4. `4_demand_trend_over_time.png`: Daily average demand progression across the test window.
5. `5_demand_spikes.png`: Identification of statistical demand spikes ($>95^{\text{th}}$ percentile) and event surges.
6. `6_prediction_error.png`: Residual histogram centered tightly at 0 with Gaussian bell distribution.
7. `7_model_comparison.png`: Side-by-side bar comparison of MAE, RMSE, and $R^2$.
8. `8_feature_importance.png`: Top 15 influential features (lag-1 demand, historical baseline, and rush hour).
9. `9_zone_level_performance.png`: Zone-by-zone breakdown of MAE and RMSE.

---

## 🛡️ 7. Fair & Safe Driver Positioning System
Rather than forcing all drivers into the highest surge hotspot, the dispatch system employs a **Constrained Optimization Strategy**:

### Core Constraints:
1. **Minimum Service Guarantee:** Every zone receives a baseline minimum of **$\ge 6$ drivers**, ensuring peripheral residential and university areas are not abandoned.
2. **Anti-Gridlock Cap:** No zone may consume more than **35% of the total active fleet**, preventing localized traffic jams.
3. **Proportional Marginal Allocation:** Remaining drivers are distributed based on unmet marginal demand $\max(0, \hat{y}_z - 6)$.
4. **Advisory Decision Support:** Recommendations serve as heatmaps, repositioning bonuses, and surge incentives—preserving driver labor autonomy without coercive automated commands.

### Operational Dispatch Matrix:
| Net Deficit / Surplus ($\Delta$) | Action Recommendation | Operational Meaning |
| :---: | :--- | :--- |
| $\Delta \ge +8$ | **Deploy more drivers** | Critical shortage; high wait times imminent |
| $+3 \le \Delta < +8$ | **Deploy drivers** | Moderate deficit; reposition idle drivers |
| $-3 \le \Delta < +3$ | **Maintain/increase coverage** | Supply and demand in equilibrium |
| $-8 \le \Delta < -3$ | **Normal coverage** | Slight surplus; adequate supply buffer |
| $\Delta < -8$ | **Reduce excess drivers** | Substantial surplus; incentivize rebalancing |

---

## ⚠️ 8. Distribution Shift & Anomaly Detection
- **Scenario:** During a major public concert in Downtown, demand surges from **50 rides to 125 rides**.
- **Detection:** The dashboard runs an automated **Two-Sample Kolmogorov-Smirnov (KS) Test** between recent operational windows and training baselines.
- **Alert:** When $p < 0.05$ or events trigger anomalous residual spikes, the dashboard renders an active warning banner:
  `"⚠️ POTENTIAL DISTRIBUTION SHIFT DETECTED!"`

---

## 💻 9. Real-Time Dashboard Architecture
The Streamlit application (`app.py`) provides 9 interactive sections:
- **Section 1 – Executive Overview:** Total predicted requests, highest-demand zone, avg demand, model accuracy.
- **Section 2 – Demand Forecast:** 15-minute zone forecasts and interactive Plotly time-series trajectories.
- **Section 3 – Zone Analysis & Positioning:** Ranked dispatch table with status badges and fleet distribution donut chart.
- **Section 4 – Error Monitoring:** Live MAE, RMSE, and residual distribution histograms.
- **Section 5 – Demand Spikes:** Anomaly detection highlighting event-driven surges.
- **Section 6 – Model Drift:** Real-time Kolmogorov-Smirnov test and Normal vs Event KDE distribution overlay.
- **Section 7 – Over vs Underprediction:** Asymmetric cost analysis and why RMSE alone is insufficient.
- **Section 8 – Assessment Q&A:** Full interactive technical solutions to Case Study questions Q1–Q8.
- **Section 9 – Presentation Slides:** 14 structured presentation points ready for classroom showcase.

---

## 📂 10. Project Directory Structure
```
urban_mobility_prediction/
│
├── data/
│   └── ride_demand.csv             # 42,240 records across 8 zones (55 days)
│
├── notebooks/
│   └── analysis.ipynb              # Executed Jupyter Notebook with outputs & plots
│
├── models/
│   ├── best_model.pkl              # Saved best Gradient Boosting model
│   ├── rf_model.pkl                # Saved Random Forest model
│   ├── gb_model.pkl                # Saved Gradient Boosting model
│   ├── pipeline.pkl                # Fitted FeaturePipeline (Encoders & transforms)
│   ├── model_metrics.json          # Actual test metrics and feature names
│   └── test_predictions.csv        # Pre-computed predictions for test window
│
├── reports/
│   └── figures/                    # 9 high-resolution visualization charts
│       ├── 1_actual_vs_predicted.png
│       ├── 2_demand_by_hour.png
│       ├── 3_demand_by_zone.png
│       ├── 4_demand_trend_over_time.png
│       ├── 5_demand_spikes.png
│       ├── 6_prediction_error.png
│       ├── 7_model_comparison.png
│       ├── 8_feature_importance.png
│       └── 9_zone_level_performance.png
│
├── src/
│   ├── __init__.py
│   ├── data_generation.py          # Synthetic dataset generator
│   ├── preprocessing.py            # Data loader & chronological splitter
│   ├── feature_engineering.py      # Feature engineering pipeline
│   ├── train_model.py              # ML training, evaluation, & figure generator
│   ├── prediction.py               # Production inference engine
│   └── driver_positioning.py       # Fair & safe driver allocation logic
│
├── app.py                          # Full-featured Streamlit real-time dashboard
├── requirements.txt                # Pinned dependencies
├── README.md                       # Comprehensive project documentation
└── assessment_answers.md           # Complete solutions to Case Study questions Q1–Q8
```

---

## 🚀 11. Quickstart: How to Run the Project Locally

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd "d:\case study"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3 (Optional): Re-Generate Dataset & Retrain Models
```powershell
# 1. Generate synthetic dataset (42,240 records)
python urban_mobility_prediction/src/data_generation.py

# 2. Train baseline, Random Forest, & Gradient Boosting models
python urban_mobility_prediction/src/train_model.py
```

### Step 4: Launch the Streamlit Interactive Dashboard
Run either from the root or inside the project folder:
```powershell
streamlit run app.py
```
*The interactive dashboard will automatically open in your default browser at `http://localhost:8501`.*

### Step 5: Open the Jupyter Notebook
```powershell
jupyter notebook urban_mobility_prediction/notebooks/analysis.ipynb
```

---

## 🎓 12. Academic Presentation Summary (14 Pillars)
1. **Problem Statement:** Ride-demand fluctuations create acute localized driver-passenger imbalances every 15 minutes.
2. **Objective:** Build a short-term 15-minute forecasting model and translate predictions into fair driver positioning recommendations.
3. **Dataset:** 42,240 records across 8 city zones spanning 55 continuous days.
4. **Features:** Timestamp, zone, weather, temperature, rainfall, cancellation rates, event indicators, and prior demand lags.
5. **Feature Engineering:** Five core time features (hour, day_of_week, weekend, rush_hour, cyclical sin/cos), lag demands ($t-1, t-2, t-4$), rolling statistics, and domain interactions.
6. **ML Models:** Persistence Baseline, Historical Mean Baseline, Random Forest Regressor, and Gradient Boosting Regressor.
7. **Model Evaluation:** Gradient Boosting achieved the lowest RMSE (3.56) and MAE (2.68) with $R^2 = 96.6\%$, outperforming the baseline by 38.8%.
8. **Prediction Results:** Accurate capture of rush-hour commutes, adverse weather demand spikes, and weekend entertainment surges.
9. **Driver Positioning Strategy:** Constrained optimization with minimum baseline guarantees (6 drivers/zone) and anti-congestion caps (35% max).
10. **Distribution Shift:** Modeled major public events (Downtown 50 -> 120 rides) and monitored structural divergence with KS testing.
11. **Dashboard:** Streamlit real-time dashboard featuring live dispatch tables, error trends, anomaly spikes, and drift alerts.
12. **Ethical Considerations:** Guaranteed non-discriminatory service across all zones and preserved driver labor autonomy.
13. **Conclusion:** 15-minute predictive positioning creates a win-win: drivers earn higher hourly wages with less idle fuel burn, while passengers enjoy rapid pickup times.
14. **Future Enhancements:** Ingestion of real-time GPS trajectory streams, deep Spatio-Temporal Graph Neural Networks (ST-GNNs), and reinforcement learning for dynamic incentive pricing.
