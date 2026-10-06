# Urban Mobility: Predicting Ride Demand
## Academic Assessment Solutions & Technical Explanations (Q1 – Q8)
**Program:** B.Sc. Computer Science with Artificial Intelligence  
**Domain:** Transportation / Smart City / Spatio-Temporal Forecasting  

---

## Question 1: Construct five time-based features for short-term ride-demand prediction.

### Detailed Technical Explanation:
In short-term urban transportation forecasting (15-minute resolution), demand fluctuates based on human routines, work schedules, school hours, and nightlife activities. Time-based features transform raw timestamps into structured mathematical signals that capture these periodic cycles.

### The Five Core Time-Based Features:

#### 1. `hour` (Discrete Integer: $0, 1, 2, \dots, 23$)
- **Purpose:** Captures the primary 24-hour diurnal cycle of urban movement.
- **Physical Dynamics:** Commuter movement drops to minimal baselines between 01:00 and 05:00 AM, ramps up during morning business arrivals (08:00–10:00), maintains moderate midday activity, surges during evening departures (17:00–20:00), and sustains late-night entertainment peaks in entertainment corridors.
- **Implementation:**
  ```python
  df['hour'] = df['timestamp'].dt.hour
  ```

#### 2. `day_of_week` (Discrete Integer: $0 = \text{Monday}, \dots, 6 = \text{Sunday}$)
- **Purpose:** Disentangles business-day traffic from leisure and shopping travel patterns.
- **Physical Dynamics:** IT Parks and industrial zones experience heavy demand Monday through Friday but drop sharply on weekends. In contrast, shopping districts and dining strips surge on Friday evening, Saturday, and Sunday afternoon.
- **Implementation:**
  ```python
  df['day_of_week'] = df['timestamp'].dt.weekday
  ```

#### 3. `weekend` (Binary Indicator: $\{0, 1\}$)
- **Purpose:** Provides a categorical indicator distinguishing workdays from rest days.
- **Mathematical Representation:**
  $$\text{weekend}_t = \begin{cases} 1 & \text{if } \text{day\_of\_week} \in \{5, 6\} \\ 0 & \text{otherwise} \end{cases}$$
- **Significance:** Allows tree-based split algorithms (e.g., Random Forest) to partition working-day commute models from weekend leisure models in a single split.

#### 4. `rush_hour` (Binary Indicator: $\{0, 1\}$)
- **Purpose:** Explicitly marks the acute morning and evening congestion windows where demand spikes non-linearly.
- **Domain Criteria:** On weekdays, peak morning rush runs from 08:00 to 10:45 AM, and peak evening rush runs from 17:00 to 20:45 PM.
- **Implementation:**
  ```python
  is_weekday = df['day_of_week'] < 5
  morning_rush = (df['hour'] >= 8) & (df['hour'] <= 10)
  evening_rush = (df['hour'] >= 17) & (df['hour'] <= 20)
  df['rush_hour'] = (is_weekday & (morning_rush | evening_rush)).astype(int)
  ```

#### 5. Cyclical Time Encodings: `sin_hour` and `cos_hour` (Continuous Floating Point: $[-1.0, +1.0]$)
- **Purpose:** Solves the boundary discontinuity problem inherent in numerical hours.
- **Problem with Raw Integer Hour:** Hour 23 (11:45 PM) and Hour 0 (12:00 AM) are separated by only 15 minutes in reality, but Euclidean algorithms treat them as having a massive distance $|23 - 0| = 23$.
- **Trigonometric Solution:**
  $$\sin\text{\_hour}_t = \sin\left(\frac{2\pi \cdot (hour + minute/60)}{24}\right), \quad \cos\text{\_hour}_t = \cos\left(\frac{2\pi \cdot (hour + minute/60)}{24}\right)$$
- **Effect:** Maps time onto a 2D unit circle where 23:45 and 00:00 are naturally adjacent.

---

## Question 2: Show how previous-interval demand can be represented as a lag feature.

### Mathematical Formulation:
Let $y_{z, t}$ denote the true number of ride requests observed in city zone $z$ during the 15-minute interval ending at timestamp $t$.

The lag-$k$ demand feature is defined as:
$$L_k(z, t) = y_{z, t - k}$$

For short-term 15-minute forecasting, the previous-interval demand is the **Lag-1 feature ($k=1$)**:
$$\text{previous\_15min\_demand}_{z, t} = y_{z, t-1}$$

Similarly:
- **Lag-2 (30 minutes prior):** $\text{previous\_30min\_demand}_{z, t} = y_{z, t-2}$
- **Lag-4 (1 hour prior):** $\text{previous\_1hr\_demand}_{z, t} = y_{z, t-4}$

### Python Implementation Without Data Leakage:
```python
# Ensure the dataset is strictly sorted chronologically per zone
df = df.sort_values(by=["zone", "timestamp"]).reset_index(drop=True)

# Generate lag features strictly within each zone boundary
df["previous_15min_demand"] = df.groupby("zone")["ride_requests"].shift(1)
df["previous_30min_demand"] = df.groupby("zone")["ride_requests"].shift(2)
df["previous_1hr_demand"]   = df.groupby("zone")["ride_requests"].shift(4)

# Handle boundary rows (earliest intervals of the dataset) using historical baseline
df["previous_15min_demand"] = df["previous_15min_demand"].fillna(df["historical_demand"]).astype(int)
```

### Why This Strictly Prevents Data Leakage:
1. **Temporal Causality:** The value $y_{z, t-1}$ represents completed trips from the elapsed interval $[t - 15\text{min}, t]$. When predicting demand for interval $[t, t + 15\text{min}]$, $y_{z, t-1}$ is fully realized historical data.
2. **Zone Partitioning:** By using `groupby("zone")`, we guarantee that lag values never bleed across geographic boundaries (e.g., Downtown's previous demand is never assigned to Airport).
3. **No Random Shuffling:** Random cross-validation (`train_test_split(shuffle=True)`) would place interval $t+1$ in the training set and interval $t$ in the test set, creating catastrophic future leakage. We enforce strict **chronological train-test splitting**.

---

## Question 3: Analyze why two city zones may require different features or models.

### Case Comparison: `IT Park` vs. `Downtown Commercial Core`

| Characteristic | **IT Park Zone** | **Downtown Zone** |
| :--- | :--- | :--- |
| **Primary Land Use** | High-density corporate offices, technology campuses | Entertainment, dining, retail, corporate, residential |
| **Weekly Demand Pattern** | Strict Monday–Friday concentration; dead on weekends | High 7 days a week; major Friday/Saturday night surges |
| **Diurnal Curve** | **Bi-Modal Commuter Spikes:**<br>• Inbound: 08:15 – 10:15 AM<br>• Outbound: 17:30 – 20:15 PM | **Multi-Modal Extended Plateau:**<br>• Lunch peak (12:30 – 14:00)<br>• Evening & Nightlife (18:30 – 01:30) |
| **Sensitivity to Weather** | Low elasticity (workers must commute regardless of light rain) | High elasticity (leisure pedestrians immediately book cabs in rain) |
| **Sensitivity to Public Events**| Minimal (rarely hosts stadium sports or music festivals) | Massive (concerts, parades, festivals trigger 100%+ surges) |

### Why Single Uniform Features or Global Models May Fall Short:
1. **Opposing Feature Correlations:**
   - In `IT Park`, the feature `is_weekend = 1` correlates with a **-75% drop** in demand.
   - In `Downtown`, `is_weekend = 1` correlates with a **+40% increase** in evening demand.
   - A linear model without zone interaction features cannot reconcile these opposite coefficients.
2. **Zone-Specific Exogenous Drivers:**
   - The `Airport` zone is driven by flight radar arrival schedules, baggage carousel delays, and airline hubs.
   - The `Railway Station` zone is governed by long-distance express train timetables.
3. **Architectural Solutions:**
   - **Enriched Global Tree Model (Our Approach):** Using One-Hot Encoded zone indicators crossed with interaction features (`event * is_downtown`, `rain * rush_hour`) allowing non-linear tree branches to learn distinct rules per zone.
   - **Hierarchical / Multi-Task Learning:** Training individual zone-specialized sub-models or fine-tuning local model heads on top of shared spatio-temporal representations.

---

## Question 4: Analyze how a large public event can create distribution shift.

### Concept of Distribution Shift:
Supervised machine learning algorithms operate under the **Stationarity Assumption**: that the joint probability distribution of features and targets during test time is identical to the training distribution:
$$P_{\text{train}}(X, Y) = P_{\text{test}}(X, Y)$$

A major public event (such as a 60,000-seat stadium concert, football final, or convention) creates a severe **Covariate and Concept Shift**:
$$P_{\text{event}}(Y \mid X) \neq P_{\text{normal}}(Y \mid X)$$

### Quantitative Demonstration:

```
Normal Friday 21:00 at Downtown:
  - Hour = 21, Rain = 0 mm, Lag-1 Demand = 48
  - Model Expected Demand = ~50 rides
  - Available Stationed Drivers = 55 drivers  --> Equilibrium (Wait time: 3 mins)

Event Friday 21:00 at Downtown (Concert Let-Out):
  - Hour = 21, Rain = 0 mm, Lag-1 Demand = 48 (Event just ended, lag hasn't caught up)
  - Actual Realized Demand = 125 rides (+150% spike!)
  - Model Prediction (Without Event Feature) = 52 rides
  - Real Shortage = 125 - 52 = 73 unserved passengers!
  - Operational Outcome: Surge multiplier jumps to 3.8x, wait times hit 25 mins, cancellations spike to 30%.
```

### Analytical Breakdown of the Shift:
1. **Lag Latency (The Blind Spot):** Because the concert finishes abruptly within 10 minutes, `previous_15min_demand` reflects the calm interval *before* the crowd exited. The lag feature alone lags behind reality.
2. **Feature-Target Relationship Breakdown:** On normal days, an hour feature of 21:00 corresponds to gradual tapering. On event days, the exact same timestamp represents peak outflow.
3. **Mitigation:**
   - Ingestion of explicit external metadata: `event_indicator = 1`, venue capacity, scheduled end-time.
   - Non-linear interaction features: `event_in_downtown = event_indicator * is_downtown`.
   - Continuous drift monitoring using two-sample statistical tests (e.g., Kolmogorov-Smirnov test or Population Stability Index - PSI) to detect anomalies in real-time.

---

## Question 5: Evaluate whether minimizing RMSE is enough for driver positioning.

### Evaluation: **No, minimizing RMSE alone is fundamentally insufficient for operational driver positioning.**

### 1. Mathematical Vulnerability: Quadratic Outlier Penalty
Root Mean Squared Error is defined as:
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

Because errors are squared:
- A single error of $10$ rides contributes $10^2 = 100$ to the sum of squares.
- Ten errors of $1$ ride each contribute $10 \times 1^2 = 10$ to the sum of squares.
- **The Pitfall:** An algorithm optimized purely for RMSE will obsess over avoiding a rare 15-ride prediction error during a freak storm, even if doing so degrades everyday precision by 2–3 rides across all 8 zones. But from a fleet operations standpoint, a consistent 3-ride error across the city drains millions in daily idle driver fuel.

### 2. Economic Asymmetry (RMSE is Directionally Blind):
RMSE treats overprediction and underprediction as identical because $(-5)^2 = (+5)^2 = 25$.
In ride-hailing economics, however, the real-world operational costs of these two errors are drastically asymmetric:

$$\text{Operational Cost}(\hat{y} > y) \neq \text{Operational Cost}(\hat{y} < y)$$

- **Overpredicting by 10 rides:** 10 drivers idle for 15 minutes. Cost = driver frustration and idle fuel burn.
- **Underpredicting by 10 rides:** 10 passengers stranded in the rain. Wait times escalate, passengers abandon the app for competitors, and future lifetime customer value (LTV) is destroyed.

### 3. Metric Portfolio Required for Dispatch Optimization:
To support real-world driver dispatching, operations teams must monitor a multi-dimensional metric scorecard:
1. **MAE (Mean Absolute Error):** Measures expected physical driver deficit/surplus per zone.
2. **RMSE:** Tracks stability and penalizes catastrophic forecasting breakdowns.
3. **Directional Error Ratio (Underprediction % vs Overprediction %):** Verifies that the model maintains a controlled, intentional bias.
4. **Service Level Agreement (SLA) Pickup Compliance:** Percentage of intervals where available drivers meet at least 90% of actual realized ride requests.

---

## Question 6: Evaluate the consequences of overpredicting versus underpredicting demand.

A machine learning system operating in production does not produce predictions in a vacuum; its outputs directly drive the physical repositioning of human drivers across city streets.

### Comparative Consequences Matrix:

| Operational Dimension | **Overprediction ($\hat{y} > y$)** | **Underprediction ($\hat{y} < y$)** |
| :--- | :--- | :--- |
| **Dispatched Supply** | Excess vehicles deployed to zone | Insufficient vehicles stationed in zone |
| **Driver Impact** | • **High Idle Time:** Drivers sit parked waiting for non-existent trip pings.<br>• **Wasted Fuel & Dead-Heading:** Uncompensated repositioning miles.<br>• **Reduced Earnings/Hour:** Leads to driver complaints and app disengagement. | • **High Stress:** Drivers bombarded with back-to-back pickup pings.<br>• **Congested Pickups:** High street-side passenger crowding.<br>• **Lost Revenue:** Drivers miss peak earning surge potential. |
| **Passenger Impact** | • **Ultra-Fast Pickups:** Vehicle arrives in 1–3 minutes.<br>• **Standard Fares:** Baseline pricing without surge multipliers. | • **Long Wait Times:** ETA increases to 15–25 minutes.<br>• **Surge Price Shocks:** Dynamic pricing surges 2.0x–3.5x.<br>• **Driver Cancellations:** Drivers cancel distant pickups. |
| **Platform Business Impact**| • **Subsidies/Guarantees Paid:** Platform pays hourly guarantees to idle drivers.<br>• **Sub-optimal Fleet Efficiency:** Other zones are starved of drivers. | • **Ride Cancellations:** Frustrated riders switch to competing apps.<br>• **Customer Churn:** Long-term brand erosion during critical moments.<br>• **Unrealized Commissions:** Direct revenue lost from unfulfilled trips. |

### Strategic Asymmetric Loss Design:
Because customer acquisition costs are high and rider churn is permanent, mobility platforms deliberately tune their dispatching loss functions to prefer **slight overprediction (a small safety buffer)** over underprediction during high-value periods (rush hour, bad weather, airport connections).

---

## Question 7: Create a driver-positioning strategy that uses predictions while avoiding unsafe or unfair allocation.

### The Problem of Naive Optimization:
A naive dispatch algorithm simply calculates:
$$\text{Target Drivers}_z = \text{Predicted Demand}_z$$
and commands all idle drivers to rush to whichever zone has the highest forecasted surge.

**Why Naive Allocation Fails in the Real World:**
1. **Spatial Abandonment:** Low-demand zones (Residential, University, Peripheral Industrial) are allocated 0 drivers, stranding elderly residents, hospital workers, and students.
2. **Gridlock and Bottlenecks:** Concentrating 120 cars into 4 blocks of Downtown causes localized traffic jams, preventing drivers from physically reaching pickup points.
3. **Worker Exploitation:** Treating independent drivers as automated robots violates labor autonomy.

---

### The 4-Pillar Fair & Safe Dispatching Framework:

```
                           [Predicted 15-Min Demand (All Zones)]
                                             │
                                             ▼
                 ┌───────────────────────────────────────────────────────┐
                 │ Pillar 1: Minimum Baseline Guarantee (Equity Reserve)  │
                 │ Allocate M_min = 6 drivers to EVERY zone first        │
                 └───────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                 ┌───────────────────────────────────────────────────────┐
                 │ Pillar 2: Proportional Marginal Allocation            │
                 │ Distribute remaining fleet based on (Predicted - M_min)│
                 └───────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                 ┌───────────────────────────────────────────────────────┐
                 │ Pillar 3: Anti-Congestion Safety Cap                  │
                 │ Cap any single zone at <= 35% of total active fleet    │
                 └───────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                 ┌───────────────────────────────────────────────────────┐
                 │ Pillar 4: Advisory Nudges (Preserving Driver Autonomy) │
                 │ Repositioning Incentives, Heatmaps, and Surge Bonuses  │
                 └───────────────────────────┘
```

#### Mathematical Formulation:
Let $N$ be total active available drivers across the city (e.g., $N = 300$).  
Let $Z = \{z_1, \dots, z_K\}$ be the set of $K = 8$ zones.

1. **Step 1: Baseline Equity Reserve:**
   Allocate $M_{\min} = 6$ drivers to each zone:
   $$R_0(z) = M_{\min}, \quad \forall z \in Z$$
   $$\text{Reserved Fleet} = K \cdot M_{\min} = 8 \times 6 = 48 \text{ drivers}$$
   $$\text{Remaining Fleet } N_{\text{rem}} = N - 48 = 252 \text{ drivers}$$

2. **Step 2: Marginal Demand Allocation:**
   Calculate unmet marginal demand:
   $$\tilde{d}(z) = \max\left(0, \hat{y}(z) - M_{\min}\right)$$
   Allocate remaining fleet proportionally:
   $$R_1(z) = M_{\min} + \text{round}\left(N_{\text{rem}} \cdot \frac{\tilde{d}(z)}{\sum_{j} \tilde{d}(j)}\right)$$

3. **Step 3: Anti-Gridlock Safety Cap:**
   Ensure no zone exceeds $C_{\max} = 0.35 \cdot N$:
   $$R_{\text{final}}(z) = \min\left(R_1(z), \lfloor 0.35 \cdot N \rfloor\right)$$
   Any capped surplus is redistributed to the next-highest deficit zones below their cap.

4. **Step 4: Operational Action Rules (Advisory Decision Support):**
   Compare target allocation $R_{\text{final}}(z)$ with current stationed drivers $C(z)$:
   $$\Delta(z) = R_{\text{final}}(z) - C(z)$$

   $$\text{Operational Action} = \begin{cases}
   \text{"Deploy more drivers"} & \text{if } \Delta(z) \ge +8 \\
   \text{"Deploy drivers"} & \text{if } +3 \le \Delta(z) < +8 \\
   \text{"Maintain/increase coverage"} & \text{if } -3 \le \Delta(z) < +3 \\
   \text{"Normal coverage"} & \text{if } -8 \le \Delta(z) < -3 \\
   \text{"Reduce excess drivers"} & \text{if } \Delta(z) < -8
   \end{cases}$$

---

## Question 8: Design a real-time dashboard for prediction error, demand spikes, zone performance, and drift.

### End-to-End System Architecture:

```
  ┌───────────────────────┐       ┌────────────────────────┐
  │  Streaming Telemetry  │       │ Historical Feature Lag │
  │  (GPS, App Requests)  │       │      (CSV/SQLite)      │
  └───────────┬───────────┘       └───────────┬────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │ Feature Engineering Pipeline │
               │ (Lags, Cyclical, One-Hot)    │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │ Scikit-Learn Inference Engine│
               │ (Gradient Boosting / RF)     │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │ Driver Positioning Engine    │
               │ (Constrained Optimization)   │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │     Streamlit Dashboard      │
               │  (Interactive Plotly & CSS)  │
               └──────────────────────────────┘
```

### Dashboard Core Functional Sections:

#### 1. Executive Overview & Live KPIs
- **Total Predicted Requests (15m):** Aggregated fleet-wide demand gauge.
- **Highest-Demand Zone:** Highlights immediate focal hotspot.
- **Average Zone Demand:** Baseline reference metric.
- **Model Accuracy ($R^2$):** Real-time tracking of predictive confidence.

#### 2. Demand Forecast & Real-Time Actual vs. Predicted Graphs
- Interactive multi-line chart comparing actual ground-truth against model predictions across a rolling 40-hour window.
- Visual inspection of morning and evening commute peak alignments.

#### 3. Zone Ranking & Recommended Driver Positioning
- Live ranked operational table displaying Zone, Predicted Demand, Target Drivers, Stationed Drivers, Net Deficit/Surplus, and Dispatch Action.
- Donut chart depicting fleet distribution across city sectors.
- Net shortage/surplus horizontal bar chart highlighting urgent zones in red and balanced zones in green.

#### 4. Error & Residual Monitoring
- Real-time MAE and RMSE KPI cards with delta improvements over persistence baselines.
- Residual Histogram ($y - \hat{y}$) with a zero-error target line, verifying that error distribution is centered tightly near zero without fat tails.
- Chronological error timeline detecting time periods with elevated residuals.

#### 5. Demand Spike & Anomaly Detection
- Dynamic statistical threshold ($95^{\text{th}}$ percentile of historical demand).
- Scatter plot flagging public event surges and torrential rainstorm anomalies in bright red markers.
- Bar chart identifying zones most susceptible to acute spikes.

#### 6. Model Drift & Distribution Shift Detector
- **Two-Sample Kolmogorov-Smirnov (KS) Test:** Compares recent 72-hour demand distribution against the training baseline.
- **Alert Banner:** Renders an amber/red warning:
  `"⚠️ POTENTIAL DISTRIBUTION SHIFT DETECTED!"`
  whenever public events or weather shocks cause $p$-value $< 0.05$.
- KDE distribution plot overlaying normal day demand against event day demand curves.

---
**Document Status:** Complete & Verified  
**Project:** Urban Mobility – Predicting Ride Demand  
