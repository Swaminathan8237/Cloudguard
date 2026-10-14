# Report 1 — ML for Cloud Services

**Course**: Machine Learning for Cloud Services  
**Project Title**: **CloudGuard-Water: Proactive Water Quality Deterioration Prediction Using IoT Data and Machine Learning**  
**Submission Date**: 05 September 2026  
**Candidate Name / Roll No**: E0224033  

---

## 1. Project Title

**CloudGuard-Water: Proactive Water Quality Deterioration Prediction Using IoT Data and Machine Learning**

---

## 2. Dataset

### 2.1 Dataset Name
**Water Quality Dataset from Water Sources in Kenya**

### 2.2 Dataset Source
**Mendeley Data (Elsevier)**  
- **Contributors / Authors**: Samson Otieno Ooko, Laura Cheptegei, Simon Karume  
- **Affiliated Institution**: Kabarak University, Nakuru, Kenya  
- **Funder**: Adventist University of Africa  
- **Dataset Version**: Version 2 (Published: 07 April 2026)  
- **Digital Object Identifier (DOI)**: `10.17632/575ctg3nnr.2`  
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

### 2.3 Dataset Download Link
- **Mendeley Data Repository**: [https://data.mendeley.com/datasets/575ctg3nnr/2](https://data.mendeley.com/datasets/575ctg3nnr/2)  
- **Direct S3 File Download**: [https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/b12f1afe-b88c-409c-bca6-61fec2dd3fbc](https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/b12f1afe-b88c-409c-bca6-61fec2dd3fbc)

### 2.4 Dataset Description
The dataset contains continuous, in-situ physicochemical water quality measurements captured using an IoT-based multi-sensor telemetry probe deployed across diverse aquatic environments throughout Kenya during February–March 2026. Data logging was performed at a rigorous sampling cadence of **2-minute intervals**, yielding a benchmark collection of **100,800 observations**.

The physical sensing platform captured four primary water quality indicators:
1. **pH**: Acidity/alkalinity balance.
2. **Total Dissolved Solids (TDS)**: Aggregate mineral, salt, and dissolved metal concentration.
3. **Turbidity**: Suspended particulate matter and water clarity.
4. **Temperature**: Ambient water temperature in degrees Celsius.

The dataset covers four contrasting environmental water source classes:
- **Treated Municipal and Bottled Water**: Monitored safe drinking sources compliant with standard potability criteria.
- **Surface Water (Rivers and Lakes)**: Natural runoff water with seasonal silt and organic loading.
- **Groundwater (Boreholes)**: Deep aquifer extraction characterized by high dissolved mineralization.
- **Marine Water (Indian Ocean)**: High-salinity coastal seawater.

Each record is labeled with a binary safety outcome (`Safe` vs. `Unsafe`), established against national and World Health Organization (WHO) potable water guidelines.

### 2.5 Dataset Shape
- **Rows**: **100,800**
- **Columns**: **5**

### 2.6 Number of Rows
**100,800 observations** (Verified by programmatic audit; zero truncation).

### 2.7 Number of Columns
**5 columns** (4 numerical continuous sensor features + 1 categorical target label).

### 2.8 Total Storage Size
- **On-Disk File Size (CSV)**: **2,964,409 bytes** (**2.83 MB**)
- **In-Memory Size (Pandas DataFrame)**: **9.07 MB**

### 2.9 Data Dictionary

| Variable Name | Data Type | Physical Meaning & Description | Example | Role |
| :--- | :---: | :--- | :---: | :---: |
| **`pH`** | `float64` | Dimensionless logarithmic scale measuring hydrogen-ion activity (acidity: <7.0; alkalinity: >7.0). WHO potable target is 6.5–8.5. | `7.38` | Feature (Input) |
| **`TDS_ppm`** | `float64` | Total Dissolved Solids measured in parts per million ($\text{ppm} = \text{mg/L}$). Reflects dissolved inorganic salts (calcium, magnesium, chlorides, sulfates). Desirable drinking limit is <300–500 ppm. | `138.7` | Feature (Input) |
| **`Turbidity_NTU`** | `float64` | Nephelometric Turbidity Units measuring light attenuation caused by suspended particles, clay, and microorganisms. Potable threshold is <5.0 NTU. | `0.25` | Feature (Input) |
| **`Temperature_C`** | `float64` | Water temperature measured in degrees Celsius (°C). Governs chemical dissolution equilibria, microbial growth rates, and electrical conductivity calibration. | `20.25` | Feature (Input) |
| **`Label`** | `object` (string) | Ground truth potability classification: `'Safe'` denotes potable treated water; `'Unsafe'` denotes untreated or contaminated surface, marine, or mineral groundwater. | `'Safe'` | Target (Output) |

### 2.10 Target Variable
- **Variable Name**: `Label`
- **Type**: Binary Categorical (`object` / string)
- **Unique Classes**:
  - **`Unsafe`**: **70,560 rows (70.00%)**
  - **`Safe`**: **30,240 rows (30.00%)**
- **Class Balance**: 70:30 ratio representing typical operational imbalance in environmental water monitoring.

### 2.11 Feature Variables & Statistical Summary

| Feature | Min | 25% | Median (50%) | Mean | 75% | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pH`** | 6.09 | 7.01 | 7.38 | 7.59 | 8.10 | 9.96 | 0.80 |
| **`TDS_ppm`** | 28.60 | 163.00 | 472.85 | 692.65 | 1023.33 | 2806.10 | 647.33 |
| **`Turbidity_NTU`** | 0.00 | 1.92 | 9.97 | 11.31 | 18.54 | 72.75 | 9.87 |
| **`Temperature_C`** | 16.56 | 23.09 | 25.04 | 24.88 | 26.93 | 31.51 | 2.69 |

#### Structural Architecture Discovery (10 Deployment Blocks)
Forensic auditing revealed that the dataset is organized into **10 contiguous 14-day monitoring campaigns** of exactly **10,080 observations** each ($10,080 \times 2 \text{ minutes} = 20,160 \text{ minutes} = 336 \text{ hours} = \mathbf{14 \text{ days}}$):
- **Blocks 1–3 (Rows 0 to 30,239)**: Three `Safe` water sources (municipal tap, bottled, spring supply), with TDS < 211 ppm and Turbidity < 4.45 NTU.
- **Blocks 4–10 (Rows 30,240 to 100,799)**: Seven `Unsafe` environmental sources:
  - *Blocks 4–6*: High turbidity river/lake runoff (turbidity reaching 72.75 NTU).
  - *Blocks 7–8*: High salinity marine/coastal incursion (TDS up to 2,806.1 ppm, pH up to 9.96).
  - *Block 9*: High mineral groundwater borehole (TDS up to 913 ppm).
  - *Block 10*: Agricultural/industrial drainage water.

---

## 3. Problem Statement

### 3.1 The Existing Problem
Clean water accessibility is a critical global sustainability challenge recognized by the United Nations Sustainable Development Goal 6 (Clean Water and Sanitation). In many regions, clean water supplies are vulnerable to sudden contamination caused by flash flooding, agricultural pesticide runoff, sewage overflow, and industrial effluents.

In conventional water supply management, testing relies on periodic manual grab sampling followed by laboratory wet-chemistry assays (such as microbial plate counts, gravimetric solids analysis, and spectrometry). These laboratory tests are time-consuming (taking 24–48 hours) and labor-intensive, meaning contamination is often detected long after water has already reached consumers.

### 3.2 Limitation of Current-State IoT Monitoring
Recent IoT water quality deployments address latency by streaming sensor measurements (pH, TDS, turbidity, temperature) to cloud platforms and running machine-learning classifiers to categorize the *current* state:

$$\mathbf{x}_t = [\text{pH}_t, \text{TDS}_t, \text{Turbidity}_t, \text{Temperature}_t] \xrightarrow{\text{ML Classifier}} \hat{y}_t \in \{\text{Safe}, \text{Unsafe}\}$$

While this provides instantaneous classification, **current-state classification is inherently reactive**:
1. **Zero Lead Time**: An alert is triggered only *after* the water passing through the sensor has already crossed the contamination threshold.
2. **Failure to Prevent Exposure**: By the time treatment plants, valve actuators, or community managers receive a "Water Unsafe" notification, contaminated water has already entered distribution conduits.
3. **No Trend Recognition**: A current reading of $\text{pH} = 7.1$ and $\text{Turbidity} = 2.0 \text{ NTU}$ is reported as `Safe`. However, if turbidity was $0.5 \text{ NTU}$ twenty minutes ago and $1.2 \text{ NTU}$ ten minutes ago, the water is clearly deteriorating rapidly toward an unsafe condition. Static classifiers miss this critical trend entirely.

### 3.3 Why Machine Learning is Needed
Water quality parameters do not change independently; they exhibit complex, non-linear interactions:
- Rainfall introduces particulate soil runoff that simultaneously increases turbidity, alters dissolved ion concentrations (TDS), and shifts pH depending on soil chemistry.
- Water temperature influences gas solubility and microbial metabolic kinetics, accelerating chemical degradation at higher temperatures.
- Simple static threshold rules (e.g., "Flag if $\text{TDS} > 500$") frequently cause false alarms during transient spikes or fail to detect composite contamination where several parameters remain slightly below individual thresholds but collectively signal dangerous deterioration.

Machine learning models learn non-linear decision boundaries directly from multi-sensor telemetry, enabling both accurate classification and early pattern detection.

### 3.4 Why Classification is Appropriate
The operational goal of water utility distribution and cloud-connected IoT monitoring is to drive binary, actionable decisions:
- **`Safe` (0)**: Normal operational state; water distribution continues without intervention.
- **`Unsafe` (1)**: Contaminated or deteriorating state; trigger automated intake shutoff valves, initiate chemical coagulant dosing, or dispatch field maintenance alerts.

Because health guidelines and treatment workflows are structured around discrete compliance boundaries, binary classification provides clear, unambiguous operational triggers.

### 3.5 Suitability of Decision Trees and Random Forests
For cloud-connected IoT water safety, **Decision Trees** and **Random Forests** are ideal for several key reasons:
1. **Interpretability and Regulatory Transparency**: A Decision Tree produces clear, human-readable if-then decision rules. Operators can see exactly why a warning was generated (e.g., *"If $\text{TDS} > 250 \text{ ppm}$ AND $\Delta\text{Turbidity} > +1.5 \text{ NTU/20 min}$, flag high risk"*).
2. **Handling Non-Linearities and Complex Feature Thresholds**: Tree-based partitioning naturally captures sharp environmental thresholds without requiring non-linear kernel transformations.
3. **Robustness to Differing Scales**: Physical sensor variables vary across very different scales ($\text{pH} \in [6, 10]$ vs. $\text{TDS} \in [20, 2800]$). Tree models are scale-invariant, eliminating the need for feature normalization and reducing edge compute overhead.
4. **Resilience to Sensor Noise**: Random Forest ensembles aggregate dozens of randomized decision trees, dampening individual sensor noise, transient drift, and sensor glitches common in field IoT deployments.

### 3.6 Value of Historical Sensor Behavior
A single snapshot in time ($\mathbf{x}_t$) carries no information about direction or velocity of change. Incorporating historical sensor behavior enables the model to capture:
- **Rate of Change (Trends)**: Distinguishing between stable baseline readings and sharp upward spikes caused by sudden contamination events:
  $$\Delta \text{Turbidity} = \text{Turbidity}_t - \text{Turbidity}_{t-k}$$
- **Rolling Volatility**: Elevated variance in turbidity and TDS often precedes severe contamination events:
  $$\sigma_{\text{rolling}}(\text{TDS}, W=10)$$
- **Lagged Telemetry**: Comparing sensor readings at $t-10 \text{ min}$ and $t-20 \text{ min}$ captures environmental inertia and diffusion dynamics.

### 3.7 The Novel Component: Proactive Early-Warning Deterioration Prediction
The primary novel contribution of **CloudGuard-Water** is extending traditional current-state water quality classification into a **proactive, cloud-based early-warning deterioration prediction system**:

$$\underbrace{[\mathbf{x}_t, \mathbf{x}_{t-1}, \dots, \mathbf{x}_{t-k}, \Delta\mathbf{x}, \boldsymbol{\sigma}_{\text{rolling}}]}_{\text{Current + Historical Telemetry}} \xrightarrow{\text{Cloud ML Model}} \mathbf{\hat{y}_{t + 20\text{min}}} \in \{\text{Will Remain Safe}, \text{Will Deteriorate to Unsafe}\}$$

Rather than waiting for contamination to occur, the system answers:
> *"Given sensor readings and recent trajectory up to time $t$, will this water source become unsafe 20 minutes from now?"*

This 20-minute operational lead time transforms water safety management from **reactive damage control** to **preventative automated protection**.

### 3.8 Expected Results
1. **Baseline Classification Benchmark**:
   - Establish current-state classification using Logistic Regression, Decision Tree, and Random Forest.
   - Initial empirical benchmarking on the Kenya dataset achieved **>99.9% accuracy**, with TDS ($67.4\%$) and Turbidity ($25.7\%$) emerging as the dominant current-state predictors. This demonstrates that current-state classification is essentially a solved problem.
2. **Proactive 20-Minute Prediction**:
   - Develop temporal features (lags at $t-2, t-4, t-10, t-20$ minutes; rolling averages; rates of change) strictly within continuous deployment blocks to prevent data leakage.
   - Construct future target labels: $\text{Unsafe}_{t+20\text{min}}$.
   - Chronological train/test split ($80\%$ earliest observations for training, $20\%$ unseen future observations for evaluation).
   - Expected Recall on the `Unsafe` class $\ge 95\%$, ensuring nearly all impending contamination events are caught before water quality degrades.

### 3.9 Cloud Services Integration & Real-World Utility
The end-to-end architecture is designed as a modular, scalable cloud service:
1. **IoT Edge Gateway**: Microcontroller (e.g., ESP32 / Arduino / Raspberry Pi) reads analog sensor probes, applies calibration offsets, and publishes JSON telemetry payloads every 2 minutes via MQTT or HTTP POST to an IoT ingestion endpoint (such as ThingSpeak or AWS IoT Core / Google Cloud Pub/Sub).
2. **Cloud Storage & Ingestion Pipeline**: Ingestion buffers stream data into scalable storage (e.g., Google Cloud Storage / Amazon S3 bucket) and an operational database.
3. **Automated Feature Pipeline**: Cloud function or lightweight worker computes rolling window statistics and lag features on incoming telemetry streams.
4. **Cloud ML Inference Microservice**: A RESTful FastAPI / Flask endpoint hosts the trained Random Forest model. Upon receiving newly engineered features, it computes:
   - Current Water Safety State (`Safe` / `Unsafe`).
   - Predicted 20-Minute Future Safety State.
   - Contamination Risk Probability ($P(\text{Unsafe}_{t+20})$).
5. **Real-Time Monitoring Dashboard & Alert Dispatch**:
   - Web-based operations dashboard displays real-time time-series sensor charts and live risk indicators.
   - When risk exceeds a designated threshold (e.g., $P > 0.70$), automated webhooks dispatch SMS/email early warnings and trigger relay switches to close physical intake valves before contaminated water enters distribution networks.

---

## 4. Verification and Reproducibility

All dataset audit numbers, statistics, and baseline ML results reported above were verified through reproducible Python scripts executed in the project workspace:
- **`audit_dataset.py`**: Validates file integrity, hashes, row/column counts, missing values, and block partitioning.
- **`train_baseline.py`**: Executes 5-fold cross-validation across baseline classifiers and extracts feature importance rankings.
- **`generate_eda_plots.py`**: Generates publication-ready figures saved in [`eda_plots/`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/eda_plots/).
