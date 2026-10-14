# Phase 2 — Exploratory Data Analysis (EDA) Report

**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Course**: Machine Learning for Cloud Services (Use Case #5: IoT-Based Water Quality Classification)  
**Dataset**: [`data/water_quality_dataset1.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/data/water_quality_dataset1.csv) (100,800 rows × 5 columns)  
**Code Artifacts**: [`src/eda.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/eda.py), [`notebooks/02_eda.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/02_eda.ipynb)  
**Figures Location**: [`reports/figures/`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/)  

---

## 1. Executive Summary & Dataset Overview

This report documents the rigorous exploratory data analysis conducted for the **CloudGuard-Water** intelligent water quality classification system. All analyses, statistics, and visualizations presented herein are derived directly from programmatic execution against the authoritative dataset (`water_quality_dataset1.csv`).

### Dataset Quick Facts
- **Total Observations**: 100,800
- **Total Variables**: 5 (4 continuous physicochemical features + 1 binary categorical target)
- **Missing / Corrupt Values**: Exactly 0 across all columns
- **Duplicate Records**: Exactly 0
- **Target Variable**: `Label` (`Safe` vs. `Unsafe`)
- **Temporal Nature**: Strictly cross-sectional / tabular IoT telemetry snapshot; **no timestamp or temporal ordering** exists.

---

## 2. Class Distribution Analysis

### Quantitative Findings
From the direct evaluation of `df['Label'].value_counts()`:

```text
Safe:   30,240 samples, 30.00%
Unsafe: 70,560 samples, 70.00%
Class Imbalance Ratio (Unsafe : Safe): 2.33 : 1
```

| Class Label | Frequency (Count) | Proportion (%) | Operational Meaning |
|:------------|------------------:|---------------:|:-------------------|
| **`Unsafe`** | 70,560 | 70.00% | Non-potable water exceeding physical contamination limits |
| **`Safe`** | 30,240 | 30.00% | Potable water complying with standard drinking guidelines |
| **Total** | 100,800 | 100.00% | Complete monitored population |

### Visualization
The visualization is saved at: [`reports/figures/class_distribution.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/class_distribution.png).

### Analytical Interpretation: Why Class Imbalance Matters
1. **The Accuracy Paradox**: If a naive model simply predicts the majority class (`Unsafe`) for every incoming record, it will theoretically achieve a **70.00% accuracy** while demonstrating **zero utility** for recognizing potable safe water. Therefore, raw accuracy alone is insufficient.
2. **Asymmetric Cost of Classification Errors**:
   - **False Positive** (Predicting `Unsafe` when water is actually `Safe`): Causes unnecessary alert notifications or redundant filtration cycles, leading to minor operational costs.
   - **False Negative** (Predicting `Safe` when water is actually `Unsafe`): Represents a catastrophic failure mode where contaminated water passes into human consumption or municipal distribution conduits without warning.
3. **Implications for Phase 3/4**:
   - Model evaluation must prioritize **Recall on the `Unsafe` class**, **Precision**, and the **Macro/Weighted F1-score**.
   - Cross-validation and train/test splits must use **Stratification** (`StratifiedKFold` / `train_test_split(stratify=y)`) to ensure both splits preserve the true 70:30 population proportion.

---

## 3. Individual Feature Distributions

Each of the four continuous features was analyzed for central tendency, dispersion, range, and skewness:

| Feature Name | Minimum | Maximum | Mean | Median | Std Dev | Skewness | Primary Physical Trait |
|:-------------|--------:|--------:|-----:|-------:|--------:|---------:|:-----------------------|
| **`pH`** | 6.09 | 9.96 | 7.59 | 7.38 | 0.80 | **+0.80** | Moderately right-skewed; mostly neutral to alkaline |
| **`TDS_ppm`** | 28.60 | 2,806.10 | 692.65 | 472.85 | 647.33 | **+1.27** | Highly right-skewed with long upper tail |
| **`Turbidity_NTU`**| 0.00 | 72.75 | 11.31 | 9.97 | 9.87 | **+0.62** | Moderately right-skewed; heavy concentration near 0 |
| **`Temperature_C`**| 16.56 | 31.51 | 24.88 | 25.04 | 2.69 | **-0.32** | Nearly symmetric / bell-shaped distribution |

### Visualization
The combined distribution plot (Histograms + KDE curves overlaid by class) is saved at: [`reports/figures/feature_distributions.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/feature_distributions.png).

### Feature-by-Feature Interpretation:
- **`pH`**: Centered around 7.38, well within the normal environmental spectrum. Safe samples show a compact distribution tightly centered around neutral pH ($7.17 \pm 0.22$), whereas Unsafe samples extend considerably toward alkaline levels up to 9.96.
- **`TDS_ppm`**: Demonstrates the strongest skewness (+1.27) and highest variance ($\sigma = 647.33$). Safe water is strictly compressed at low mineral levels (<211 ppm), while Unsafe observations span an enormous range up to 2,806 ppm (typical of mineral groundwater and marine environments).
- **`Turbidity_NTU`**: Safe samples cluster tightly below 4.5 NTU with a mean of 1.28 NTU (fully compliant with the WHO <5.0 NTU potability rule). Unsafe observations reach up to 72.75 NTU with an elevated mean of 15.60 NTU.
- **`Temperature_C`**: Exhibits a near-normal, symmetric bell curve ($\text{skew} = -0.32$) spanning 16.56 °C to 31.51 °C. Unsafe water tends to be slightly warmer on average (26.03 °C vs. 22.21 °C), reflecting surface/shallow open water environments exposed to ambient heating.

---

## 4. Safe vs. Unsafe Comparative Analysis

### Quantitative Comparison: Per-Class Central Tendencies

| Feature | Safe Mean | Unsafe Mean | Difference ($\Delta = \text{Unsafe} - \text{Safe}$) | Safe Median | Unsafe Median |
|:--------|----------:|------------:|----------------------------------------------------:|------------:|--------------:|
| **`pH`** | 7.17 | 7.78 | **+0.61** | 7.17 | 7.74 |
| **`TDS_ppm`** | 118.34 | 938.78 | **+820.43** | 114.60 | 702.30 |
| **`Turbidity_NTU`** | 1.28 | 15.60 | **+14.32** | 1.13 | 15.34 |
| **`Temperature_C`** | 22.21 | 26.03 | **+3.82** | 22.21 | 26.13 |

### Visualization
Box plots comparing Safe vs. Unsafe across all four features are saved at: [`reports/figures/safe_vs_unsafe_boxplots.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/safe_vs_unsafe_boxplots.png).

### Key Observations:
1. **Total Dissolved Solids (TDS)** provides the largest absolute shift ($+820.43\text{ ppm}$). The interquartile range (IQR) of `Safe` water does not overlap with the upper 75% of `Unsafe` water.
2. **Turbidity** shows an order-of-magnitude increase ($1.28 \to 15.60\text{ NTU}$). In potable water, suspended particulate matter is virtually absent; in contaminated sources, runoff and microbial/sediment load drastically elevate light scattering.
3. **Separation Quality**: Both TDS and Turbidity display substantial separation margins between the two classes.

---

## 5. Input Feature Correlation Analysis

### Quantitative Correlation Matrix (Input Features Only)
*Note: Target variable is strictly excluded from this heatmap to focus purely on input feature interdependencies.*

| Pearson $r$ | pH | TDS_ppm | Turbidity_NTU | Temperature_C |
|:------------|---:|--------:|--------------:|--------------:|
| **`pH`** | **1.0000** | +0.8041 | +0.2194 | +0.3933 |
| **`TDS_ppm`** | +0.8041 | **1.0000** | +0.4529 | +0.5361 |
| **`Turbidity_NTU`** | +0.2194 | +0.4529 | **1.0000** | +0.5608 |
| **`Temperature_C`** | +0.3933 | +0.5361 | +0.5608 | **1.0000** |

### Visualization
The heatmap is saved at: [`reports/figures/correlation_heatmap.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/correlation_heatmap.png).

### Interpretation (Avoiding Causal Claims):
- **pH and TDS ($r = +0.8041$)**: Displays a strong positive linear association. In natural waters, high dissolved mineral salts (carbonates, bicarbonates, and alkaline earth metals) frequently co-occur with alkaline pH. This reflects common geochemical hydro-environments rather than one directly "causing" the other.
- **Turbidity and Temperature ($r = +0.5608$)**: Moderately positive association, likely reflecting seasonal runoff where warm summer waters receive storm silt runoff.
- **TDS and Turbidity ($r = +0.4529$)**: Moderate positive association.
- **Multicollinearity Impact**: While strong collinearity (+0.804) can inflate variance in linear model weights (Logistic Regression coefficients), **tree-based classifiers (Decision Trees, Random Forests) are invariant to monotonic collinearity**, as they evaluate features via orthogonal axis splits.

---

## 6. Pairwise Relationships (5% Stratified Representative Sample)

### Visualization
To prevent visual overplotting and ensure responsive visualization performance, a **5% stratified sample (5,040 observations: 1,512 Safe, 3,528 Unsafe)** was plotted.  
The pairwise scatter and KDE grid is saved at: [`reports/figures/feature_pairplot.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/feature_pairplot.png).

### Key Observations:
- **Cluster Demarcation**: In the 2D subspaces of `(TDS_ppm, Turbidity_NTU)` and `(pH, TDS_ppm)`, the `Safe` observations (green) collapse into a tightly bounded, dense cluster in the lower-left corner (low TDS, low turbidity, neutral pH).
- **Unsafe Region Dispersion**: The `Unsafe` observations (red) form dispersed clusters occupying high-TDS, high-turbidity, and high-pH spaces.
- **Separability Implication**: The classes do not form complex, intertwined non-convex spirals; instead, they occupy predominantly distinct regions of the feature space. This suggests that basic tree models and decision boundaries will be effective and interpretable.

---

## 7. Statistical Outlier Inspection (Tukey's IQR Method)

### Quantitative Summary of IQR Fences

| Feature | Q1 | Q3 | IQR | Lower Fence ($Q_1 - 1.5\text{IQR}$) | Upper Fence ($Q_3 + 1.5\text{IQR}$) | Outliers Detected | % of Dataset |
|:--------|---:|---:|----:|-------------------------------------:|-------------------------------------:|------------------:|-------------:|
| **`pH`** | 7.01 | 8.10 | 1.09 | 5.38 | 9.73 | 423 | 0.42% |
| **`TDS_ppm`** | 163.00 | 1,023.32 | 860.32 | -1,127.49 | 2,313.81 | 3,096 | 3.07% |
| **`Turbidity_NTU`** | 1.92 | 18.54 | 16.62 | -23.01 | 43.47 | 118 | 0.12% |
| **`Temperature_C`** | 23.09 | 26.93 | 3.84 | 17.33 | 32.69 | 72 | 0.07% |

### Visualization
The boxplot showing statistical fences and points is saved at: [`reports/figures/outlier_analysis.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/outlier_analysis.png).

### Why Outliers Must NOT Be Removed:
1. **Statistical Outlier $\neq$ Invalid Data**: A Tukey IQR fence is a mechanical mathematical threshold based on the assumption of normality. Environmental water quality is inherently multi-modal and non-Gaussian.
2. **Representation of Real Water Conditions**:
   - The 3,096 TDS observations exceeding 2,313 ppm correspond to real high-salinity coastal water and deep mineral aquifers.
   - The 118 Turbidity observations exceeding 43.5 NTU represent extreme runoff/silt loading events.
3. **Critical for Safety Classification**: If we artificially trimmed or deleted high TDS or high turbidity rows, we would be stripping out the clearest signatures of unsafe water.

---

## 8. Master Statistical Reference Table

| Metric | `pH` | `TDS_ppm` | `Turbidity_NTU` | `Temperature_C` |
|:-------|-----:|----------:|----------------:|----------------:|
| **Count** | 100,800 | 100,800 | 100,800 | 100,800 |
| **Mean** | 7.59 | 692.65 | 11.31 | 24.88 |
| **Median** | 7.38 | 472.85 | 9.97 | 25.04 |
| **Standard Deviation** | 0.80 | 647.33 | 9.87 | 2.69 |
| **Minimum** | 6.09 | 28.60 | 0.00 | 16.56 |
| **Maximum** | 9.96 | 2,806.10 | 72.75 | 31.51 |
| **Safe Mean** | 7.17 | 118.34 | 1.28 | 22.21 |
| **Unsafe Mean** | 7.78 | 938.78 | 15.60 | 26.03 |
| **Difference ($\Delta$)** | **+0.61** | **+820.43** | **+14.32** | **+3.82** |
| **Safe Median** | 7.17 | 114.60 | 1.13 | 22.21 |
| **Unsafe Median** | 7.74 | 702.30 | 15.34 | 26.13 |

---

## 9. Key Findings for ML Modeling (Transition to Phase 3/4)

### 1. Most Discriminative Features
- **TDS_ppm** and **Turbidity_NTU** demonstrate the strongest statistical separation between Safe and Unsafe classes.
- Safe water strictly occupies low ranges ($TDS < 211\text{ ppm}, Turbidity < 4.5\text{ NTU}$), making simple threshold-based decision rules in Decision Trees intuitive.

### 2. Class Imbalance Handling
- The 70:30 imbalance ratio (2.33:1) is moderate.
- Oversampling techniques (e.g., SMOTE) are **unnecessary** and might synthesize artificial data points in empty regions.
- Instead, **Stratified K-Fold splitting** and potential model class weighting (`class_weight='balanced'`) should be explored in Phase 3/4.
- Primary evaluation must report **Recall on Unsafe**, **Precision**, and **F1-score**.

### 3. Feature Scaling Strategy
- **Logistic Regression**: Essential. The feature ranges vary drastically ($\text{pH} \in [6, 10]$ vs. $\text{TDS} \in [28, 2806]$). Without scaling (`StandardScaler`), the gradient updates would be dominated by TDS.
- **Decision Tree & Random Forest**: Invariant to monotonic scale transformations. No scaling is needed, preserving the original physical engineering units (ppm, NTU, °C) for explainability.

### 4. Treatment of Outliers
- All 3,600+ IQR outlier observations must be **retained**. Tree-based partition algorithms handle non-Gaussian tails naturally without suffering from distortion.

### 5. Multicollinearity Consideration
- The +0.804 correlation between pH and TDS should be kept in mind when interpreting Logistic Regression coefficients, but will not hinder tree-based ensemble performance.

### 6. Dataset Suitability & Limitations
- **Suitability**: Excellent. Large sample size (100,800), zero missing records, clear physical features, distinct decision boundaries.
- **Limitation**: Cross-sectional only. No temporal variables exist, confirming that time-series forecasting cannot be performed.

---

**Phase 2 Complete.** Ready for Phase 3 (Data Preprocessing & Baseline Model Training).
