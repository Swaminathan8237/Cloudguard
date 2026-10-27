"""
Script: src/generate_phase5_artifacts.py
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: Generate all required Phase 5 explainability artifacts:
         - Decision tree rules text (reports/decision_tree_rules.txt)
         - Tree architecture visualization (reports/figures/decision_tree_explainability.png)
         - DT feature importance plot (reports/figures/decision_tree_feature_importance.png)
         - RF SHAP feature importance table & plots (reports/random_forest_shap_importance.csv, figures)
         - Representative local prediction explanations (reports/local_explanations.csv)
         - Decision boundary sensitivity analysis (reports/decision_boundary_sensitivity.csv, figures)
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, plot_tree
import shap

from src.explain import (
    FEATURE_NAMES,
    CLASS_LABELS,
    load_model,
    validate_input,
    extract_decision_path,
    get_feature_importances,
    explain_sample,
)

# Setup directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
COLOR_SAFE = "#2ecc71"
COLOR_UNSAFE = "#e74c3c"
COLOR_PRIMARY = "#3498db"
COLOR_SECONDARY = "#9b59b6"


def run_phase5_generation():
    print("=" * 80)
    print("PHASE 5: ARTIFACT GENERATION & EXPLAINABILITY ENGINE")
    print("=" * 80)

    # 1. Load canonical dataset and partition identically to Phases 3-4
    csv_path = DATA_DIR / "water_quality_dataset1.csv"
    print(f"Loading canonical dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES]
    y = (df["Label"] == "Unsafe").astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train = pd.DataFrame(X_train_raw, columns=FEATURE_NAMES)
    X_test = pd.DataFrame(X_test_raw, columns=FEATURE_NAMES)
    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    # 2. Load Models
    dt = load_model(MODELS_DIR / "decision_tree.joblib")
    rf = load_model(MODELS_DIR / "random_forest.joblib")
    print(f"Loaded Decision Tree: depth={dt.get_depth()}, leaves={dt.get_n_leaves()}, nodes={dt.tree_.node_count}")
    print(f"Loaded Random Forest: estimators={rf.n_estimators}, max_depth={rf.max_depth}")

    # -----------------------------------------------------------------------
    # Artifact 1: Decision Tree Rules Text
    # -----------------------------------------------------------------------
    print("\n--- Generating reports/decision_tree_rules.txt ---")
    tree_rules_text = export_text(dt, feature_names=FEATURE_NAMES)
    rules_header = (
        "================================================================================\n"
        "CloudGuard-Water: Decision Tree Full Architecture & Learned Rules\n"
        "Model: DecisionTreeClassifier(max_depth=5, random_state=42)\n"
        f"Actual Depth: {dt.get_depth()} | Leaves: {dt.get_n_leaves()} | Total Nodes: {dt.tree_.node_count}\n"
        "Class Encoding: 0 = Safe, 1 = Unsafe\n"
        "Feature Order: ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']\n"
        "================================================================================\n\n"
    )
    rules_path = REPORTS_DIR / "decision_tree_rules.txt"
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(rules_header + tree_rules_text)
    print(f"Saved: {rules_path}")

    # -----------------------------------------------------------------------
    # Artifact 2: Decision Tree Visual Diagram
    # -----------------------------------------------------------------------
    print("\n--- Generating reports/figures/decision_tree_explainability.png ---")
    fig, ax = plt.subplots(figsize=(24, 11), dpi=200)
    plot_tree(
        dt,
        feature_names=FEATURE_NAMES,
        class_names=["Safe", "Unsafe"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        precision=2,
    )
    ax.set_title(
        "CloudGuard-Water: Learned Decision Tree (max_depth=5, 14 Leaves)\n"
        "Deterministic Transparent Decision Boundaries for Water Potability Classification",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    tree_fig_path = FIGURES_DIR / "decision_tree_explainability.png"
    fig.savefig(tree_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {tree_fig_path}")

    # -----------------------------------------------------------------------
    # Artifact 3: Decision Tree Feature Importance Bar Chart
    # -----------------------------------------------------------------------
    print("\n--- Generating reports/figures/decision_tree_feature_importance.png ---")
    fi_dt = get_feature_importances(dt)
    print("Decision Tree Feature Importances:")
    print(fi_dt.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.barh(
        fi_dt["Feature"],
        fi_dt["Percentage"],
        color=[COLOR_PRIMARY if p > 50 else COLOR_SECONDARY if p > 0 else "#95a5a6" for p in fi_dt["Percentage"]],
        edgecolor="#2c3e50",
        height=0.55,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Gini Impurity Decrease Relative Importance (%)", fontsize=11, fontweight="bold")
    ax.set_title("Decision Tree Global Feature Importance (Impurity Reduction)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", ha="left", fontsize=10, fontweight="bold")

    plt.tight_layout()
    dt_fi_path = FIGURES_DIR / "decision_tree_feature_importance.png"
    fig.savefig(dt_fi_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {dt_fi_path}")

    # -----------------------------------------------------------------------
    # Artifact 4: Random Forest SHAP Analysis
    # -----------------------------------------------------------------------
    print("\n--- Calculating Random Forest SHAP Values ---")
    # Take a reproducible stratified sample of 1,000 observations from X_train
    shap_sample_X = X_train.sample(n=1000, random_state=42)
    explainer = shap.TreeExplainer(rf)
    shap_explanation = explainer(shap_sample_X)

    # Extract SHAP values for Class 1 (Unsafe)
    # Shape is (1000, 4, 2)
    values_arr = np.asarray(shap_explanation.values)
    shap_values_unsafe = values_arr[:, :, 1]
    mean_abs_shap = np.abs(shap_values_unsafe).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Mean_Absolute_SHAP": mean_abs_shap,
        "Relative_Percentage": (mean_abs_shap / mean_abs_shap.sum()) * 100.0,
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

    shap_csv_path = REPORTS_DIR / "random_forest_shap_importance.csv"
    shap_df.to_csv(shap_csv_path, index=False)
    print("Random Forest SHAP Feature Importance (Class 1 = Unsafe):")
    print(shap_df.to_string(index=False))
    print(f"Saved: {shap_csv_path}")

    # SHAP Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.barh(
        shap_df["Feature"],
        shap_df["Mean_Absolute_SHAP"],
        color=[COLOR_UNSAFE, COLOR_PRIMARY, COLOR_SECONDARY, "#95a5a6"],
        edgecolor="#2c3e50",
        height=0.55,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value| (Impact on Class: Unsafe)", fontsize=11, fontweight="bold")
    ax.set_title("Random Forest Global SHAP Feature Importance\n(Sample N=1,000 from Training Partition)", fontsize=12, fontweight="bold", pad=12)
    max_shap = shap_df["Mean_Absolute_SHAP"].max()
    ax.set_xlim(0, max_shap * 1.22)

    for bar, pct in zip(bars, shap_df["Relative_Percentage"]):
        w = bar.get_width()
        ax.text(w + (max_shap * 0.02), bar.get_y() + bar.get_height() / 2, f"{w:.4f} ({pct:.1f}%)", va="center", ha="left", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    shap_bar_path = FIGURES_DIR / "random_forest_shap_importance.png"
    fig.savefig(shap_bar_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {shap_bar_path}")

    # SHAP Summary Beeswarm Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    shap.summary_plot(
        shap_values_unsafe,
        shap_sample_X,
        feature_names=FEATURE_NAMES,
        plot_type="dot",
        show=False,
    )
    plt.title("Random Forest SHAP Summary (Beeswarm) Plot\nFeature Impact Distribution on Unsafe Classification", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    shap_summary_path = FIGURES_DIR / "random_forest_shap_summary.png"
    plt.savefig(shap_summary_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {shap_summary_path}")

    # -----------------------------------------------------------------------
    # Artifact 5: Local Explanations for Real Samples
    # -----------------------------------------------------------------------
    print("\n--- Generating reports/local_explanations.csv ---")
    sample_definitions = [
        {"Category": "Safe (Conforming)", "Index": 0, "Description": "Representative Potable Water Sample from Start of Dataset"},
        {"Category": "Unsafe (Severe TDS Contamination)", "Index": 30240, "Description": "Representative Contaminated Water Sample from Transition Boundary"},
        {"Category": "Borderline False Negative (Test Set)", "Index": 55037, "Description": "Borderline Physical Reading Labeled Unsafe but within Safe Envelope"},
    ]

    local_rows = []
    for s_def in sample_definitions:
        idx = int(s_def["Index"])
        row_feat = X.iloc[idx:idx+1]
        ground_truth = df["Label"].iloc[idx]
        exp_result = explain_sample(dt, row_feat, actual_label=ground_truth)

        dt_pred = exp_result["predicted_class"]
        p_safe = exp_result["probabilities"]["Safe"]
        p_unsafe = exp_result["probabilities"]["Unsafe"]
        steps = exp_result["decision_path"]["steps"]
        steps_summary = " -> ".join([s["rule"] for s in steps])
        leaf_id = exp_result["decision_path"]["leaf_node_id"]

        # Also get RF prediction & RF SHAP for this sample
        rf_pred = CLASS_LABELS[int(rf.predict(row_feat)[0])]
        rf_probas = rf.predict_proba(row_feat)[0]
        sample_shap_arr = np.asarray(explainer(row_feat).values)
        sample_shap = sample_shap_arr[0, :, 1]
        shap_breakdown = ", ".join([f"{name}: {val:+.4f}" for name, val in zip(FEATURE_NAMES, sample_shap)])

        local_rows.append({
            "Sample_Category": s_def["Category"],
            "Dataset_Index": idx,
            "pH": row_feat["pH"].iloc[0],
            "TDS_ppm": row_feat["TDS_ppm"].iloc[0],
            "Turbidity_NTU": row_feat["Turbidity_NTU"].iloc[0],
            "Temperature_C": row_feat["Temperature_C"].iloc[0],
            "Ground_Truth": ground_truth,
            "DT_Prediction": dt_pred,
            "DT_Prob_Safe": round(p_safe, 4),
            "DT_Prob_Unsafe": round(p_unsafe, 4),
            "DT_Leaf_Node_ID": leaf_id,
            "DT_Decision_Path": steps_summary,
            "RF_Prediction": rf_pred,
            "RF_Prob_Safe": round(float(rf_probas[0]), 4),
            "RF_Prob_Unsafe": round(float(rf_probas[1]), 4),
            "RF_SHAP_Values_Unsafe": shap_breakdown,
            "Human_Explanation": exp_result["human_explanation"].replace("\n", " | "),
        })

    local_df = pd.DataFrame(local_rows)
    local_csv_path = REPORTS_DIR / "local_explanations.csv"
    local_df.to_csv(local_csv_path, index=False)
    print(f"Saved: {local_csv_path}")

    # -----------------------------------------------------------------------
    # Artifact 6: Decision Boundary Sensitivity Experiment
    # -----------------------------------------------------------------------
    print("\n--- Running Decision Boundary Sensitivity Experiments ---")
    sensitivity_rows = []

    # Experiment A: Sweep TDS across primary threshold 211.05 ppm
    # Base: pH=7.20, Turbidity=2.00, Temp=22.00
    tds_sweep = np.round(np.arange(205.0, 216.5, 0.5), 2)
    for val in tds_sweep:
        s = pd.DataFrame([[7.20, val, 2.00, 22.00]], columns=FEATURE_NAMES)
        path_info = extract_decision_path(dt, s)
        rf_p = rf.predict_proba(s)[0]
        sensitivity_rows.append({
            "Experiment": "TDS_Boundary_Sweep",
            "Variable_Feature": "TDS_ppm",
            "Variable_Value": val,
            "pH": 7.20,
            "TDS_ppm": val,
            "Turbidity_NTU": 2.00,
            "Temperature_C": 22.00,
            "DT_Prediction": path_info["predicted_class"],
            "DT_Prob_Safe": round(path_info["probabilities"]["Safe"], 4),
            "DT_Prob_Unsafe": round(path_info["probabilities"]["Unsafe"], 4),
            "DT_Leaf_ID": path_info["leaf_node_id"],
            "RF_Prediction": CLASS_LABELS[int(rf.predict(s)[0])],
            "RF_Prob_Safe": round(float(rf_p[0]), 4),
            "RF_Prob_Unsafe": round(float(rf_p[1]), 4),
        })

    # Experiment B: Sweep Turbidity across threshold 4.49 NTU
    # Base: pH=7.20, TDS=150.00, Temp=22.00
    turb_sweep = np.round(np.arange(4.20, 4.72, 0.02), 2)
    for val in turb_sweep:
        s = pd.DataFrame([[7.20, 150.00, val, 22.00]], columns=FEATURE_NAMES)
        path_info = extract_decision_path(dt, s)
        rf_p = rf.predict_proba(s)[0]
        sensitivity_rows.append({
            "Experiment": "Turbidity_Boundary_Sweep",
            "Variable_Feature": "Turbidity_NTU",
            "Variable_Value": val,
            "pH": 7.20,
            "TDS_ppm": 150.00,
            "Turbidity_NTU": val,
            "Temperature_C": 22.00,
            "DT_Prediction": path_info["predicted_class"],
            "DT_Prob_Safe": round(path_info["probabilities"]["Safe"], 4),
            "DT_Prob_Unsafe": round(path_info["probabilities"]["Unsafe"], 4),
            "DT_Leaf_ID": path_info["leaf_node_id"],
            "RF_Prediction": CLASS_LABELS[int(rf.predict(s)[0])],
            "RF_Prob_Safe": round(float(rf_p[0]), 4),
            "RF_Prob_Unsafe": round(float(rf_p[1]), 4),
        })

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_csv_path = REPORTS_DIR / "decision_boundary_sensitivity.csv"
    sens_df.to_csv(sens_csv_path, index=False)
    print(f"Saved: {sens_csv_path}")

    # Plot Sensitivity Figures
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

    # Subplot 1: TDS Sweep
    tds_data = sens_df[sens_df["Experiment"] == "TDS_Boundary_Sweep"]
    axes[0].plot(tds_data["Variable_Value"], tds_data["DT_Prob_Unsafe"], "o-", color=COLOR_UNSAFE, label="Decision Tree P(Unsafe)", linewidth=2.5, markersize=5)
    axes[0].plot(tds_data["Variable_Value"], tds_data["RF_Prob_Unsafe"], "s--", color="#8e44ad", label="Random Forest P(Unsafe)", linewidth=2, markersize=4)
    axes[0].axvline(211.05, color="#e67e22", linestyle=":", linewidth=2, label="DT Threshold (211.05 ppm)")
    axes[0].set_xlabel("TDS (ppm) [pH=7.20, Turbidity=2.0 NTU, Temp=22.0°C]", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Predicted Probability P(Unsafe)", fontsize=10, fontweight="bold")
    axes[0].set_title("Decision Boundary Sensitivity: TDS Sweep (205 to 216 ppm)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper left", frameon=True)

    # Subplot 2: Turbidity Sweep
    turb_data = sens_df[sens_df["Experiment"] == "Turbidity_Boundary_Sweep"]
    axes[1].plot(turb_data["Variable_Value"], turb_data["DT_Prob_Unsafe"], "o-", color=COLOR_UNSAFE, label="Decision Tree P(Unsafe)", linewidth=2.5, markersize=5)
    axes[1].plot(turb_data["Variable_Value"], turb_data["RF_Prob_Unsafe"], "s--", color="#8e44ad", label="Random Forest P(Unsafe)", linewidth=2, markersize=4)
    axes[1].axvline(4.49, color="#e67e22", linestyle=":", linewidth=2, label="DT Split Threshold (4.49 NTU)")
    axes[1].set_xlabel("Turbidity (NTU) [pH=7.20, TDS=150.0 ppm, Temp=22.0°C]", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Predicted Probability P(Unsafe)", fontsize=10, fontweight="bold")
    axes[1].set_title("Decision Boundary Sensitivity: Turbidity Sweep (4.2 to 4.7 NTU)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper left", frameon=True)

    plt.tight_layout()
    sens_fig_path = FIGURES_DIR / "decision_boundary_sensitivity.png"
    fig.savefig(sens_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {sens_fig_path}")

    print("\n" + "=" * 80)
    print("ALL PHASE 5 ARTIFACTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase5_generation()
