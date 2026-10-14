# CloudGuard-Water — Project Instructions

## 1. Project Identity

Project: **CloudGuard-Water: Intelligent and Explainable Water Quality Classification Using Machine Learning**

Course: Machine Learning for Cloud Services.

Objective: Train and evaluate basic ML models to classify water-quality samples as Safe or Unsafe, provide understandable explanations for predictions, and eventually deploy the trained model as a cloud service connected to a physical IoT prototype.

## 2. Verified Dataset

Authoritative dataset:

`data/water_quality_dataset1.csv`

Verified structure:
- 100,800 rows
- 5 columns
- `pH`
- `TDS_ppm`
- `Turbidity_NTU`
- `Temperature_C`
- `Label`

Target classes:
- `Safe`
- `Unsafe`

Verified distribution:
- Safe: 30,240 samples (30%)
- Unsafe: 70,560 samples (70%)

The CSV has no timestamp or source identifier.

Always inspect the actual dataset before reporting statistics. Do not invent columns, labels, results, measurements, or file properties.

## 3. Critical Dataset Rules

- Never treat CSV row order as chronological time.
- Never fabricate timestamps.
- Do not create `pH_change`, `TDS_change`, rolling statistics, or `unsafe_in_next_20_minutes` as temporal features from this dataset.
- Do not claim that this dataset supports future water-quality forecasting.
- Distinguish original dataset columns from any features genuinely engineered during development.
- Do not remove outliers without a justified experiment.
- Avoid target leakage and train/test contamination.
- Remember that the dataset's Safe/Unsafe label does not independently certify real-world drinking-water safety.

## 4. Machine Learning Requirements

Use basic, explainable supervised ML.

Required baseline models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Preprocessing:
- Use `StandardScaler` inside a scikit-learn Pipeline for Logistic Regression.
- Do not scale features unnecessarily for tree-based models.
- Use a reproducible, stratified train/test split.
- Preserve the test set for final evaluation.
- Fit all preprocessing exclusively on training data.
- Do not apply SMOTE, oversampling, undersampling, or class weighting automatically.

Evaluate using:
- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Weighted F1
- Confusion matrix
- False-positive and false-negative counts

Report Safe and Unsafe metrics separately.

Because Unsafe represents 70% of this dataset, never assume `class_weight='balanced'` specifically prioritizes Unsafe detection. Explain class weighting and decision-threshold changes before using them.

Never fabricate model metrics or claim an experiment was run when it was not.

## 5. Explainability

The project must go beyond returning a class label.

Where supported by the selected model, provide:
- Predicted class
- Class probabilities
- Model feature importance or another appropriate explanation method
- Human-readable interpretation

Clearly distinguish feature importance and association from causation.

Do not hard-code prediction probabilities, feature rankings, or sample outputs.

## 6. Cloud and IoT Roadmap

The eventual system should support this workflow:

Physical sensors → ESP32 → Wi-Fi/cloud ingestion → ML prediction API → prediction explanation → dashboard or alert.

The planned sensor inputs are:
- pH sensor
- TDS sensor
- Turbidity sensor
- Water-temperature sensor

The physical implementation and cloud integration must use real sensor readings when available.

Do not claim hardware integration or live cloud operation is complete until it has been implemented and tested.

Do not automatically actuate valves or other physical equipment based on predictions.

## 7. Jupyter Notebook Requirement

Every substantive project phase must have a numbered Jupyter Notebook under `notebooks/`.

Each notebook must contain:
- A clear title and objective
- Markdown explanations
- Readable, commented Python code
- Actual execution outputs
- Relevant charts and result tables
- Interpretation of findings
- Limitations and next steps

Ensure notebooks run from top to bottom using project-relative paths and existing dependencies.

Do not leave important work only in standalone Python scripts when it should also be documented in a notebook.

## 8. Markdown Documentation Requirement

Every completed phase must produce a detailed Markdown report under `reports/`.

Include:
- Objective and methodology
- Actual files used and created
- Dataset or model details
- Actual results, tables, and metrics
- Key findings
- Known errors, assumptions, and limitations
- Decisions and their reasoning
- Recommended next steps

For each completed phase, create a handoff report that ChatGPT can use to independently review the work without access to the local project folder.

Never fabricate missing details to make a report appear complete.

## 9. Development Practices

- Use the existing project structure; inspect files before editing them.
- Reuse working code and avoid unnecessary duplication.
- Keep code modular and understandable to a college student.
- Use relative paths instead of machine-specific absolute paths.
- Validate model inputs and handle invalid API requests.
- Keep model preprocessing consistent between training and inference.
- Save trained models and their required preprocessing pipelines.
- Update `requirements.txt` when dependencies change.
- Do not overwrite good working results without a reason.
- Do not commit, push, delete, or destructively modify project files without appropriate user authorization.
- Run relevant tests and report genuine errors.

## 10. Work in Phases

Follow the phase requested by the user.

Current roadmap:
1. Dataset inspection — completed
2. Exploratory Data Analysis — completed
3. Baseline ML training — completed
4. Detailed evaluation, validation, and model selection
5. Model explainability
6. IoT circuit and sensor integration
7. Cloud API integration
8. Dashboard and end-to-end testing
9. Final documentation and presentation

Before each phase, inspect the existing reports, notebooks, code, and artifacts relevant to that phase.

Do not silently skip phases or implement later phases without approval.

At the end of each phase:
1. Run the relevant code.
2. Check outputs and generated files.
3. Produce the numbered notebook and detailed Markdown handoff report.
4. Summarize what was completed and any unresolved issues.
5. Stop and wait for the user's instruction.

## 11. Teaching Style

The user is a college student learning ML for Cloud Services.

Explain technical decisions clearly, including:
- What is being done
- Why it is necessary
- How the code works
- How results should be interpreted
- What can be explained during a project viva

Prioritize correctness, reproducibility, interpretability, and a working end-to-end project over unnecessary complexity.