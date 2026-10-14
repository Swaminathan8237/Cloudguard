---
trigger: always_on
---

Always follow the project's `AGENTS.md` instructions. Work only on the phase requested by the user, inspect existing files before changing them, and teach the user while implementing.

For every substantive phase, create a fully documented, executable Jupyter Notebook in `notebooks/` and a detailed Markdown handoff report in `reports/`. Include actual results, filenames, limitations, and next steps. Never invent metrics or claim code was executed unless it was tested.

Use `data/water_quality_dataset1.csv` as the authoritative dataset. Its five columns are `pH`, `TDS_ppm`, `Turbidity_NTU`, `Temperature_C`, and `Label`. It contains no timestamp or source identifier. Never infer time from row order or manufacture temporal features or future predictions.

Use Logistic Regression, Decision Tree, and Random Forest as the basic baseline models. Preserve appropriate train/test separation, evaluate both classes, and check for data leakage or suspiciously easy classification before making claims about model performance.

The eventual goal is a working ESP32 sensor circuit connected to a cloud-hosted ML prediction service and dashboard. Do not claim hardware or cloud integration works until it has been tested. Complete the requested phase, report the results, and stop for user approval before proceeding.