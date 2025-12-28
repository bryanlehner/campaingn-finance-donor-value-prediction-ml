# Donor Value Prediction (Per‑Donor Total Contributions)

Predict **per‑donor total contribution value** from transaction‑level contribution records using **feature engineering**, **machine learning**, and **model explainability** (permutation feature importance).

> Portfolio note: This repo is structured to be readable by hiring teams—clear workflow, reproducible setup, and interpretable results.

---

## What this project does

**Goal:** Estimate a donor’s **total contributions** (e.g., per cycle / dataset period) from their observed contribution records.

**High-level approach**
1. Load and clean transaction data (contributions).
2. Build a stable `donor_id` (e.g., name + address + ZIP; adjustable).
3. Aggregate transactions into a **donor-level table** (totals + behavioral features).
4. Train models to predict **`log(1 + donor_total)`** to handle heavy-tailed donation distributions.
5. Explain results with **permutation feature importance** and diagnostic charts.

---

## Repository contents

- `Predict_Contribution_DonorTotal_Insights_with_charts.ipynb`  
  End‑to‑end notebook: cleaning → aggregation → modeling → explainability → visual insights
- `contributions.csv` *(optional)*  
  Sample dataset used by the notebook (remove if sensitive)

---

## Quickstart

### 1) Create a Python environment
```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the notebook
```bash
jupyter notebook
```
Open: `Predict_Contribution_DonorTotal_Insights_with_charts.ipynb`

---

## Methodology

### Target
- **Donor total** = sum of positive contribution amounts (`AMNT > 0`) per donor.
- Model target = **`log1p(donor_total)`** to reduce skew and stabilize training.

### Donor-level features (examples)
- **Behavioral:** number of transactions, active span in days, number of unique recipients/committees
- **Donor attributes:** occupation/job category, geography fields
- **Context:** office / recipient category (where available)
- **Payment signals:** pay method, diversity of payment methods

### Why log(1 + total)?
Contribution totals are typically **heavy‑tailed** (many small donors, few large donors). Modeling on log scale improves:
- stability
- robustness to outliers
- interpretability of errors for the majority of donors

---

## Explainability: Feature Importance

This repo uses **permutation importance** on the fitted pipeline to identify which features most influence predictions.

Why permutation importance?
- Works with many model types
- Measures importance **on the final model** (not just raw coefficients)
- Helps detect **leaky features** that artificially inflate performance

---

## Visual Insights

The notebook includes plots designed to answer practical questions like:
- How skewed are transaction amounts and donor totals?
- Which occupations/jobs associate with higher totals?
- How do repeat behavior signals (transaction count, span) relate to total value?
- Which features drive the model predictions most?

> Add screenshots/GIFs here for quick scanning by reviewers:
>
> - `docs/img/donor_total_distribution.png`
> - `docs/img/feature_importance.png`

---

## Notes on data leakage (important)

Some features can “leak” future information (e.g., cumulative-to-date amounts or variables derived from the target period).  
If you want a **first-touch** scorer (“predict future value from the first donation”), restrict features to what is known at the time of scoring.

The notebook is structured to let you:
- run a **full-history** model (best performance, less realistic for early scoring), and/or
- run a **first-touch/early-signal** model (more realistic for production scoring)

---

## Roadmap ideas

- Two-stage modeling: classify **“whale vs normal”** donors, then regress totals within segments
- Time-horizon prediction: “total in next 6/12 months” or “this cycle” (more actionable than lifetime)
- Add SHAP explanations for local interpretability (optional)

---

## Reproducibility

- Random seeds are set in the modeling section where applicable.
- Sampling is used for expensive plots/models to keep the notebook responsive on large datasets.

---

## License

Choose a license before publishing (MIT is common for portfolio repos).

---

## Author

**Bryan Lehner**  
If you’re reviewing this for a role, feel free to reach out with questions or suggestions.
