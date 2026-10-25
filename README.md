# Heart Disease Prediction: EDA, Preprocessing & Modeling

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-2E8B57)


An end-to-end machine learning workflow on the **UCI Heart Disease dataset**: exploratory data analysis, missing-value handling, outlier treatment, scaling, encoding, and a baseline **LightGBM** classifier that predicts the severity of heart disease.

Built by **Team Data Dynamics** for the **GDSC UofK AI Bootcamp**, where it ranked **#1** with a score of **9.6** and won the **Best Documentation and Solution** medal.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Key Findings from EDA](#key-findings-from-eda)
- [Model Results](#model-results)
- [Limitations & Notes](#limitations--notes)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Team](#team)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

Heart disease is one of the leading causes of death worldwide, and early detection depends on interpreting many clinical measurements together. This project explores how well routine clinical attributes (age, chest pain type, blood pressure, cholesterol, ECG results, exercise response, and more) can be used to predict the presence and severity of heart disease.

The target variable (`num`, renamed `target`) is **multiclass (0 to 4)**:

| Value | Meaning |
|-------|---------|
| 0 | No heart disease |
| 1 to 4 | Increasing severity of heart disease |

The notebook covers:

1. Loading and inspecting the data
2. Exploratory data analysis and visualization
3. Handling missing values and outliers
4. Feature scaling and categorical encoding
5. Training and evaluating a baseline LightGBM model

---

## Dataset

- **Source:** [Heart Disease Data on Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?select=heart_disease_uci.csv) (derived from the UCI Machine Learning Repository)
- **File:** `heart_disease_uci.csv`
- **Size:** 920 patients, 16 columns
- **Origin:** Four clinical sites: Cleveland (33.0%), Hungary (31.8%), VA Long Beach (21.7%), Switzerland (13.4%)
- **Gender split:** 78.9% male, 21.1% female

### Features

| Column | Description | Type |
|--------|-------------|------|
| `id` | Unique patient identifier | Numerical |
| `age` | Age in years | Numerical |
| `sex` | Male / Female | Categorical |
| `dataset` | Site where the data was collected | Categorical |
| `cp` | Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Boolean |
| `restecg` | Resting ECG results (normal, ST-T abnormality, LV hypertrophy) | Categorical |
| `thalch` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise-induced angina | Boolean |
| `oldpeak` | ST depression induced by exercise relative to rest | Numerical |
| `slope` | Slope of the peak exercise ST segment (upsloping, flat, downsloping) | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy (0 to 3) | Numerical |
| `thal` | Thalassemia (normal, fixed defect, reversible defect) | Categorical |
| `num` | **Target**: heart disease severity (0 to 4) | Numerical |

For readability, the notebook renames columns (for example `trestbps` to `resting_blood_pressure`, `chol` to `serum_cholesterol`, `thalch` to `max_heart_rate`, `oldpeak` to `st_depression`, `num` to `target`).

---

## Workflow

### 1. Exploratory Data Analysis
- Inspected structure with `df.head()`, `df.info()`, and `describe()`; found **2 duplicate rows**
- Pair plots of the numerical features
- Count plots of categorical features split by sex
- Pie charts for gender distribution and data collection site
- Sunburst chart (sex, site) colored by target
- Bar plot of target by site and count plot of thalassemia by site
- Correlation heatmap of numerical features

### 2. Missing Values
- **Dropped:** `num_major_vessels` (66.4% missing), too sparse to impute reliably
- **Numerical features** (`resting_blood_pressure`, `serum_cholesterol`, `max_heart_rate`, `st_depression`; 3.3% to 6.7% missing): imputed with `IterativeImputer`
- **Categorical features** (`thalassemia`, `slope_st_segment`, `fasting_blood_sugar`, `exercise_induced_angina`, `resting_ecg_results`): imputed with the most frequent value

### 3. Outliers and Cleaning
- Dropped the `id` column (no predictive value)
- Box plots used to inspect outliers in numerical features
- Removed the single record with `resting_blood_pressure = 0` (physiologically impossible), leaving **919 rows**
- Other extreme values were kept because they may carry meaningful clinical signal

### 4. Scaling and Encoding
- `StandardScaler` applied to: `age`, `resting_blood_pressure`, `serum_cholesterol`, `max_heart_rate`, `st_depression`
- `LabelEncoder` applied to all categorical and boolean columns

### 5. Modeling
- 80/20 train/test split (`random_state=42`): 735 training rows, 184 test rows
- **LightGBM** (`LGBMClassifier`) with default parameters
- Evaluation with a classification report and confusion matrix
- Feature importance plot from the trained model

---

## Key Findings from EDA

- **Strongest correlations with the target** (numerical features): number of major vessels (0.52, later dropped due to missingness), ST depression (0.44), max heart rate (-0.37), and age (0.34).
- **Disease severity varies a lot by site.** Switzerland (mean target about 1.8) and VA Long Beach (about 1.5) show higher severity than Cleveland (about 0.9) and Hungary (about 0.4).
- **The data is heavily male** (78.9%), which limits how well conclusions generalize to women.
- **Cholesterol has many zero values**, a clear sign of missing data recorded as 0 (visible as a spike in the histograms).
- **Thalassemia and ST-slope are sparsely recorded**, and mostly in the Cleveland subset.

---

## Model Results

LightGBM on the held-out test set (184 samples):

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.72 | 0.79 | 0.75 | 75 |
| 1 | 0.55 | 0.53 | 0.54 | 59 |
| 2 | 0.38 | 0.43 | 0.41 | 23 |
| 3 | 0.44 | 0.35 | 0.39 | 23 |
| 4 | 0.00 | 0.00 | 0.00 | 4 |
| **Accuracy** | | | **0.59** | 184 |
| **Macro avg** | 0.42 | 0.42 | 0.42 | 184 |
| **Weighted avg** | 0.57 | 0.59 | 0.58 | 184 |

**Confusion matrix** (rows = true class, columns = predicted class):

```
[[59 11  4  1  0]
 [18 31  8  2  0]
 [ 1  6 10  5  1]
 [ 4  8  2  8  1]
 [ 0  0  2  2  0]]
```

**Interpretation:** the model separates "no disease" from "disease" reasonably well, but struggles to distinguish between the severity levels, especially the rare class 4 (only 4 test samples). This is expected for a small, imbalanced multiclass dataset.

---

## Limitations & Notes

- **Preprocessing before splitting.** Imputation and scaling are fit on the full dataset before the train/test split, which can leak information from the test set. A scikit-learn `Pipeline` fit on the training data only would fix this.
- **No stratified split.** With a rare class 4, a stratified split is preferable.
- **Label encoding of nominal features.** `LabelEncoder` imposes an artificial order on features such as `dataset`, `thalassemia`, and `chest_pain_type`. One-hot encoding is more appropriate for linear and distance-based models.
- **Zero cholesterol values** are kept as-is rather than treated as missing.
- **Heavy imputation on `thalassemia`** (52.8% missing) with the mode may add noise.
- **Class imbalance** is not yet addressed (class weights, SMOTE, or merging severity levels).
- Only LightGBM is trained and evaluated in the notebook; the other imported models and tuning tools are not used.

---

## Getting Started

### Run in Google Colab

1. Open the notebook in Google Colab (upload `ML_Project_Bootcamp.ipynb` or open it directly from GitHub).
2. Upload `heart_disease_uci.csv` to the Colab session.
3. Run all cells in order.

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/Nadaali1232/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn lightgbm xgboost jupyter

# 4. Launch the notebook
jupyter notebook ML_Project_Bootcamp.ipynb
```

> **Note:** the notebook reads the data from `/content/heart_disease_uci.csv` (the Colab path). When running locally, change this to `heart_disease_uci.csv`.

---

## Repository Structure

```
.
├── README.md
├── heart_disease_uci.csv          # Dataset (from Kaggle)
└── Heart_Disease_Prediction.ipynb      # EDA, preprocessing and modeling notebook
```

---

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** scikit-learn, LightGBM, XGBoost
- **Environment:** Google Colab / Jupyter

---

## Team

**Data Dynamics**

| Name | Links |
|------|-------|
| Faisal Ali | [LinkedIn](https://www.linkedin.com/in/bawazirfaisal/) |
| Lina Ahmed | [LinkedIn](https://www.linkedin.com/in/lina-ahmed-alnoosh-0b2085159/) |
| Nada Ali | [LinkedIn](https://linkedin.com/in/nada-ali) |
| Abdallah Mohammed | [LinkedIn](#) |
| Samar | [LinkedIn](#) |

---

## Acknowledgments

- Dataset: [UCI Machine Learning Repository, Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) via the [Kaggle upload by Redwan Karimsony](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- The original data was collected by clinicians at the Cleveland Clinic, the Hungarian Institute of Cardiology, University Hospital Zurich/Basel, and the VA Medical Center, Long Beach.

> **Disclaimer:** this project is for educational purposes only and is not a medical tool. It must not be used for diagnosis or treatment decisions.
