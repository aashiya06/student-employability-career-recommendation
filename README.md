# 🎓 Student Employability Prediction and Career Recommendation System Using Machine Learning

An end-to-end Machine Learning project developed as part of the **YuvaIntern Internship**, designed to predict undergraduate employability status and deliver transparent, explainable career pathway recommendations.

---

## 📌 Project Overview

Traditional placement prediction systems primarily rely on raw academic scores (such as GPA). This system incorporates modern employability dimensions—including programming abilities, software tool proficiency, practical project experience, internships, certifications, soft skills, and industry awareness.

The project follows a rigorous, leak-free machine learning pipeline centered on **Logistic Regression**, enhanced by systematic hyperparameter optimization (**Grid Search**, **Random Search**, and **Bayesian Optimization via Optuna**), and paired with a rule-based **Career Recommendation Engine** and an interactive **Streamlit** dashboard.

---

## 🔄 Documented 11-Stage Technical Workflow

```text
Data Preparation
      ↓
Preprocessing (Imputation & Scaling)
      ↓
Exploratory Data Analysis (EDA)
      ↓
Feature Engineering (Composite Competencies)
      ↓
Logistic Regression (Primary Model)
      ↓
Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
      ↓
5-Fold Stratified Cross Validation
      ↓
Grid Search CV
      ↓
Random Search CV
      ↓
Bayesian Optimization (Optuna)
      ↓
Employability Prediction & Confidence Scoring
      ↓
Career Recommendation & Skill-Gap Analysis
```

---

## 📊 Dataset Specification

* **Dataset Name**: Undergraduate Employability Dataset (`UECD_1200_dataset.csv`)
* **Source**: Public Kaggle Undergraduate Employability Dataset by `programmer3`
* **Raw Records**: 1,200 rows × 26 columns
* **Target Labeled Records**: 620 non-null instances for `Employment_Competitiveness_Score`
* **Target Classification Logic**:
  $$\text{Employable} = \begin{cases} 1 & \text{if } \text{Employment\_Competitiveness\_Score} \ge 60 \\ 0 & \text{if } \text{Employment\_Competitiveness\_Score} < 60 \end{cases}$$
* **Class Balance**: 488 Class 0 (78.7% Needs Improvement), 132 Class 1 (21.3% Employable)

---

## 🛠️ Data Preprocessing & Feature Engineering

### 1. Leak-Free Preprocessing Pipeline
To prevent data leakage:
* **Train/Test Splitting**: Stratified 80/20 split executed **before** fitting scalers and imputers.
* **Missing Value Imputation**: `SimpleImputer(strategy='median')` fit strictly on training folds.
* **Feature Standardization**: `StandardScaler()` fit strictly on training folds.
* **Duplicate Detection**: Verified and audited dataset integrity.

### 2. Domain Composite Feature Engineering
Three mathematically bounded composite indicators (0–100 scale) were created without target leakage:
1. **Technical Readiness Index ($TRI$)**:
   $$TRI = \frac{\text{Programming\_Skill} + \text{Domain\_Knowledge} + \text{Tool\_Proficiency}}{3.0}$$
2. **Soft Skill Score ($SSS$)**:
   $$SSS = \frac{\text{Communication\_Skill} + \text{Teamwork\_Ability} + \text{Leadership\_Potential} + \text{Adaptability}}{4.0}$$
3. **Practical Exposure Score ($PES$)**:
   $$PES = \left(\frac{\min(\text{Project\_Experience}, 7)}{7} \times 30\right) + \left(\frac{\min(\text{Internship\_Duration}, 18)}{18} \times 30\right) + \left(\frac{\min(\text{Certification\_Count}, 5)}{5} \times 20\right) + (\text{Industry\_Relevance} \times 20)$$

---

## 🔬 Experimental Evaluation & Optimization Results

All models were evaluated on the exact same stratified held-out test set ($N=124$) and 5-fold cross-validation folds.

| Optimization Strategy | Selected Hyperparameters | 5-Fold CV F1 Score | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | $C=1.0$, `penalty='l2'`, `solver='liblinear'` | 0.8852 (±0.0165) | 96.77% | 86.67% | 100.0% | 0.9286 | 0.9957 |
| **Grid Search CV** | $C=10.0$, `penalty='l1'`, `solver='saga'` | **0.9492 (±0.0176)** | **98.39%** | **96.15%** | **96.15%** | **0.9615** | **0.9992** |
| **Random Search CV** | $C=4.4062$, `penalty='l1'`, `solver='saga'` | 0.9454 (±0.0185) | 99.19% | 96.30% | 100.0% | 0.9811 | 0.9992 |
| **Optuna (Bayesian Optimization)** | $C=8.4310$, `penalty='l1'`, `solver='saga'` | **0.9492 (±0.0176)** | **98.39%** | **96.15%** | **96.15%** | **0.9615** | **0.9992** |

---

## 🧭 Career Recommendation Methodology

The career recommendation engine is an explainable, transparent role-matching module distinct from the binary employability classifier. It evaluates student skill profiles against industry competency vectors across 6 key career domains:

1. **Software Developer / Engineer**: Evaluates Programming, Tool Proficiency, Modular Project Experience, Core CS Concepts, and Collaborative Coding.
2. **Data Analyst / Data Scientist**: Evaluates Domain Knowledge, Quantitative Foundations, Data Programming (SQL/Python), BI Tools, and Data Storytelling.
3. **AI / ML Engineer**: Evaluates ML Programming, Mathematical Foundations, MLOps Tools, Applied ML Projects, and Research Experimentation.
4. **Cloud / DevOps Engineer**: Evaluates Cloud & Automation Tools, Hands-on Infrastructure Exposure, Scripting, System Troubleshooting, and Industry Certifications.
5. **Business Analyst**: Evaluates Stakeholder Communication, Market Awareness, Functional Domain Knowledge, Strategic Problem Solving, and Cross-Functional Collaboration.
6. **Project / Product Management**: Evaluates Leadership Potential, Executive Communication, Agile Teamwork, Strategic Roadmapping, and Change Management.

**Output**: Returns the **Top 3 matching career tracks**, each with:
* Percentage suitability score ($0 - 100\%$)
* Identified skill gaps (deficits relative to role benchmarks)
* Actionable improvement suggestions and milestones

---

## 🖥️ Streamlit Web Application

The interactive web application (`app.py`) provides:
* **Tab 1 — Employability Assessment**: Real-time student profile input via sliders, binary prediction (`Employable` vs `Needs Targeted Improvement`), confidence percentage, composite index cards ($TRI, SSS, PES$), and profile drivers (strengths, opportunities, next actions).
* **Tab 2 — Career Recommendations**: Ranks top 3 matching roles, displays competency breakdowns versus benchmarks, and highlights personalized skill-gap roadmaps.
* **Tab 3 — Model Analytics & Optimization**: Displays comparative performance tables, confusion matrix heatmap, ROC curves, cross-validation distributions, and standardized feature coefficients.

---

## 📂 Project Structure

```plaintext
AI-Employability-Predictor/
├── assets/                                      # Exported evaluation visualizations
│   ├── confusion_matrix.png
│   ├── cv_comparison.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── roc_curve.png
├── data/                                        # Dataset directory
│   └── UECD_1200_dataset.csv
├── model/                                       # Trained pipeline and metadata
│   ├── employability_model.pkl
│   └── model_metadata.json
├── notebooks/                                   # Jupyter notebooks
│   ├── Employability_Prediction_model_.ipynb    # Original exploratory notebook
│   └── Student_Employability_Internship_Workflow.ipynb # Complete executed workflow
├── src/                                         # Modular project codebase
│   ├── __init__.py
│   ├── career_recommender.py                    # Role matching & skill gap engine
│   ├── preprocessing.py                         # Data loading & feature engineering
│   └── train.py                                 # Training, tuning & visualization script
├── .streamlit/
│   └── config.toml
├── app.py                                       # Streamlit web application
├── README.md                                    # Comprehensive technical documentation
└── requirements.txt                             # Production dependencies
```

---

## ▶️ Execution Instructions

### 1. Set Up Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the End-to-End Training Pipeline

```bash
python src/train.py
```

### 4. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

---

## 📜 Attribution & License

* **Base Dataset**: [Undergraduate Employability Dataset](https://www.kaggle.com/datasets/programmer3/undergraduate-employability-dataset) published by `programmer3` on Kaggle.
* **Base Project Origin**: Adapted from the open-source repository by Alok Ranjan ([alokrj01](https://github.com/alokrj01/AI-Employability-Predictor)), re-engineered and extended for the **YuvaIntern Machine Learning Internship** with leak-free pipelines, feature engineering, Grid/Random/Optuna optimization, and explainable career recommendations.
