"""
Training and Optimization Script for Student Employability Prediction.
Executes the documented workflow:
1. Baseline Logistic Regression
2. 5-Fold Stratified Cross Validation
3. GridSearchCV
4. RandomizedSearchCV
5. Optuna Bayesian Optimization
6. Metric recording and visualization export
7. Best pipeline serialization and metadata logging
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import optuna

from src.preprocessing import prepare_data, engineer_features

# Filter scikit-learn penalty deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def evaluate_pipeline(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes standard classification evaluation metrics on the held-out test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def run_training_pipeline(data_path: str = "data/UECD_1200_dataset.csv", output_dir: str = "model"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    print("=" * 60)
    print("STEP 1: Data Preparation & Leak-Free Splitting")
    print("=" * 60)
    X_train, X_test, y_train, y_test, feature_names = prepare_data(data_path=data_path, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Class counts in training: Class 0 = {(y_train == 0).sum()}, Class 1 = {(y_train == 1).sum()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---------------------------------------------------------
    # 1. Baseline Logistic Regression Pipeline
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Baseline Logistic Regression Pipeline")
    print("=" * 60)
    baseline_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            penalty='l2',
            C=1.0,
            random_state=42,
            max_iter=5000
        ))
    ])

    baseline_cv_scores = cross_val_score(baseline_pipe, X_train, y_train, cv=cv, scoring='f1')
    baseline_pipe.fit(X_train, y_train)
    baseline_eval = evaluate_pipeline("Baseline Logistic Regression", baseline_pipe, X_test, y_test)
    baseline_eval["cv_f1_mean"] = round(float(baseline_cv_scores.mean()), 4)
    baseline_eval["cv_f1_std"] = round(float(baseline_cv_scores.std()), 4)
    baseline_eval["best_params"] = {"C": 1.0, "penalty": "l2", "solver": "liblinear"}
    print(f"Baseline Test F1: {baseline_eval['f1']} | CV F1: {baseline_eval['cv_f1_mean']} (+/- {baseline_eval['cv_f1_std']})")

    # ---------------------------------------------------------
    # 2. GridSearchCV
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Hyperparameter Optimization — GridSearchCV")
    print("=" * 60)
    param_grid = [
        {
            'classifier__solver': ['liblinear'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]
        },
        {
            'classifier__solver': ['saga'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]
        }
    ]

    grid_base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=5000))
    ])

    grid_search = GridSearchCV(
        grid_base_pipe,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_best_pipe = grid_search.best_estimator_
    grid_eval = evaluate_pipeline("Grid Search CV", grid_best_pipe, X_test, y_test)
    grid_eval["cv_f1_mean"] = round(float(grid_search.best_score_), 4)
    grid_cv_scores = cross_val_score(grid_best_pipe, X_train, y_train, cv=cv, scoring='f1')
    grid_eval["cv_f1_std"] = round(float(grid_cv_scores.std()), 4)
    grid_eval["best_params"] = {
        k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()
    }
    print(f"GridSearch Best Params: {grid_eval['best_params']}")
    print(f"GridSearch Test F1: {grid_eval['f1']} | CV F1: {grid_eval['cv_f1_mean']}")

    # ---------------------------------------------------------
    # 3. RandomizedSearchCV
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Hyperparameter Optimization — RandomizedSearchCV")
    print("=" * 60)
    param_dist = [
        {
            'classifier__solver': ['liblinear'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.logspace(-3, 2, 60)
        },
        {
            'classifier__solver': ['saga'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.logspace(-3, 2, 60)
        }
    ]

    random_search = RandomizedSearchCV(
        grid_base_pipe,
        param_dist,
        n_iter=30,
        cv=cv,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    random_best_pipe = random_search.best_estimator_
    random_eval = evaluate_pipeline("Random Search CV", random_best_pipe, X_test, y_test)
    random_eval["cv_f1_mean"] = round(float(random_search.best_score_), 4)
    random_cv_scores = cross_val_score(random_best_pipe, X_train, y_train, cv=cv, scoring='f1')
    random_eval["cv_f1_std"] = round(float(random_cv_scores.std()), 4)
    random_eval["best_params"] = {
        k.replace("classifier__", ""): (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
        for k, v in random_search.best_params_.items()
    }
    print(f"RandomSearch Best Params: {random_eval['best_params']}")
    print(f"RandomSearch Test F1: {random_eval['f1']} | CV F1: {random_eval['cv_f1_mean']}")

    # ---------------------------------------------------------
    # 4. Bayesian Optimization using Optuna
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Bayesian Optimization — Optuna")
    print("=" * 60)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optuna_objective(trial):
        c = trial.suggest_float('C', 1e-3, 100.0, log=True)
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

        candidate_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=c,
                penalty=penalty,
                solver=solver,
                class_weight='balanced',
                random_state=42,
                max_iter=5000
            ))
        ])

        scores = cross_val_score(candidate_pipe, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(optuna_objective, n_trials=50)

    optuna_best_raw = study.best_params
    optuna_best_params = {
        "C": round(float(optuna_best_raw["C"]), 4),
        "penalty": optuna_best_raw["penalty"],
        "solver": optuna_best_raw["solver"]
    }

    optuna_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=optuna_best_raw["C"],
            penalty=optuna_best_raw["penalty"],
            solver=optuna_best_raw["solver"],
            class_weight='balanced',
            random_state=42,
            max_iter=5000
        ))
    ])
    optuna_pipe.fit(X_train, y_train)
    optuna_eval = evaluate_pipeline("Optuna Bayesian Optimization", optuna_pipe, X_test, y_test)
    optuna_eval["cv_f1_mean"] = round(float(study.best_value), 4)
    optuna_cv_scores = cross_val_score(optuna_pipe, X_train, y_train, cv=cv, scoring='f1')
    optuna_eval["cv_f1_std"] = round(float(optuna_cv_scores.std()), 4)
    optuna_eval["best_params"] = optuna_best_params
    print(f"Optuna Best Params: {optuna_eval['best_params']}")
    print(f"Optuna Test F1: {optuna_eval['f1']} | CV F1: {optuna_eval['cv_f1_mean']}")

    # ---------------------------------------------------------
    # 5. Model Selection & Export
    # ---------------------------------------------------------
    all_models = [baseline_eval, grid_eval, random_eval, optuna_eval]
    pipelines_map = {
        "Baseline Logistic Regression": baseline_pipe,
        "Grid Search CV": grid_best_pipe,
        "Random Search CV": random_best_pipe,
        "Optuna Bayesian Optimization": optuna_pipe
    }

    # Best model selection by CV F1 score
    best_eval = max(all_models, key=lambda x: (x["cv_f1_mean"], x["f1"]))
    best_pipe = pipelines_map[best_eval["model_name"]]
    print("\n" + "=" * 60)
    print(f"SELECTED FINAL MODEL: {best_eval['model_name']}")
    print("=" * 60)

    # Serialize final pipeline
    model_path = os.path.join(output_dir, "employability_model.pkl")
    joblib.dump(best_pipe, model_path)
    print(f"Saved best pipeline to: {model_path}")

    # ---------------------------------------------------------
    # 6. Generate and Save Visualizations
    # ---------------------------------------------------------
    print("\nGenerating evaluation and comparison plots...")

    # Plot 1: Confusion Matrix for Best Model
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, best_eval["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Employable', 'Employable'],
                yticklabels=['Not Employable', 'Employable'])
    plt.title(f"Confusion Matrix — {best_eval['model_name']}", fontsize=12, pad=12)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.savefig("assets/confusion_matrix.png", dpi=300)
    plt.close()

    # Plot 2: ROC Curves Comparison
    plt.figure(figsize=(7, 6))
    for m in all_models:
        fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
        plt.plot(fpr, tpr, label=f"{m['model_name']} (AUC = {m['roc_auc']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    plt.ylabel('True Positive Rate (Recall)', fontsize=10)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.savefig("assets/roc_curve.png", dpi=300)
    plt.close()

    # Plot 3: Metrics Comparison Bar Chart
    comp_df = pd.DataFrame([
        {
            "Model": m["model_name"].replace(" Logistic Regression", "").replace(" Bayesian Optimization", ""),
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1"],
            "ROC-AUC": m["roc_auc"]
        }
        for m in all_models
    ])
    melted = pd.melt(comp_df, id_vars=["Model"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="crest")
    plt.title("Comparative Performance across Optimization Methods", fontsize=12, pad=12)
    plt.ylim([0.7, 1.02])
    plt.ylabel("Metric Score", fontsize=10)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    comp_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.savefig("assets/model_comparison.png", dpi=300)
    plt.close()

    # Plot 4: Feature Importance (Logistic Regression Coefficients)
    lr_model = best_pipe.named_steps['classifier']
    coefs = lr_model.coef_[0]
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs
    }).sort_values(by="Coefficient", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp, x="Coefficient", y="Feature", palette="coolwarm")
    plt.title(f"Logistic Regression Feature Coefficients ({best_eval['model_name']})", fontsize=12, pad=12)
    plt.xlabel("Standardized Logistic Regression Coefficient", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.tight_layout()
    feat_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(feat_path, dpi=300)
    plt.savefig("assets/feature_importance.png", dpi=300)
    plt.close()

    # Plot 5: Cross-Validation Boxplot Comparison
    cv_data = {
        "Baseline": baseline_cv_scores,
        "Grid Search": grid_cv_scores,
        "Random Search": random_cv_scores,
        "Optuna": optuna_cv_scores
    }
    cv_df = pd.DataFrame(cv_data)
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=cv_df, palette="Set2")
    plt.title("5-Fold Cross-Validation F1 Score Distribution", fontsize=12, pad=12)
    plt.ylabel("F1 Score", fontsize=10)
    plt.ylim([0.80, 1.0])
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    cv_box_path = os.path.join(output_dir, "cv_comparison.png")
    plt.savefig(cv_box_path, dpi=300)
    plt.savefig("assets/cv_comparison.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 7. Metadata Serialization
    # ---------------------------------------------------------
    comparison_summary = []
    for m in all_models:
        comparison_summary.append({
            "model_name": m["model_name"],
            "best_params": m["best_params"],
            "cv_f1_mean": m["cv_f1_mean"],
            "cv_f1_std": m["cv_f1_std"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"]
        })

    metadata = {
        "project_title": "Student Employability Prediction and Career Recommendation System Using Machine Learning",
        "training_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "Logistic Regression within Scikit-Learn Pipeline",
        "scikit_learn_version": joblib.__name__,
        "dataset_source": "data/UECD_1200_dataset.csv (1200 rows raw, 620 non-null target ground truth)",
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features_count": int(len(feature_names)),
        "feature_names": feature_names,
        "preprocessing_pipeline": [
            "SimpleImputer(strategy='median')",
            "StandardScaler()"
        ],
        "feature_engineering": [
            "Technical_Readiness_Index = (Programming_Skill + Domain_Knowledge + Tool_Proficiency) / 3.0",
            "Soft_Skill_Score = (Communication_Skill + Teamwork_Ability + Leadership_Potential + Adaptability) / 4.0",
            "Practical_Exposure_Score = (Project_Experience / 7 * 30) + (Internship_Duration / 18 * 30) + (Certification_Count / 5 * 20) + (Industry_Relevance * 20)"
        ],
        "best_model_name": best_eval["model_name"],
        "best_hyperparameters": best_eval["best_params"],
        "best_model_metrics": {
            "cv_f1_mean": best_eval["cv_f1_mean"],
            "cv_f1_std": best_eval["cv_f1_std"],
            "test_accuracy": best_eval["accuracy"],
            "test_precision": best_eval["precision"],
            "test_recall": best_eval["recall"],
            "test_f1": best_eval["f1"],
            "test_roc_auc": best_eval["roc_auc"]
        },
        "all_experiments": comparison_summary
    }

    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {meta_path}")

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
