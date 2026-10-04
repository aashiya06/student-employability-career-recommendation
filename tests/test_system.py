"""
Unit and integration tests for the Student Employability Prediction
and Career Recommendation System.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import prepare_data, engineer_features, load_and_clean_data
from src.career_recommender import evaluate_career_recommendations, ROLE_PROFILES


def test_dataset_loading_and_cleaning():
    df, feature_cols, initial_shape, duplicates = load_and_clean_data("data/UECD_1200_dataset.csv")
    assert initial_shape[0] == 1200
    assert initial_shape[1] == 26
    assert df.shape[0] == 620
    assert "Employable" in df.columns
    assert "Technical_Readiness_Index" in df.columns
    assert "Soft_Skill_Score" in df.columns
    assert "Practical_Exposure_Score" in df.columns
    assert len(feature_cols) == 28


def test_model_loading_and_artifacts():
    assert os.path.exists("model/employability_model.pkl"), "Trained model pkl not found"
    assert os.path.exists("model/model_metadata.json"), "Metadata json not found"

    model = joblib.load("model/employability_model.pkl")
    with open("model/model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "feature_names" in meta
    assert len(meta["feature_names"]) == 28
    assert meta["best_model_metrics"]["test_accuracy"] >= 0.90
    assert meta["best_model_metrics"]["test_f1"] >= 0.90


def test_employability_prediction_flow():
    model = joblib.load("model/employability_model.pkl")
    with open("model/model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # High-performing profile
    high_student = {
        'GPA': 8.8, 'Core_Subject_Score': 85, 'Academic_Consistency': 1.1,
        'Research_Participation': 1.0, 'Programming_Skill': 90, 'Domain_Knowledge': 85,
        'Tool_Proficiency': 85, 'Certification_Count': 4, 'Internship_Count': 2,
        'Internship_Duration': 12, 'Industry_Relevance': 0.9, 'Project_Experience': 6,
        'Communication_Skill': 80, 'Teamwork_Ability': 85, 'Leadership_Potential': 80,
        'Adaptability': 85, 'Learning_Motivation': 90, 'Career_Clarity': 85,
        'Market_Awareness': 80, 'Higher_Study_Interest': 0.0, 'Target_Industry_Code': 1.0,
        'Workshop_Participation': 6, 'Online_Course_Hours': 180, 'Networking_Activity': 80,
        'Learning_Path_Generation_Time': 6.0
    }
    df_high = engineer_features(pd.DataFrame([high_student]))[meta["feature_names"]]
    pred_high = model.predict(df_high)[0]
    prob_high = model.predict_proba(df_high)[0][1]

    assert pred_high == 1
    assert prob_high > 0.80

    # Low-performing profile
    low_student = {
        'GPA': 5.8, 'Core_Subject_Score': 55, 'Academic_Consistency': 2.2,
        'Research_Participation': 0.0, 'Programming_Skill': 42, 'Domain_Knowledge': 45,
        'Tool_Proficiency': 35, 'Certification_Count': 0, 'Internship_Count': 0,
        'Internship_Duration': 0, 'Industry_Relevance': 0.2, 'Project_Experience': 1,
        'Communication_Skill': 45, 'Teamwork_Ability': 50, 'Leadership_Potential': 40,
        'Adaptability': 45, 'Learning_Motivation': 50, 'Career_Clarity': 40,
        'Market_Awareness': 40, 'Higher_Study_Interest': 0.0, 'Target_Industry_Code': 6.0,
        'Workshop_Participation': 0, 'Online_Course_Hours': 20, 'Networking_Activity': 30,
        'Learning_Path_Generation_Time': 6.0
    }
    df_low = engineer_features(pd.DataFrame([low_student]))[meta["feature_names"]]
    pred_low = model.predict(df_low)[0]
    prob_low = model.predict_proba(df_low)[0][1]

    assert pred_low == 0
    assert prob_low < 0.20


def test_missing_and_imputed_inputs():
    model = joblib.load("model/employability_model.pkl")
    with open("model/model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Input with NaN fields
    partial_student = {
        'GPA': np.nan, 'Core_Subject_Score': 75, 'Academic_Consistency': 1.4,
        'Research_Participation': 0.0, 'Programming_Skill': np.nan, 'Domain_Knowledge': 70,
        'Tool_Proficiency': 68, 'Certification_Count': 2, 'Internship_Count': 1,
        'Internship_Duration': 6, 'Industry_Relevance': 0.7, 'Project_Experience': np.nan,
        'Communication_Skill': 68, 'Teamwork_Ability': 72, 'Leadership_Potential': 62,
        'Adaptability': 70, 'Learning_Motivation': 75, 'Career_Clarity': 68,
        'Market_Awareness': 65, 'Higher_Study_Interest': 0.0, 'Target_Industry_Code': 1.0,
        'Workshop_Participation': 4, 'Online_Course_Hours': 120, 'Networking_Activity': 60,
        'Learning_Path_Generation_Time': 6.0
    }
    df_partial = engineer_features(pd.DataFrame([partial_student]))[meta["feature_names"]]
    # Imputer handles NaNs without raising exception
    pred = model.predict(df_partial)
    proba = model.predict_proba(df_partial)
    assert len(pred) == 1
    assert proba.shape == (1, 2)


def test_career_recommendation_module():
    student = {
        'Programming_Skill': 85, 'Tool_Proficiency': 80, 'Domain_Knowledge': 75,
        'Project_Experience': 5, 'Communication_Skill': 65, 'Teamwork_Ability': 70,
        'Core_Subject_Score': 78, 'Research_Participation': 1, 'Internship_Duration': 6,
        'Certification_Count': 2, 'Industry_Relevance': 0.8, 'Adaptability': 70,
        'Market_Awareness': 60, 'Career_Clarity': 65, 'Leadership_Potential': 60
    }

    recommendations = evaluate_career_recommendations(student, top_n=3)
    assert len(recommendations) == 3
    for rec in recommendations:
        assert "role_title" in rec
        assert "suitability_score" in rec
        assert 0.0 <= rec["suitability_score"] <= 100.0
        assert "skill_gaps" in rec
        assert "skill_breakdown" in rec

    # Scores should be sorted descending
    scores = [r["suitability_score"] for r in recommendations]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    test_dataset_loading_and_cleaning()
    test_model_loading_and_artifacts()
    test_employability_prediction_flow()
    test_missing_and_imputed_inputs()
    test_career_recommendation_module()
    print("ALL TESTS COMPLETED AND PASSED!")
