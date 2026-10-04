"""
Data Loading, Cleaning, Preprocessing, and Feature Engineering Module.
Follows the documented YuvaIntern internship project workflow without data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs meaningful composite features to capture holistic student competencies:
    1. Technical_Readiness_Index: Average of Programming, Domain Knowledge, and Tool Proficiency.
    2. Soft_Skill_Score: Average of Communication, Teamwork, Leadership, and Adaptability.
    3. Practical_Exposure_Score: Bounded composite (0-100) combining Project Experience,
       Internship Duration, Certifications, and Industry Relevance.
    
    All engineered features are strictly computed from input variables without target leakage.
    """
    df = df.copy()

    # 1. Technical Readiness Index (Scale: ~30-100)
    df['Technical_Readiness_Index'] = (
        df['Programming_Skill'] + df['Domain_Knowledge'] + df['Tool_Proficiency']
    ) / 3.0

    # 2. Soft Skill Score (Scale: ~35-100)
    df['Soft_Skill_Score'] = (
        df['Communication_Skill'] + df['Teamwork_Ability'] +
        df['Leadership_Potential'] + df['Adaptability']
    ) / 4.0

    # 3. Practical Exposure Score (Scale: 0-100)
    # Weights: Projects (30%), Internship Duration (30%), Certifications (20%), Industry Relevance (20%)
    df['Practical_Exposure_Score'] = (
        (np.clip(df['Project_Experience'], 0, 7.0) / 7.0 * 30.0) +
        (np.clip(df['Internship_Duration'], 0, 18.0) / 18.0 * 30.0) +
        (np.clip(df['Certification_Count'], 0, 5.0) / 5.0 * 20.0) +
        (np.clip(df['Industry_Relevance'], 0, 1.0) * 20.0)
    )

    return df


def load_and_clean_data(data_path: str = "data/UECD_1200_dataset.csv"):
    """
    Loads raw dataset, audits duplicates, handles missing targets, and constructs the target label.
    """
    df = pd.read_csv(data_path)
    initial_shape = df.shape
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()

    # Drop records missing the target ground truth
    df_clean = df.dropna(subset=['Employment_Competitiveness_Score']).copy()

    # Documented binary classification threshold: Employable = score >= 60
    df_clean['Employable'] = (df_clean['Employment_Competitiveness_Score'] >= 60).astype(int)

    # Apply feature engineering
    df_engineered = engineer_features(df_clean)

    # Separate features and target
    drop_cols = ['Employment_Competitiveness_Score', 'Employable']
    feature_cols = [c for c in df_engineered.columns if c not in drop_cols]

    return df_engineered, feature_cols, initial_shape, duplicate_count


def prepare_data(data_path: str = "data/UECD_1200_dataset.csv", test_size: float = 0.2, random_state: int = 42):
    """
    Performs train-test split before fitting imputers and scalers to prevent data leakage.
    """
    df, feature_cols, _, _ = load_and_clean_data(data_path)

    X = df[feature_cols]
    y = df['Employable']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols


def build_preprocessing_pipeline():
    """
    Builds a leak-free preprocessing pipeline:
    - SimpleImputer with median strategy (fit strictly on training folds/sets)
    - StandardScaler (fit strictly on training folds/sets)
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
