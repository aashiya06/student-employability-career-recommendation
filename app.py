"""
Student Employability Prediction and Career Recommendation System
Interactive Streamlit Application

Workflow:
Data Preparation -> Preprocessing -> EDA -> Feature Engineering ->
Logistic Regression -> Model Evaluation -> Cross Validation ->
Grid Search -> Random Search -> Bayesian Optimization / Optuna ->
Employability Prediction -> Career Recommendation
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.preprocessing import engineer_features
from src.career_recommender import evaluate_career_recommendations, ROLE_PROFILES


# Set page config
st.set_page_config(
    page_title="Student Employability & Career Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional layout
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-employable {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.1rem;
    }
    .badge-improvement {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.1rem;
    }
    .career-card {
        border-left: 4px solid #3B82F6;
        background-color: #F8FAFC;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }
    .footer-text {
        text-align: center;
        color: #6B7280;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Loads the serialized Scikit-learn Pipeline (SimpleImputer + StandardScaler + LogisticRegression)."""
    model_path = os.path.join("model", "employability_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_model_metadata():
    """Loads model evaluation metadata and hyperparameter optimization results."""
    meta_path = os.path.join("model", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


model = load_trained_model()
metadata = load_model_metadata()

# Header Section
st.markdown('<div class="main-header">🎓 Student Employability Prediction & Career Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An End-to-End Machine Learning System based on Logistic Regression, Hyperparameter Optimization, and Explainable Career Matching</div>', unsafe_allow_html=True)

# Application Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Employability Assessment",
    "🧭 Career Recommendations",
    "📊 Model Analytics & Optimization"
])

# -------------------------------------------------------------
# SIDEBAR / SHARED INPUT CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.header("📋 Student Profile Inputs")
    st.caption("Adjust student metrics below to evaluate employability and career pathways.")

    with st.expander("📚 Academic Foundations", expanded=True):
        gpa = st.slider("GPA (Grade Point Average)", min_value=5.0, max_value=10.0, value=7.8, step=0.1)
        core_subject_score = st.slider("Core Subject Score (Marks)", min_value=50, max_value=100, value=75, step=1)
        academic_consistency = st.slider("Academic Consistency (Variance)", min_value=0.5, max_value=2.5, value=1.4, step=0.1,
                                         help="Lower variance indicates higher academic consistency across semesters.")
        research_participation = st.checkbox("Research Project Participation", value=False)
        higher_study = st.checkbox("Interested in Higher Studies", value=False)

    with st.expander("💻 Technical & Practical Skills", expanded=True):
        programming_skill = st.slider("Programming Skill (0-100)", 30, 100, 72)
        domain_knowledge = st.slider("Domain Knowledge (0-100)", 30, 100, 70)
        tool_proficiency = st.slider("Tool Proficiency (0-100)", 30, 100, 68)
        project_experience = st.slider("Completed Projects Count", 0, 7, 4)
        online_course_hours = st.slider("Online Course / Self-Study Hours", 0, 250, 120)

    with st.expander("💼 Industry Exposure", expanded=False):
        internship_count = st.slider("Internship Count", 0, 4, 1)
        internship_duration = st.slider("Total Internship Duration (Months)", 0, 18, 6)
        certification_count = st.slider("Certifications Count", 0, 5, 2)
        industry_relevance = st.slider("Industry Alignment Relevance", 0.0, 1.0, 0.7, step=0.05,
                                       help="How closely current coursework and projects align with industry standards.")
        workshop_participation = st.slider("Workshops Attended", 0, 10, 4)

    with st.expander("🤝 Soft Skills & Career Readiness", expanded=False):
        communication_skill = st.slider("Communication Skill (0-100)", 30, 100, 68)
        teamwork_ability = st.slider("Teamwork Ability (0-100)", 30, 100, 72)
        leadership_potential = st.slider("Leadership Potential (0-100)", 30, 100, 62)
        adaptability = st.slider("Adaptability (0-100)", 30, 100, 70)
        learning_motivation = st.slider("Learning Motivation (0-100)", 30, 100, 75)
        career_clarity = st.slider("Career Clarity (0-100)", 30, 100, 68)
        market_awareness = st.slider("Market Awareness (0-100)", 30, 100, 65)
        networking_activity = st.slider("Networking Activity (0-100)", 20, 100, 60)
        industry_map = {
            "IT / Software": 1, "Finance & FinTech": 2, "Healthcare": 3,
            "Manufacturing": 4, "Consulting": 5, "Other": 6
        }
        target_industry = st.selectbox("Target Industry Domain", list(industry_map.keys()))
        target_industry_code = industry_map[target_industry]

# Prepare feature dictionary
student_inputs = {
    'GPA': gpa,
    'Core_Subject_Score': core_subject_score,
    'Academic_Consistency': academic_consistency,
    'Research_Participation': 1.0 if research_participation else 0.0,
    'Programming_Skill': programming_skill,
    'Domain_Knowledge': domain_knowledge,
    'Tool_Proficiency': tool_proficiency,
    'Certification_Count': certification_count,
    'Internship_Count': internship_count,
    'Internship_Duration': internship_duration,
    'Industry_Relevance': industry_relevance,
    'Project_Experience': project_experience,
    'Communication_Skill': communication_skill,
    'Teamwork_Ability': teamwork_ability,
    'Leadership_Potential': leadership_potential,
    'Adaptability': adaptability,
    'Learning_Motivation': learning_motivation,
    'Career_Clarity': career_clarity,
    'Market_Awareness': market_awareness,
    'Higher_Study_Interest': 1.0 if higher_study else 0.0,
    'Target_Industry_Code': float(target_industry_code),
    'Workshop_Participation': workshop_participation,
    'Online_Course_Hours': online_course_hours,
    'Networking_Activity': networking_activity,
    'Learning_Path_Generation_Time': 6.0  # Dataset median benchmark
}

# Construct DataFrame and apply Feature Engineering
raw_input_df = pd.DataFrame([student_inputs])
engineered_input_df = engineer_features(raw_input_df)

# Align with training feature order if metadata available
if metadata and "feature_names" in metadata:
    feature_cols = metadata["feature_names"]
    input_data = engineered_input_df[feature_cols]
else:
    input_data = engineered_input_df


# -------------------------------------------------------------
# TAB 1: EMPLOYABILITY ASSESSMENT
# -------------------------------------------------------------
with tab1:
    st.subheader("🎯 Student Employability Prediction")
    st.write("Calculates employability status and probability using the optimized Logistic Regression classification pipeline.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        if model is not None:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            prob_employable = probabilities[1]
            confidence_pct = prob_employable * 100.0

            st.write("### Assessment Result")
            if prediction == 1:
                st.markdown(
                    f'<div class="badge-employable">✅ Employable Candidate ({confidence_pct:.1f}% Confidence)</div>',
                    unsafe_allow_html=True
                )
                st.write("")
                st.success(
                    f"The model predicts high employability competitiveness based on the student's profile "
                    f"with a **{confidence_pct:.1f}%** probability of securing placement."
                )
            else:
                prob_unemployable = (1.0 - prob_employable) * 100.0
                st.markdown(
                    f'<div class="badge-improvement">⚠️ Needs Targeted Improvement ({prob_unemployable:.1f}% Confidence)</div>',
                    unsafe_allow_html=True
                )
                st.write("")
                st.warning(
                    f"The model indicates this student is currently in the **Needs Improvement** category. "
                    f"Targeting key skill gaps and practical exposure will significantly boost employability."
                )

            st.write("#### Employability Probability Score")
            st.progress(float(prob_employable))
            st.caption(f"Estimated Probability: {confidence_pct:.2f}% | Decision Threshold: 50.0%")

        else:
            st.error("Trained model not found at `model/employability_model.pkl`. Please execute `src/train.py`.")

    with col2:
        st.write("### 📐 Composite Feature Indicators")
        tri = engineered_input_df['Technical_Readiness_Index'].iloc[0]
        sss = engineered_input_df['Soft_Skill_Score'].iloc[0]
        pes = engineered_input_df['Practical_Exposure_Score'].iloc[0]

        st.metric(
            label="Technical Readiness Index",
            value=f"{tri:.1f} / 100",
            delta=f"{tri - 68.3:+.1f} vs Avg",
            help="Average of Programming Skill, Domain Knowledge, and Tool Proficiency."
        )
        st.metric(
            label="Soft Skill Score",
            value=f"{sss:.1f} / 100",
            delta=f"{sss - 68.3:+.1f} vs Avg",
            help="Average of Communication, Teamwork, Leadership, and Adaptability."
        )
        st.metric(
            label="Practical Exposure Score",
            value=f"{pes:.1f} / 100",
            delta=f"{pes - 50.0:+.1f} vs Avg",
            help="Composite combining Projects, Internship Duration, Certifications, and Industry Relevance."
        )

    st.divider()

    st.write("### 🔍 Key Profile Driver Analysis")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Top Strengths**")
        strengths = []
        if programming_skill >= 75: strengths.append(f"Strong Programming Foundation ({programming_skill}/100)")
        if gpa >= 8.0: strengths.append(f"Strong Academic Record (GPA: {gpa:.2f})")
        if internship_duration >= 6: strengths.append(f"Solid Internship Experience ({internship_duration} mos)")
        if project_experience >= 4: strengths.append(f"Substantial Project Portfolio ({project_experience} projects)")
        if communication_skill >= 70: strengths.append(f"Effective Communication ({communication_skill}/100)")
        if not strengths: strengths.append("Consistent baseline across standard indicators.")
        for s in strengths:
            st.markdown(f"- ✅ {s}")

    with c2:
        st.write("**Primary Growth Opportunities**")
        weaknesses = []
        if programming_skill < 65: weaknesses.append("Programming proficiency below industry threshold")
        if internship_count == 0: weaknesses.append("No recorded internship experience")
        if project_experience < 3: weaknesses.append("Limited practical project work")
        if tool_proficiency < 65: weaknesses.append("Tool & developer environment proficiency can be improved")
        if communication_skill < 60: weaknesses.append("Soft skill and interview communication needs practice")
        if not weaknesses: weaknesses.append("Well-rounded profile across all core evaluation metrics.")
        for w in weaknesses:
            st.markdown(f"- ⚠️ {w}")

    with c3:
        st.write("**Recommended Next Actions**")
        actions = []
        if internship_count == 0: actions.append("Target a 2-3 month internship or capstone industry project.")
        if programming_skill < 70: actions.append("Practice daily problem solving on LeetCode/HackerRank.")
        if certification_count < 2: actions.append("Complete 1-2 recognized certifications in your target domain.")
        if not actions: actions.append("Focus on advanced system design and mock interview preparation.")
        for a in actions:
            st.markdown(f"- 📌 {a}")


# -------------------------------------------------------------
# TAB 2: CAREER RECOMMENDATIONS
# -------------------------------------------------------------
with tab2:
    st.subheader("🧭 Explainable Career Recommendations")
    st.write("Evaluates the student's skill portfolio against benchmark competency vectors across standard technology and business roles.")

    recommendations = evaluate_career_recommendations(student_inputs, top_n=3)

    st.write("### Top 3 Career Role Matches")
    for rank, rec in enumerate(recommendations, 1):
        with st.container():
            st.markdown(f"""
            <div class="career-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; color: #1E3A8A;">#{rank} {rec['role_title']}</h3>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #2563EB;">{rec['suitability_score']}% Match</span>
                </div>
                <p style="color: #4B5563; margin-top: 4px; font-size: 0.95rem;"><b>Category:</b> {rec['category']} &nbsp;|&nbsp; {rec['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            col_chart, col_gaps = st.columns([1, 1.2])

            with col_chart:
                st.write("**Competency Breakdown vs Benchmark**")
                breakdown_df = pd.DataFrame([
                    {
                        "Skill": item["skill_name"],
                        "Student": item["student_score"],
                        "Benchmark": item["benchmark"]
                    }
                    for item in rec["skill_breakdown"]
                ])
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            with col_gaps:
                st.write("**Identified Skill Gaps & Actionable Advice**")
                if rec["skill_gaps"]:
                    for gap in rec["skill_gaps"]:
                        st.markdown(f"- **{gap['skill_name']}** (Current: {gap['student_score']:.1f}, Benchmark: {gap['benchmark']:.1f}):")
                        st.caption(f"  💡 {gap['advice']}")
                else:
                    st.success("🎉 Outstanding fit! Your current competency scores meet or exceed all core benchmarks for this role.")

            st.write("")


# -------------------------------------------------------------
# TAB 3: MODEL ANALYTICS & OPTIMIZATION
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 Model Analytics, Evaluation & Hyperparameter Optimization")
    st.write("Presents empirical cross-validation metrics, hyperparameter optimization comparisons, and interpretability plots.")

    if metadata is not None:
        st.write("### 🔬 Optimization Comparison Summary")
        exp_list = metadata.get("all_experiments", [])
        if exp_list:
            table_rows = []
            for e in exp_list:
                table_rows.append({
                    "Optimization Method": e["model_name"],
                    "Selected Hyperparameters": str(e["best_params"]),
                    "5-Fold CV F1": f"{e['cv_f1_mean']:.4f} (±{e['cv_f1_std']:.4f})",
                    "Accuracy": f"{e['accuracy'] * 100:.2f}%",
                    "Precision": f"{e['precision'] * 100:.2f}%",
                    "Recall": f"{e['recall'] * 100:.2f}%",
                    "F1-Score": f"{e['f1']:.4f}",
                    "ROC-AUC": f"{e['roc_auc']:.4f}"
                })
            comp_df = pd.DataFrame(table_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.divider()

        st.write("### 📈 Evaluation Visualizations")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.write("**Confusion Matrix (Held-out Test Set)**")
            cm_img = os.path.join("assets", "confusion_matrix.png")
            if os.path.exists(cm_img):
                st.image(cm_img, caption="Confusion Matrix on Held-Out Test Set (124 samples)", use_container_width=True)
            else:
                st.info("Confusion matrix plot not available.")

        with viz_col2:
            st.write("**Receiver Operating Characteristic (ROC) Curves**")
            roc_img = os.path.join("assets", "roc_curve.png")
            if os.path.exists(roc_img):
                st.image(roc_img, caption="ROC Curves comparing Baseline, GridSearch, RandomSearch, and Optuna", use_container_width=True)
            else:
                st.info("ROC plot not available.")

        st.divider()

        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            st.write("**Optimization Metric Comparison**")
            comp_img = os.path.join("assets", "model_comparison.png")
            if os.path.exists(comp_img):
                st.image(comp_img, caption="Classification Metrics across Hyperparameter Optimization Strategies", use_container_width=True)
            else:
                st.info("Comparison chart not available.")

        with viz_col4:
            st.write("**5-Fold Cross-Validation Distribution**")
            cv_img = os.path.join("assets", "cv_comparison.png")
            if os.path.exists(cv_img):
                st.image(cv_img, caption="Variance and Median F1 Across 5 Stratified CV Folds", use_container_width=True)
            else:
                st.info("CV chart not available.")

        st.divider()

        st.write("### 🔍 Model Interpretability — Feature Coefficients")
        feat_img = os.path.join("assets", "feature_importance.png")
        if os.path.exists(feat_img):
            st.image(feat_img, caption="Standardized Logistic Regression Coefficients", use_container_width=True)

    else:
        st.info("Model metadata not found. Run `src/train.py` to generate complete evaluation analytics.")


# Footer Section
st.markdown("""
<div class="footer-text">
    Student Employability Prediction and Career Recommendation System Using Machine Learning<br>
    YuvaIntern Internship Project | Built with Scikit-learn, Optuna, and Streamlit
</div>
""", unsafe_allow_html=True)