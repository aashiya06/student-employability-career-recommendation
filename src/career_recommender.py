"""
Career Recommendation System.
Provides transparent, rule-based role matching and skill-gap analysis.
Note: Logistic Regression determines binary employability; this dedicated module
matches the student's skill portfolio to appropriate industry career profiles.
"""

from typing import Dict, List, Any


ROLE_PROFILES = {
    "Software Developer": {
        "title": "Software Developer / Engineer",
        "category": "Software Engineering",
        "description": "Focuses on designing, building, testing, and maintaining software applications and systems.",
        "skills": {
            "Programming_Skill": {
                "benchmark": 75.0, "weight": 0.35, "label": "Programming Skills",
                "advice": "Strengthen core programming (Data Structures, Algorithms, OOP) and build full-stack projects."
            },
            "Tool_Proficiency": {
                "benchmark": 70.0, "weight": 0.25, "label": "Tool Proficiency",
                "advice": "Gain proficiency with Git, GitHub, IDEs, debugging tools, and containerization (Docker)."
            },
            "Project_Experience_Norm": {
                "benchmark": 65.0, "weight": 0.20, "label": "Project Experience",
                "advice": "Develop at least 2 complete, deployed software projects with clean modular architecture."
            },
            "Domain_Knowledge": {
                "benchmark": 65.0, "weight": 0.10, "label": "Core CS Concepts",
                "advice": "Review operating systems, networking, databases (SQL), and system design fundamentals."
            },
            "Teamwork_Ability": {
                "benchmark": 65.0, "weight": 0.10, "label": "Collaborative Coding",
                "advice": "Contribute to open-source repositories or participate in team hackathons and code reviews."
            }
        }
    },
    "Data Analyst / Data Scientist": {
        "title": "Data Analyst / Data Scientist",
        "category": "Data & Analytics",
        "description": "Analyzes complex datasets, builds statistical models, and translates data into strategic business insights.",
        "skills": {
            "Domain_Knowledge": {
                "benchmark": 75.0, "weight": 0.25, "label": "Domain & Statistical Knowledge",
                "advice": "Master descriptive/inferential statistics, hypothesis testing, and exploratory data analysis."
            },
            "Programming_Skill": {
                "benchmark": 70.0, "weight": 0.20, "label": "Data Programming (Python/R)",
                "advice": "Deepen skills in Pandas, NumPy, Scikit-learn, and database querying with SQL."
            },
            "Tool_Proficiency": {
                "benchmark": 70.0, "weight": 0.20, "label": "BI & Analytics Tools",
                "advice": "Master data visualization tools such as Power BI, Tableau, and Excel dashboards."
            },
            "Core_Subject_Score": {
                "benchmark": 70.0, "weight": 0.20, "label": "Quantitative Foundations",
                "advice": "Revisit linear algebra, probability distributions, and quantitative problem-solving."
            },
            "Communication_Skill": {
                "benchmark": 65.0, "weight": 0.15, "label": "Data Storytelling",
                "advice": "Practice presenting analytical findings clearly to non-technical stakeholders."
            }
        }
    },
    "AI / ML Engineer": {
        "title": "AI / ML Engineer",
        "category": "Artificial Intelligence",
        "description": "Designs and deploys machine learning models, predictive algorithms, and AI systems into production.",
        "skills": {
            "Programming_Skill": {
                "benchmark": 80.0, "weight": 0.30, "label": "Machine Learning Programming",
                "advice": "Deepen practical coding in Python, Scikit-learn, PyTorch/TensorFlow, and ML pipelines."
            },
            "Core_Subject_Score": {
                "benchmark": 75.0, "weight": 0.25, "label": "Mathematical Foundations",
                "advice": "Strengthen understanding of calculus, optimization algorithms, and probability theory."
            },
            "Tool_Proficiency": {
                "benchmark": 75.0, "weight": 0.15, "label": "MLOps & Environment Tools",
                "advice": "Familiarize yourself with ML lifecycle tools (MLflow, Git, Docker, cloud ML services)."
            },
            "Project_Experience_Norm": {
                "benchmark": 70.0, "weight": 0.15, "label": "Applied ML Projects",
                "advice": "Build end-to-end ML projects covering data cleaning, training, tuning, and API deployment."
            },
            "Research_Participation_Norm": {
                "benchmark": 50.0, "weight": 0.15, "label": "Research & Experimentation",
                "advice": "Read recent peer-reviewed ML papers, reproduce benchmarks, and explore research initiatives."
            }
        }
    },
    "Cloud / DevOps Engineer": {
        "title": "Cloud / DevOps Engineer",
        "category": "Infrastructure & Cloud",
        "description": "Manages cloud infrastructure, automates CI/CD pipelines, and ensures high availability and scalability.",
        "skills": {
            "Tool_Proficiency": {
                "benchmark": 75.0, "weight": 0.30, "label": "Cloud & Automation Tools",
                "advice": "Gain hands-on experience with Linux, Docker, Kubernetes, Terraform, and CI/CD tools."
            },
            "Practical_Exposure_Norm": {
                "benchmark": 65.0, "weight": 0.25, "label": "Hands-on Infrastructure Exposure",
                "advice": "Deploy and manage small applications on AWS, Azure, or GCP with automated pipelines."
            },
            "Programming_Skill": {
                "benchmark": 65.0, "weight": 0.20, "label": "Scripting (Python/Bash)",
                "advice": "Master bash scripting and Python automation for infrastructure monitoring and maintenance."
            },
            "Adaptability": {
                "benchmark": 70.0, "weight": 0.15, "label": "System Troubleshooting & Agility",
                "advice": "Practice real-time debugging, log monitoring, and incident mitigation."
            },
            "Certification_Norm": {
                "benchmark": 60.0, "weight": 0.10, "label": "Industry Cloud Certifications",
                "advice": "Target entry certifications like AWS Certified Cloud Practitioner or Azure Fundamentals."
            }
        }
    },
    "Business Analyst": {
        "title": "Business Analyst",
        "category": "Business & Strategy",
        "description": "Bridges the gap between business stakeholders and technology teams by defining requirements and workflows.",
        "skills": {
            "Communication_Skill": {
                "benchmark": 75.0, "weight": 0.25, "label": "Stakeholder Communication",
                "advice": "Refine client-facing communication, requirement gathering, and structured writing."
            },
            "Market_Awareness": {
                "benchmark": 70.0, "weight": 0.25, "label": "Industry & Market Awareness",
                "advice": "Study target industry trends, business models, competitor dynamics, and KPI frameworks."
            },
            "Domain_Knowledge": {
                "benchmark": 70.0, "weight": 0.20, "label": "Functional Domain Knowledge",
                "advice": "Learn standard business processes, requirement documentation (BRD/FRD), and agile user stories."
            },
            "Career_Clarity": {
                "benchmark": 70.0, "weight": 0.15, "label": "Strategic Problem Solving",
                "advice": "Engage in business case studies, process mapping, and root-cause problem breakdown."
            },
            "Teamwork_Ability": {
                "benchmark": 70.0, "weight": 0.15, "label": "Cross-Functional Collaboration",
                "advice": "Practice mediating cross-team discussions and driving consensus between technical and business groups."
            }
        }
    },
    "Project / Product Management": {
        "title": "Project / Product Management",
        "category": "Management & Leadership",
        "description": "Oversees project execution, resource allocation, team coordination, and product delivery lifecycle.",
        "skills": {
            "Leadership_Potential": {
                "benchmark": 75.0, "weight": 0.25, "label": "Leadership & Initiative",
                "advice": "Take lead roles in student clubs, academic projects, or volunteer coordination teams."
            },
            "Communication_Skill": {
                "benchmark": 75.0, "weight": 0.25, "label": "Executive Communication",
                "advice": "Enhance presentation clarity, negotiation skills, and conflict-resolution abilities."
            },
            "Teamwork_Ability": {
                "benchmark": 75.0, "weight": 0.20, "label": "Team Empowerment",
                "advice": "Practice agile ceremonies, sprint planning, task delegation, and tracking deliverables."
            },
            "Career_Clarity": {
                "benchmark": 70.0, "weight": 0.15, "label": "Vision & Planning Clarity",
                "advice": "Formulate roadmaps, prioritize backlogs, and analyze user feedback loops."
            },
            "Adaptability": {
                "benchmark": 70.0, "weight": 0.15, "label": "Risk & Change Management",
                "advice": "Learn change management frameworks and contingency planning for project roadblocks."
            }
        }
    }
}


def normalize_student_profile(raw_profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalizes student attributes into a unified 0-100 scale for transparent comparison.
    """
    profile = {}
    for k, v in raw_profile.items():
        try:
            profile[k] = float(v)
        except (ValueError, TypeError):
            profile[k] = 0.0

    # Project experience (0 to 7) -> scaled to 0-100
    proj = profile.get("Project_Experience", 0.0)
    profile["Project_Experience_Norm"] = min(100.0, (proj / 7.0) * 100.0)

    # Certifications (0 to 5) -> scaled to 0-100 (target 3+ for full norm score)
    certs = profile.get("Certification_Count", 0.0)
    profile["Certification_Norm"] = min(100.0, (certs / 3.0) * 100.0)

    # Research participation (0 or 1) -> 0 or 100
    res = profile.get("Research_Participation", 0.0)
    profile["Research_Participation_Norm"] = 100.0 if res >= 1.0 else 0.0

    # Internship duration (0 to 18 months) -> scaled (target 6+ months for good score)
    dur = profile.get("Internship_Duration", 0.0)
    profile["Internship_Duration_Norm"] = min(100.0, (dur / 12.0) * 100.0)

    # Practical exposure composite
    prac = (
        profile["Project_Experience_Norm"] * 0.4 +
        profile["Internship_Duration_Norm"] * 0.3 +
        profile["Certification_Norm"] * 0.3
    )
    profile["Practical_Exposure_Norm"] = min(100.0, prac)

    return profile


def evaluate_career_recommendations(student_inputs: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Evaluates student competencies against industry career benchmarks.
    Returns ranked list with suitability scores, identified skill gaps, and recommendations.
    """
    norm_profile = normalize_student_profile(student_inputs)
    results = []

    for role_key, role_meta in ROLE_PROFILES.items():
        total_weight = 0.0
        weighted_score = 0.0
        skill_gaps = []
        skill_breakdown = []

        for skill_key, skill_spec in role_meta["skills"].items():
            weight = skill_spec["weight"]
            benchmark = skill_spec["benchmark"]
            label = skill_spec["label"]
            advice = skill_spec["advice"]

            student_val = norm_profile.get(skill_key, 50.0)
            # Achievement ratio (capped at 100% contribution per skill)
            achievement_pct = min(100.0, (student_val / benchmark) * 100.0) if benchmark > 0 else 100.0

            weighted_score += achievement_pct * weight
            total_weight += weight

            # Check if there's a skill deficit
            deficit = benchmark - student_val
            is_gap = deficit > 5.0

            skill_entry = {
                "skill_name": label,
                "student_score": round(student_val, 1),
                "benchmark": round(benchmark, 1),
                "is_gap": is_gap,
                "deficit": round(deficit, 1) if is_gap else 0.0,
                "advice": advice if is_gap else "Good proficiency demonstrated in this area."
            }
            skill_breakdown.append(skill_entry)

            if is_gap:
                skill_gaps.append(skill_entry)

        final_suitability = round(weighted_score / total_weight, 1) if total_weight > 0 else 0.0

        results.append({
            "role_key": role_key,
            "role_title": role_meta["title"],
            "category": role_meta["category"],
            "description": role_meta["description"],
            "suitability_score": final_suitability,
            "skill_gaps": skill_gaps,
            "total_gaps": len(skill_gaps),
            "skill_breakdown": skill_breakdown
        })

    # Sort descending by suitability score
    results.sort(key=lambda x: x["suitability_score"], reverse=True)

    # Return top_n recommendations (default 3)
    return results[:top_n]
