import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# === Load hybrid recommender artifacts ===
with open("hybrid_recommender.pkl", "rb") as f:
    artifacts = pickle.load(f)

jobs_df = artifacts["jobs_df"]
tfidf = artifacts["tfidf"]

# === Predefined options ===
course_university = [
    "Arts - Information Technology - University of Sri Jayewardenepura",
    "Computer Science - University of Colombo School of Computing (UCSC)",
    "Computer Science - University of Jaffna",
    "Computer Science - University of Ruhuna",
    "Trincomalee Campus - Eastern University, Sri Lanka",
    "Physical Science - ICT - University of Kelaniya",
    "Artificial Intelligence - University of Moratuwa",
    "Electronics and Computer Science - University of Kelaniya",
    "Information Systems - University of Colombo School of Computing (UCSC)",
    "Data Science - Sabaragamuwa University of Sri Lanka",
    "Information Technology (IT) - University of Moratuwa",
    "Management and Information Technology (MIT) - University of Kelaniya",
    "Computer Science & Technology - Uva Wellassa University of Sri Lanka",
    "Information Communication Technology - University of Sri Jayewardenepura",
    "Information Communication Technology - University of Kelaniya",
    "Information Communication Technology - University of Vavuniya, Sri Lanka",
    "Information Communication Technology - University of Ruhuna",
    "Information Communication Technology - South Eastern University of Sri Lanka",
    "Information Communication Technology - Rajarata University of Sri Lanka",
    "Information Communication Technology - University of Colombo",
    "Information Communication Technology - Uva Wellassa University of Sri Lanka",
    "Information Communication Technology - Eastern University, Sri Lanka"
]

languages = ["English", "Sinhala", "Tamil"]

skills_list = [
    "Python", "Java", "SQL", "JavaScript", "TensorFlow", "Pandas", "Docker",
    "Kubernetes", "HTML/CSS", "Power BI", "Spark", "AWS", "Azure",
    "Linux", "Tableau", "React", "Node.js"
]

internships = [
    "Software Intern", "Data Analyst Intern", "ML Intern", "QA Intern",
    "BI Intern", "Cloud Intern", "Network Intern", "Cybersecurity Intern",
    "UI/UX Intern", "None"
]

# === Recommendation function ===
def recommend_roles_new_candidate(candidate_info, jobs_df, tfidf, top_n=5, alpha=0.6):
    # Combine features for content similarity
    combined_features = (
        candidate_info["Skills"].lower() + ", " +
        candidate_info["Current_Role"].lower() + ", " +
        candidate_info["Course_University"].lower() + ", " +
        candidate_info["Language_Proficiency"].lower()
    )
    candidate_vec = tfidf.transform([combined_features])
    job_tfidf_matrix = tfidf.transform(jobs_df["combined_features"])
    content_scores = cosine_similarity(candidate_vec, job_tfidf_matrix).flatten()

    # Collaborative fallback: popularity of Target_Role
    top_roles_by_popularity = jobs_df["Target_Role"].value_counts().to_dict()
    collab_scores = [top_roles_by_popularity.get(role, 0) for role in jobs_df["Target_Role"]]

    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores

    # Get top N recommended roles (remove duplicates)
    sorted_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)
    recommended_roles = []
    for idx in sorted_indices:
        role = jobs_df.iloc[idx]["Target_Role"]
        if role not in recommended_roles:
            recommended_roles.append(role)
        if len(recommended_roles) >= top_n:
            break
    return recommended_roles

# === Streamlit UI ===
st.title("🧑‍💼 Hybrid Job Recommendation System")

with st.form("candidate_form"):
    candidate_id = st.number_input("Candidate ID", min_value=1, step=1)
    course_univ = st.selectbox("Course & University", course_university)
    language = st.multiselect("Languages", languages)
    internship = st.selectbox("Previous Internship", internships)
    experience = st.number_input("Experience Years", min_value=0.0, step=0.1)
    skills = st.multiselect("Skills", skills_list)
    current_role = st.text_input("Current Role")
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    candidate_info = {
        "Candidate_ID": candidate_id,
        "Course_University": course_univ,
        "Language_Proficiency": ", ".join(language),
        "Previous_Internship": internship,
        "Experience_Years": experience,
        "Skills": ", ".join(skills),
        "Current_Role": current_role,
        "Target_Role": ""  # unknown for new candidate
    }
    recommendations = recommend_roles_new_candidate(candidate_info, jobs_df, tfidf, top_n=5, alpha=0.6)
    st.success("Top Recommended Roles:")
    for i, role in enumerate(recommendations, 1):
        st.write(f"{i}. {role}")
