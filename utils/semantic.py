# utils/semantic.py
from sentence_transformers import SentenceTransformer, util
from utils.preprocess import clean_text

model = SentenceTransformer('all-mpnet-base-v2')  # stronger model

# Predefined skill keywords (optional)
SKILL_KEYWORDS = ["python", "flask", "django", "aws", "sql", "javascript"]

def get_similarity(resume_text, job_desc):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_desc)

    embeddings = model.encode([resume_clean, job_clean])
    score = float(util.cos_sim(embeddings[0], embeddings[1]))

    # Check missing skills
    missing_skills = [skill for skill in SKILL_KEYWORDS if skill not in resume_clean]

    return score, missing_skills
