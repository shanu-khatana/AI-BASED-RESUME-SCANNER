from sentence_transformers import SentenceTransformer, util
from utils.preprocess import clean_text

# Define skill keywords for analysis
SKILL_KEYWORDS = ["python", "flask", "django", "aws", "sql", "javascript", "html", "css"]

# Load SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')  # Strong semantic model

def get_similarity(resume_text, job_desc):
    """
    Returns similarity score and missing skills
    """
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_desc)

    embeddings = model.encode([resume_clean, job_clean])
    score = float(util.cos_sim(embeddings[0], embeddings[1]))

    # Find missing skills
    missing_skills = [skill for skill in SKILL_KEYWORDS if skill not in resume_clean]

    return score, missing_skills
