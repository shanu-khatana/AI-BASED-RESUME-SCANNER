from flask import Flask, render_template, request, redirect, url_for, session, flash 
import sqlite3
import os
import fitz  # PyMuPDF for PDF
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils.extract import extract_text
from utils.preprocess import clean_text
import re
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from itsdangerous import URLSafeTimedSerializer
import datetime
import spacy

# ----------------- FLASK CONFIG -----------------
app = Flask(__name__)
app.secret_key = "super_secret_key"
UPLOAD_FOLDER = "resumes"
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure password reset
app.config['SECURITY_PASSWORD_SALT'] = 'your_salt_here'
app.config['RESET_PASSWORD_EXPIRATION'] = 3600  

# ----------------- SENTENCE-BERT MODEL -----------------
# Load the model once at startup
print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-mpnet-base-v2')
print("Model loaded successfully!")

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except:
    logger.warning("spaCy model not found. Some features may not work optimally.")
    nlp = None

# ----------------- DATABASE SETUP -----------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add password reset table
    c.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            token TEXT UNIQUE NOT NULL,
            expiration TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Create scans table to store scan history
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            job_description TEXT,
            similarity_score REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()

init_db()

# ----------------- HELPER FUNCTIONS -----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_token(user_id):
    serializer = URLSafeTimedSerializer(app.secret_key)
    return serializer.dumps(str(user_id), salt=app.config['SECURITY_PASSWORD_SALT'])

def verify_token(token):
    serializer = URLSafeTimedSerializer(app.secret_key)
    try:
        user_id = serializer.loads(
            token,
            salt=app.config['SECURITY_PASSWORD_SALT'],
            max_age=app.config['RESET_PASSWORD_EXPIRATION']
        )
        return user_id
    except:
        return None

def save_reset_token(user_id, token):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    # Delete any existing tokens for this user
    c.execute("DELETE FROM password_resets WHERE user_id=?", (user_id,))
    
    # Calculate expiration time
    expiration = datetime.datetime.now() + datetime.timedelta(seconds=app.config['RESET_PASSWORD_EXPIRATION'])
    
    # Save the new token
    c.execute("INSERT INTO password_resets (user_id, token, expiration) VALUES (?, ?, ?)", 
              (user_id, token, expiration))
    
    conn.commit()
    conn.close()

def is_valid_token(token):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    # Check if token exists and is not expired
    c.execute("SELECT user_id FROM password_resets WHERE token=? AND expiration > datetime('now')", (token,))
    result = c.fetchone()
    
    conn.close()
    
    return result is not None

def delete_reset_token(token):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("DELETE FROM password_resets WHERE token=?", (token,))
    conn.commit()
    conn.close()

def send_reset_email(email, token):
    # In a real application, you would send an email with the reset link
    # For development, we'll just print the link
    reset_link = url_for('reset_password', token=token, _external=True)
    print(f"Password reset link for {email}: {reset_link}")
    # In production, you would use something like:
    # send_email(to=email, subject="Password Reset", body=f"Click here to reset your password: {reset_link}")

def extract_years_experience(text):
    """Extract years of experience from resume text"""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience\s*:\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:of\s*)?work',
        r'total\s*experience\s*:\s*(\d+)\+?\s*years?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return 0

def extract_education_level(text):
    """Extract education level from resume text"""
    education_keywords = {
        'phd': 5,
        'doctorate': 5,
        'postgraduate': 4,
        'master': 4,
        'mtech': 4,
        'msc': 4,
        'mca': 4,
        'graduate': 3,
        'btech': 3,
        'be': 3,
        'bca': 3,
        'bachelor': 3,
        'diploma': 2,
        'intermediate': 1,
        '12th': 1,
        '10th': 0
    }
    
    text_lower = text.lower()
    for keyword, level in education_keywords.items():
        if keyword in text_lower:
            return level
    return 0

# Enhanced skill definitions with variations and contexts
SKILL_DEFINITIONS = {
    "python": {
        "variations": ["python", "py"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "programming"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "java": {
        "variations": ["java", "java8", "java11", "java17"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "programming"],
        "exclusions": ["javascript", "course", "class", "training", "learned", "studied"]
    },
    "javascript": {
        "variations": ["javascript", "js", "ecmascript", "es6", "es7"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "programming"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "html": {
        "variations": ["html", "html5", "xhtml"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "markup"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "css": {
        "variations": ["css", "css3", "sass", "scss", "less"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "styling"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "sql": {
        "variations": ["sql", "mysql", "postgresql", "sqlite", "tsql", "pl/sql"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "database", "query"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "nosql": {
        "variations": ["nosql", "mongodb", "cassandra", "dynamodb", "redis", "neo4j"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "database"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "flask": {
        "variations": ["flask", "flask-sqlalchemy", "flask-restful"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "framework"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "django": {
        "variations": ["django", "django-rest", "djangorestframework"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "framework"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "react": {
        "variations": ["react", "reactjs", "react.js", "react-native"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "framework"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "angular": {
        "variations": ["angular", "angularjs", "angular.js", "angular2", "angular4", "angular6", "angular8"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "framework"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "nodejs": {
        "variations": ["nodejs", "node.js", "node", "express", "expressjs", "express.js"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "backend"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "mongodb": {
        "variations": ["mongodb", "mongo", "mongoose"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "database"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "mysql": {
        "variations": ["mysql", "my-sql"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "database"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "postgresql": {
        "variations": ["postgresql", "postgres", "postgre"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "database"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "aws": {
        "variations": ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "cloudformation"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "cloud"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "azure": {
        "variations": ["azure", "microsoft azure", "azure cloud"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "cloud"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "gcp": {
        "variations": ["gcp", "google cloud", "google cloud platform"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "cloud"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "docker": {
        "variations": ["docker", "docker-compose", "dockerfile", "container"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "containerization"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "kubernetes": {
        "variations": ["kubernetes", "k8s", "kube", "kubectl", "helm"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "orchestration"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "machine learning": {
        "variations": ["machine learning", "ml", "ml algorithms"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "implemented", "models"],
        "exclusions": ["course", "class", "training", "learned", "studied", "ai ml", "ai & ml"]
    },
    "ai": {
        "variations": ["ai", "artificial intelligence"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "implemented"],
        "exclusions": ["course", "class", "training", "learned", "studied", "ai ml", "ai & ml", "artificial intelligence machine learning"]
    },
    "deep learning": {
        "variations": ["deep learning", "dl", "neural networks", "cnn", "rnn", "lstm"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "implemented", "models"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "nlp": {
        "variations": ["nlp", "natural language processing", "text processing", "text mining"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "implemented"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "data science": {
        "variations": ["data science", "data analysis", "data analytics"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "analysis"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "tensorflow": {
        "variations": ["tensorflow", "tf", "tensorflow2", "tf2"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "implemented"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "pytorch": {
        "variations": ["pytorch", "torch"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "implemented"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "scikit-learn": {
        "variations": ["scikit-learn", "sklearn"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "implemented"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "pandas": {
        "variations": ["pandas", "pd"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "data manipulation"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "numpy": {
        "variations": ["numpy", "np"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "numerical computing"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "git": {
        "variations": ["git", "gitlab", "bitbucket"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "version control"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "github": {
        "variations": ["github", "gh"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "version control"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "linux": {
        "variations": ["linux", "ubuntu", "centos", "redhat", "debian", "shell", "bash"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "administration"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "ubuntu": {
        "variations": ["ubuntu"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "administration"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "windows": {
        "variations": ["windows", "win", "powershell"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "administration"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "api": {
        "variations": ["api", "apis", "restful", "rest", "soap"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "integration"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "rest": {
        "variations": ["rest", "restful", "rest-api"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "api"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "microservices": {
        "variations": ["microservices", "micro-service", "micro service"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "architecture"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "agile": {
        "variations": ["agile", "scrum", "kanban"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "methodology"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "scrum": {
        "variations": ["scrum", "agile", "sprint"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "methodology"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "devops": {
        "variations": ["devops", "dev-ops", "ci/cd", "cicd"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "practices"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    },
    "ci/cd": {
        "variations": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "pipeline"],
        "exclusions": ["course", "class", "training", "learned", "studied"]
    }
}

def extract_skills_from_text(text, skill_definitions):
    """
    Advanced skill extraction using multiple strategies and context analysis.
    Returns matched and missing skills with confidence scores.
    """
    text_lower = text.lower()
    sentences = re.split(r'[.!?]+', text_lower)
    
    matched_skills = []
    missing_skills = []
    skill_confidence = {}
    
    logger.debug(f"Starting skill extraction from text: {text_lower[:200]}...")
    
    for skill_name, skill_info in skill_definitions.items():
        confidence = 0
        found_in_context = False
        
        # Strategy 1: Direct pattern matching with variations
        for variation in skill_info["variations"]:
            # Create regex pattern for word boundaries
            pattern = r'\b' + re.escape(variation) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
            
            for match in matches:
                # Get context around the match
                start = max(0, match.start() - 50)
                end = min(len(text_lower), match.end() + 50)
                context = text_lower[start:end]
                
                # Check for exclusion patterns
                has_exclusion = any(exc in context for exc in skill_info["exclusions"])
                
                if not has_exclusion:
                    # Check for positive context indicators
                    has_context = any(ctx in context for ctx in skill_info["contexts"])
                    
                    if has_context:
                        confidence += 0.8
                        found_in_context = True
                        logger.debug(f"Found {skill_name} (variation: {variation}) with context: {context}")
                    else:
                        # Still count but with lower confidence
                        confidence += 0.4
                        logger.debug(f"Found {skill_name} (variation: {variation}) without context: {context}")
        
        # Strategy 2: NER-based extraction (if spaCy is available)
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PRODUCT", "ORG", "TECHNOLOGY"]:
                    for variation in skill_info["variations"]:
                        if variation in ent.text.lower():
                            confidence += 0.6
                            logger.debug(f"NER found {skill_name} in entity: {ent.text}")
        
        # Strategy 3: Section-based analysis
        sections = {
            'skills': re.search(r'(skills|technical skills|core competencies)(.*?)(experience|education|projects|$)', text_lower, re.IGNORECASE | re.DOTALL),
            'experience': re.search(r'(experience|work experience|professional experience)(.*?)(education|skills|projects|$)', text_lower, re.IGNORECASE | re.DOTALL),
            'projects': re.search(r'(projects|personal projects|academic projects)(.*?)(education|skills|experience|$)', text_lower, re.IGNORECASE | re.DOTALL)
        }
        
        for section_name, section_match in sections.items():
            if section_match:
                section_text = section_match.group(2)
                for variation in skill_info["variations"]:
                    if variation in section_text:
                        if section_name in ['skills', 'experience', 'projects']:
                            confidence += 0.7
                            logger.debug(f"Found {skill_name} in {section_name} section")
        
        # Normalize confidence
        confidence = min(confidence, 1.0)
        
        # Determine if skill is present based on confidence threshold
        if confidence >= 0.5:  # Adjustable threshold
            matched_skills.append(skill_name)
            skill_confidence[skill_name] = confidence
            logger.debug(f"Skill {skill_name} matched with confidence: {confidence}")
        else:
            missing_skills.append(skill_name)
            logger.debug(f"Skill {skill_name} not matched (confidence: {confidence})")
    
    logger.debug(f"Final matched skills: {matched_skills}")
    logger.debug(f"Final missing skills: {missing_skills}")
    
    return matched_skills, missing_skills

def calculate_semantic_similarity(job_desc, resume_text):
    """Calculate semantic similarity using Sentence-BERT"""
    # Generate embeddings
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_score = util.pytorch_cos_sim(job_embedding, resume_embedding)
    return cosine_score.item() * 100  # Convert to percentage

# ----------------- ROUTES -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ----------------- REGISTER -----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        email = request.form.get("email", "").strip()
        
        # Validation
        if not username or not password:
            flash("Username and password are required!", "error")
            return render_template("register.html")
        
        if len(password) < 6:
            flash("Password must be at least 6 characters long!", "error")
            return render_template("register.html")
        
        password_hash = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", 
                     (username, password_hash, email))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "error")
        finally:
            conn.close()
    return render_template("register.html")

# ----------------- LOGIN -----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if not username or not password:
            flash("Please enter both username and password!", "error")
            return render_template("login.html")

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            session["username"] = username
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password!", "error")

    return render_template("login.html")

# ----------------- FORGOT PASSWORD -----------------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username_email = request.form.get("username_email", "").strip()
        
        if not username_email:
            flash("Please enter your username or email!", "error")
            return render_template("forgot_password.html")
        
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        # Try to find user by username or email
        c.execute("SELECT id, email FROM users WHERE username=? OR email=?", (username_email, username_email))
        user = c.fetchone()
        
        if user:
            user_id, email = user
            token = generate_token(user_id)
            save_reset_token(user_id, token)
            
            # In a real application, you would send an email
            # For development, we'll just show a success message and print the link
            send_reset_email(email, token)
            
            flash("Password reset instructions have been sent to your email.", "success")
            
            # For development, show the reset link directly
            reset_link = url_for('reset_password', token=token, _external=True)
            logger.info(f"Password reset link: {reset_link}")
        else:
            # Don't reveal whether the user exists for security
            flash("If your username or email is in our system, you will receive password reset instructions.", "info")
        
        conn.close()
        return redirect(url_for("login"))
    
    return render_template("forgot_password.html")

# ----------------- RESET PASSWORD -----------------
@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if not is_valid_token(token):
        flash("Invalid or expired reset token. Please try again.", "error")
        return redirect(url_for("forgot_password"))
    
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        if not password or not confirm_password:
            flash("Please enter both password fields!", "error")
            return render_template("reset_password.html", token=token)
        
        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return render_template("reset_password.html", token=token)
        
        if len(password) < 6:
            flash("Password must be at least 6 characters long!", "error")
            return render_template("reset_password.html", token=token)
        
        # Get user ID from token
        user_id = verify_token(token)
        if not user_id:
            flash("Invalid or expired reset token. Please try again.", "error")
            return redirect(url_for("forgot_password"))
        
        # Update password
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("UPDATE users SET password=? WHERE id=?", (password_hash, user_id))
        conn.commit()
        conn.close()
        
        # Delete the used token
        delete_reset_token(token)
        
        flash("Your password has been reset successfully. Please login with your new password.", "success")
        return redirect(url_for("login"))
    
    return render_template("reset_password.html", token=token)

# ----------------- LOGOUT -----------------
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# ----------------- DASHBOARD -----------------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    # Get user's scan history
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        SELECT filename, similarity_score, scan_date, matched_skills, missing_skills
        FROM scans 
        WHERE user_id = ? 
        ORDER BY scan_date DESC 
        LIMIT 10
    """, (session["user_id"],))
    scans = c.fetchall()
    conn.close()
    
    return render_template("dashboard.html", 
                         username=session["username"], 
                         scans=scans)

# ----------------- RESUME SCANNER -----------------
@app.route("/upload", methods=["POST"])
def upload_resume():
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    job_desc = request.form.get("jobdesc", "").strip()
    resume = request.files.get("resume")

    if not job_desc:
        flash("Please enter a job description!", "error")
        return redirect(url_for("dashboard"))
    
    if not resume or resume.filename == '':
        flash("Please upload a resume!", "error")
        return redirect(url_for("dashboard"))
    
    if not allowed_file(resume.filename):
        flash("Invalid file type! Please upload PDF or DOCX files only.", "error")
        return redirect(url_for("dashboard"))

    try:
        # Secure filename and save file
        filename = secure_filename(resume.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        resume.save(file_path)

        # Extract text from resume
        resume_text = extract_text(file_path)
        
        if not resume_text.strip():
            flash("Could not extract text from the uploaded file. Please try another file.", "error")
            return redirect(url_for("dashboard"))

        # Clean and preprocess text
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_desc)

        # Calculate semantic similarity using Sentence-BERT
        try:
            similarity = calculate_semantic_similarity(job_desc, resume_text)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            # Fallback to TF-IDF if Sentence-BERT fails
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                tfidf_matrix = vectorizer.fit_transform([job_clean, resume_clean])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
            except:
                similarity = 0.0

        # Use advanced skills analysis
        matched_skills, missing_skills = extract_skills_from_text(resume_text, SKILL_DEFINITIONS)

        # Extract additional metrics
        years_experience = extract_years_experience(resume_text)
        education_level = extract_education_level(resume_text)

        # Calculate comprehensive score
        skills_score = len(matched_skills) / len(SKILL_DEFINITIONS) * 100
        experience_score = min(years_experience * 10, 30)  # Max 30 points for experience
        education_score = education_level * 10  # Max 50 points for education
        
        # Weighted final score (using the same weights as the original project)
        final_score = (similarity * 0.5) + (skills_score * 0.3) + (experience_score * 0.1) + (education_score * 0.1)

        # Save scan to database
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("""
            INSERT INTO scans (user_id, filename, job_description, similarity_score, 
                             matched_skills, missing_skills)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            filename,
            job_desc[:500],  # Limit job description length
            round(final_score, 2),
            ','.join(matched_skills[:20]),  # Limit skills stored
            ','.join(missing_skills[:20])
        ))
        conn.commit()
        conn.close()

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass

        return render_template("result.html",
                             filename=filename,
                             score=round(final_score, 2),
                             similarity_score=round(similarity, 2),
                             skills_score=round(skills_score, 2),
                             matched_count=len(matched_skills),
                             missing_count=len(missing_skills),
                             matched_skills=matched_skills[:10],  # Show top 10
                             missing_skills=missing_skills[:10],  # Show top 10
                             years_experience=years_experience,
                             education_level=education_level)

    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        flash("An error occurred while processing your resume. Please try again.", "error")
        return redirect(url_for("dashboard"))

# ----------------- ERROR HANDLERS -----------------
@app.errorhandler(413)
def too_large(e):
    flash("File too large! Please upload a file smaller than 16MB.", "error")
    return redirect(url_for("dashboard"))

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    flash("An internal server error occurred. Please try again later.", "error")
    return redirect(url_for("home"))

# ----------------- RUN APP -----------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)