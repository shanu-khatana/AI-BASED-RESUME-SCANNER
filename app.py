#from skill_extractor import extract_skills
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
import sqlite3
import os
import fitz  # PyMuPDF for PDF
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import re
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from itsdangerous import URLSafeTimedSerializer
import datetime
import spacy
import csv
from io import StringIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import io

# ----------------- FLASK CONFIG -----------------
app = Flask(__name__)
app.secret_key = "your_super_secret_key_change_this_in_production"
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
app.config['RESET_PASSWORD_EXPIRATION'] = 3600  # 1 hour in seconds

# ----------------- SENTENCE-BERT MODEL -----------------
# Load model once at startup
print("Loading Sentence-BERT model...")
try:
    model = SentenceTransformer('all-mpnet-base-v2')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except:
    logger.warning("spaCy model not found. Some features may not work optimally.")
    nlp = None

# ----------------- DATABASE SETUP -----------------
def get_db_connection():
    """Get a database connection with proper error handling"""
    try:
        conn = sqlite3.connect("users.db", timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds timeout
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def init_db():
    """Initialize database with proper schema including all required columns"""
    conn = get_db_connection()
    if conn is None:
        return
    
    c = conn.cursor()
    
    try:
        # Create users table
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create password reset table
        c.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                token TEXT UNIQUE NOT NULL,
                expiration TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Create scans table with ALL required columns
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
                years_experience INTEGER DEFAULT 0,
                education_level INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()
        print("Database initialized successfully!")
        
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()

def check_and_add_columns():
    """Check and add missing columns if they don't exist"""
    conn = get_db_connection()
    if conn is None:
        return
    
    c = conn.cursor()
    
    try:
        # Check existing columns
        c.execute("PRAGMA table_info(scans)")
        existing_columns = [row[1] for row in c.fetchall()]
        
        # Add missing columns if they don't exist
        if 'years_experience' not in existing_columns:
            print("Adding years_experience column...")
            c.execute("ALTER TABLE scans ADD COLUMN years_experience INTEGER DEFAULT 0")
        
        if 'education_level' not in existing_columns:
            print("Adding education_level column...")
            c.execute("ALTER TABLE scans ADD COLUMN education_level INTEGER DEFAULT 0")
        
        conn.commit()
        print("Database schema updated successfully!")
        
    except Exception as e:
        print(f"Error updating database schema: {e}")
    finally:
        if conn:
            conn.close()

# Initialize database with schema check
init_db()
check_and_add_columns()

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
    conn = get_db_connection()
    if conn is None:
        return
    
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
    conn = get_db_connection()
    if conn is None:
        return False
    
    c = conn.cursor()
    
    # Check if token exists and is not expired
    c.execute("SELECT user_id FROM password_resets WHERE token=? AND expiration > datetime('now')", (token,))
    result = c.fetchone()
    
    conn.close()
    
    return result is not None

def delete_reset_token(token):
    conn = get_db_connection()
    if conn is None:
        return
    
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

def clean_text(text):
    """Clean and preprocess text for analysis"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_text(file_path):
    """Extract text from PDF or DOCX file"""
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""
    
    return text

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
        "exclusions": ["course", "class", "training", "learned", "studied"]
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
        "variations": ["angular", "angularjs", "angular2", "angular4", "angular6", "angular8"],
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
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "database", "query"],
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
        "exclusions": ["course", "class", "training", "learned", "studied", "ai ml", "ai & ml"],
    },
    "ai": {
        "variations": ["ai", "artificial intelligence"],
        "contexts": ["experience", "skilled", "proficient", "knowledge", "worked", "developed", "implemented"],
        "exclusions": ["course", "class", "training", "learned", "studied", "artificial intelligence machine learning"]
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
            
            # Additional check: ensure it's not part of a larger phrase
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
    if not model:
        logger.warning("Sentence-BERT model not loaded. Using fallback method.")
        return 0.0
    
    try:
        # Generate embeddings
        job_embedding = model.encode(job_desc, convert_to_tensor=True)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_score = util.pytorch_cos_sim(job_embedding, resume_embedding)
        return cosine_score.item() * 100  # Convert to percentage
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {str(e)}")
        return 0.0

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

        conn = get_db_connection()
        if conn is None:
            return
        
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

        conn = get_db_connection()
        if conn is None:
            return
        
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
        
        conn = get_db_connection()
        if conn is None:
            return
        
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
        conn = get_db_connection()
        if conn is None:
            return
        
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
    
    # Get user's scan statistics
    conn = get_db_connection()
    if conn is None:
        return
    
    c = conn.cursor()
    
    try:
        # Get scan statistics - handle missing columns gracefully
        c.execute("SELECT COUNT(*) FROM scans WHERE user_id=?", (session["user_id"],))
        total_scans = c.fetchone()[0]
        
        # Check if column exists before querying
        c.execute("PRAGMA table_info(scans)")
        columns = [row[1] for row in c.fetchall()]
        
        has_years_experience = 'years_experience' in columns
        has_education_level = 'education_level' in columns
        
        if has_years_experience and has_education_level:
            c.execute("SELECT AVG(similarity_score) FROM scans WHERE user_id=?", (session["user_id"],))
            avg_score_result = c.fetchone()
            avg_score = round(avg_score_result[0], 1) if avg_score_result[0] else 0
            
            c.execute("SELECT MAX(similarity_score) FROM scans WHERE user_id=?", (session["user_id"],))
            high_score_result = c.fetchone()
            high_score = round(high_score_result[0], 1) if high_score_result[0] else 0
        else:
            # Fallback if columns don't exist
            avg_score = 0
            high_score = 0
        
        # Get recent batch results from session
        recent_batch = session.get('batch_results', [])
        
        # Get individual scan history - handle missing columns gracefully
        if has_years_experience and has_education_level:
            c.execute("""
                SELECT id, filename, similarity_score, scan_date, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE user_id = ? 
                ORDER BY scan_date DESC 
                LIMIT 10
            """, (session["user_id"],))
        else:
            # Original query without new columns
            c.execute("""
                SELECT id, filename, similarity_score, scan_date, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE user_id = ? 
                ORDER BY scan_date DESC 
                LIMIT 10
            """, (session["user_id"],))
        
        scans = c.fetchall()
        conn.close()
        
        # Format scans for display
        formatted_scans = []
        for scan in scans:
            formatted_scan = {
                'id': scan[0],
                'filename': scan[1],
                'score': scan[2],
                'date': scan[3],
                'matched_skills': scan[4].split(',') if scan[4] else [],
                'missing_skills': scan[5].split(',') if scan[5] else [],
                'years_experience': scan[6] if len(scan) > 6 else 0,
                'education_level': scan[7] if len(scan) > 7 else 0
            }
            formatted_scans.append(formatted_scan)
        
        return render_template("dashboard.html", 
                         username=session["username"], 
                         scans=formatted_scans,
                         total_scans=total_scans,
                         avg_score=avg_score,
                         high_score=high_score,
                         recent_batch=recent_batch)
    
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        flash("An error occurred while loading dashboard.", "error")
        return redirect(url_for("login"))

# ----------------- CHECK RECENT BATCHES -----------------
@app.route("/check_recent_batches")
def check_recent_batches():
    if "username" not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    
    # Check if user has recent batch results
    recent_batch = session.get('batch_results', [])
    has_recent = len(recent_batch) > 0
    
    return jsonify({"has_recent": has_recent})

# ----------------- VIEW SCAN DETAILS -----------------
# ----------------- VIEW SCAN DETAILS -----------------
@app.route("/view_scan/<int:scan_id>")
def view_scan(scan_id):
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    conn = get_db_connection()
    if conn is None:
        flash("Database connection error", "error")
        return redirect(url_for("dashboard"))
    
    try:
        c = conn.cursor()
        
        # Check if columns exist
        c.execute("PRAGMA table_info(scans)")
        columns = [row[1] for row in c.fetchall()]
        
        has_years_experience = 'years_experience' in columns
        has_education_level = 'education_level' in columns
        
        # Query with proper column handling
        if has_years_experience and has_education_level:
            c.execute("""
                SELECT id, filename, job_description, similarity_score, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE id=? AND user_id=?
            """, (scan_id, session["user_id"]))
        else:
            # Fallback for older database schema
            c.execute("""
                SELECT id, filename, job_description, similarity_score, matched_skills, missing_skills,
                       0 as years_experience, 0 as education_level
                FROM scans 
                WHERE id=? AND user_id=?
            """, (scan_id, session["user_id"]))
        
        scan = c.fetchone()
        conn.close()
        
        if not scan:
            flash("Scan not found or you don't have permission to view it.", "error")
            return redirect(url_for("dashboard"))
        
        # Parse skills for display
        matched_skills = [s.strip() for s in scan[4].split(',') if s.strip()] if scan[4] else []
        missing_skills = [s.strip() for s in scan[5].split(',') if s.strip()] if scan[5] else []
        
        logger.info(f"Viewing scan {scan_id} - {scan[1]}")
        
        return render_template("scan_details.html",
                             scan=scan,
                             matched_skills=matched_skills,
                             missing_skills=missing_skills)
    
    except Exception as e:
        logger.error(f"Error loading scan details: {str(e)}", exc_info=True)
        flash("Error loading scan details", "error")
        return redirect(url_for("dashboard"))
    
# ----------------- DOWNLOAD SCAN -----------------
@app.route("/download_scan/<int:scan_id>")
def download_scan(scan_id):
    if "username" not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    
    # Get scan details from database
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "message": "Scan not found"})
    
    try:
        # Check if columns exist
        c = conn.cursor()
        c.execute("PRAGMA table_info(scans)")
        columns = [row[1] for row in c.fetchall()]
        
        has_years_experience = 'years_experience' in columns
        has_education_level = 'education_level' in columns
        
        if has_years_experience and has_education_level:
            c.execute("""
                SELECT id, filename, job_description, similarity_score, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE id=? AND user_id=?
            """, (scan_id, session["user_id"]))
        else:
            # Query without new columns
            c.execute("""
                SELECT id, filename, job_description, similarity_score, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE id=? AND user_id=?
            """, (scan_id, session["user_id"]))
        
        scan = c.fetchone()
        conn.close()
        
        if not scan:
            return jsonify({"success": False, "message": "Scan not found"})
        
        # Convert to CSV format
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Filename', 'Job Description', 'Score', 'Matched Skills', 'Missing Skills'])
        
        # Write data
        matched_skills = scan[4].split(',') if scan[4] and len(scan) > 4 else []
        missing_skills = scan[5].split(',') if scan[5] and len(scan) > 5 else []
        writer.writerow([
            scan[1] if len(scan) > 1 else "Unknown",
            scan[2][:100] + ('...' if len(scan[2]) > 100 else ''),  # Truncate long job descriptions
            f"{scan[3] if len(scan) > 3 else 0}%",
            ', '.join(matched_skills),
            ', '.join(missing_skills)
        ])
        
        # Create response
        response = app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={scan[1] if len(scan) > 1 else "scan"}_scan_report.csv'}
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error downloading scan: {str(e)}")
        return jsonify({"success": False, "message": "Error downloading scan"})

# ----------------- DELETE SCAN -----------------
@app.route("/delete_scan/<int:scan_id>", methods=["POST"])
def delete_scan(scan_id):
    if "username" not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "message": "Scan not found"})
    
    try:
        # Check if scan belongs to user
        c = conn.cursor()
        c.execute("SELECT id FROM scans WHERE id=? AND user_id=?", (scan_id, session["user_id"]))
        scan = c.fetchone()
        
        if not scan:
            conn.close()
            return jsonify({"success": False, "message": "Scan not found"})
        
        # Delete scan
        c.execute("DELETE FROM scans WHERE id=? AND user_id=?", (scan_id, session["user_id"]))
        conn.commit()
        
        return jsonify({"success": True, "message": "Scan deleted successfully"})
    
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"success": False, "message": "Error deleting scan"})

# ----------------- EXPORT DATA -----------------
@app.route("/export")
def export_data():
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    # Get all user's scans
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "message": "Not logged in"})
    
    try:
        # Check if columns exist
        c = conn.cursor()
        c.execute("PRAGMA table_info(scans)")
        columns = [row[1] for row in c.fetchall()]
        
        has_years_experience = 'years_experience' in columns
        has_education_level = 'education_level' in columns
        
        if has_years_experience and has_education_level:
            c.execute("""
                SELECT id, filename, similarity_score, scan_date, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE user_id=? 
                ORDER BY scan_date DESC
            """, (session["user_id"],))
        else:
            # Query without new columns
            c.execute("""
                SELECT id, filename, similarity_score, scan_date, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE user_id=? 
                ORDER BY scan_date DESC
            """, (session["user_id"],))
        
        scans = c.fetchall()
        conn.close()
        
        # Convert to CSV format
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Filename', 'Similarity Score', 'Scan Date', 'Matched Skills', 'Missing Skills', 
                        'Years Experience', 'Education Level'])
        
        # Write data
        for scan in scans:
            matched_skills = scan[4].split(',') if scan[4] and len(scan) > 4 else []
            missing_skills = scan[5].split(',') if scan[5] and len(scan) > 5 else []
            writer.writerow([
                scan[1] if len(scan) > 1 else "Unknown",
                f"{scan[2] if len(scan) > 2 else 0}%",
                scan[3] if len(scan) > 3 else "",
                ', '.join(matched_skills),
                ', '.join(missing_skills),
                scan[5] if len(scan) > 5 else 0,
                scan[6] if len(scan) > 6 else 0
            ])
        
        # Create response
        response = app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=scan_history.csv'}
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({"success": False, "message": "Error exporting data"})

# ----------------- EXPORT DATA TO PDF -----------------
@app.route("/export_pdf")
def export_data_pdf():
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    # Get all user's scans
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "message": "Not logged in"})
    
    try:
        # Check if columns exist
        c = conn.cursor()
        c.execute("PRAGMA table_info(scans)")
        columns = [row[1] for row in c.fetchall()]
        
        has_years_experience = 'years_experience' in columns
        has_education_level = 'education_level' in columns
        
        if has_years_experience and has_education_level:
            c.execute("""
                SELECT id, filename, similarity_score, scan_date, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE user_id=? 
                ORDER BY similarity_score DESC
            """, (session["user_id"],))
        else:
            # Query without new columns
            c.execute("""
                SELECT id, filename, similarity_score, scan_date, matched_skills, missing_skills,
                       years_experience, education_level
                FROM scans 
                WHERE user_id=? 
                ORDER BY similarity_score DESC
            """, (session["user_id"],))
        
        scans = c.fetchall()
        conn.close()
        
        # Create a PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Container for 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = styles['h1']
        normal_style = styles['Normal']
        
        # Add title
        elements.append(Paragraph("AI Resume Scanner - Scan History", title_style))
        elements.append(Spacer(1, 12))
        
        # Add user info
        elements.append(Paragraph(f"User: {session['username']}", normal_style))
        elements.append(Paragraph(f"Export Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        elements.append(Spacer(1, 12))
        
        # Create table data
        table_data = [['ID', 'Filename', 'Score', 'Date', 'Matched Skills', 'Missing Skills', 'Experience', 'Education']]
        
        for scan in scans:
            matched_skills = scan[4].split(',') if scan[4] and len(scan) > 4 else []
            missing_skills = scan[5].split(',') if scan[5] and len(scan) > 5 else []
            
            # Format education level
            education_levels = ['None', 'High School', 'Diploma', 'Bachelor', 'Master', 'PhD']
            education = education_levels[scan[6]] if scan[6] < len(education_levels) else 'Unknown'
            
            table_data.append([
                str(scan[0]),
                scan[1] if len(scan) > 1 else "Unknown",
                f"{scan[2] if len(scan) > 2 else 0}%",
                scan[3] if len(scan) > 3 else "",
                ', '.join(matched_skills[:3]) + ('...' if len(matched_skills) > 3 else ''),
                ', '.join(missing_skills[:3]) + ('...' if len(missing_skills) > 3 else ''),
                f"{scan[5] if len(scan) > 5 else 0} years",
                education
            ])
        
        # Create table
        table = Table(table_data, colWidths=[0.5*inch, 1.5*inch, 0.7*inch, 1*inch, 1.5*inch, 1.5*inch, 0.8*inch, 0.8*inch])
        
        # Style table
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])
        
        table.setStyle(style)
        
        # Add table to elements
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        
        # Get value of BytesIO buffer
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        
        # Create response
        response = Response(
            pdf_data,
            mimetype='application/pdf',
            headers={'Content-Disposition': 'attachment; filename=scan_history.pdf'}
        )
        
        return response
    
    except Exception as e:
        logger.error(f"PDF export error: {str(e)}")
        return jsonify({"success": False, "message": "Error exporting data to PDF"})

# ----------------- NEW BATCH -----------------
@app.route("/new_batch")
def new_batch():
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    # Clear previous batch results
    if 'batch_results' in session:
        session.pop('batch_results', None)
    if 'job_description' in session:
        session.pop('job_description', None)
    
    return redirect(url_for("batch_upload"))

# ----------------- BATCH UPLOAD -----------------
# Replace the batch_upload route in your app.py with this improved version

@app.route("/batch_upload", methods=["GET", "POST"])
def batch_upload():
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    if request.method == "GET":
        return render_template("batch_upload.html")
    
    # Handle POST request for batch upload
    try:
        job_desc = request.form.get("jobdesc", "").strip()
        
        # Get all files from the request
        resume_files = request.files.getlist("resumes")
        
        logger.info(f"Received job description: {len(job_desc)} characters")
        logger.info(f"Number of files received: {len(resume_files)}")
        
        # Validation
        if not job_desc:
            logger.warning("No job description provided")
            return jsonify({"success": False, "message": "Please enter a job description!"})
        
        if not resume_files or len(resume_files) == 0:
            logger.warning("No files uploaded")
            return jsonify({"success": False, "message": "Please upload at least one resume!"})
        
        # Check if any file was actually selected
        valid_files = []
        for file in resume_files:
            if file and file.filename and file.filename != '':
                if allowed_file(file.filename):
                    valid_files.append(file)
                    logger.info(f"Valid file: {file.filename}")
                else:
                    logger.warning(f"Invalid file type: {file.filename}")
        
        if not valid_files:
            return jsonify({"success": False, "message": "No valid files were uploaded! Please upload PDF or DOCX files."})
        
        if len(valid_files) > 10:
            return jsonify({"success": False, "message": "Please upload no more than 10 resumes at once!"})
        
        logger.info(f"Processing {len(valid_files)} valid files")
        
        # Create a batch ID to group these scans
        batch_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Process each file and store results
        results = []
        processed_count = 0
        failed_files = []
        
        for file in valid_files:
            try:
                # Save file temporarily
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{batch_id}_{filename}")
                file.save(file_path)
                
                logger.info(f"Saved file: {file_path}")
                
                # Extract text from resume
                resume_text = extract_text(file_path)
                
                if not resume_text.strip():
                    logger.warning(f"Could not extract text from {filename}")
                    failed_files.append(filename)
                    # Clean up file
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    continue
                
                logger.info(f"Extracted {len(resume_text)} characters from {filename}")
                
                # Clean and preprocess text
                resume_clean = clean_text(resume_text)
                job_clean = clean_text(job_desc)
                
                # Calculate semantic similarity using Sentence-BERT
                similarity = calculate_semantic_similarity(job_desc, resume_text)
                
                # Use advanced skills analysis
                matched_skills, missing_skills = extract_skills_from_text(resume_text, SKILL_DEFINITIONS)
                # 1) Extract required skills from job description
                '''gemini_output = extract_skills(job_desc)
                required_skills = [s.strip("-• ").strip() for s in gemini_output.split("\n") if s.strip()]

                # 2) Match resume with required skills
                resume_text_lower = resume_text.lower()
                matched_skills = []
                missing_skills = []

                for skill in required_skills:
                    if skill.lower() in resume_text_lower:
                        matched_skills.append(skill)
                    else:
                        missing_skills.append(skill)'''
                        
                # Extract additional metrics
                years_experience = extract_years_experience(resume_text)
                education_level = extract_education_level(resume_text)
                
                # Calculate comprehensive score
                skills_score = len(matched_skills) / len(SKILL_DEFINITIONS) * 100
                experience_score = min(years_experience * 10, 30)  # Max 30 points for experience
                education_score = education_level * 10  # Max 50 points for education
                
                # Weighted final score
                final_score = (similarity * 0.5) + (skills_score * 0.3) + (experience_score * 0.1) + (education_score * 0.1)
                
                logger.info(f"Calculated score for {filename}: {final_score}")
                
                # Store result
                result = {
                    'filename': filename,
                    'score': round(final_score, 2),
                    'similarity_score': round(similarity, 2),
                    'skills_score': round(skills_score, 2),
                    'matched_count': len(matched_skills),
                    'total_skills': len(SKILL_DEFINITIONS),
                    'missing_skills': len(missing_skills),
                    'years_experience': years_experience,
                    'education_level': education_level,
                    'matched_skills': matched_skills[:10],
                    'missing_skills': missing_skills[:10]
                }
                results.append(result)
                
                # Save to database
                conn = get_db_connection()
                if conn is not None:
                    c = conn.cursor()
                    
                    # Check if columns exist
                    c.execute("PRAGMA table_info(scans)")
                    columns = [row[1] for row in c.fetchall()]
                    
                    has_years_experience = 'years_experience' in columns
                    has_education_level = 'education_level' in columns
                    
                    if has_years_experience and has_education_level:
                        c.execute("""
                            INSERT INTO scans (user_id, filename, job_description, similarity_score, 
                                             matched_skills, missing_skills, years_experience, education_level)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            session["user_id"],
                            filename,
                            job_desc[:500],
                            round(final_score, 2),
                            ','.join(matched_skills[:20]),
                            ','.join(missing_skills[:20]),
                            years_experience,
                            education_level
                        ))
                    
                    conn.commit()
                    conn.close()
                
                processed_count += 1
                logger.info(f"Successfully processed {filename}")
                
                # Clean up temporary file
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {file_path}: {e}")
            
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
                failed_files.append(file.filename)
                continue
        
        # Store results in session
        if results:
            session['batch_results'] = results
            session['job_description'] = job_desc
            
            logger.info(f"Successfully processed {processed_count}/{len(valid_files)} resumes")
            
            message = f"Successfully processed {processed_count} resume(s)!"
            if failed_files:
                message += f" Failed to process: {', '.join(failed_files)}"
            
            return jsonify({
                "success": True, 
                "message": message,
                "processed": processed_count,
                "total": len(valid_files),
                "failed": failed_files
            })
        else:
            logger.error("No resumes were successfully processed")
            return jsonify({
                "success": False, 
                "message": "Failed to process any resumes. Please check file formats and try again."
            })
    
    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False, 
            "message": f"An error occurred during upload: {str(e)}"
        })

# ----------------- BATCH RESULTS -----------------
@app.route("/batch_results")
@app.route("/batch_results/<int:page>")
def batch_results(page=1):
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    # Get batch results from session
    results = session.get('batch_results', [])
    job_description = session.get('job_description', '')
    
    if not results:
        flash("No batch results to display. Please start a new batch scan.", "info")
        return redirect(url_for("dashboard"))
    
    # Sort results by score in descending order (highest first)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Pagination settings
    results_per_page = 10
    total_results = len(sorted_results)
    total_pages = (total_results + results_per_page - 1) // results_per_page
    start_idx = (page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    
    # Get results for current page
    paginated_results = sorted_results[start_idx:end_idx]
    
    return render_template("batch_results.html", 
                         results=paginated_results,
                         job_description=job_description,
                         page=page,
                         total_pages=total_pages)

# ----------------- SINGLE SCAN UPLOAD -----------------
@app.route("/single_scan", methods=["GET", "POST"])
def single_scan():
    """Single resume scan page"""
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    if request.method == "GET":
        return render_template("single_scan.html")
    
    # Handle POST request for single scan
    job_desc = request.form.get("jobdesc", "").strip()
    resume = request.files.get("resume")

    if not job_desc:
        return jsonify({"success": False, "message": "Please enter a job description!"})
    
    if not resume or resume.filename == '':
        return jsonify({"success": False, "message": "Please upload a resume!"})
    
    if not allowed_file(resume.filename):
        return jsonify({"success": False, "message": "Invalid file type! Please upload PDF or DOCX files only."})
    
    try:
        # Secure filename and save file
        filename = secure_filename(resume.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        resume.save(file_path)

        # Extract text from resume
        resume_text = extract_text(file_path)
        
        if not resume_text.strip():
            logger.warning(f"Could not extract text from {filename}")
            return jsonify({"success": False, "message": "Could not extract text from resume. Please try another file."})
        
        # Clean and preprocess text
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_desc)
        
        # Calculate semantic similarity using Sentence-BERT
        similarity = calculate_semantic_similarity(job_desc, resume_text)
        
        # Use advanced skills analysis
        matched_skills, missing_skills = extract_skills_from_text(resume_text, SKILL_DEFINITIONS)
        # 1) Extract required skills from Job Description using Gemini
        '''gemini_output = extract_skills(job_desc)
        required_skills = [s.strip("-• ").strip() for s in gemini_output.split("\n") if s.strip()]

        # 2) Match resume with required skills
        resume_text_lower = resume_text.lower()
        matched_skills = []
        missing_skills = []

        for skill in required_skills:
            if skill.lower() in resume_text_lower:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)'''
        
        
        
        # Extract additional metrics
        years_experience = extract_years_experience(resume_text)
        education_level = extract_education_level(resume_text)
        
        # Calculate comprehensive score
        skills_score = len(matched_skills) / len(SKILL_DEFINITIONS) * 100
        experience_score = min(years_experience * 10, 30)  # Max 30 points for experience
        education_score = education_level * 10  # Max 50 points for education
        
        # Weighted final score
        final_score = (similarity * 0.5) + (skills_score * 0.3) + (experience_score * 0.1) + (education_score * 0.1)
        
        # Save to database
        conn = get_db_connection()
        if conn is None:
            return jsonify({"success": False, "message": "Database connection failed"})
        
        c = conn.cursor()
        
        # Check if columns exist
        c.execute("PRAGMA table_info(scans)")
        columns = [row[1] for row in c.fetchall()]
        
        has_years_experience = 'years_experience' in columns
        has_education_level = 'education_level' in columns
        
        if has_years_experience and has_education_level:
            c.execute("""
                INSERT INTO scans (user_id, filename, job_description, similarity_score, 
                                 matched_skills, missing_skills, years_experience, education_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                filename,
                job_desc[:500],  # Limit job description length
                round(final_score, 2),
                ','.join(matched_skills[:20]),
                ','.join(missing_skills[:20]),
                years_experience,
                education_level
            ))
        
        conn.commit()
        conn.close()
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        # Return success response with score details
        return jsonify({
            "success": True,
            "score": round(final_score, 2),
            "similarity_score": round(similarity, 2),
            "skills_score": round(skills_score, 2),
            "matched_count": len(matched_skills),
            "total_skills": len(SKILL_DEFINITIONS),
            "missing_skills": len(missing_skills),
            "years_experience": years_experience,
            "education_level": education_level,
            "matched_skills": matched_skills[:10],  # Include actual skill names
            "missing_skills": missing_skills[:10]   # Include actual skill names
        })
    
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        return jsonify({"success": False, "message": "Error processing resume"})

# ----------------- SAVE SCAN -----------------
@app.route("/save_scan", methods=["POST"])
def save_scan():
    if "username" not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    
    try:
        # Get scan data from request
        scan_data = request.get_json()
        
        if not scan_data:
            return jsonify({"success": False, "message": "No scan data provided"})
        
        # Save to database
        conn = get_db_connection()
        if conn is None:
            return jsonify({"success": False, "message": "Database connection failed"})
        
        c = conn.cursor()
        
        # Check if columns exist
        c.execute("PRAGMA table_info(scans)")
        columns = [row[1] for row in c.fetchall()]
        
        has_years_experience = 'years_experience' in columns
        has_education_level = 'education_level' in columns
        
        if has_years_experience and has_education_level:
            c.execute("""
                INSERT INTO scans (user_id, filename, job_description, similarity_score, 
                                 matched_skills, missing_skills, years_experience, education_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                scan_data.get("filename", "unknown"),
                scan_data.get("jobdesc", "")[:500],  # Limit job description length
                scan_data.get("score", 0),
                ','.join(scan_data.get("matched_skills", [])[:20]),
                ','.join(scan_data.get("missing_skills", [])[:20]),
                scan_data.get("years_experience", 0),
                scan_data.get("education_level", 0)
            ))
        
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "message": "Scan saved successfully"})
    
    except Exception as e:
        logger.error(f"Error saving scan: {str(e)}")
        return jsonify({"success": False, "message": "Error saving scan"})

# ----------------- GET SCAN DETAILS -----------------
@app.route("/get_scan_details")
def get_scan_details():
    if "username" not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    
    filename = request.args.get("filename", "")
    job_desc = request.args.get("jobdesc", "")
    
    if not filename or not job_desc:
        return jsonify({"success": False, "message": "Missing parameters"})
    
    try:
        # Find file path
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({"success": False, "message": "File not found"})
        
        # Extract text from resume
        resume_text = extract_text(file_path)
        
        if not resume_text.strip():
            return jsonify({"success": False, "message": "Could not extract text from resume"})
        
        # Extract skills
        matched_skills, missing_skills = extract_skills_from_text(resume_text, SKILL_DEFINITIONS)
        
        return jsonify({
            "success": True,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })
    
    except Exception as e:
        logger.error(f"Error getting scan details: {str(e)}")
        return jsonify({"success": False, "message": "Error getting scan details"})

# ----------------- SETTINGS -----------------
@app.route("/settings")
def settings():
    if "username" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    # Get user details from database
    conn = get_db_connection()
    if conn is None:
        return render_template("settings.html", username=session.get("username", "User"))
    
    try:
        c = conn.cursor()
        c.execute("SELECT username, email FROM users WHERE id=?", (session["user_id"],))
        user = c.fetchone()
        conn.close()
        
        if user:
            return render_template("settings.html", 
                             username=session["username"], 
                             email=user[1] if user[1] else "")
        else:
            return render_template("settings.html", username=session.get("username", "User"))
    
    except Exception as e:
        logger.error(f"Error loading user settings: {str(e)}")
        return render_template("settings.html", username=session.get("username", "User"))

# ----------------- UPDATE PROFILE -----------------
@app.route("/update_profile", methods=["POST"])
def update_profile():
    if "username" not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    
    email = request.form.get("email", "").strip()
    
    if not email:
        return jsonify({"success": False, "message": "Email is required"})
    
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"success": False, "message": "Database connection failed"})
        
        c = conn.cursor()
        c.execute("UPDATE users SET email=? WHERE id=?", (email, session["user_id"]))
        conn.commit()
        conn.close()
        
        flash("Profile updated successfully!", "success")
        return jsonify({"success": True, "message": "Profile updated successfully"})
    
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        return jsonify({"success": False, "message": "Error updating profile"})

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