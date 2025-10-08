import os
from flask import Flask, render_template, request
from utils.extract import extract_text
from utils.semantic import get_similarity

app = Flask(__name__)

# ✅ Folder where uploaded resumes are stored
UPLOAD_FOLDER = "resumes"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # auto-create folder if missing

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    job_desc = request.form['jobdesc']
    resume = request.files['resume']

    if resume:
        # ✅ Save uploaded file to "resumes" folder
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
        resume.save(file_path)

        # ✅ Extract text from resume
        text = extract_text(file_path)

        # ✅ Get similarity score and missing skills
        score, missing_skills = get_similarity(text, job_desc)

        # ✅ Convert score to percentage
        score_percentage = round(score * 100, 2)

        # ✅ Render result page with animated circular progress bar
        return render_template('result.html',
                               filename=resume.filename,
                               score=score_percentage,
                               missing_skills=missing_skills)

    return "No resume uploaded!"

if __name__ == '__main__':
    app.run(debug=True)
