import pdfplumber
import docx

def extract_text(path):
    text = ""
    if path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif path.endswith(".docx"):
        doc = docx.Document(path)
        text = " ".join([p.text for p in doc.paragraphs])
    return text
