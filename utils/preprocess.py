import re
import nltk
from nltk.corpus import stopwords

# ✅ Download stopwords automatically if not present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans and preprocesses text:
    - Converts to lowercase
    - Removes numbers and punctuation
    - Removes stopwords
    """
    # Convert to lowercase
    text = text.lower()

    # Remove anything that is not a-z or space
    text = re.sub(r'[^a-z\s]', '', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [w for w in words if w not in stop_words]

    # Join back to a single string
    return " ".join(words)
