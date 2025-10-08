import spacy, re
nlp = spacy.load("en_core_web_sm")

def anonymize(text):
    doc = nlp(text)
    result = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG", "EMAIL", "PHONE"]:
            result = result.replace(ent.text, "[REDACTED]")
    result = re.sub(r'\S+@\S+', "[REDACTED]", result)  # remove emails
    result = re.sub(r'\d{10}', "[REDACTED]", result)   # remove phone numbers
    return result
