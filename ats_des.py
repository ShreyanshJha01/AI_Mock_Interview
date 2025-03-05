import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


model = joblib.load("resume_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text.lower())
    return {token.lemma_ for token in doc if token.is_alpha and not token.is_stop}

def ats_with_jd(resume_text, job_desc_text):
    res_keywords = extract_keywords(resume_text)
    jobdes_keywords = extract_keywords(job_desc_text)
    missing_keywords = jobdes_keywords - res_keywords


    resume_vector = vectorizer.transform([resume_text])
    job_desc_vector = vectorizer.transform([job_desc_text])


    predicted_category = model.predict(resume_vector)[0]


    similarity_score = cosine_similarity(resume_vector, job_desc_vector)[0][0]
    keyword_match_score = len(res_keywords & jobdes_keywords) / len(jobdes_keywords) if jobdes_keywords else 0

    ats_score = round((0.7 * similarity_score + 0.3 * keyword_match_score) * 100, 2)

    return {"ats_score": ats_score, "missing_keywords": list(missing_keywords), "predicted_category": predicted_category}
