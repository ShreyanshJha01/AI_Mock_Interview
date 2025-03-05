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

def ats_without_jd(resume_text):
    resume_keywords = extract_keywords(resume_text)

    resume_vector = vectorizer.transform([resume_text])

    predicted_category = model.predict(resume_vector)[0]

    dataset_text = " ".join(vectorizer.get_feature_names_out())
    dataset_vector = vectorizer.transform([dataset_text])
    similarity_score = cosine_similarity(resume_vector, dataset_vector)[0][0]

    ats_score = round(similarity_score * 100, 2)
    return {"ats_score": ats_score, "predicted_category": predicted_category}
