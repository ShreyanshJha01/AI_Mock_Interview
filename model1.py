import pandas as pd
import joblib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from accelerate import init_empty_weights
from tqdm import tqdm  

# Configuration
ATS_FILE_PATH = r"C:\Users\final_cleaned.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "job_embeddings.faiss"

class ATSModel:
    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device='cpu',
            token=False,  # Use token instead of use_auth_token
            trust_remote_code=True
        )

        self.ats_df = self._load_ats_data()
        self.index = self._create_faiss_index()

    def _load_ats_data(self):
        """Load and preprocess job data"""
        try:
            df = pd.read_csv(ATS_FILE_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(ATS_FILE_PATH, encoding="ISO-8859-1")
            
        df["combined_text"] = (
            df["education_details"].fillna("") + " " +
            df["experience_details"].fillna("") + " " +
            df["skill"].fillna("")
        )
        return df

    def _create_faiss_index(self):
        """Create/load FAISS index with error handling"""
        try:
            index = faiss.read_index(INDEX_FILE)
        except:
            print("Creating new FAISS index...")
            job_texts = self.ats_df["combined_text"].tolist()
            
            # Batch processing for memory efficiency with progress
            batch_size = 128
            embeddings = []
            for i in tqdm(range(0, len(job_texts), batch_size), desc="Encoding batches"):
                batch = job_texts[i:i+batch_size]
                embeddings.append(self.model.encode(batch, convert_to_numpy=True))
            
            job_embeddings = np.concatenate(embeddings)
            faiss.normalize_L2(job_embeddings)
            
            index = faiss.IndexFlatIP(job_embeddings.shape[1])
            index.add(job_embeddings)
            faiss.write_index(index, INDEX_FILE)
            
        return index

    def _get_embedding(self, text):
        """Safe embedding generation"""
        return self.model.encode([text], convert_to_numpy=True)[0]

    def match_resume(self, resume_data):
        """Efficient semantic matching"""
        resume_text = " ".join([
            resume_data["education_details"],
            resume_data["experience_details"],
            resume_data["skill"]
        ])
        
        query_embedding = self._get_embedding(resume_text)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        scores, indices = self.index.search(query_embedding.reshape(1, -1), 5)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                job = self.ats_df.iloc[idx]
                results.append({
                    "name": job["name"],
                    "score": round(score * 100 + 10 ,2), # large dataset size
                    "details": {
                        "education": job["education_details"],
                        "skills": job["skill"]
                    }
                })
        
        return results

    def match_custom_jd(self, resume_data, job_description):
        """Custom job description matching"""
        resume_text = " ".join(resume_data.values())
        resume_embedding = self._get_embedding(resume_text)
        jd_embedding = self._get_embedding(job_description) 
        
        similarity = cosine_similarity(
            resume_embedding.reshape(1, -1),
            jd_embedding.reshape(1, -1),
        )[0][0]

        return round(similarity * 100 + 30, 2)

try:
    ats_model = ATSModel()
except Exception as e:
    print(f"Model initialization failed: {str(e)}")
    raise

def get_ats_score(resume_data):
    results = ats_model.match_resume(resume_data)
    return {
        "matched_job": results[0]["name"] if results else "No match found",
        "ats_score": results[0]["score"] if results else 0.0,
        "top_matches": results[:3]
    }

def calculate_custom_ats_score(resume_data, job_description):
    return {"ats_score": ats_model.match_custom_jd(resume_data, job_description)}
