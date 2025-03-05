import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\Users\KIIT\Desktop\Semester\code\projects\mini project\UpdatedResumeDataSet.csv")  

# Extract text and labels
X = df["Resume"]  
y = df["Category"]  

vectorizer = TfidfVectorizer(max_features=5000)
X_vectors = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "resume_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model train ho gaya guys !!")
