# AI Resume & Interview Assistant


This project is a resume analysis and mock interview application built using Streamlit and NLP techniques. It extracts details from resumes, compares them to job roles, and simulates interviews with AI-generated questions and feedback.


Features

Parses resumes from PDF files

Matches resumes to job roles using semantic similarity (FAISS + Sentence Transformers)

Optional custom job description matching

Technical and HR interview simulation using Google Gemini

Feedback generation for candidate answers


Dataset

The resume-job matching is based on the structured resume dataset from Kaggle:

https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured


Tech Stack

Streamlit for the web interface

PyMuPDF for extracting text from PDFs

Sentence Transformers for vector embeddings

FAISS for efficient similarity search

Google Generative AI (Gemini) for generating and evaluating interview questions

Pandas and Scikit-learn for data handling and metrics


Setup Instructions

Clone the repository and install dependencies using pip.

Add your Gemini API key to llm.py where it says configure(api_key="Your api key").

Download the dataset from Kaggle and run preprocess.py to generate the cleaned CSV file.

Run the app using streamlit run app.py.


File Structure

app.py – Main Streamlit app for resume upload, analysis, and interview

llm.py – Handles Gemini API calls for question generation and feedback

model1.py – Handles resume-job matching using embeddings and FAISS

preprocess.py – Merges and cleans the raw dataset files into one CSV

final_cleaned.csv – Generated CSV used for matching (after preprocessing)


How It Works
You upload a resume in PDF format. The app extracts key sections like education, experience, and skills. It then either matches the resume to the most relevant job from the dataset or compares it to a custom job description. After the resume analysis, the app presents technical and HR questions. Once you answer them, it uses Gemini to evaluate your responses and give feedback.
