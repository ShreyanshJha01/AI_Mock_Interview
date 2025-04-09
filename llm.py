import google.generativeai as genai
from google.generativeai import configure

configure(api_key="Your api key")  
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Function to generate response from Gemini
def generate_gemini_response(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()

# Generate technical questions from role/resume
def get_technical_questions(role: str) -> list:
    prompt = f"Generate 5 technical interview questions for the role of {role}. Provide detailed and practical questions suitable for an experienced candidate."
    response = generate_gemini_response(prompt)
    return response.split('\n')

# Generate HR questions
def get_hr_questions() -> list:
    prompt = "Generate 5 HR interview questions suitable for evaluating a candidate's soft skills, teamwork, adaptability, and motivation."
    response = generate_gemini_response(prompt)
    return response.split('\n')

# Evaluate answers with basic feedback
def evaluate_candidate_answers(questions: list, answers: list, round_type="technical") -> str:
    combined_input = ""
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        combined_input += f"Q{i}: {q}\nA{i}: {a}\n"

    prompt = f"Evaluate the following {round_type} interview answers. Provide constructive feedback and a rating out of 10 for each:\n\n{combined_input}"
    return generate_gemini_response(prompt)
