import streamlit as st
import fitz  # PyMuPDF
from model1 import get_ats_score, calculate_custom_ats_score
from llm import get_technical_questions, get_hr_questions, evaluate_candidate_answers

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc])

def parse_resume_sections(text):
    sections = {"education_details": "", "experience_details": "", "skill": ""}
    current_section = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        header_check = line.upper()
        if "EDUCATION" in header_check:
            current_section = "education_details"
        elif "EXPERIENCE" in header_check:
            current_section = "experience_details"
        elif "SKILL" in header_check or "TECHNICAL SKILLS" in header_check:
            current_section = "skill"
        elif current_section:
            sections[current_section] += line + "\n"

    return sections

# Streamlit UI
st.set_page_config(page_title="AI Resume & Interview Assistant", layout="wide")
st.title(" AI-Powered Resume & Interview Assistant")
st.markdown("Get resume analysis + technical & HR mock interviews powered by Gemini")

uploaded_file = st.file_uploader(" Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area(" Optional Job Description", placeholder="Paste job requirements here...", height=150)

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    parsed_resume = parse_resume_sections(resume_text)

    with st.expander(" Resume Breakdown", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 🎓 Education")
            st.write(parsed_resume["education_details"] or "No education details found")
        with col2:
            st.write("###  Experience")
            st.write(parsed_resume["experience_details"] or "No experience details found")

        st.write("### 🛠️ Skills")
        st.write(parsed_resume["skill"] or "No skills listed")

    # Job Matching
    if job_description.strip():
        score = calculate_custom_ats_score(parsed_resume, job_description)
        rounded_score = round(float(score["ats_score"]), 2)
        st.subheader(" Custom Job Match")
        st.metric(label="Relevance Score", value=f"{rounded_score}%")
        role = "custom role"
    else:
        results = get_ats_score(parsed_resume)
        best_job = results["matched_job"]
        best_score = round(float(results["ats_score"]), 2)
        st.subheader(" Top Job Match")
        st.write(f"**Best Match:** `{best_job}` — **{best_score}% match**")
        role = best_job

    st.success("✅ Resume analysis complete!")

    # Initialize session state
    if "tech_questions" not in st.session_state:
        st.session_state.tech_questions = get_technical_questions(role)
        st.session_state.tech_answers = [""] * len(st.session_state.tech_questions)
        st.session_state.tech_submitted = False
        st.session_state.tech_feedback = ""

    if "hr_questions" not in st.session_state:
        st.session_state.hr_questions = get_hr_questions()
        st.session_state.hr_answers = [""] * len(st.session_state.hr_questions)
        st.session_state.hr_submitted = False
        st.session_state.hr_feedback = ""

    # Technical Round
    st.divider()
    with st.expander(" Mock Technical Interview", expanded=True):
        if not st.session_state.tech_submitted:
            for i, question in enumerate(st.session_state.tech_questions):
                st.session_state.tech_answers[i] = st.text_area(
                    f"Q{i+1}: {question}",
                    key=f"tech_q{i}",
                    value=st.session_state.tech_answers[i]
                )

            if st.button(" Submit Technical Answers"):
                questions = st.session_state.tech_questions
                answers = st.session_state.tech_answers
                st.session_state.tech_feedback = evaluate_candidate_answers(questions, answers, round_type="technical")

                st.session_state.tech_submitted = True
                st.rerun()
        else:
            st.write("###  Technical Feedback")
            st.markdown(st.session_state.tech_feedback)

    # HR Round
    st.divider()
    with st.expander(" Mock HR Interview", expanded=True):
        if not st.session_state.hr_submitted:
            for i, question in enumerate(st.session_state.hr_questions):
                st.session_state.hr_answers[i] = st.text_area(
                    f"HR Q{i+1}: {question}",
                    key=f"hr_q{i}",
                    value=st.session_state.hr_answers[i]
                )

            if st.button(" Submit HR Answers"):
                questions = st.session_state.hr_questions
                answers = st.session_state.hr_answers
                st.session_state.hr_feedback = evaluate_candidate_answers(questions, answers, round_type="hr")

                st.session_state.hr_submitted = True
                st.rerun()
        else:
            st.write("###  HR Feedback")
            st.markdown(st.session_state.hr_feedback)
