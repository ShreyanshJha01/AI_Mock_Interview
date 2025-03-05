import streamlit as st
import PyPDF2
import docx
import ats_des
import ats_without_des


def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

st.title("Resume ATS Scorer")
st.write("Upload your resume and job description (optional) to get the ATS match score.")

resume_file = st.file_uploader(" Upload Your Resume (PDF or DOCX)", type=["pdf", "docx"])
job_desc_file = st.file_uploader(" Upload the Job Description (Optional)", type=["pdf", "docx"])

if st.button("Submit"):
    if resume_file:
       
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_docx(resume_file)

       
        if job_desc_file:
            if job_desc_file.type == "application/pdf":
                job_desc_text = extract_text_from_pdf(job_desc_file)
            else:
                job_desc_text = extract_text_from_docx(job_desc_file)

            
            result = ats_des.ats_with_jd(resume_text, job_desc_text)

            st.success(f"ATS Score: {result['ats_score']}%")
            st.info(f" Missing Keywords: {', '.join(result['missing_keywords']) if result['missing_keywords'] else 'None'}")
            st.write(f" Predicted Category: {result['predicted_category']}")

        else:
            
            result = ats_without_des.ats_without_jd(resume_text)

            st.success(f" ATS Score: {result['ats_score']}%")
            st.write(f" Predicted Category: {result['predicted_category']}")

    else:
        st.error("Please upload your resume.")
