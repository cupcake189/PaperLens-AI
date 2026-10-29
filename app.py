
import os
import re
import io
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from xml.sax.saxutils import escape

MODEL_NAME = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="PaperLens AI",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.hero {
    padding: 28px;
    border-radius: 16px;
    background: linear-gradient(120deg, #102b50, #2864a5);
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📚 PaperLens AI</h1>
    <p>Turn complex research papers into clear, structured summaries.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Upload a research paper in PDF format. PaperLens AI "
    "will summarize its objectives, methodology, findings, "
    "limitations, and key takeaways."
)

summary_length = st.selectbox(
    "Choose summary length",
    ["Short", "Balanced", "Detailed"],
    index=1
)

uploaded_file = st.file_uploader(
    "Upload your research paper (PDF)",
    type=["pdf"]
)

def split_chunks(text, chunk_size=12000, overlap=500):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks

def summarize_paper(text, client):
    chunks = split_chunks(text)
    notes = []

    progress = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(chunks):
        status.write(
            f"Analyzing section {i + 1} of {len(chunks)}..."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are PaperLens AI, an academic research "
                        "assistant. Use only facts supported by the "
                        "provided research paper text. Do not invent "
                        "results or conclusions."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize this paper section. Include its "
                        "main ideas, methods, results, and limitations "
                        "where available.\n\n" + chunk
                    )
                }
            ],
            temperature=0.2,
            max_tokens=1000
        )

        notes.append(response.choices[0].message.content)
        progress.progress((i + 1) / len(chunks))

    lengths = {
        "Short": 250,
        "Balanced": 550,
        "Detailed": 900
    }

    final_prompt = f"""
Create a research paper summary of approximately
{lengths[summary_length]} words using the notes below.

Use these headings:
1. Paper Overview
2. Problem and Objectives
3. Methodology
4. Key Findings
5. Limitations and Future Work
6. Plain-Language Takeaway
7. Evidence Check

Do not invent facts. Mention when information is unavailable.
Treat the notes as evidence, and flag important claims that
should be checked against the original paper.

NOTES:
{"".join(notes)}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful academic summarization assistant."
                )
            },
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2,
        max_tokens=2500
    )

    status.success("Paper analysis complete!")
    return response.choices[0].message.content

def make_pdf(summary):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PaperHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#17365D"),
        spaceBefore=12,
        spaceAfter=6
    ))

    story = [
        Paragraph("PaperLens AI", styles["Title"]),
        Paragraph("Research Paper Summary", styles["Heading2"]),
        Spacer(1, 12)
    ]

    for line in summary.splitlines():
        line = line.strip()

        if not line:
            story.append(Spacer(1, 5))
            continue

        line = line.replace("**", "").replace("*", "")

        if re.match(r"^#{1,6}\s", line) or re.match(r"^\d+\.\s", line):
            line = re.sub(r"^#{1,6}\s*", "", line)
            story.append(Paragraph(escape(line), styles["PaperHeading"]))
        else:
            story.append(Paragraph(escape(line), styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue()

if uploaded_file:
    if st.button("✨ Generate Summary", type="primary"):
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

        if not api_key:
            st.error(
                "API key not configured. Add GROQ_API_KEY in "
                "your Streamlit app Secrets settings."
            )
            st.stop()

        try:
            reader = PdfReader(uploaded_file)
            paper_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

            if not paper_text.strip():
                st.error(
                    "No readable text found. This PDF may be scanned "
                    "and require OCR before summarization."
                )
                st.stop()

            with st.spinner("Reading and summarizing your paper..."):
                client = Groq(api_key=api_key)
                summary = summarize_paper(paper_text, client)

            st.session_state["paperlens_summary"] = summary

        except Exception as e:
            st.error(f"Could not summarize the paper: {e}")

if st.session_state.get("paperlens_summary"):
    st.subheader("Your Research Summary")
    st.markdown(st.session_state["paperlens_summary"])

    pdf_bytes = make_pdf(st.session_state["paperlens_summary"])

    st.download_button(
        "📄 Download Summary as PDF",
        data=pdf_bytes,
        file_name="PaperLens_AI_Summary.pdf",
        mime="application/pdf"
    )
