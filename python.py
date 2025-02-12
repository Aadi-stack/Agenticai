import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)


# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


# Function to split text using selected method
def split_text(text, method, chunk_size, overlap):
    if method == "Character Splitter":
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    elif method == "Recursive Splitter":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    elif method == "Token Splitter":
        splitter = TokenTextSplitter(chunk_size=chunk_size)
    else:
        return ["Invalid method selected"]

    return splitter.split_text(text)


# Streamlit UI
st.title("🔍 Text Splitter Playground")

# User input: Textbox or File Upload
input_option = st.radio("Choose Input Method", ("Enter Text", "Upload PDF"))

if input_option == "Enter Text":
    user_text = st.text_area("Enter your text here:")
elif input_option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        user_text = extract_text_from_pdf(uploaded_file)
    else:
        user_text = ""

# Select text splitter method
splitter_method = st.selectbox("Select Text Splitting Method",
                               ["Character Splitter", "Recursive Splitter", "Token Splitter"])

# Parameters
chunk_size = st.slider("Chunk Size", min_value=50, max_value=1000, value=200, step=50)
overlap = st.slider("Chunk Overlap", min_value=0, max_value=100, value=20, step=10)

# Process Text
if st.button("Split Text"):
    if user_text.strip():
        chunks = split_text(user_text, splitter_method, chunk_size, overlap)

        # Analytics Panel
        total_chunks = len(chunks)
        avg_words_per_chunk = sum(len(chunk.split()) for chunk in chunks) / total_chunks if total_chunks > 0 else 0

        st.subheader("📊 Analytics Panel")
        st.write(f"**Total Chunks:** {total_chunks}")
        st.write(f"**Avg. Words per Chunk:** {avg_words_per_chunk:.2f}")

        # Color-coded Visualization
        st.subheader(f"📌 Chunks Generated ({total_chunks})")
        colors = ["#FFD700", "#FF6347", "#4682B4", "#32CD32", "#DA70D6"]
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            st.markdown(
                f'<div style="background-color:{colors[i % len(colors)]}; padding:10px; border-radius:5px; margin:5px 0;">'
                f'<b>Chunk {i + 1}:</b> {chunk}</div>', unsafe_allow_html=True)

        # Search within Chunks
        search_query = st.text_input("🔍 Search within chunks:")
        if search_query:
            matching_chunks = [chunk for chunk in chunks if search_query.lower() in chunk.lower()]
            st.write(f"**Found {len(matching_chunks)} matching chunks**")
            for chunk in matching_chunks[:5]:  # Show top 5 matches
                st.write(chunk)

        # Download as CSV
        df = pd.DataFrame({"Chunk": chunks})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Chunks as CSV", csv, "chunks.csv", "text/csv")

    else:
        st.warning("Please enter text or upload a PDF.")
