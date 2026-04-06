import streamlit as st
from rag import load_pdf, chunk_text, get_embeddings, create_index, ask_question

st.title("📄 Chat with Your PDF (Free Local AI)")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if st.session_state.index is None:
        with st.spinner("Processing PDF..."):
            text = load_pdf(uploaded_file)
            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks)
            index = create_index(embeddings)

            st.session_state.index = index
            st.session_state.chunks = chunks

        st.success("PDF processed! ✅")

if st.session_state.index is not None:
    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Thinking..."):
            answer = ask_question(
                query,
                st.session_state.chunks,
                st.session_state.index
            )

        st.write("🤖", answer)