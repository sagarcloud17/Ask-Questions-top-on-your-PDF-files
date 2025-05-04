import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import openai

#load
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="PDF Q&A App", page_icon="📄", layout="wide")
st.title("📄 AI-Powered PDF Q&A")
st.markdown("---")

# Ask user for OpenAI API key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # Verify API Key using OpenAIEmbeddings first
    try:
        embeddings = OpenAIEmbeddings()
        embeddings.embed_query("Test")  # Validate API Key with a test query
    except openai.AuthenticationError:
        st.error("❌ Incorrect API Key provided. Please enter a valid API key.")
        st.stop()
    except openai.OpenAIError as e:
        st.error(f"⚠️ OpenAI API Error: {str(e)}")
        st.stop()
else:
    st.warning("⚠️ Please enter your OpenAI API key to proceed.")
    st.stop()

# Upload PDF file
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("⏳ Processing PDF..."):
        # Save uploaded file temporarily
        pdf_path = f"./{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and process the document
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        # Create embeddings and vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

        # Define LLM and prompt
        llm = ChatOpenAI(model="gpt-4o")  # Initialize LLM only after API verification
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            ("user","{input}")
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        st.success("✅ PDF Processed Successfully! 🎉")

        # User Input
        question = st.text_input("💬 Ask a question about the document:")
        if st.button("🔍 Get Answer") and question:
            with st.spinner("🤖 Fetching answer..."):
                result = retrieval_chain.invoke({"input": question})
                st.subheader("📢 Answer:")
                st.write(result['answer'])

st.markdown("---")
st.markdown("👨‍💻 Developed with ❤️ using LangChain & Streamlit")