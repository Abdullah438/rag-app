import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import tempfile

# Load environment variables
load_dotenv()

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set up the page
st.title("📚 Document Q&A with RAG")
st.write("Upload PDF documents and ask questions about them!")

def process_documents(uploaded_files):
    # Create a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        documents = []
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)
            
            # Save uploaded file temporarily
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load PDF
            loader = PyPDFLoader(temp_filepath)
            documents.extend(loader.load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(documents)

        # Create embeddings and store in vector database
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        vectorstore = Chroma.from_documents(splits, embeddings)

        # Create conversation chain
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

# File uploader
uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            process_documents(uploaded_files)
        st.success("Documents processed! You can now ask questions.")

# Query input
if st.session_state.conversation:
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        with st.spinner("Generating response..."):
            # Generate response
            response = st.session_state.conversation({
                "question": user_question,
                "chat_history": st.session_state.chat_history
            })
            
            # Update chat history
            st.session_state.chat_history.append((user_question, response["answer"]))

            # Display response as a stream
            response_placeholder = st.empty()
            streamed_response = ""

            for part in response["answer"]:
                streamed_response += part
                response_placeholder.markdown(streamed_response)
            
            # Display source documents
            with st.expander("View Source Documents"):
                for source_doc in response["source_documents"]:
                    st.write(source_doc.page_content)
                    st.write("---")

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History:")
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.write(f"**Question {i+1}:** {question}")
        st.write(f"**Answer {i+1}:** {answer}")
        st.write("---")
