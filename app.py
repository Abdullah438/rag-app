import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PDFPlumberLoader
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
            
            # Load PDF with PDFPlumberLoader
            loader = PDFPlumberLoader(temp_filepath)
            docs = loader.load()
            documents.extend(docs)
            
            print(f"Loaded {len(documents)} pages from {uploaded_file.name}")
            print(f"First page: {documents[0].page_content}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        print(f"Split into {len(splits)} chunks: {splits[0].page_content}")
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
                for i, source_doc in enumerate(response["source_documents"], 1):
                    st.markdown(f"**Source {i}:**")
                    # Use a code block to preserve formatting
                    st.markdown(f"```\n{source_doc.page_content}\n```")
                    st.markdown("---")

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History:")
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.write(f"**Question {i+1}:** {question}")
        st.write(f"**Answer {i+1}:** {answer}")
        st.write("---")
