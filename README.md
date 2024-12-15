# PDF Question Answering with RAG

This is a Streamlit-based application that implements Retrieval-Augmented Generation (RAG) for answering questions about PDF documents. The application allows users to upload PDF files and ask questions about their content, providing accurate answers with source references.

## Features

- PDF document upload and processing
- Natural language question answering
- Source document references
- Chat history tracking
- Streaming response display

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abdullah438/rag-app.git
cd rag-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the `.env.example` file to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
```

## Configuration

Update the `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Upload your PDF documents using the file uploader

4. Ask questions about the content of your documents

5. View the answers and their source references

## How it Works

1. **Document Processing**: The application uses PyPDF loader to extract text from uploaded PDF documents.

2. **Text Splitting**: The extracted text is split into manageable chunks using RecursiveCharacterTextSplitter.

3. **Embeddings**: Document chunks are converted into embeddings using OpenAI's embedding model.

4. **Vector Storage**: Embeddings are stored in a Chroma vector store for efficient retrieval.

5. **Question Answering**: When a question is asked, the application:
   - Retrieves relevant document chunks
   - Uses OpenAI's GPT model to generate an answer
   - Provides source references for verification

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
