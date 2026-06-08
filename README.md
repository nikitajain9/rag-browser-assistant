# Website Q&A Chrome Extension using LangChain RAG

## Overview

This project is a Chrome Extension that allows users to ask questions about the website they are currently viewing.

The extension extracts webpage content, sends it to a FastAPI backend, and uses a Retrieval-Augmented Generation (RAG) pipeline built with LangChain to generate accurate answers based on the webpage content.

## Features

* Ask questions about any webpage
* Retrieval-Augmented Generation (RAG)
* Semantic search using embeddings
* Keyword search using BM25
* FastAPI backend
* Gemini LLM integration
* FAISS vector database
* Chrome Extension frontend
* Source-aware question answering

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript
* Chrome Extension (Manifest V3)

### Backend

* Python
* FastAPI

### AI & RAG

* LangChain
* Google Gemini
* Sentence Transformers
* FAISS

## Project Structure

```text
website-rag-extension/
│
├── backend/
│   ├── api.py
│   ├── rag.py
│   ├── requirements.txt
│   ├── .env
│   └── vector_store/
│
└── extension/
    ├── manifest.json
    ├── popup.html
    ├── popup.css
    ├── popup.js
    └── content.js
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd website-rag-extension
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

## Running the Backend

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

Server will be available at:

```text
http://127.0.0.1:8000
```

## Chrome Extension Setup

1. Open Chrome.
2. Navigate to:

```text
chrome://extensions
```

3. Enable **Developer Mode**.
4. Click **Load Unpacked**.
5. Select the `extension` folder.

The extension should now appear in the browser toolbar.

## Workflow

1. User opens a webpage.
2. Extension extracts webpage text.
3. Text is sent to the FastAPI backend.
4. Backend chunks the content.
5. Embeddings are generated.
6. Chunks are stored in FAISS.
7. Relevant chunks are retrieved.
8. Gemini generates an answer using retrieved context.
9. Response is returned to the extension.

## Future Improvements

* Chat history support
* PDF question answering
* YouTube transcript support
* Website-wide indexing
* Caching of embeddings
* Streaming responses
* User authentication
* Multi-page knowledge base

## Learning Objectives

This project demonstrates:

* Chrome Extension Development
* FastAPI Development
* LangChain Fundamentals
* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embeddings
* Large Language Models
* API Integration

## License

This project is licensed under the MIT License.
