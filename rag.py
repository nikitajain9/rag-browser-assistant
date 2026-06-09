import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()


embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

def rag_pipeline(content: str, query: str):

    doc = Document(page_content=content)

    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    keyword_search = BM25Retriever.from_documents(chunks)
    keyword_search.k = 3

    final_retriever = EnsembleRetriever(
        retrievers=[retriever, keyword_search],
        weights=[0.5, 0.5]
    )

    retrieved_docs = final_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    result = chain.invoke({
        "context": context,
        "question": query
    })

    return result

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    (
    'system',
    '''
    Answer the user's question using only the
    provided webpage context.

    If the answer is not present in the context,
    say that the information was not found on
    the webpage.
    '''
    ),
    ('user', """
        Webpage Content:
        {context}

        Question:
        {question}
        """)
])

parser = StrOutputParser()

chain = prompt | llm | parser
