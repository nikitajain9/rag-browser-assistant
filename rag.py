import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever

doc = Document(page_content=content)

load_dotenv()

parser = StrOutputParser()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap= 200
)

chunks = splitter.split_documents([doc])

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    persist_directory="./chroma_data"
)

base_retriver = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

keyword_search = BM25Retriever.from_documents(chunks)
keyword_search.k = 3

final_retriever = EnsembleRetriever(
    retrievers = [base_retriver,keyword_search],
    weights = [0.5,0.5]
)

retrieved_docs = final_retriever.invoke(query)

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

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

chain = prompt | llm | parser
result = chain.invoke({
    "context": context,
    "question": query
})
print(result)