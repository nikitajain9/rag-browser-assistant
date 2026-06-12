import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
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

parser = StrOutputParser()
chain = prompt | llm | parser


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