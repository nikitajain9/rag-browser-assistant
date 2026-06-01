import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


doc = Document(page_content=content)

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ('system','Act as an AI assistent and answer the user query'),
    ('user', """
        Webpage Content:
        {content}

        Question:
        {question}
        """)
])


parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({'content':doc.page_content, 'question':'what is in this web page'})
