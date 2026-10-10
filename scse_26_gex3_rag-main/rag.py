import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from policy_loader import load_policy_documents

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()

MODEL = "qwen3-0.6b"
EMBED_MODEL = "nomic-embed-text"
DB_DIR = "chroma_db"

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# building the embeddings takes a while, so reuse the db if it is already there
if Path(DB_DIR).exists():
    store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

retriever = store.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model=MODEL, temperature=0)

prompt = ChatPromptTemplate.from_template(
    """You are the University IT Service Desk assistant.
Answer the question using only the policy context below.
If the context does not contain the answer, say exactly: I don't know.
Do not guess and do not invent phone numbers, emails or any other detail.

Context:
{context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | strip_thinking
)


if __name__ == "__main__":
    while True:
        question = input("Question (q to quit): ")
        if question == "q":
            break
        print("Answer:", chain.invoke(question))
