from langchain_core.tools import tool

from rag import retriever


@tool
def policy_search(question: str) -> str:
    """Search the university IT policy knowledge base for questions about
    university IT rules or the IT Service Desk."""
    docs = retriever.invoke(question)
    return "\n\n".join(f"[{d.metadata['policy_id']}] {d.page_content}" for d in docs)
