from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from rag import MODEL, strip_thinking
from rag_tool import policy_search

SYSTEM_PROMPT = """You are the University IT Service Desk assistant.
Always use the policy_search tool and answer only from what it returns.
If the answer is not there, say: I don't know.
Never invent a phone number or any other detail."""

agent = create_agent(
    ChatOllama(model=MODEL, temperature=0), [policy_search], system_prompt=SYSTEM_PROMPT
)

if __name__ == "__main__":
    while True:
        question = input("Question (q to quit): ")
        if question == "q":
            break
        result = agent.invoke({"messages": [("user", question)]})
        print("Answer:", strip_thinking(result["messages"][-1].content))
