from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    # return "Tokyo weather is sunny"
    return tavily.search(query=query)

llm = ChatGroq(model="openai/gpt-oss-20b")
tools = [search]
prompt = ChatPromptTemplate.from_messages([("human", "{input}"),MessagesPlaceholder(variable_name="agent_scratchpad")])
agent = create_tool_calling_agent(llm=llm,tools=tools,prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    result = agent_executor.invoke({"input": HumanMessage(content="What's the weather in Tokyo?")})
    print(result)

if __name__ == "__main__":
    main()
