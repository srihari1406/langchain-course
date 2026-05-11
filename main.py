from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     # return "Tokyo weather is sunny"
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for the scource used by the agents"""

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's answer and source"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list,description="List of sources used to generate answers")

llm = ChatGroq(model="openai/gpt-oss-20b")
structured_llm = llm.with_structured_output(AgentResponse)
# tools = [search]
tools = [TavilySearch(max_results=3)]
prompt = ChatPromptTemplate.from_messages([("human", "{input}"),MessagesPlaceholder(variable_name="agent_scratchpad")])
agent = create_tool_calling_agent(llm=llm,tools=tools,prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    result = agent_executor.invoke({"input": HumanMessage(content="What's the weather in Tokyo?")})
    print(result)
    final_output = structured_llm.invoke(
        f"Based on this research: {result}, format the answer and list the URLs."
    )
    print("\n--- Structured Result ---")
    print(f"Answer: {final_output.answer}")
    print(f"Sources: {final_output.sources}")

if __name__ == "__main__":
    main()
