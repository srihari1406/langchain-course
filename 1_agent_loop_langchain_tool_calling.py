from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# -------- Tools (LangChain @tools decorator) --------

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >>Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product,0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available Tiers: bronze, silver, gold"""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier,0)
    print(f"discount_pct = {discount}")
    return round(price * (1 - discount / 100), 2)

# -------- Agent loop --------

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount] #Creatikng List of Tools
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature = 0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content="""You are a helpful assistant
You have access to a product catalog tool and a discount tool
strict rules - you must follow these exactly:
1. Never guess or assume any product price.
2. You must call get_product_price first to get the real price.
3. only call apply_discount after you have received a price from get_product_price. pass the exact price returned by the get_product_price - do not pass a made-up number
4. If the user does not specify a discount tier, ask them which tier to use - do not assume one."""
        ),
        HumanMessage(
            content= question
        )
    ]

    for iterations in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iterations} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_call_lst = ai_message.tool_calls
        print(tool_call_lst)

        if not tool_call_lst:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_call_lst[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)
        print(f"    [Tool result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    
    print("ERROR: MAX iterations reached without a final answer")
    return None

 

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools!)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold tier discount?")
