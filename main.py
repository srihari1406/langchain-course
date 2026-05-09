from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    # print("Hello from langchain-course!")
    information = """Elon Musk is a South African-born entrepreneur and business magnate who has become one of the most influential figures in modern technology and industry. Born on June 28, 1971, in Pretoria, he displayed an early aptitude for computing, teaching himself to code and selling his first video game, Blastar, at the age of twelve. After moving to Canada and then the United States, he pursued degrees in physics and economics at the University of Pennsylvania. Musk first achieved significant financial success during the dot-com boom with Zip2 and later X.com, which merged to become PayPal before being acquired by eBay in 2002.
    His subsequent ventures have aimed at addressing existential threats to humanity by revolutionizing transportation and energy. As the CEO and chief engineer of SpaceX, Musk has pioneered reusable rocket technology to reduce the cost of space flight, with the ultimate goal of colonizing Mars. Simultaneously, as the CEO and product architect of Tesla, he transformed the perception of electric vehicles from niche products into high-performance, mass-market cars, while also pushing for a global shift toward sustainable energy through solar power and battery storage solutions.
    In recent years, Musk’s influence has expanded further into social media and artificial intelligence. Following his acquisition of Twitter (now rebranded as X) in 2022, he has implemented sweeping changes to the platform’s operations and content moderation policies. He also leads Neuralink, which develops brain-computer interface technology, and The Boring Company, focused on tunnel construction to alleviate urban traffic. Despite being a polarizing figure due to his outspoken public persona and management style, he remains the world's wealthiest individual and a central driver of 21st-century innovation."""

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"],template=summary_template)
    
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response)
    print(response.content)

if __name__ == "__main__":
    main()
