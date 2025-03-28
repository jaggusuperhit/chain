import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load API key from .env
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Define prompt template
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

# Use OpenRouter with a valid model
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API URL
    api_key=OPENROUTER_API_KEY,
    model="gpt-3.5-turbo"  # ✅ Use a valid model ID
)

# Output parser to convert model output to string
parser = StrOutputParser()

# Create the chain: prompt -> model -> parser
chain = prompt | model | parser

# Invoke the chain with a topic
result = chain.invoke({"topic": "cricket"})

# Print the result
print(result)

chain.get_graph().print_ascii()