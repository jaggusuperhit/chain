import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Define the first prompt template
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# Define the second prompt template
prompt2 = PromptTemplate(
    template='Generate a 5-pointer summary from the following text: \n{text}',
    input_variables=['text']
)

# Initialize the ChatOpenAI model for OpenRouter
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API URL
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Ensure OPENROUTER_API_KEY is in your .env file
    model="gpt-3.5-turbo"  # Use a valid model ID
)

# Initialize the output parser
parser = StrOutputParser()

# Create a single chain: prompt1 -> model -> parser -> prompt2 -> model -> parser
chain = (
    prompt1 | model | parser | 
    {"text": lambda x: x} |  # Pass the output of the first chain as input to the second chain
    prompt2 | model | parser
)

# Invoke the chain with a topic
result = chain.invoke({'topic': 'Unemployment in India'})

# Print the result
print(result)

chain.get_graph().print_ascii()