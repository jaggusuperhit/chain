import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

# Load environment variables
load_dotenv()

# Initialize the ChatOpenAI model for OpenRouter
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API URL
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Ensure OPENROUTER_API_KEY is in your .env file
    model="gpt-3.5-turbo"  # Use a smaller model to reduce token usage
)


parser = StrOutputParser()


class Feedback(BaseModel):

    sentiment:Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)



classifier_chain = prompt1 | model | parser2


prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch (
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser ),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser ),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':'This is a terrible phone'}))

chain.get_graph().print_ascii()