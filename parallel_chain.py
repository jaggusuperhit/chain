import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

# Load environment variables
load_dotenv()

# Initialize the ChatOpenAI model for OpenRouter
model1 = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API URL
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Ensure OPENROUTER_API_KEY is in your .env file
    model="gpt-3.5-turbo"  # Use a smaller model to reduce token usage
)

model2 = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API URL
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Ensure OPENROUTER_API_KEY is in your .env file
    model="gpt-3.5-turbo"  # Use the same smaller model for both chains
)

# Define prompt templates
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text: \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question-answers from the following text: \n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document. \nNotes: {notes} \nQuiz: {quiz}',
    input_variables=['notes', 'quiz']
)

# Initialize the output parser
parser = StrOutputParser()

# Create parallel chains for notes and quiz
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# Create the merge chain
merge_chain = prompt3 | model1 | parser

# Combine the parallel chains and merge chain
chain = parallel_chain | merge_chain

# Input text (shortened to reduce token usage)
text = """ 
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Invoke the chain with the input text
result = chain.invoke({'text': text})

# Print the result
print(result)

chain.get_graph().print_ascii()