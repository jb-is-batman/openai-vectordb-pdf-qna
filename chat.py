# Importing necessary libraries and modules
import  pinecone  # Pinecone for vector search and indexing
from    dotenv import load_dotenv  # To load environment variables from a .env file
import  os  # For interacting with the operating system environment
from    openai import OpenAI  # OpenAI for AI tasks, especially embeddings and chat completions

# Load environment variables from a .env file
load_dotenv()

# Retrieve API keys and other configuration details from environment variables
PINECONE_API_KEY      = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT  = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME   = os.getenv('PINECONE_INDEX_NAME')
OPENAI_API_KEY        = os.getenv('OPENAI_API_KEY')
MODEL                 = "text-embedding-ada-002"  # Model for text embedding
CHAT_MODEL            = "gpt-4-1106-preview"  # Model for chat completion

# Setting the API key for OpenAI
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Initialize Pinecone client with the specified API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX_NAME)

# Function to query the Pinecone index and generate a response using OpenAI ChatCompletion
def query_index(query):
    
    # Generate an embedding for the user's query
    openai_embedding = client.embeddings.create(
      input=query, 
      model=MODEL
    )

    vector = openai_embedding.data[0].embedding

    # Query the Pinecone index with the generated embedding
    query_response = index.query(
        top_k            = 3,  # Number of top results to retrieve
        include_values   = True,
        include_metadata = True,
        vector           = vector,
    )

    # Start building the system message with context from query results
    completion_system_query = "Based on the document's query results, the main points to consider are: \n\n"

    # Iterate through the matches and append their text to the system message
    for match in query_response['matches']:
        text                    = match['metadata']['text']
        text                    = text.replace("\n", " ")  # Remove newlines for cleaner formatting
        completion_system_query += "- " + text + "\n\n"

    # Generate a response using OpenAI ChatCompletion
    response = client.chat.completions.create(
        model=CHAT_MODEL,  # Specify the chat model
        messages=[
            {"role": "system", "content": completion_system_query},  # System message providing context
            {"role": "user", "content": query},  # User's actual query
        ],
        temperature=0,  # Controls the randomness/creativity of the response
    )

    # Return the AI-generated response
    return response.choices[0].message.content

# Main loop for continuous querying
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    answer = query_index(query)
    print("Answer:\n", answer, "\n\n")
