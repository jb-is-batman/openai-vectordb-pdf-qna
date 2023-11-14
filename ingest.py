# Importing necessary libraries and modules
from pdfminer.high_level import extract_pages  # Used for extracting pages from a PDF
from pdfminer.layout import LTTextBox, LTTextLine  # Layout objects from PDFs

import  pinecone  # Pinecone is used for vector search
from    dotenv import load_dotenv  # To load environment variables from .env file
import  os  # To interact with the operating system
from    openai import OpenAI  # OpenAI for AI tasks, especially embeddings and chat completions
import  uuid  # For generating unique identifiers

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI and Pinecone with keys and configurations from environment variables
OPENAI_API_KEY        = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY      = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT  = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX        = os.getenv('PINECONE_INDEX_NAME')
MODEL                 = "text-embedding-ada-002"

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Initialize Pinecone client with the specified API key and environment
pinecone.init(
    api_key     =PINECONE_API_KEY,
    environment =PINECONE_ENVIRONMENT
)
index = pinecone.Index(PINECONE_INDEX)

# Define function to extract text from PDF, along with page numbers
def extract_text_from_pdf_with_page_numbers(pdf_path):
    chunks_with_pages = []
    for page_number, page_layout in enumerate(extract_pages(pdf_path)):
        page_text = ""
        for element in page_layout:
            if isinstance(element, (LTTextBox, LTTextLine)):
                page_text += element.get_text()
        chunks_with_pages.extend(split_text(page_text, page_number))
    return chunks_with_pages

# Define function to split text into chunks with a minimum size and overlap
def split_text(text, page_number, min_chunk=1500, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + min_chunk
        # Extend to the end of the word to avoid cutting words in half
        while end < len(text) and text[end].isalpha():
            end += 1
        # Append the chunk along with its corresponding page number
        chunk = (text[start:end], page_number)
        chunks.append(chunk)
        # Move start for the next chunk, with an overlap
        start = end - overlap if end - overlap > start else end
    return chunks

# Path to the PDF file to be processed
pdf_path                = './example.pdf'
# Extract text from the specified PDF file
text_chunks_with_pages  = extract_text_from_pdf_with_page_numbers(pdf_path)
# Extract only the text part for creating embeddings
text_only_chunks        = [chunk[0] for chunk in text_chunks_with_pages]

# Generate embeddings for each text chunk using OpenAI's API
openai_embedding = client.embeddings.create(
    input=text_only_chunks, 
    model=MODEL
)

# Extract embeddings to a list for further processing
embeddings = [record.embedding for record in openai_embedding.data]

mapped_data = []
for embedding, (text, page_number) in zip(embeddings, text_chunks_with_pages):
    vector_id = str(uuid.uuid4())  # Generate a unique ID for each vector
    metadata  = {"text": text, "page_number": page_number, "file_path": pdf_path}  # Attach metadata including page number
    mapped_data.append((vector_id, embedding, metadata))

# Insert the data into the Pinecone index for search and retrieval
upsert_response = index.upsert(vectors=mapped_data)