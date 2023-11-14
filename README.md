# OpenAI PDF Chat

## Description
"OpenAI PDF Chat" is a Python-based terminal application designed to facilitate advanced document analysis and querying capabilities. The core functionality of the application revolves around processing PDF documents, extracting text, and leveraging the power of OpenAI's models to generate meaningful embeddings from the extracted text.

### Key Features and Workflow:
**1. Document Upload and Text Extraction:**
 - The program begins by uploading a PDF document (pdf_path = './example.pdf') through the ingest.py file.
 - The text extraction process involves breaking down the document into defined character chunks. This ensures a detailed and comprehensive analysis of the document’s content.

**2. Intelligent Text Chunking:**
  - A crucial aspect of the extraction process is the intelligent chunking of text. The application ensures that each chunk extends to the end of a word, especially when a word falls in the middle of a chunk boundary. This feature preserves the integrity and context of the text being analyzed.

**3. Text Embedding Using OpenAI Models:**
  - Once the text is extracted and chunked, the application utilizes OpenAI's advanced models to create embeddings from these text chunks. These embeddings capture the semantic essence of the text, enabling deeper and more contextual analysis.

**4. Integration with Pinecone Vector Database:**
  - The generated embeddings, along with the original text and its corresponding page number, are stored in a Pinecone vector database. This integration facilitates efficient storage and retrieval of information, making it ideal for applications requiring fast and accurate text search and analysis.

## Environment Setup

### Prerequisites
- Miniconda (Version 23.9.0)
- Python (Version 3.10.13)

### Setting Up the Python Environment
Follow these steps to set up the required environment for the project:

1. **Create a Conda Environment**  
   To create a new environment, use the following command:
   
   ```bash
   conda create --name [environment-name] python=3.10
   ```

1. **Activate the newly created environment with**:
Activate the newly created environment with:
    ```bash
    conda activate [environment-name]```
1. **Install Required Packages**
Install the necessary packages using the commands below:
```bash
# Install Pinecone client
pip install pinecone-client

# Install dotenv and python-dotenv for environment variable management
conda install dotenv
conda install -c conda-forge python-dotenv

# Install OpenAI package for AI-related functionalities
conda install openai

# Install pdfminer.six for PDF processing capabilities
conda install -c conda-forge pdfminer.six
```

## Application Use Case:
"OpenAI PDF Chat" is particularly useful for users who need to process and analyze large volumes of PDF documents. Whether it's extracting insights from academic papers, analyzing reports, or querying large document sets, this application streamlines the process, making it efficient and user-friendly.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.