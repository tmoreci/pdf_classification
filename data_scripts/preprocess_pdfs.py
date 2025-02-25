from tqdm import tqdm
import pymupdf
import os
from google import genai
from google.genai import types
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Function to process and add a paper
def add_paper(pdf_path, paper_id, collection, metadata=None, add_context=False):
    # Load and split the PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Add to collection with metadata
    for i, chunk in enumerate(chunks):
        if add_context:
            chunk.page_content = gemini_add_context(
                pdf_path, chunk.page_content
            )
        collection.add(
            ids=[f"{paper_id}_chunk_{i}"],
            documents=[chunk.page_content],
            metadatas=[
                {
                    "paper_id": paper_id,
                    "page": chunk.metadata.get("page", 0),
                    "chunk_id": i,
                    **(metadata or {}),
                }
            ],
        )

    return len(chunks)


def gemini_add_context(file: str, chunk):
    chunk_addition = f"""
<chunk> 
{chunk}
</chunk> 
The provided chunk is a small subsection of the full PDF document you have been provided. Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
    # Convert the file path string to a Path object
    file_path = Path(file)
    client = genai.Client(api_key="")
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            types.Part.from_bytes(
                data=file_path.read_bytes(),
                mime_type="application/pdf",
            ),
            chunk_addition,
        ],
    )
    output = f"{chunk}\n\n{response.text}"
    print(response.text)
    return output


def preprocess_pdf(file_path: str, max_block_size: int = 300) -> dict:
    doc = pymupdf.Document(filename=file_path)
    references = False
    # Iterate through each page in the PDF
    blocks = {}
    section = 0
    for i, page in enumerate(doc):
        section += 1
        blocks[f"page_{i}"] = []
        text = page.get_text()
        # Keep 'References' upper case to avoid any confusion with the verb
        if i >= len(doc) - 3 and ("References" in text or "REFERENCES" in text):
            index = text.lower().find("references")
            text = text[:index]
            references = True
            # Don't include empty chunk when references has its own page
            if len(text) < 100:
                break
        words = text.split()
        word_count = len(words)
        if word_count > max_block_size:
            start_index = 0
            while start_index < word_count:
                end_index = min(start_index + max_block_size, word_count)
                text_section = " ".join(words[start_index:end_index])
                blocks[f"page_{i}"].append(
                    {"section_title": f"doc_{section}", "snippet": text_section}
                )
                section += 1
                start_index = end_index
        else:
            blocks[f"page_{i}"].append(
                {"section_title": f"doc_{section}", "snippet": text}
            )
        if references:
            break

    return blocks


def gemini_extraction(file: str):
    prompt = """
## Task
The provided PDF is a scientific article. Your task is to extract the text from this article into a JSON object. You must extract the following information:
1.) Title: the title of the paper
2.) Abstract: The abstract of the paper
3.) Paragraphs: Each individual paragraph in the paper. For each paragraph, you must also give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.

## Output Format
The output should be a JSON object with the following structure:

```json
{
  "title": "string",
  "abstract": "string",
  "paragraphs": [
    {
      "text": "string",
      "context": "string"
    }
  ]
}
```

Where:
- `title`: The full title of the scientific article
- `abstract`: The complete abstract text of the article
- `paragraphs`: An array of paragraph objects, each containing:
  - `text`: The full text content of the paragraph
  - `context`: A brief description (1-2 sentences) that places this paragraph in the document's context to improve search retrieval

The paragraphs should be extracted in the order they appear in the document, maintaining the original structure of the article.
Output only the JSON object
    """
    client = genai.Client(api_key="AIzaSyB6fss_57uiraamCB6VdItD-LJrchtA4e0")

    # Convert the file path string to a Path object
    file_path = Path(file)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=file_path.read_bytes(),
                mime_type="application/pdf",
            ),
            prompt,
        ],
    )
    return response.text


def query_papers(query_text, collection, n_results=5):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results


if __name__ == "__main__":
    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # Free, lightweight model
        )
    )

    # Initialize ChromaDB (persistent)
    client = chromadb.PersistentClient(path="../data/chroma_db")

    # Create or get collection
    collection = client.get_or_create_collection(
        name="academic_papers", embedding_function=embedding_function
    )
    pdfs_path = "../data"
    files = os.listdir(pdfs_path)
    output = []
    # query = "Long context input vs retrieval augmented generation"
    # print(query_papers(query, collection)["documents"])
    # for file in tqdm(files):
    #     full_file_path = os.path.join(pdfs_path, file)
    #     response = preprocess_pdf(full_file_path)
    #     output.append({"file": file, "chunks": response})
    # with open("test_outputs.json", "w") as f:
    #     json.dump(output, f)
    for i, file in tqdm(enumerate(files[:2])):
        full_file_path = os.path.join(pdfs_path, file)
        add_paper(full_file_path, i, collection, add_context=False)
    # gemini_output = gemini_extraction(full_file_path)
    # doc_json = parse_json_response(gemini_output)
    # output.append(doc_json)

    # with open("test_outputs.json", "w") as f:
    #     json.dump(output, f)
