from pathlib import Path
import re
from time import time

from pathlib import Path
import hashlib

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


DB_PATH = "./chroma_db_digital_twin"
COLLECTION_NAME = "knowledge_base"
MARKDOWN_FOLDER = "./digital-twin-knowledge-base"
SOURCE_PREFIX = "https://github.com/ed-roof-maker/ai-resume/blob/main/digital-twin-knowledge-base"

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 400

HEADERS_TO_SPLIT_ON = [
    ("#", "heading_1"),
    ("##", "heading_2"),
    ("###", "heading_3"),
    ("####", "heading_4"),
]

# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


def slugify(text: str, max_length: int = 80) -> str:
    """Convert text into a safe filename."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug[:max_length].strip("-")


# --------------------------------------------------
# Text chunking
# --------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_markdown(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):

    # Split by Markdown heading hierarchy
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )

    sections = markdown_splitter.split_text(text)

    # Split oversized sections into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(sections)

    results = []

    for chunk in chunks:

        metadata = chunk.metadata.copy()

        # Build heading hierarchy
        headings = [
            metadata[key]
            for key in (
                "heading_1",
                "heading_2",
                "heading_3",
                "heading_4",
            )
            if key in metadata
        ]

        heading_path = " > ".join(headings)

        # Prepend heading context to chunk text
        if heading_path:
            document = (
                f"Context: {heading_path}\n"
                f"{chunk.page_content}"
            )

        else:
            document = chunk.page_content

        metadata["heading_path"] = heading_path
        

        results.append({
            "document": document,
            "metadata": metadata,
        })

    return results

# --------------------------------------------------
# Unique document IDs
# --------------------------------------------------

def generate_chunk_id(
    file_path: Path,
    chunk_index: int
) -> str:

    # Stable identifier based on full file path
    file_hash = hashlib.sha256(
        str(file_path.resolve()).encode("utf-8")
    ).hexdigest()[:16]

    return f"{file_hash}-{chunk_index}"


# --------------------------------------------------
# Load Markdown
# --------------------------------------------------

def load_markdown(file_path: Path) -> int:

    print(f"Indexing file: {file_path}")

    source = str(file_path.resolve())

    text = file_path.read_text(
        encoding="utf-8"
    )

    chunks = chunk_markdown(text)

    if not chunks:
        return 0

    ids = [
        generate_chunk_id(file_path, i)
        for i in range(len(chunks))
    ]

    documents = [
        chunk["document"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": f"""{SOURCE_PREFIX}/{get_filename_from_path(source)}""",
            "filename": file_path.name,
            "chunk": i,
            **chunk["metadata"],
        }
        for i, chunk in enumerate(chunks)
    ]

    # Remove previously indexed chunks
    collection.delete(
        where={"source": source}
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Indexed {len(chunks)} chunks.")

    return len(chunks)


# --------------------------------------------------
# Load all Markdown files
# --------------------------------------------------

def load_all_facts():

    folder = Path(MARKDOWN_FOLDER)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder does not exist: {folder}"
        )

    files = sorted(folder.glob("*.md"))

    total_files = 0
    total_chunks = 0

    for file in files:

        chunks = load_markdown(file)

        total_files += 1
        total_chunks += chunks

    print("\nIndexing completed.")
    print(f"Markdown files processed: {total_files}")
    print(f"Chunks processed: {total_chunks}")
    print(f"ChromaDB records: {collection.count()}")


def retrieve_knowledge(
    query: str,
    n_results: int = 5,
) -> list[dict]:

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):

        retrieved.append({
            "content": document,
            "source": metadata.get("source"),
            "heading_path": metadata.get("heading_path"),
            "distance": distance,
        })

    return retrieved

def get_filename_from_path(path):
    return path.split('/')[-1]

