from debater import debate, summarise_debate, judge_debate, transcript_to_markdown
from debater import WEAK_MODEL

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


DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
MARKDOWN_FOLDER = "./ai-schizophrenia-facts-gpt-4.1-mini-1791419905"
SOURCE_PREFIX = "https://github.com/ed-roof-maker/ai-resume/blob/main/ai-schizophrenia-facts-gpt-4.1-mini-1791419905"

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


claims = [
    'Schizophrenia is a complex mental disorder that affects how a person thinks, feels, and behaves.',
    'Schizophrenia typically emerges in late adolescence or early adulthood.',
    'Genetic factors play a significant role in the risk of developing schizophrenia.',
    'Environmental factors, such as prenatal stress or infections, can contribute to the onset of schizophrenia.',
    'Schizophrenia is often associated with hallucinations, delusions, and cognitive impairments.',
    'Early intervention and treatment can improve outcomes for individuals with schizophrenia.',
    'Schizophrenia is a lifelong condition that requires ongoing management.',
    'Antipsychotic medications are commonly used to manage symptoms of schizophrenia.',
    'Psychosocial interventions, such as cognitive-behavioral therapy, can complement medication in managing schizophrenia.',
    'Research is ongoing to better understand the underlying causes and mechanisms of schizophrenia.',
    'Stigma and misunderstanding surrounding schizophrenia can affect social interactions and access to care.',
    'In Australia, support from family, friends, and mental health professionals is crucial for individuals with schizophrenia.',
    'Lifestyle factors, such as regular exercise, a balanced diet, and adequate sleep, can influence the management of schizophrenia.',
    'Schizophrenia research includes the study of neurobiological mechanisms, aiming to develop more effective treatments.',
    'Emerging therapies, such as neuromodulation and personalized medicine, hold promise for improving schizophrenia care.',
    'Ongoing research aims to identify biomarkers for early detection and personalized treatment of schizophrenia.',
    'In Australia, collaboration between GPs, clinicians, and patients is essential for advancing schizophrenia care.',
    'Schizophrenics are dangerous people',
    'Antipsychotics, like paliperidone, adversely affect metabolism and can lead to weight gain.',
    'Aripiprazole has less side effects than paliperidone.',
    'Certain antipsychotics can make your muscles twitch.',
    'Australian advocacy groups like QAI, can help prevent forced ECT under a treatment order for schizophrenics.',
    'ECT (Electroconvulsive Therapy) is sometimes used for treatment-resistant schizophrenia, but its benefits do not outweigh the potential risks.',
    'Eleanor Longden is an unmedicated schizophrenic psychologist, and it is possible to live a fulfilling life without medication.',
    'Burn out, sleep deprivation, and stress, can trigger schizophrenia episodes (psychosis).',
    'There are plenty of supports for schizophrenia, in Australia.',
    'Antipsychotics can affect libido.',
    'Antipsychotics can raise liver enzymes and cause organ damage.',
    'It is possible to be an unmedicated schizophrenic and live a normal happy life.',
    'Low dose antipsychotics, therapy, and supports, are more effective than high dose medication alone.',
    'High dose antipsychotics for short term benefits, then low dose antipsychotics for long term maintenance, is an effective strategy.',
    'Today, the forefront of schizophrenia research centers on non-dopaminergic drug discovery, synaptic connectivity, neuroinflammation, and AI-driven precision psychiatry.',
    'Individuals with a genetic predisposition may be more susceptible to developing schizophrenia as a result of cannabis use.',
    'Trauma significantly increases the vulnerability and risk of developing schizophrenia in individuals with a genetic or biological predisposition.',
    'Schizophrenics have multiple personalities.',
    'For support, you can get the Australian disability pension, if you are schizophrenic.',
    'For support, in Australia, you can claim TPD insurance through your super, if you were diagnosed with schizophrenia when employed.',
    'For support, there are Australian charities that can help you if you have schizophrenia.',
    'Schizophrenic hallucinations, are god talking to you.',
    'Schizophrenic hallucinations, are a form of mind control and/or electronic harassment.',
    'I should employ someone with schizophrenia, in Australia.',
    'It is safe to employ someone with schizophrenia.'
    'In Australia, it is illegal to deny employment to someone based on disability status.',
]
def slugify(text: str, max_length: int = 80) -> str:
    """Convert text into a safe filename."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug[:max_length].strip("-")

def debate_claims_and_save_to_file(claims):
    output_dir = Path(f"ai-schizophrenia-facts-{WEAK_MODEL}-{int(time())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, claim in enumerate(claims, start=1):
        try:

            print(f"\n{'=' * 80}")
            print(f"Debate {index}: {claim}")
            print(f"{'=' * 80}")

            # --------------------------------------------------
            # Run debate
            # --------------------------------------------------

            transcript = debate(
                claim=claim,
                rounds=6,
            )

            # --------------------------------------------------
            # Judge debate
            # --------------------------------------------------
            judgement = judge_debate(
                claim=claim,
                transcript=transcript,
            )

            # --------------------------------------------------
            # Write debate to Markdown
            # --------------------------------------------------

            filename = f"{index:03d}-{slugify(claim)}.md"
            output_path = output_dir / filename

            markdown = f"""# AI Debates
## Claim

{claim}


## Debate

{transcript_to_markdown(transcript)}


## Judgement

{judgement}
"""

            output_path.write_text(
                markdown,
                encoding="utf-8",
            )

            print(f"Saved: {output_path}")

        except Exception as exc:
            print(f"ERROR processing {claim}: {exc}")

def generate_debates():
    debate_claims_and_save_to_file(claims)
    print("Debates generated for all claims.")  


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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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
    claim = None

    for section in sections:

        metadata = section.metadata

        if metadata.get("heading_2", "").strip().lower() == "claim":
            claim = section.page_content.strip()
            break

        if metadata.get("heading_3", "").strip().lower() == "claim":
                    claim = section.page_content.strip()
                    break


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
                f"""Claim: "{claim}"\n"""
                f"{chunk.page_content}"
            )

        else:
            document = chunk.page_content

        if claim:
            metadata["claim"] = claim

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
            "claim": metadata.get("claim"),
            "distance": distance,
        })

    return retrieved

def get_filename_from_path(path):
    return path.split('/')[-1]

