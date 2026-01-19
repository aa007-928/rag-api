import chromadb
from pathlib import Path
import logging
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
from tiktoken import encoding_for_model
import hashlib
import sys

# ------------------------
# CONFIGURATION
# ------------------------
KB_DIR = Path("KB")                  # Folder containing text files
DB_PATH = "./db"                     # ChromaDB persistent path
CHUNK_SIZE_TOKENS = 200              # Max tokens per chunk
BATCH_SIZE = 50                      # Chunks per DB insert
EMBEDDING_MODEL = "text-embedding-3-small"

# ------------------------
# LOGGING SETUP
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("kb_ingest.log"),
        logging.StreamHandler()
    ]
)

# ------------------------
# CHROMADB SETUP
# ------------------------
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("docs")

# ------------------------
# NLP / TOKENIZER SETUP
# ------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

enc = encoding_for_model(EMBEDDING_MODEL)

def num_tokens(text: str) -> int:
    return len(enc.encode(text))

# ------------------------
# CHUNKING FUNCTION
# ------------------------
def chunk_text(file_path: Path, chunk_size_tokens: int):
    """
    Stream file line-by-line, chunk by sentences, respect token limit.
    Never loads full file into memory.
    """
    buffer = ""

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            buffer += line.strip() + " "

            try:
                sentences = sent_tokenize(buffer)
            except Exception as e:
                logging.warning(f"Sentence tokenization failed in {file_path.name}: {e}")
                sentences = [buffer]

            chunk = ""
            chunk_tokens = 0

            # Keep last sentence in buffer to avoid cutting it
            for sentence in sentences[:-1]:
                s_tokens = num_tokens(sentence)

                if chunk_tokens + s_tokens > chunk_size_tokens:
                    if chunk:
                        yield chunk.strip()
                    chunk = sentence
                    chunk_tokens = s_tokens
                else:
                    chunk += sentence + " "
                    chunk_tokens += s_tokens

            buffer = sentences[-1] if sentences else ""

    # Yield remaining content
    if buffer.strip():
        yield buffer.strip()

# ------------------------
# DETERMINE FILES TO PROCESS
# ------------------------
if len(sys.argv) > 1:
    files_to_process = [Path(p) for p in sys.argv[1:]]
    logging.info(f"Processing specific files: {[f.name for f in files_to_process]}")
else:
    files_to_process = list(KB_DIR.glob("*.txt"))
    logging.info("Processing all KB files")

# ------------------------
# MAIN INGESTION LOOP
# ------------------------
for file_path in files_to_process:
    if not file_path.exists():
        logging.warning(f"File not found: {file_path}")
        continue

    logging.info(f"Processing file: {file_path.name}")

    # Delete existing embeddings for this file (overwrite behavior)
    try:
        collection.delete(where={"source_file": file_path.name})
        logging.info(f"Deleted previous embeddings for {file_path.name}")
    except Exception as e:
        logging.warning(f"Could not delete previous embeddings for {file_path.name}: {e}")

    chunk_docs, chunk_ids, chunk_metas = [], [], []

    try:
        for idx, chunk in enumerate(chunk_text(file_path, CHUNK_SIZE_TOKENS)):
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:8]
            chunk_id = f"{file_path.stem}_chunk{idx}_{chunk_hash}"

            chunk_docs.append(chunk)
            chunk_ids.append(chunk_id)
            chunk_metas.append({
                "source_file": file_path.name,
                "chunk_index": idx,
                "ingested_at": datetime.utcnow().isoformat()
            })

            # Batch insert
            if len(chunk_docs) >= BATCH_SIZE:
                collection.add(
                    documents=chunk_docs,
                    ids=chunk_ids,
                    metadatas=chunk_metas
                )
                logging.info(f"Inserted batch of {len(chunk_docs)} chunks")

                chunk_docs, chunk_ids, chunk_metas = [], [], []

        # Insert remaining chunks
        if chunk_docs:
            collection.add(
                documents=chunk_docs,
                ids=chunk_ids,
                metadatas=chunk_metas
            )
            logging.info(f"Inserted final batch of {len(chunk_docs)} chunks")

        logging.info(f"Finished processing {file_path.name}")

    except Exception as e:
        logging.error(f"Failed to process {file_path.name}: {e}")

logging.info("Embedding ingestion completed successfully")
