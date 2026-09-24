import json
import re
from pathlib import Path


# --------------------------------------------------
# PROJECT FOLDERS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "extracted" / "IT_Act_2000.json"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chunks.json"


# Create processed folder if it doesn't exist
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD EXTRACTED DOCUMENT
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    document = json.load(file)


# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------

def clean_text(text):

    # Replace multiple spaces with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple blank lines with one newline
    text = re.sub(r"\n\s*\n+", "\n", text)

    # Remove spaces at beginning/end
    text = text.strip()

    return text


# --------------------------------------------------
# SPLIT TEXT INTO CHUNKS
# --------------------------------------------------

def create_chunks(text, page_number, chunk_size=1200, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end]

        chunk_text = chunk_text.strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": page_number
            })

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# PROCESS ALL PAGES
# --------------------------------------------------

all_chunks = []

for page in document["pages"]:

    page_number = page["page"]

    raw_text = page["text"]

    cleaned_text = clean_text(raw_text)

    page_chunks = create_chunks(
        cleaned_text,
        page_number
    )

    all_chunks.extend(page_chunks)


# --------------------------------------------------
# ADD IDs AND METADATA
# --------------------------------------------------

final_chunks = []

for index, chunk in enumerate(all_chunks, start=1):

    final_chunks.append({
        "chunk_id": f"chunk_{index}",
        "document": document["filename"],
        "page": chunk["page"],
        "text": chunk["text"]
    })


# --------------------------------------------------
# SAVE CHUNKS
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_chunks,
        file,
        ensure_ascii=False,
        indent=4
    )


print("\nChunking completed!")
print(f"Total chunks created: {len(final_chunks)}")
print(f"Saved to: {OUTPUT_FILE}")