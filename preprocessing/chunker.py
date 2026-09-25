import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "extracted" / "IT_Act_2000.json"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_FOLDER / "chunks.json"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load extracted PDF data
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

# The JSON contains:
# filename
# total_pages
# pages

pages = data["pages"]

print(f"Total pages loaded: {len(pages)}")


# --------------------------------------------------
# Detect legal section headings
# Examples:
# 66C. Punishment for identity theft.
# 66F. Punishment for cyber terrorism.
# 43. Penalty and compensation...
# --------------------------------------------------

section_pattern = re.compile(
    r'(?m)^\s*(\d+[A-Z]?(?:\([0-9A-Za-z]+\))?)\.\s+'
)


# --------------------------------------------------
# Create section-aware chunks
# --------------------------------------------------

chunks = []
chunk_id = 0

for page_data in pages:

    page_number = page_data["page"]
    text = page_data["text"]

    matches = list(section_pattern.finditer(text))

    # If no section heading is found on this page
    if not matches:

        cleaned_text = text.strip()

        if cleaned_text:
            chunks.append(
                {
                    "chunk_id": str(chunk_id),
                    "document": "IT_Act_2000.pdf",
                    "page": page_number,
                    "section": "Unknown",
                    "text": cleaned_text
                }
            )

            chunk_id += 1

        continue


    # --------------------------------------------------
    # Create one chunk for each section found on page
    # --------------------------------------------------

    for i, match in enumerate(matches):

        section_number = match.group(1)

        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        if section_text:

            chunks.append(
                {
                    "chunk_id": str(chunk_id),
                    "document": "IT_Act_2000.pdf",
                    "page": page_number,
                    "section": section_number,
                    "text": section_text
                }
            )

            chunk_id += 1


# --------------------------------------------------
# Save chunks
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4, ensure_ascii=False)


print(f"Total chunks created: {len(chunks)}")
print(f"Saved to: {OUTPUT_FILE}")