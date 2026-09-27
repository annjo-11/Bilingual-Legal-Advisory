import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "extracted" / "IT_Act_2000.json"

OUTPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_FOLDER / "chunks.json"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD EXTRACTED PDF DATA
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

pages = data["pages"]

print(f"Total pages loaded: {len(pages)}")


# ============================================================
# SECTION HEADING DETECTION
# ============================================================

# Matches headings such as:
#
# 43. Penalty and compensation...
# 43A. Compensation...
# 66C. Punishment...
#
# It will NOT match:
#
# (a)
# (b)
# 1.
# 2.
# 3.
#
# because actual IT Act sections are numbered 10 or above
# after the introductory sections, and footnote numbers
# are handled separately below.

section_pattern = re.compile(
    r"(?m)^\s*(\d+[A-Z]?)\.\s+"
)


# ============================================================
# VARIABLES
# ============================================================

chunks = []

current_section = None
current_text = []
current_page_start = None

chunk_id = 0


# ============================================================
# SAVE CURRENT SECTION
# ============================================================

def save_current_section():

    global chunk_id

    if current_section is None:
        return

    text = "\n".join(current_text).strip()

    if not text:
        return

    chunks.append({
        "chunk_id": str(chunk_id),
        "document": "IT_Act_2000.pdf",
        "page": current_page_start,
        "section": current_section,
        "text": text
    })

    chunk_id += 1


# ============================================================
# PROCESS EACH PAGE
# ============================================================

for page_data in pages:

    page_number = int(page_data["page"])
    text = page_data["text"]

    # --------------------------------------------------------
    # Skip page 3
    # --------------------------------------------------------
    #
    # Page 3 is the table of contents / section list.
    # We don't want those entries to become legal chunks.
    #
    if page_number in [2, 3]:
        print(f"Skipping page {page_number}: table of contents")
        continue


    # --------------------------------------------------------
    # Remove page-number-only lines
    # --------------------------------------------------------
    #
    # Example:
    #
    # 19
    #
    # These are PDF page numbers, not section numbers.
    #
    text = re.sub(
        r"(?m)^\s*\d+\s*$",
        "",
        text
    )


    # --------------------------------------------------------
    # Find possible section headings
    # --------------------------------------------------------

    raw_matches = list(section_pattern.finditer(text))

    matches = []


    # --------------------------------------------------------
    # Filter false section headings
    # --------------------------------------------------------

    for match in raw_matches:

        section_number = match.group(1)

        # Footnote numbers such as:
        #
        # 1. Ins. by Act...
        # 2. Subs. by...
        # 3. Ins. by...
        #
        # should NOT be treated as sections.

        if section_number.isdigit():

            number = int(section_number)

            if number < 10:
                continue

        matches.append(match)


    # ========================================================
    # PAGE CONTAINS SECTION HEADING(S)
    # ========================================================

    if matches:

        # ----------------------------------------------------
        # Text before first detected section
        # ----------------------------------------------------

        prefix = text[:matches[0].start()].strip()

        if prefix and current_section is not None:
            current_text.append(prefix)


        # ----------------------------------------------------
        # Process detected sections
        # ----------------------------------------------------

        for i, match in enumerate(matches):

            section_number = match.group(1)

            start = match.start()


            if i + 1 < len(matches):

                end = matches[i + 1].start()

            else:

                end = len(text)


            section_text = text[start:end].strip()


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Sometimes the PDF contains the section number
            # twice.
            #
            # Example:
            #
            # 43. Penalty and compensation...
            #
            # 43. [Penalty and compensation] for damage...
            #
            # The second "43." is NOT a new section.
            #
            # If the detected section number is the same as
            # the section currently being collected, merge it.
            # ------------------------------------------------

            if current_section == section_number:

                current_text.append(section_text)

                continue


            # ------------------------------------------------
            # New section detected
            # ------------------------------------------------

            if current_section is not None:

                save_current_section()


            # Start collecting new section

            current_section = section_number

            current_page_start = page_number

            current_text = [section_text]


    # ========================================================
    # PAGE HAS NO SECTION HEADING
    # ========================================================

    else:

        cleaned_text = text.strip()

        if cleaned_text and current_section is not None:

            # This is continuation text from the previous page.
            current_text.append(cleaned_text)


# ============================================================
# SAVE FINAL SECTION
# ============================================================

save_current_section()


# ============================================================
# WRITE OUTPUT
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        chunks,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("CHUNKING COMPLETE")
print("========================================")

print(f"Total chunks created: {len(chunks)}")

print(f"Saved to: {OUTPUT_FILE}")

print("========================================")