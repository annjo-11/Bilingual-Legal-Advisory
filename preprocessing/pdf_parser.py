import pymupdf
from pathlib import Path
import json


# --------------------------------------------------
# PROJECT FOLDERS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FOLDER = BASE_DIR / "data" / "raw"
OUTPUT_FOLDER = BASE_DIR / "data" / "extracted"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# EXTRACT TEXT FROM ONE PDF
# --------------------------------------------------

def extract_pdf(pdf_path):

    print(f"\nProcessing: {pdf_path.name}")

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        page_data = {
            "page": page_number,
            "text": text
        }

        pages.append(page_data)

        print(
            f"Page {page_number}: "
            f"{len(text)} characters extracted"
        )

    document.close()

    document_data = {
        "filename": pdf_path.name,
        "total_pages": len(pages),
        "pages": pages
    }

    return document_data


# --------------------------------------------------
# PROCESS ALL PDFs
# --------------------------------------------------

def process_all_pdfs():

    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF file(s).")

    if len(pdf_files) == 0:
        print("No PDF files found.")
        return

    for pdf_path in pdf_files:

        document_data = extract_pdf(pdf_path)

        output_filename = pdf_path.stem + ".json"

        output_path = OUTPUT_FOLDER / output_filename

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                document_data,
                file,
                ensure_ascii=False,
                indent=4
            )

        print(f"Saved: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    process_all_pdfs()