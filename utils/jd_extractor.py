import os
from pypdf import PdfReader
import pytesseract

from docx import Document
from pdf2image import convert_from_path


class JDExtractor:

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = " ".join(text.split())
        return text

    def extract_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        return self.clean_text(text)

    def extract_docx(self, file_path):
        doc = Document(file_path)

        paragraphs = []

        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        return self.clean_text("\n".join(paragraphs))

    def extract_pdf_text(self, file_path):
        reader = PdfReader(file_path)

        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        return self.clean_text("\n".join(pages))

    def extract_pdf_ocr(self, file_path):
        images = convert_from_path(file_path)

        text = []

        for image in images:
            extracted = pytesseract.image_to_string(image)
            text.append(extracted)

        return self.clean_text("\n".join(text))

    def extract_pdf(self, file_path):

        text = self.extract_pdf_text(file_path)

        # If almost no text found,
        # assume scanned PDF

        if len(text.strip()) < 100:
            print("Scanned PDF detected -> OCR fallback")
            text = self.extract_pdf_ocr(file_path)

        return text
    

    def extract(self, file_path):

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self.extract_pdf(file_path)

        elif ext == ".docx":
            return self.extract_docx(file_path)

        elif ext == ".txt":
            return self.extract_txt(file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {ext}"
            )
