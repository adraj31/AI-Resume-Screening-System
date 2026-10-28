"""Extract raw text from an uploaded resume file (.pdf via pdfplumber, .docx via python-docx)."""
import io

import pdfplumber
import docx

ALLOWED_EXTENSIONS = {'pdf', 'docx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _extract_pdf_text(data):
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or '')
    return '\n'.join(text_parts)


def _extract_docx_text(data):
    document = docx.Document(io.BytesIO(data))
    return '\n'.join(p.text for p in document.paragraphs)


def extract_text_from_file(file_storage):
    """file_storage: a werkzeug FileStorage from request.files. Returns plain text."""
    filename = file_storage.filename.lower()
    data = file_storage.read()

    if filename.endswith('.pdf'):
        return _extract_pdf_text(data)

    if filename.endswith('.docx'):
        return _extract_docx_text(data)

    raise ValueError('Unsupported file type. Please upload a .pdf or .docx file.')
