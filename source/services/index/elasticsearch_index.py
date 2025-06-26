"""
Elasticsearch indexer for RAG system.
Processes PDF documents, extracts text, splits into chunks, generates embeddings, and indexes to Elasticsearch.
"""

import os
from docling.document_converter import DocumentConverter


def convert_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF document to markdown format using docling.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Markdown content of the PDF
    """
    converter = DocumentConverter()
    doc = converter.convert(pdf_path).document
    return doc.export_to_markdown()


def convert_attention_pdf_to_markdown() -> str:
    """
    Convert the 'Attention Is All You Need.pdf' to markdown.
    
    Returns:
        str: Markdown content of the PDF
    """
    # Get the path to the PDF file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "..", "..", "Attention Is All You Need.pdf")
    
    return convert_pdf_to_markdown(pdf_path)


if __name__ == "__main__":
    print("Converting 'Attention Is All You Need.pdf' to markdown...")
    try:
        markdown_content = convert_attention_pdf_to_markdown()
        print("Conversion successful!")
        print("\n" + "="*50)
        print("MARKDOWN CONTENT:")
        print("="*50)
        print(markdown_content)
    except Exception as e:
        print(f"Error converting PDF: {e}")

