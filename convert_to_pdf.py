#!/usr/bin/env python3
"""
Professional PDF converter for QBITEL AI documentation
Converts markdown to beautifully formatted PDF
"""

import sys
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

def convert_markdown_to_pdf(
    input_file: str,
    output_file: str = None
):
    """
    Convert markdown file to professional PDF

    Args:
        input_file: Path to markdown file
        output_file: Output PDF path (optional)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    # Default output file
    if output_file is None:
        output_file = str(input_path.with_suffix('.pdf'))

    print(f"Converting {input_file} to PDF...")
    print(f"Output: {output_file}")

    try:
        # Read markdown content
        with open(input_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Create PDF
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(markdown_content))
        pdf.save(output_file)

        file_size = Path(output_file).stat().st_size / 1024
        print(f"\nSuccess! PDF created: {output_file}")
        print(f"File size: {file_size:.2f} KB")

    except Exception as e:
        print(f"Error converting to PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Convert COMPLETE_PRODUCT_SUMMARY.md
    convert_markdown_to_pdf(
        "COMPLETE_PRODUCT_SUMMARY.md",
        "QBITEL_AI_Product_Summary_Professional.pdf"
    )
