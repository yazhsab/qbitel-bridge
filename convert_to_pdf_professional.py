#!/usr/bin/env python3
"""
Professional PDF Converter for QBITEL AI Documentation
Creates beautifully formatted PDFs with custom styling and layout
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import sys

# Professional CSS styling for the PDF
PROFESSIONAL_CSS = """
@page {
    size: A4;
    margin: 2.5cm 2cm 2cm 2cm;

    @top-center {
        content: "QBITEL AI - Product Summary";
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 9pt;
        color: #666;
    }

    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 9pt;
        color: #666;
    }
}

@page:first {
    margin: 0;
    @top-center { content: none; }
    @bottom-right { content: none; }
}

body {
    font-family: 'Segoe UI', 'Calibri', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #333;
    text-align: justify;
}

h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
    page-break-after: avoid;
    border-bottom: 3px solid #0066cc;
    padding-bottom: 0.3em;
}

h2 {
    font-size: 20pt;
    font-weight: 600;
    color: #0066cc;
    margin-top: 1.2em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
}

h3 {
    font-size: 15pt;
    font-weight: 600;
    color: #0052a3;
    margin-top: 1em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
}

h4 {
    font-size: 12pt;
    font-weight: 600;
    color: #004080;
    margin-top: 0.8em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}

p {
    margin: 0.5em 0;
    orphans: 3;
    widows: 3;
}

ul, ol {
    margin: 0.5em 0;
    padding-left: 2em;
}

li {
    margin: 0.3em 0;
}

code {
    background-color: #f5f5f5;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 9pt;
    color: #d14;
}

pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-left: 4px solid #0066cc;
    border-radius: 4px;
    padding: 1em;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 1em 0;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: #333;
    font-size: 9pt;
}

blockquote {
    border-left: 4px solid #0066cc;
    background-color: #f0f7ff;
    padding: 1em 1.5em;
    margin: 1em 0;
    font-style: italic;
    page-break-inside: avoid;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
    page-break-inside: avoid;
    font-size: 9.5pt;
}

th {
    background-color: #0066cc;
    color: white;
    padding: 0.6em;
    text-align: left;
    font-weight: 600;
}

td {
    border: 1px solid #ddd;
    padding: 0.5em;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

hr {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 2em 0;
}

a {
    color: #0066cc;
    text-decoration: none;
}

strong {
    color: #1a1a1a;
    font-weight: 600;
}

em {
    color: #555;
}

/* Special styling for warnings and notes */
p:has(> strong:first-child) {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 0.8em 1em;
    margin: 1em 0;
}

/* Cover page styling (first h1) */
h1:first-of-type {
    font-size: 36pt;
    text-align: center;
    margin-top: 5cm;
    border: none;
    color: #0066cc;
}

/* Checkmarks and emoji styling */
.emoji {
    font-size: 1.2em;
}
"""

def convert_markdown_to_pdf(
    input_file: str,
    output_file: str = None,
    include_toc: bool = False
):
    """
    Convert markdown file to professionally formatted PDF

    Args:
        input_file: Path to markdown file
        output_file: Output PDF path (optional)
        include_toc: Include table of contents (default: False, as doc already has one)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    # Default output file
    if output_file is None:
        output_file = str(input_path.stem + "_Professional.pdf")

    print(f"Converting {input_file} to professional PDF...")
    print(f"Output: {output_file}")

    try:
        # Read markdown content
        with open(input_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Convert markdown to HTML
        md = markdown.Markdown(extensions=[
            'extra',          # Tables, code blocks, etc.
            'nl2br',          # New line to break
            'sane_lists',     # Better list handling
            'smarty',         # Smart quotes
            'toc'             # Table of contents
        ])
        html_content = md.convert(markdown_content)

        # Wrap in HTML structure
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>QBITEL AI - Product Summary</title>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Convert HTML to PDF with custom CSS
        HTML(string=full_html).write_pdf(
            output_file,
            stylesheets=[CSS(string=PROFESSIONAL_CSS)]
        )

        file_size = Path(output_file).stat().st_size / 1024 / 1024
        print(f"\n✓ Success! Professional PDF created")
        print(f"  Location: {Path(output_file).absolute()}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"\nThe PDF includes:")
        print(f"  ✓ Professional typography and layout")
        print(f"  ✓ Page numbers and headers")
        print(f"  ✓ Styled tables and code blocks")
        print(f"  ✓ Color-coded headings")
        print(f"  ✓ Optimized for printing and screen reading")

    except Exception as e:
        print(f"Error converting to PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Convert COMPLETE_PRODUCT_SUMMARY.md to professional PDF
    convert_markdown_to_pdf(
        "COMPLETE_PRODUCT_SUMMARY.md",
        "QBITEL_AI_Product_Summary_Professional.pdf"
    )
