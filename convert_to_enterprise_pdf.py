#!/usr/bin/env python3
"""
Enterprise-Grade PDF Converter for QBITEL AI Investor Documents
Converts Markdown to professionally formatted PDF with custom styling
"""

import os
import sys
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from datetime import datetime

# Custom CSS for enterprise document styling
ENTERPRISE_CSS = """
@page {
    size: Letter;
    margin: 1in 0.75in 1in 0.75in;

    @top-left {
        content: "QBITEL AI - Confidential";
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 9pt;
        color: #666;
    }

    @top-right {
        content: "December 2025";
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 9pt;
        color: #666;
    }

    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 9pt;
        color: #666;
    }
}

body {
    font-family: 'Segoe UI', 'Calibri', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #2c3e50;
    max-width: 100%;
}

h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 24pt;
    margin-bottom: 16pt;
    padding-bottom: 8pt;
    border-bottom: 3px solid #0066cc;
    page-break-after: avoid;
}

h2 {
    font-size: 20pt;
    font-weight: 600;
    color: #0066cc;
    margin-top: 20pt;
    margin-bottom: 12pt;
    page-break-after: avoid;
}

h3 {
    font-size: 16pt;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 16pt;
    margin-bottom: 10pt;
    page-break-after: avoid;
}

h4 {
    font-size: 13pt;
    font-weight: 600;
    color: #34495e;
    margin-top: 14pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}

h5 {
    font-size: 12pt;
    font-weight: 600;
    color: #34495e;
    margin-top: 12pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}

p {
    margin-bottom: 10pt;
    text-align: justify;
    orphans: 2;
    widows: 2;
}

ul, ol {
    margin-bottom: 10pt;
    padding-left: 24pt;
}

li {
    margin-bottom: 6pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 16pt 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

th {
    background-color: #0066cc;
    color: white;
    font-weight: 600;
    padding: 10pt 8pt;
    text-align: left;
    border: 1px solid #0052a3;
}

td {
    padding: 8pt;
    border: 1px solid #ddd;
    vertical-align: top;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

code {
    font-family: 'Consolas', 'Courier New', monospace;
    background-color: #f4f4f4;
    padding: 2pt 4pt;
    border-radius: 3pt;
    font-size: 10pt;
}

pre {
    background-color: #f8f9fa;
    border-left: 4px solid #0066cc;
    padding: 12pt;
    margin: 12pt 0;
    overflow-x: auto;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    line-height: 1.4;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    padding: 0;
}

blockquote {
    border-left: 4px solid #0066cc;
    padding-left: 16pt;
    margin: 12pt 0;
    color: #555;
    font-style: italic;
}

hr {
    border: none;
    border-top: 2px solid #e1e8ed;
    margin: 24pt 0;
}

strong, b {
    font-weight: 600;
    color: #1a1a1a;
}

em, i {
    font-style: italic;
    color: #2c3e50;
}

a {
    color: #0066cc;
    text-decoration: none;
    border-bottom: 1px dotted #0066cc;
}

a:hover {
    border-bottom: 1px solid #0066cc;
}

/* Cover page styling */
.cover-page {
    page-break-after: always;
    text-align: center;
    padding-top: 200pt;
}

.cover-title {
    font-size: 36pt;
    font-weight: 700;
    color: #0066cc;
    margin-bottom: 24pt;
}

.cover-subtitle {
    font-size: 18pt;
    color: #2c3e50;
    margin-bottom: 12pt;
}

.cover-date {
    font-size: 14pt;
    color: #666;
    margin-top: 40pt;
}

/* Special formatting for financial tables */
.financial-table {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
}

/* Executive summary box */
.executive-summary {
    background-color: #f0f7ff;
    border: 2px solid #0066cc;
    padding: 16pt;
    margin: 20pt 0;
    page-break-inside: avoid;
}

/* Status indicators */
.status-complete::before {
    content: "✅ ";
}

.status-progress::before {
    content: "🔄 ";
}

.status-pending::before {
    content: "⚠️ ";
}

/* Page breaks */
.page-break {
    page-break-after: always;
}

/* Prevent breaks in specific elements */
.no-break {
    page-break-inside: avoid;
}
"""


def convert_markdown_to_pdf(markdown_file: Path, output_dir: Path = None):
    """
    Convert a markdown file to a professionally formatted PDF

    Args:
        markdown_file: Path to the markdown file
        output_dir: Output directory for the PDF (defaults to same directory as input)
    """
    if output_dir is None:
        output_dir = markdown_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read markdown content
    print(f"Reading {markdown_file.name}...")
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # Convert markdown to HTML
    print(f"Converting {markdown_file.name} to HTML...")
    html_content = markdown.markdown(
        markdown_content,
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'nl2br',
            'sane_lists'
        ]
    )

    # Wrap HTML with proper structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{markdown_file.stem}</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate PDF
    pdf_filename = output_dir / f"{markdown_file.stem}.pdf"
    print(f"Generating PDF: {pdf_filename.name}...")

    HTML(string=full_html).write_pdf(
        pdf_filename,
        stylesheets=[CSS(string=ENTERPRISE_CSS)]
    )

    print(f"✅ Successfully created: {pdf_filename}")
    return pdf_filename


def main():
    """Main function to convert all investor documents"""

    # Get the project root directory
    project_root = Path(__file__).parent

    # Files to convert
    files_to_convert = [
        "INVESTOR_PITCH_DECK.md",
        "PRODUCT_WALKTHROUGH.md",
        "IMPLEMENTATION_PLAN_ROADMAP.md"
    ]

    print("=" * 80)
    print("QBITEL AI - Enterprise PDF Generator")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    print("=" * 80)
    print()

    output_dir = project_root / "investor_documents_pdf"

    generated_files = []
    for filename in files_to_convert:
        markdown_file = project_root / filename
        if markdown_file.exists():
            try:
                pdf_file = convert_markdown_to_pdf(markdown_file, output_dir)
                generated_files.append(pdf_file)
                print()
            except Exception as e:
                print(f"❌ Error converting {filename}: {e}")
                print()
        else:
            print(f"⚠️  File not found: {filename}")
            print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files converted: {len(generated_files)}/{len(files_to_convert)}")
    print(f"Output directory: {output_dir}")
    print()
    print("Generated PDFs:")
    for pdf_file in generated_files:
        file_size = pdf_file.stat().st_size / 1024  # KB
        print(f"  - {pdf_file.name} ({file_size:.1f} KB)")
    print("=" * 80)

    return len(generated_files) == len(files_to_convert)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
