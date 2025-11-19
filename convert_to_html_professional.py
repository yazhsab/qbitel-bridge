#!/usr/bin/env python3
"""
Enterprise-Grade HTML Converter for QBITEL AI Investor Documents
Converts Markdown to professionally formatted HTML that can be printed to PDF
"""

import os
import sys
from pathlib import Path
import markdown
from datetime import datetime

# Custom CSS for enterprise document styling
ENTERPRISE_CSS = """
<style>
@media print {
    @page {
        size: Letter;
        margin: 1in 0.75in 1in 0.75in;
    }

    .no-print {
        display: none;
    }

    .page-break {
        page-break-after: always;
    }

    h1, h2, h3, h4, h5 {
        page-break-after: avoid;
    }

    table, pre, blockquote {
        page-break-inside: avoid;
    }
}

body {
    font-family: 'Segoe UI', 'Calibri', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #2c3e50;
    max-width: 8.5in;
    margin: 0 auto;
    padding: 20px;
    background: white;
}

.document-header {
    text-align: center;
    border-bottom: 3px solid #0066cc;
    padding-bottom: 20px;
    margin-bottom: 40px;
}

.company-name {
    font-size: 32pt;
    font-weight: 700;
    color: #0066cc;
    margin-bottom: 10px;
}

.document-title {
    font-size: 20pt;
    color: #2c3e50;
    margin-bottom: 10px;
}

.document-meta {
    font-size: 10pt;
    color: #666;
    margin-top: 10px;
}

.confidential-notice {
    background-color: #fff3cd;
    border: 2px solid #856404;
    padding: 15px;
    margin: 20px 0;
    border-radius: 5px;
    font-size: 10pt;
    text-align: center;
    font-weight: 600;
    color: #856404;
}

h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 40px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 3px solid #0066cc;
}

h2 {
    font-size: 20pt;
    font-weight: 600;
    color: #0066cc;
    margin-top: 32px;
    margin-bottom: 16px;
}

h3 {
    font-size: 16pt;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 24px;
    margin-bottom: 12px;
}

h4 {
    font-size: 13pt;
    font-weight: 600;
    color: #34495e;
    margin-top: 20px;
    margin-bottom: 10px;
}

h5 {
    font-size: 12pt;
    font-weight: 600;
    color: #34495e;
    margin-top: 16px;
    margin-bottom: 8px;
}

p {
    margin-bottom: 12px;
    text-align: justify;
}

ul, ol {
    margin-bottom: 12px;
    padding-left: 30px;
}

li {
    margin-bottom: 8px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 10pt;
}

th {
    background-color: #0066cc;
    color: white;
    font-weight: 600;
    padding: 12px 10px;
    text-align: left;
    border: 1px solid #0052a3;
}

td {
    padding: 10px;
    border: 1px solid #ddd;
    vertical-align: top;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

code {
    font-family: 'Consolas', 'Courier New', monospace;
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10pt;
}

pre {
    background-color: #f8f9fa;
    border-left: 4px solid #0066cc;
    padding: 15px;
    margin: 15px 0;
    overflow-x: auto;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    line-height: 1.4;
    border-radius: 3px;
}

pre code {
    background-color: transparent;
    padding: 0;
}

blockquote {
    border-left: 4px solid #0066cc;
    padding-left: 20px;
    margin: 15px 0;
    color: #555;
    font-style: italic;
}

hr {
    border: none;
    border-top: 2px solid #e1e8ed;
    margin: 30px 0;
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
}

a:hover {
    text-decoration: underline;
}

.executive-summary {
    background-color: #f0f7ff;
    border: 2px solid #0066cc;
    padding: 20px;
    margin: 25px 0;
    border-radius: 5px;
}

.highlight-box {
    background-color: #fff9e6;
    border-left: 4px solid #ffc107;
    padding: 15px;
    margin: 20px 0;
}

.footer {
    margin-top: 60px;
    padding-top: 20px;
    border-top: 2px solid #e1e8ed;
    text-align: center;
    font-size: 9pt;
    color: #666;
}

.print-button {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 12px 24px;
    background-color: #0066cc;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14pt;
    font-weight: 600;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    z-index: 1000;
}

.print-button:hover {
    background-color: #0052a3;
}

@media screen {
    body {
        background: #f5f5f5;
        padding: 40px 20px;
    }

    .document-container {
        background: white;
        padding: 60px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 8.5in;
        margin: 0 auto;
    }
}
</style>
"""

PRINT_SCRIPT = """
<script>
function printDocument() {
    window.print();
}

// Add keyboard shortcut Ctrl+P
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'p') {
        e.preventDefault();
        printDocument();
    }
});
</script>
"""


def get_document_title(filename):
    """Get appropriate title for each document"""
    titles = {
        "INVESTOR_PITCH_DECK": "Series A Investment Deck",
        "PRODUCT_WALKTHROUGH": "Product & Technology Overview",
        "IMPLEMENTATION_PLAN_ROADMAP": "5-Year Implementation Roadmap"
    }
    for key, title in titles.items():
        if key in filename:
            return title
    return filename


def convert_markdown_to_html(markdown_file: Path, output_dir: Path = None):
    """
    Convert a markdown file to a professionally formatted HTML

    Args:
        markdown_file: Path to the markdown file
        output_dir: Output directory for the HTML (defaults to same directory as input)
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
            'sane_lists',
            'attr_list'
        ]
    )

    document_title = get_document_title(markdown_file.stem)
    current_date = datetime.now().strftime("%B %d, %Y")

    # Wrap HTML with proper structure
    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QBITEL AI - {document_title}</title>
    {ENTERPRISE_CSS}
</head>
<body>
    <button class="print-button no-print" onclick="printDocument()">Print to PDF</button>

    <div class="document-container">
        <div class="document-header">
            <div class="company-name">QBITEL AI</div>
            <div class="document-title">{document_title}</div>
            <div class="document-meta">
                Confidential & Proprietary | {current_date}
            </div>
        </div>

        <div class="confidential-notice no-print">
            CONFIDENTIAL - For Investor Use Only | Not for Distribution
        </div>

        <div class="content">
            {html_content}
        </div>

        <div class="footer">
            <p><strong>© 2025 QBITEL AI. All rights reserved.</strong></p>
            <p>This document contains confidential and proprietary information.</p>
            <p>Generated: {current_date}</p>
        </div>
    </div>

    {PRINT_SCRIPT}
</body>
</html>
"""

    # Save HTML file
    html_filename = output_dir / f"{markdown_file.stem}.html"
    print(f"Generating HTML: {html_filename.name}...")

    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"[OK] Successfully created: {html_filename}")
    print(f"   > Open in browser and press Ctrl+P to save as PDF")
    print(f"   > Or click the 'Print to PDF' button in the top-right")
    return html_filename


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
    print("QBITEL AI - Enterprise HTML Generator (Print to PDF)")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    print("=" * 80)
    print()

    output_dir = project_root / "investor_documents_html"

    generated_files = []
    for filename in files_to_convert:
        markdown_file = project_root / filename
        if markdown_file.exists():
            try:
                html_file = convert_markdown_to_html(markdown_file, output_dir)
                generated_files.append(html_file)
                print()
            except Exception as e:
                print(f"[ERROR] Error converting {filename}: {e}")
                import traceback
                traceback.print_exc()
                print()
        else:
            print(f"[WARNING] File not found: {filename}")
            print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files converted: {len(generated_files)}/{len(files_to_convert)}")
    print(f"Output directory: {output_dir}")
    print()
    print("Generated HTML files:")
    for html_file in generated_files:
        file_size = html_file.stat().st_size / 1024  # KB
        print(f"  - {html_file.name} ({file_size:.1f} KB)")
    print()
    print("HOW TO SAVE AS PDF:")
    print("  1. Open each HTML file in your web browser (Chrome/Edge recommended)")
    print("  2. Click the 'Print to PDF' button OR press Ctrl+P")
    print("  3. Select 'Save as PDF' as the printer")
    print("  4. Choose destination and click 'Save'")
    print("=" * 80)

    return len(generated_files) == len(files_to_convert)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
