#!/usr/bin/env python3
"""
Professional HTML Converter for QBITEL AI Documentation
Creates beautifully formatted HTML that can be printed to PDF from browser
"""

import markdown
from pathlib import Path
import sys

# Professional CSS styling
PROFESSIONAL_CSS = """
/* Print-optimized professional theme */
@media print {
    @page {
        size: A4;
        margin: 2cm;
    }

    h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
    }

    pre, blockquote, table {
        page-break-inside: avoid;
    }

    thead {
        display: table-header-group;
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', 'Calibri', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #333;
    max-width: 210mm;
    margin: 0 auto;
    padding: 20mm;
    background: #fff;
}

h1 {
    font-size: 32pt;
    font-weight: 700;
    color: #1a1a1a;
    margin: 1.5em 0 0.8em 0;
    padding-bottom: 0.3em;
    border-bottom: 4px solid #0066cc;
    page-break-after: avoid;
}

h1:first-of-type {
    font-size: 40pt;
    text-align: center;
    margin-top: 2cm;
    border: none;
    color: #0066cc;
}

h2 {
    font-size: 22pt;
    font-weight: 600;
    color: #0066cc;
    margin: 1.5em 0 0.6em 0;
    page-break-after: avoid;
}

h3 {
    font-size: 16pt;
    font-weight: 600;
    color: #0052a3;
    margin: 1.2em 0 0.5em 0;
    page-break-after: avoid;
}

h4 {
    font-size: 13pt;
    font-weight: 600;
    color: #004080;
    margin: 1em 0 0.4em 0;
    page-break-after: avoid;
}

h5 {
    font-size: 11.5pt;
    font-weight: 600;
    color: #003366;
    margin: 0.8em 0 0.3em 0;
}

p {
    margin: 0.6em 0;
    text-align: justify;
}

ul, ol {
    margin: 0.8em 0;
    padding-left: 2.5em;
}

li {
    margin: 0.4em 0;
}

code {
    background-color: #f5f5f5;
    padding: 0.2em 0.5em;
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 9.5pt;
    color: #c7254e;
    border: 1px solid #e8e8e8;
}

pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-left: 5px solid #0066cc;
    border-radius: 4px;
    padding: 1.2em;
    overflow-x: auto;
    margin: 1.2em 0;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: #333;
    border: none;
    font-size: 10pt;
}

blockquote {
    border-left: 5px solid #0066cc;
    background-color: #f0f7ff;
    padding: 1em 1.5em;
    margin: 1.2em 0;
    font-style: italic;
    page-break-inside: avoid;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.5em 0;
    page-break-inside: avoid;
    font-size: 10pt;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

th {
    background-color: #0066cc;
    color: white;
    padding: 0.8em;
    text-align: left;
    font-weight: 600;
}

td {
    border: 1px solid #ddd;
    padding: 0.7em;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

tr:hover {
    background-color: #f0f7ff;
}

hr {
    border: none;
    border-top: 3px solid #e0e0e0;
    margin: 2.5em 0;
}

a {
    color: #0066cc;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

strong {
    color: #1a1a1a;
    font-weight: 600;
}

em {
    color: #555;
}

/* Styled alerts and warnings */
p > strong:first-child {
    display: inline;
}

/* Cover page metadata */
.metadata {
    text-align: center;
    color: #666;
    margin: 2em 0;
    font-size: 10pt;
}

/* Page break helpers */
.page-break {
    page-break-after: always;
}

/* Table of contents styling */
#table-of-contents + ul {
    background-color: #f8f8f8;
    padding: 1.5em;
    border-radius: 6px;
    border: 2px solid #0066cc;
}

/* Mermaid diagram styling */
.mermaid {
    background-color: #fafafa;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 1em;
    margin: 2em 0;
    text-align: center;
    page-break-before: always;
    page-break-after: always;
}

/* Make mermaid SVGs fit page width while maximizing size */
.mermaid svg {
    max-width: 100%;
    width: 100%;
    height: auto;
}

/* Larger text in mermaid diagrams */
.mermaid text {
    font-size: 13px !important;
    font-weight: 500;
}

/* Make node labels more readable */
.mermaid .nodeLabel {
    font-size: 13px !important;
}

/* Print-specific mermaid styling - portrait orientation */
@media print {
    .mermaid {
        page-break-before: always;
        page-break-after: always;
        page-break-inside: avoid;
        background-color: white;
        border: none;
        padding: 0.5cm;
    }

    .mermaid svg {
        width: 100%;
        max-width: 100%;
        height: auto;
    }
}

/* Print button (hide on print) */
.no-print {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 15px 30px;
    background-color: #0066cc;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 14pt;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    z-index: 1000;
}

.no-print:hover {
    background-color: #0052a3;
}

@media print {
    .no-print {
        display: none;
    }
}
"""

def convert_markdown_to_html(
    input_file: str,
    output_file: str = None
):
    """
    Convert markdown file to professionally formatted HTML

    Args:
        input_file: Path to markdown file
        output_file: Output HTML path (optional)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    # Default output file
    if output_file is None:
        output_file = str(input_path.stem + "_Professional.html")

    print(f"Converting {input_file} to professional HTML...")
    print(f"Output: {output_file}")

    try:
        # Read markdown content
        with open(input_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # First, extract and replace mermaid diagrams with placeholders
        import re
        mermaid_diagrams = []

        def extract_mermaid(match):
            diagram_code = match.group(1)
            index = len(mermaid_diagrams)
            mermaid_diagrams.append(diagram_code)
            return f'<div class="mermaid" id="mermaid-{index}">\n{diagram_code}\n</div>'

        # Extract mermaid blocks before markdown processing
        markdown_content = re.sub(
            r'```mermaid\s*\n(.*?)\n```',
            extract_mermaid,
            markdown_content,
            flags=re.DOTALL
        )

        # Convert markdown to HTML (without codehilite to avoid breaking mermaid)
        md = markdown.Markdown(extensions=[
            'extra',          # Tables, code blocks, fenced code
            'nl2br',          # New line to break
            'sane_lists',     # Better list handling
            'smarty',         # Smart quotes
            'toc',            # Table of contents
            'attr_list',      # Attribute lists
        ])
        html_body = md.convert(markdown_content)

        # Create full HTML document
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QBITEL AI - Complete Product Summary</title>
    <style>
{PROFESSIONAL_CSS}
    </style>
</head>
<body>
    <button class="no-print" onclick="window.print()">🖨️ Print to PDF</button>

    {html_body}

    <!-- Mermaid JS for diagram rendering -->
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

        // Initialize Mermaid with optimized settings for PDF
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            themeVariables: {{
                primaryColor: '#0066cc',
                primaryTextColor: '#fff',
                primaryBorderColor: '#0052a3',
                lineColor: '#333',
                secondaryColor: '#f0f7ff',
                tertiaryColor: '#e8f5e9',
                fontSize: '14px',
                fontFamily: 'Arial, sans-serif'
            }},
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis',
                padding: 8,
                nodeSpacing: 50,
                rankSpacing: 50
            }},
            graph: {{
                useMaxWidth: true
            }}
        }});
    </script>

    <script>
        // Add smooth scrolling for internal links
        document.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                anchor.addEventListener('click', function (e) {{
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {{
                        target.scrollIntoView({{ behavior: 'smooth' }});
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>"""

        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)

        file_size = Path(output_file).stat().st_size / 1024
        abs_path = Path(output_file).absolute()

        print(f"\n[SUCCESS] Professional HTML created")
        print(f"  Location: {abs_path}")
        print(f"  File size: {file_size:.2f} KB")
        print(f"\nNext steps to create PDF:")
        print(f"  1. Open the HTML file in your browser:")
        print(f"     file:///{abs_path}")
        print(f"  2. Click the 'Print to PDF' button (or press Ctrl+P)")
        print(f"  3. Select 'Save as PDF' or 'Microsoft Print to PDF'")
        print(f"  4. Recommended settings:")
        print(f"     - Paper size: A4")
        print(f"     - Margins: Default")
        print(f"     - Headers/Footers: None (already styled)")
        print(f"     - Background graphics: ON")
        print(f"\nFeatures included:")
        print(f"  + Professional typography and layout")
        print(f"  + Print-optimized CSS")
        print(f"  + Styled tables and code blocks")
        print(f"  + Color-coded headings")
        print(f"  + One-click PDF export from browser")

    except Exception as e:
        print(f"Error converting to HTML: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Convert COMPLETE_PRODUCT_SUMMARY.md to professional HTML
    convert_markdown_to_html(
        "COMPLETE_PRODUCT_SUMMARY.md",
        "QBITEL_AI_Product_Summary_Professional.html"
    )
