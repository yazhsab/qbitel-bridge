"""
Build QBITEL Bridge BPO Marketing Pitch - Professional DOCX
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ─── Brand Colors ───────────────────────────────────────────────────────────
NAVY      = RGBColor(0x0D, 0x1B, 0x3E)   # Deep navy
TEAL      = RGBColor(0x00, 0x8B, 0x9A)   # Quantum teal
GOLD      = RGBColor(0xF0, 0xA5, 0x00)   # Accent gold
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG  = RGBColor(0xF4, 0xF7, 0xFA)  # Very light blue-grey
MID_GREY  = RGBColor(0x5A, 0x6A, 0x7A)
DARK_TEXT = RGBColor(0x1A, 0x1A, 0x2E)
TABLE_HDR = RGBColor(0x0D, 0x1B, 0x3E)   # Same as NAVY
TABLE_ALT = RGBColor(0xEA, 0xF3, 0xF8)   # Very light teal
TEAL_DARK = RGBColor(0x00, 0x6B, 0x7A)


def set_cell_bg(cell, color: RGBColor):
    """Set table cell background color."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    hex_color = str(color)  # RGBColor.__str__ returns hex like '0D1B3E'
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for side, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_para_spacing(para, before=0, after=0, line=None):
    pPr = para._p.get_or_add_pPr()
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:before'), str(before))
    spacing.set(qn('w:after'), str(after))
    if line:
        spacing.set(qn('w:line'), str(line))
        spacing.set(qn('w:lineRule'), 'auto')
    pPr.append(spacing)


def add_horizontal_rule(doc, color: RGBColor = TEAL, thickness: int = 18):
    """Add a colored horizontal rule paragraph."""
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pb = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    hex_col = str(color)
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), str(thickness))
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), hex_col)
    pb.append(bottom)
    pPr.append(pb)
    set_para_spacing(p, before=40, after=40)
    return p


def add_shaded_box(doc, text, bg_color=LIGHT_BG, text_color=NAVY, font_size=11, bold=False, italic=False):
    """Add a paragraph with background shading (simulated via table)."""
    table = doc.add_table(rows=1, cols=1)
    table.style = 'Table Grid'
    cell = table.cell(0, 0)
    set_cell_bg(cell, bg_color)
    set_cell_margins(cell, 120, 120, 200, 200)
    # Remove table borders
    tbl = table._tbl
    tblPr = tbl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    tblBorders = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'nil')
        tblBorders.append(node)
    tblPr.append(tblBorders)
    para = cell.paragraphs[0]
    run = para.add_run(text)
    run.font.size = Pt(font_size)
    run.font.color.rgb = text_color
    run.bold = bold
    run.italic = italic
    run.font.name = 'Calibri'
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_para_spacing(para, 0, 0)
    return table


def add_cover_page(doc):
    # Full-width navy header block via table
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_bg(cell, NAVY)
    set_cell_margins(cell, 400, 400, 400, 400)

    # Remove table borders
    tblEl = tbl._tbl
    tblPr = tblEl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tblEl.insert(0, tblPr)
    tblBorders = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'nil')
        tblBorders.append(node)
    tblPr.append(tblBorders)

    # Set column width to full page
    tblWidth = OxmlElement('w:tblW')
    tblWidth.set(qn('w:w'), '9360')
    tblWidth.set(qn('w:type'), 'dxa')
    tblPr.append(tblWidth)

    # Brand line: gold accent
    p1 = cell.paragraphs[0]
    r1 = p1.add_run('━━━━━━━━━━━━━━━━━━━━━━━━')
    r1.font.color.rgb = GOLD
    r1.font.size = Pt(14)
    r1.font.name = 'Calibri'
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p1, 0, 200)

    # QBITEL BRIDGE
    p2 = cell.add_paragraph()
    r2 = p2.add_run('QBITEL BRIDGE')
    r2.font.size = Pt(42)
    r2.font.color.rgb = WHITE
    r2.bold = True
    r2.font.name = 'Calibri'
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p2, 0, 100)

    # Tagline
    p3 = cell.add_paragraph()
    r3 = p3.add_run('BPO & CALL CENTER SECURITY PLATFORM')
    r3.font.size = Pt(16)
    r3.font.color.rgb = TEAL
    r3.bold = True
    r3.font.name = 'Calibri'
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p3, 0, 200)

    # Subtitle
    p4 = cell.add_paragraph()
    r4 = p4.add_run('Quantum-Safe Security for the Human API')
    r4.font.size = Pt(14)
    r4.font.color.rgb = RGBColor(0xCC, 0xDD, 0xEE)
    r4.italic = True
    r4.font.name = 'Calibri'
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p4, 0, 200)

    # Gold rule
    p5 = cell.add_paragraph()
    r5 = p5.add_run('━━━━━━━━━━━━━━━━━━━━━━━━')
    r5.font.color.rgb = GOLD
    r5.font.size = Pt(14)
    r5.font.name = 'Calibri'
    p5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p5, 200, 0)

    doc.add_paragraph()  # spacer

    # Key stats bar (3 cols)
    stats = [
        ('4–6 Hours', 'Full Deployment\nZero Downtime'),
        ('78%', 'Autonomous Threat\nResolution'),
        ('9 Frameworks', 'Compliance Automated\nin < 10 Minutes'),
    ]
    stats_tbl = doc.add_table(rows=1, cols=3)
    stats_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblEl2 = stats_tbl._tbl
    tblPr2 = tblEl2.find(qn('w:tblPr'))
    if tblPr2 is None:
        tblPr2 = OxmlElement('w:tblPr')
        tblEl2.insert(0, tblPr2)
    tblBorders2 = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'nil')
        tblBorders2.append(node)
    tblPr2.append(tblBorders2)

    for i, (big, small) in enumerate(stats):
        cell = stats_tbl.cell(0, i)
        bg = NAVY if i == 1 else TEAL
        set_cell_bg(cell, bg)
        set_cell_margins(cell, 200, 200, 150, 150)
        p = cell.paragraphs[0]
        r = p.add_run(big)
        r.font.size = Pt(26)
        r.font.color.rgb = GOLD if i == 1 else WHITE
        r.bold = True
        r.font.name = 'Calibri'
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p, 0, 60)
        p2 = cell.add_paragraph()
        r2 = p2.add_run(small)
        r2.font.size = Pt(9)
        r2.font.color.rgb = WHITE
        r2.font.name = 'Calibri'
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p2, 0, 0)

    doc.add_paragraph()

    # Additional stats row
    stats2 = [
        ('<2ms', 'Voice Encryption\nOverhead'),
        ('89%+', 'Protocol Discovery\nAccuracy'),
        ('20,000+', 'Concurrent Agent\nSessions'),
    ]
    stats_tbl2 = doc.add_table(rows=1, cols=3)
    stats_tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblEl3 = stats_tbl2._tbl
    tblPr3 = tblEl3.find(qn('w:tblPr'))
    if tblPr3 is None:
        tblPr3 = OxmlElement('w:tblPr')
        tblEl3.insert(0, tblPr3)
    tblBorders3 = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'nil')
        tblBorders3.append(node)
    tblPr3.append(tblBorders3)

    for i, (big, small) in enumerate(stats2):
        cell = stats_tbl2.cell(0, i)
        set_cell_bg(cell, LIGHT_BG)
        set_cell_margins(cell, 200, 200, 150, 150)
        p = cell.paragraphs[0]
        r = p.add_run(big)
        r.font.size = Pt(22)
        r.font.color.rgb = NAVY
        r.bold = True
        r.font.name = 'Calibri'
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p, 0, 40)
        p2 = cell.add_paragraph()
        r2 = p2.add_run(small)
        r2.font.size = Pt(9)
        r2.font.color.rgb = MID_GREY
        r2.font.name = 'Calibri'
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p2, 0, 0)

    doc.add_paragraph()

    # Marketing line
    p_tag = doc.add_paragraph()
    r_tag = p_tag.add_run(
        'AI-Powered • Quantum-Safe • Zero Disruption • Deployed in Hours'
    )
    r_tag.font.size = Pt(11)
    r_tag.font.color.rgb = TEAL
    r_tag.bold = True
    r_tag.font.name = 'Calibri'
    p_tag.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p_tag, 0, 0)

    doc.add_page_break()


def add_section_header(doc, title, subtitle=None):
    """Teal left-bar section header."""
    tbl = doc.add_table(rows=1, cols=2)
    tblEl = tbl._tbl
    tblPr = tblEl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tblEl.insert(0, tblPr)
    tblBorders = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'nil')
        tblBorders.append(node)
    tblPr.append(tblBorders)

    # Accent bar (narrow teal column)
    bar_cell = tbl.cell(0, 0)
    set_cell_bg(bar_cell, TEAL)
    tc = bar_cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcW = OxmlElement('w:tcW')
    tcW.set(qn('w:w'), '160')
    tcW.set(qn('w:type'), 'dxa')
    tcPr.append(tcW)
    bar_cell.paragraphs[0].add_run(' ')

    # Title column
    title_cell = tbl.cell(0, 1)
    set_cell_bg(title_cell, NAVY)
    set_cell_margins(title_cell, 120, 120, 200, 200)
    p = title_cell.paragraphs[0]
    r = p.add_run(title.upper())
    r.font.size = Pt(16)
    r.font.color.rgb = WHITE
    r.bold = True
    r.font.name = 'Calibri'
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_para_spacing(p, 0, 0 if not subtitle else 40)

    if subtitle:
        p2 = title_cell.add_paragraph()
        r2 = p2.add_run(subtitle)
        r2.font.size = Pt(10)
        r2.font.color.rgb = TEAL
        r2.font.name = 'Calibri'
        r2.italic = True
        p2.alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_para_spacing(p2, 0, 0)

    sp = doc.add_paragraph()
    set_para_spacing(sp, 60, 100)


def add_subsection(doc, number, title):
    p = doc.add_paragraph()
    r_num = p.add_run(f'{number}. ')
    r_num.font.size = Pt(13)
    r_num.font.color.rgb = GOLD
    r_num.bold = True
    r_num.font.name = 'Calibri'
    r_title = p.add_run(title)
    r_title.font.size = Pt(13)
    r_title.font.color.rgb = NAVY
    r_title.bold = True
    r_title.font.name = 'Calibri'
    set_para_spacing(p, 160, 60)
    return p


def add_body(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.color.rgb = DARK_TEXT
    r.font.name = 'Calibri'
    set_para_spacing(p, 0, 100)
    return p


def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    r0 = p.add_run('  ▪  ')
    r0.font.size = Pt(10)
    r0.font.color.rgb = TEAL
    r0.font.name = 'Calibri'
    if bold_prefix:
        rb = p.add_run(bold_prefix + ' ')
        rb.font.size = Pt(10.5)
        rb.font.color.rgb = NAVY
        rb.bold = True
        rb.font.name = 'Calibri'
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.color.rgb = DARK_TEXT
    r.font.name = 'Calibri'
    set_para_spacing(p, 0, 50)
    return p


def add_professional_table(doc, headers, rows, col_widths=None):
    """Create a styled table with navy header and alternating rows."""
    num_cols = len(headers)
    tbl = doc.add_table(rows=1 + len(rows), cols=num_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Set column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in tbl.rows:
                row.cells[i].width = Inches(w)

    # Header row
    hrow = tbl.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        set_cell_bg(cell, TABLE_HDR)
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.size = Pt(9.5)
        r.font.color.rgb = WHITE
        r.bold = True
        r.font.name = 'Calibri'
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_para_spacing(p, 0, 0)

    # Data rows
    for ri, row_data in enumerate(rows):
        drow = tbl.rows[ri + 1]
        bg = TABLE_ALT if ri % 2 == 0 else WHITE
        for ci, cell_text in enumerate(row_data):
            cell = drow.cells[ci]
            set_cell_bg(cell, bg)
            set_cell_margins(cell, 70, 70, 100, 100)
            p = cell.paragraphs[0]
            # Bold first cell
            if ci == 0:
                r = p.add_run(cell_text)
                r.font.size = Pt(9.5)
                r.font.color.rgb = NAVY
                r.bold = True
                r.font.name = 'Calibri'
            else:
                r = p.add_run(cell_text)
                r.font.size = Pt(9.5)
                r.font.color.rgb = DARK_TEXT
                r.font.name = 'Calibri'
            set_para_spacing(p, 0, 0)

    # Table borders (light grey lines)
    tblEl = tbl._tbl
    tblPr = tblEl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tblEl.insert(0, tblPr)
    tblBorders = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'single')
        node.set(qn('w:sz'), '4')
        node.set(qn('w:color'), 'C8D8E8')
        tblBorders.append(node)
    tblPr.append(tblBorders)

    sp = doc.add_paragraph()
    set_para_spacing(sp, 0, 120)
    return tbl


def add_callout_box(doc, text, icon='', bg=LIGHT_BG, text_color=NAVY):
    tbl = doc.add_table(rows=1, cols=1)
    cell = tbl.cell(0, 0)
    set_cell_bg(cell, bg)
    set_cell_margins(cell, 140, 140, 220, 220)
    tblEl = tbl._tbl
    tblPr = tblEl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tblEl.insert(0, tblPr)
    tblBorders = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'single' if side in ['left'] else 'nil')
        if side == 'left':
            node.set(qn('w:sz'), '24')
            node.set(qn('w:color'), str(TEAL))
        tblBorders.append(node)
    tblPr.append(tblBorders)

    p = cell.paragraphs[0]
    if icon:
        ri = p.add_run(icon + '  ')
        ri.font.size = Pt(11)
        ri.font.color.rgb = GOLD
        ri.font.name = 'Calibri'
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.color.rgb = text_color
    r.font.name = 'Calibri'
    r.italic = True
    set_para_spacing(p, 0, 0)

    sp = doc.add_paragraph()
    set_para_spacing(sp, 0, 120)
    return tbl


def add_scenario_box(doc, label, title, body):
    tbl = doc.add_table(rows=2, cols=1)
    # Header row
    hcell = tbl.cell(0, 0)
    set_cell_bg(hcell, TEAL_DARK)
    set_cell_margins(hcell, 100, 100, 180, 180)
    p = hcell.paragraphs[0]
    rl = p.add_run(f'{label}  |  ')
    rl.font.size = Pt(9)
    rl.font.color.rgb = GOLD
    rl.bold = True
    rl.font.name = 'Calibri'
    rt = p.add_run(title)
    rt.font.size = Pt(11)
    rt.font.color.rgb = WHITE
    rt.bold = True
    rt.font.name = 'Calibri'
    set_para_spacing(p, 0, 0)
    # Body row
    bcell = tbl.cell(1, 0)
    set_cell_bg(bcell, LIGHT_BG)
    set_cell_margins(bcell, 120, 120, 180, 180)
    p2 = bcell.paragraphs[0]
    r2 = p2.add_run(body)
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK_TEXT
    r2.font.name = 'Calibri'
    set_para_spacing(p2, 0, 0)
    # Remove borders
    tblEl = tbl._tbl
    tblPr = tblEl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tblEl.insert(0, tblPr)
    tblBorders = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'nil')
        tblBorders.append(node)
    tblPr.append(tblBorders)

    sp = doc.add_paragraph()
    set_para_spacing(sp, 0, 120)


def set_page_margins(doc):
    for section in doc.sections:
        section.page_width  = Inches(8.5)
        section.page_height = Inches(11)
        section.left_margin   = Inches(1.0)
        section.right_margin  = Inches(1.0)
        section.top_margin    = Inches(0.8)
        section.bottom_margin = Inches(0.8)


def add_footer(doc):
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.clear()
        set_cell_bg = None  # not needed here

        # Left: brand name
        r1 = p.add_run('QBITEL BRIDGE  |  BPO & Call Center Security Platform')
        r1.font.size = Pt(8)
        r1.font.color.rgb = MID_GREY
        r1.font.name = 'Calibri'

        # Tab to right
        r2 = p.add_run('        Confidential — For Authorized Recipients Only')
        r2.font.size = Pt(8)
        r2.font.color.rgb = MID_GREY
        r2.font.name = 'Calibri'

        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add top border to footer
        pPr = p._p.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        top = OxmlElement('w:top')
        top.set(qn('w:val'), 'single')
        top.set(qn('w:sz'), '6')
        top.set(qn('w:color'), str(TEAL))
        pBdr.append(top)
        pPr.append(pBdr)


def build_document():
    doc = Document()
    set_page_margins(doc)

    # Default paragraph style
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(10.5)

    # ─── COVER PAGE ───────────────────────────────────────────────────────────
    add_cover_page(doc)

    # ─── SECTION 1: THE PROBLEM ───────────────────────────────────────────────
    add_section_header(doc, '1. The Problem BPOs Cannot Ignore',
                       'Three converging threats are turning legacy contact center infrastructure into your largest risk')

    add_body(doc,
        'Business Process Outsourcing operations and contact centers are the frontline of global commerce — '
        'processing millions of sensitive customer interactions every day. Credit card payments, patient records, '
        'insurance details, and financial disputes flow through infrastructure built decades ago, protected by '
        'encryption standards that were never designed for today\'s threat landscape.')

    add_subsection(doc, 'Threat 1', 'Quantum Harvest Attacks — Your Call Recordings Are Already Being Stolen')
    add_body(doc,
        'Nation-state adversaries are executing "harvest now, decrypt later" strategies — capturing your encrypted '
        'call recordings today to decrypt when quantum computers arrive (estimated 5–10 years). SOX compliance '
        'requires 7-year call recording retention. That means recordings captured this week will still be '
        'within retention when quantum decryption becomes viable.')

    add_subsection(doc, 'Threat 2', 'SIP Toll Fraud — $10 Billion Annual Industry Loss')
    add_body(doc,
        'A single compromised PBX trunk can generate $50,000 in fraudulent premium-rate calls over a single '
        'weekend. Detection typically happens when the carrier invoice arrives 72 hours later. Traditional '
        'fraud prevention tools work on known patterns — QBITEL\'s AI detects emerging patterns after just 3 calls.')

    add_subsection(doc, 'Threat 3', 'Remote Workforce Security Gap')
    add_body(doc,
        'Post-pandemic, 60%+ of BPO agents work from home on consumer-grade internet connections, sharing '
        'home networks with family devices. One compromised home router is all a threat actor needs to intercept '
        'agent sessions, capture customer data, and exfiltrate PII undetected for weeks.')

    add_callout_box(doc,
        'Traditional response: $5M+ PBX replacement, 12–18 months of migration risk, weeks of agent retraining. '
        'QBITEL response: 4–6 hour deployment at the network layer. Nothing replaced. Nothing disrupted.',
        icon='◈')

    add_professional_table(doc,
        ['Challenge', 'Industry Impact', 'Traditional Fix', 'QBITEL Approach'],
        [
            ['Legacy PBX encryption', 'Unencrypted SIP carries 70%+ of global call traffic', '$5M+ PBX replacement', 'PQC overlay — no hardware changes'],
            ['TN3270e terminal sessions', 'Mainframe access with zero session protection', 'VPN + MFA rollout', 'PQC tunnel wrapping, transparent'],
            ['Payment DTMF exposure', '50M+ daily payment transactions over voice', 'Manual pause/resume procedures', 'Automated DTMF masking <5ms'],
            ['Remote agent risk', '60%+ agents on consumer-grade networks', 'VPN infrastructure rollout', 'VPN-less PQC tunnels, instant'],
            ['Toll fraud', '$10B+ annual industry loss', 'Carrier-level controls (72h delay)', 'AI detection in <1 second'],
            ['Insider data theft', '34% of BPO breaches are insider threats', 'Policy enforcement only', 'DLP enforcement at kernel level'],
            ['Multi-tenant compliance', 'Separate audits per client', 'Separate infrastructure per client', 'Isolated policies, one platform'],
        ],
        col_widths=[1.6, 2.0, 1.7, 1.7]
    )

    # ─── SECTION 2: QBITEL BRIDGE ─────────────────────────────────────────────
    add_section_header(doc, '2. QBITEL Bridge — How It Works',
                       'AI discovers your environment. PQC encrypts it. Agentic AI defends it.')

    add_body(doc,
        'QBITEL Bridge deploys as a network-layer security overlay. It does not require you to replace any '
        'existing infrastructure, retrain agents, or accept any downtime. In 4–6 hours, your entire contact '
        'center environment — every protocol, every agent session, every recording — is protected by '
        'NIST Level 5 post-quantum cryptography.')

    add_subsection(doc, 'Phase 1', 'AI Protocol Discovery (2–4 Hours)')
    add_body(doc,
        'QBITEL\'s AI scans your network and identifies every voice, signaling, and data protocol in use — '
        'including undocumented and legacy protocols your own IT team may not know about. 89%+ accuracy on '
        'first pass. No configuration required.')

    add_professional_table(doc,
        ['Discovery Phase', 'Duration', 'What Happens'],
        [
            ['Statistical Analysis', '5–10 seconds', 'Entropy calculation, byte frequency, binary vs. text classification'],
            ['ML Classification', '10–20 seconds', 'CNN feature extraction, BiLSTM sequence learning, 89%+ protocol family accuracy'],
            ['Grammar Learning', '1–2 minutes', 'PCFG inference, semantic learning via Transformers'],
            ['Parser Generation', '30–60 seconds', 'Auto-generate parsers at 50,000+ msg/sec throughput'],
            ['Adaptive Learning', 'Continuous', 'Error analysis, grammar refinement, field detection improvement'],
        ],
        col_widths=[2.0, 1.4, 3.6]
    )

    add_subsection(doc, 'Phase 2', 'Post-Quantum Encryption Activation (1 Hour)')
    add_body(doc,
        'Once discovered, every protocol is wrapped in NIST-standardized post-quantum cryptography — '
        'automatically, with no manual configuration. Encryption profiles are pre-optimized per subdomain '
        'to meet latency requirements.')

    add_professional_table(doc,
        ['Subdomain', 'Algorithm', 'Latency', 'Use Case'],
        [
            ['Voice Signaling', 'ML-KEM-512 + Falcon-512', '<50ms', 'SIP/SDP signaling encryption'],
            ['Voice Media (RTP)', 'ML-KEM-512 + Falcon-512', '<20ms', 'Real-time media stream encryption'],
            ['Payment Processing', 'ML-KEM-1024 + ML-DSA-87', '<100ms', 'PCI-DSS Level 1 call encryption'],
            ['Call Recording', 'ML-KEM-1024 + ML-DSA-87', '<1000ms', 'Quantum-safe long-term storage'],
            ['Remote Agent Tunnel', 'ML-KEM-768 + ML-DSA-65', '<200ms', 'VPN-less WFH agent access'],
            ['Terminal Emulation', 'ML-KEM-768 + ML-DSA-65', '<300ms', 'TN3270e/TN5250 session protection'],
            ['CTI Middleware', 'ML-KEM-512 + Falcon-512', '<100ms', 'CTI event processing'],
        ],
        col_widths=[2.0, 2.0, 1.0, 2.0]
    )

    add_subsection(doc, 'Phase 3', 'Agentic AI Security (Continuous)')
    add_body(doc,
        'QBITEL\'s agentic AI monitors all protected traffic continuously and responds to threats autonomously. '
        '78% of security events are handled without human intervention, with full LLM-generated explanations '
        'for every action taken.')

    # ─── SECTION 3: CAPABILITY DEEP DIVES ────────────────────────────────────
    doc.add_page_break()
    add_section_header(doc, '3. Capability Deep Dives',
                       'Six purpose-built modules for contact center security')

    add_subsection(doc, '3.1', 'PCI-DSS Voice Compliance — Up to 80% Scope Reduction')
    add_body(doc,
        'Contact centers handling card payments face enormous PCI-DSS audit scope. Every system that '
        'touches cardholder data — recordings, agent desktops, CRM integrations — falls within scope. '
        'QBITEL eliminates cardholder data from your environment before it can be stored or accessed.')

    add_professional_table(doc,
        ['PCI Control', 'What QBITEL Does', 'Compliance Impact'],
        [
            ['DTMF Masking', 'Card digits clamped/suppressed in headset AND recording in real-time (<5ms)', 'CHD never reaches agent or recording system'],
            ['Auto Pause/Resume', 'Recording pauses on payment detection, resumes automatically after', 'Recording system exits PCI scope during card capture'],
            ['PAN Detection', 'Real-time Luhn-validated card number detection across all data streams', 'Accidental PAN transmission caught immediately'],
            ['Screen Masking', 'Cardholder data shown as last 4 digits only on agent desktop', 'Agent screen removed from PCI scope'],
            ['Recording Encryption', 'ML-KEM-1024 for long-term recording archive', 'Recordings safe against quantum decryption'],
            ['Scope Tracking', 'Automatic PCI scope calculation per call, agent, and tenant', 'Continuous audit evidence — not just at audit time'],
        ],
        col_widths=[1.8, 2.6, 2.6]
    )

    add_subsection(doc, '3.2', 'Toll Fraud Prevention — 10 Pattern Types, <1 Second Detection')

    add_professional_table(doc,
        ['Fraud Type', 'Attack Pattern', 'QBITEL Response'],
        [
            ['IRSF', 'Calls to premium-rate numbers in 200+ country database', 'Block in <1 second; forensics preserved'],
            ['PBX Hacking', 'Unauthorized trunk access; midnight call spikes', 'Trunk quarantined; NOC alerted'],
            ['Wangiri', 'Missed call callback manipulation to premium numbers', 'Pattern detected after 3 calls; blocked'],
            ['Call Transfer Fraud', 'Transfer to premium-rate destinations', 'Destinations validated against whitelist'],
            ['Call Pumping', 'Artificially extended calls to inflate revenue share', 'Duration anomaly detection; call terminated'],
            ['Bypass Fraud', 'SIM box illegal termination; CLI manipulation', 'CLI anomaly detected; call blocked'],
            ['Wangiri / CLIP Manipulation', 'Caller ID spoofing to mask fraud origin', 'CLI format anomaly flagged immediately'],
            ['Toll-Free Abuse', 'Repeated short calls to toll-free numbers', 'Volume pattern detection; source rate-limited'],
        ],
        col_widths=[1.9, 2.8, 2.3]
    )

    add_subsection(doc, '3.3', 'Agent Desktop DLP — Six Exfiltration Channels Blocked')

    add_professional_table(doc,
        ['Threat Vector', 'Exfiltration Method', 'QBITEL Defense'],
        [
            ['Clipboard', 'Copy-paste SSNs and card numbers to personal files', 'Block clipboard for PII patterns — real-time kernel level'],
            ['Screen Capture', 'Screenshot customer data and order details', 'Block PrintScreen, Snipping Tool, third-party tools'],
            ['USB Exfiltration', 'Copy customer data to USB drive on break', 'USB storage blocked; all USB events logged'],
            ['Email / Chat', 'Email PII to personal accounts during shift', 'Monitor outbound channels for PII pattern matches'],
            ['Voice Reading', 'Read card numbers aloud during payment calls', 'Speech analytics detect agents reading CHD aloud'],
            ['Screen Scraping', 'Automated tools scraping agent desktop data', 'eBPF-based scraping pattern detection'],
        ],
        col_widths=[1.8, 2.4, 2.8]
    )

    add_body(doc, 'Forensic Watermarking: Every agent screen carries an invisible watermark with agent ID '
             'and session timestamp. Any screenshot or recording can be attributed to the exact agent, '
             'session, and moment it was taken — enabling post-incident attribution.')

    add_subsection(doc, '3.4', 'Remote Agent Security — VPN-Less. Quantum-Safe. Zero Trust.')

    add_professional_table(doc,
        ['Security Control', 'Specification'],
        [
            ['Tunnel Encryption', 'ML-KEM-768 + AES-256-GCM hybrid post-quantum encryption for all agent connections'],
            ['Endpoint Verification', 'OS version, patch level, antivirus, full-disk encryption, home WiFi WPA3 compliance'],
            ['Continuous Monitoring', 'eBPF-based runtime monitoring of agent endpoint behavior during entire session'],
            ['Geo-fencing', 'Location verification, country/city-level access restrictions, anomaly alerting'],
            ['Session Watermarking', 'Forensic agent ID and timestamp on all agent screens — visible or invisible'],
            ['Split Tunnel Prevention', 'All traffic forced through quantum-safe tunnel — no bypassing'],
        ],
        col_widths=[2.2, 4.8]
    )

    add_subsection(doc, '3.5', 'Autonomous Threat Response — 78% No-Touch Resolution')

    add_professional_table(doc,
        ['Threat Event', 'Autonomous Response', 'Time'],
        [
            ['SIP injection attack', 'Block source, alert NOC, preserve forensic evidence', '<1 second'],
            ['Terminal session hijacking', 'Terminate session, force agent re-authentication', '<2 seconds'],
            ['Bulk customer data access', 'Rate-limit access, flag for supervisor notification', '<5 seconds'],
            ['Recording tampering', 'Cryptographic integrity alert, evidence locked and preserved', '<1 second'],
            ['Rogue remote agent endpoint', 'Quarantine endpoint, suspend all sessions', '<10 seconds'],
            ['Toll fraud pattern', 'Block compromised trunk, alert NOC, preserve forensics', '<1 second'],
        ],
        col_widths=[2.4, 3.0, 1.6]
    )

    add_body(doc,
        'Every automated action generates a plain-language LLM-powered narrative — not an alert code. '
        'Your security team understands exactly what happened, why, and what QBITEL did about it. '
        'All LLM reasoning runs on-premise (Ollama / Llama 3 / Mixtral) — no customer data ever '
        'leaves your network.')

    # ─── SECTION 4: COMPLIANCE ────────────────────────────────────────────────
    doc.add_page_break()
    add_section_header(doc, '4. Compliance Coverage',
                       'Nine frameworks. Automated. Reports in under 10 minutes.')

    add_professional_table(doc,
        ['Framework', 'BPO Application', 'What QBITEL Automates'],
        [
            ['PCI-DSS 4.0', 'Voice payment processing', 'DTMF masking, recording encryption, agent desktop controls, scope reduction'],
            ['TCPA', 'Outbound calling compliance', 'Consent tracking, DNC list enforcement, time-of-day restrictions'],
            ['HIPAA', 'Healthcare BPO operations', 'PHI encryption, minimum necessary access controls, 6-year audit retention'],
            ['SOC 2 Type II', 'Service organization controls', 'Continuous monitoring, automated evidence collection, real-time alerting'],
            ['GDPR', 'EU customer data handling', 'Recording consent, DSAR processing, retention and deletion automation'],
            ['SOX', 'Financial services recording', 'Recording integrity, tamper-evident audit trails, 7-year retention'],
            ['GLBA', 'Financial data handling', 'Customer data classification, access controls, breach notification'],
            ['FCA / MiFID II', 'UK/EU financial call recording', 'All-call recording, quantum-safe encryption, regulatory retention'],
            ['NIST PQC', 'Quantum-safe transition', 'ML-KEM + ML-DSA across all voice and data channels — NIST Level 5'],
        ],
        col_widths=[1.6, 2.0, 3.4]
    )

    add_callout_box(doc,
        'Compliance reports generated on-demand in under 10 minutes. '
        'Blockchain-backed audit trails for tamper evidence. '
        'Multi-tenant isolation ensures each client receives independent compliance reporting.',
        icon='◈')

    # ─── SECTION 5: REAL-WORLD SCENARIOS ─────────────────────────────────────
    add_section_header(doc, '5. Real-World Scenarios',
                       'Proven results across financial services, healthcare, and multi-tenant BPO environments')

    add_scenario_box(doc,
        'SCENARIO A', 'Financial Services BPO — 5,000 Seats',
        'Challenge: A Fortune 500 bank\'s BPO partner handles credit card disputes. Every call carries '
        'cardholder data through legacy Avaya infrastructure. PCI-DSS audit scope covers the entire '
        'contact center. Annual audit cost: $2.3M.\n\n'
        'QBITEL Result: Network tap in 30 minutes. AI discovers SIP + TN3270e in 2 hours. PQC-SRTP '
        'and DTMF masking active before close of business. Agents continue uninterrupted. PCI-DSS '
        'audit scope reduced by 80%. Annual audit cost reduced by $1.7M.')

    add_scenario_box(doc,
        'SCENARIO B', 'Healthcare BPO — 2,000 Remote Agents',
        'Challenge: Post-COVID workforce handles patient scheduling and insurance verification from home '
        'networks. HIPAA requires audit trails, PHI encryption, and endpoint compliance — but the BPO '
        'has zero visibility into agent home environments.\n\n'
        'QBITEL Result: VPN-less quantum-safe tunnels deployed to all 2,000 agents. Endpoint compliance '
        'verified continuously — WPA3, disk encryption, antivirus. PHI exfiltration monitoring active '
        'across clipboard, USB, email, and screen channels. First HIPAA audit: zero findings.')

    add_scenario_box(doc,
        'SCENARIO C', 'Toll Fraud — The $50,000 Weekend Attack',
        'Challenge: A 500-seat contact center discovers $50,000 in fraudulent calls to Caribbean '
        'premium-rate numbers — detected Monday morning on the carrier invoice. The attack ran all '
        'weekend through a single compromised SIP trunk.\n\n'
        'With QBITEL: IRSF pattern detected after the 3rd fraudulent call. Trunk isolated automatically. '
        'NOC alerted. Forensic evidence preserved. Total loss: $200 vs. $50,000 without QBITEL.')

    add_scenario_box(doc,
        'SCENARIO D', 'Multi-Tenant BPO — Banking, Healthcare, Retail on One Floor',
        'Challenge: One BPO facility, three enterprise clients with different compliance requirements. '
        'Three separate annual audits. Separate IT infrastructure per client is cost-prohibitive.\n\n'
        'QBITEL Result: Per-tenant cryptographic key isolation. Per-tenant compliance policies: PCI-DSS '
        'for banking, HIPAA for healthcare, SOC 2 for retail. Three independent compliance reports — '
        'each generated on-demand in under 10 minutes. One infrastructure investment.')

    # ─── SECTION 6: INTEGRATION ───────────────────────────────────────────────
    doc.add_page_break()
    add_section_header(doc, '6. Integration — Works With Everything You Already Have',
                       'QBITEL deploys as a security overlay. Your existing infrastructure stays exactly as it is.')

    add_subsection(doc, '6.1', 'PBX & Telephony Platforms')
    add_professional_table(doc,
        ['Platform', 'Integration Method', 'Infrastructure Change'],
        [
            ['Avaya Aura / Avaya CM', 'TSAPI/DMCC with PQC tunnel', 'None — PQC added transparently'],
            ['Cisco CUCM', 'CTI-OS / Finesse API with PQC tunnel', 'None — security at network layer'],
            ['Genesys Cloud', 'REST API with PQC-TLS', 'None — existing API calls secured'],
            ['Asterisk / FreePBX', 'AMI/ARI with PQC tunnel', 'None — overlay protection'],
            ['Legacy PBX (any vendor)', 'Protocol-level encryption overlay', 'None — fully protocol agnostic'],
        ],
        col_widths=[2.2, 2.4, 2.4]
    )

    add_subsection(doc, '6.2', 'CRM, WFM & Mainframe Systems')
    add_professional_table(doc,
        ['System', 'Platform', 'Security Layer Added'],
        [
            ['CRM', 'Salesforce, Zendesk, ServiceNow, Dynamics 365', 'PII masking, data classification, audit trail'],
            ['WFM', 'NICE, Verint, Aspect, Calabrio', 'Schedule enforcement, recording security bridge'],
            ['Mainframe', 'IBM TN3270e / TN5250', 'PQC encryption, session monitoring, audit trail'],
        ],
        col_widths=[1.2, 3.2, 2.6]
    )

    add_subsection(doc, '6.3', 'Deployment Timeline')
    add_professional_table(doc,
        ['Step', 'Duration', 'Activity'],
        [
            ['1. Network Tap', '30 minutes', 'Non-invasive tap on voice/data network — passive observation only'],
            ['2. Protocol Discovery', '2–4 hours', 'AI identifies all protocols, traffic patterns, and data flows'],
            ['3. Encryption Activation', '1 hour', 'PQC encryption activated for all discovered protocols'],
            ['4. Policy Deployment', '30 minutes', 'BPO-specific security policies configured and validated'],
            ['Total', '4–6 hours', 'Full quantum-safe protection active. Zero downtime. No PBX replacement.'],
        ],
        col_widths=[1.8, 1.2, 4.0]
    )

    # ─── SECTION 7: PERFORMANCE ───────────────────────────────────────────────
    add_section_header(doc, '7. Performance Specifications',
                       'Enterprise scale. Zero compromise on voice quality.')

    add_professional_table(doc,
        ['Metric', 'QBITEL Performance', 'Business Impact'],
        [
            ['Voice PQC overhead', '<2ms', 'Inaudible — within ITU-T G.114 150ms quality budget'],
            ['DTMF masking latency', '<5ms', 'Callers cannot detect any delay in payment flow'],
            ['Toll fraud detection', '<1 second (3-call pattern)', 'Fraud stopped before meaningful loss occurs'],
            ['Concurrent agent sessions', '20,000+ per deployment', 'Enterprise scale from day one'],
            ['Recording encryption throughput', '10,000+ concurrent streams', 'Entire recording estate protected simultaneously'],
            ['Autonomous threat resolution', '78% no-touch', 'SOC team handles exceptions, not routine alerts'],
            ['PAN detection latency', '<50ms', 'Real-time Luhn validation across all data streams'],
            ['Compliance report generation', '<10 minutes', 'On-demand reports for any framework, any client'],
            ['Agent session validation', '<100ms', 'Per-request posture check — imperceptible to agents'],
            ['Full deployment time', '4–6 hours', 'Full protection in a single business day'],
        ],
        col_widths=[2.4, 1.8, 2.8]
    )

    # ─── SECTION 8: COMPETITIVE DIFFERENTIATION ────────────────────────────────
    doc.add_page_break()
    add_section_header(doc, '8. Why QBITEL Bridge',
                       'The only platform purpose-built for BPO contact center security')

    add_subsection(doc, '8.1', 'QBITEL Bridge vs. Traditional Security Vendors')
    add_professional_table(doc,
        ['Capability', 'Traditional Vendors', 'QBITEL Bridge'],
        [
            ['Legacy protocol support', 'Known protocols only', 'AI discovers unknown/undocumented protocols'],
            ['Quantum cryptography', 'Not available (2026)', 'NIST Level 5 — ML-KEM-1024, ML-DSA-87'],
            ['Deployment time', 'Weeks to months', '4–6 hours, zero downtime'],
            ['Infrastructure changes', 'PBX replacement required', 'Network overlay — nothing replaced'],
            ['BPO-specific controls', 'Generic security policies', 'DTMF masking, toll fraud, agent DLP, multi-tenant'],
            ['Threat response', 'Alert-based, SOC review required', '78% autonomous — LLM reasoning, not playbooks'],
            ['Compliance automation', 'Manual evidence collection', '9 frameworks automated, reports in <10 minutes'],
            ['Air-gapped AI', 'Cloud-dependent LLM', 'Full on-premise LLM — Ollama/Llama 3, no data egress'],
        ],
        col_widths=[2.4, 2.0, 2.6]
    )

    add_subsection(doc, '8.2', 'QBITEL Bridge vs. PBX Replacement')
    add_professional_table(doc,
        ['Factor', 'PBX Replacement', 'QBITEL Bridge'],
        [
            ['Cost', '$5M+ hardware + migration + training', 'Fraction of replacement cost'],
            ['Downtime risk', 'Weeks of migration with rollback risk', 'Zero downtime — 4–6 hour deployment'],
            ['Timeline', '12–18 months to production', 'Full protection same day'],
            ['Quantum-readiness', 'Depends on new vendor roadmap', 'NIST Level 5 from day one'],
            ['Agent impact', 'Full retraining required', 'Zero — agents notice nothing'],
            ['Legacy systems', 'Remaining legacy still unprotected', 'All protocols protected including undocumented'],
        ],
        col_widths=[1.8, 2.6, 2.6]
    )

    # ─── SECTION 9: NEXT STEPS ─────────────────────────────────────────────────
    add_section_header(doc, '9. Next Steps',
                       'Full quantum-safe protection in under one week')

    steps = [
        ('Step 1', 'Discovery Assessment (Free — 2 Hours)',
         'We deploy a passive network tap in your environment and deliver a report showing exactly which '
         'protocols are running, what is unencrypted, and where your PCI/HIPAA scope currently sits. '
         'No commitment. No infrastructure changes. Pure intelligence.'),
        ('Step 2', 'Proof of Concept (2 Weeks)',
         'Full QBITEL Bridge deployment against your production traffic. Real threats. Live compliance '
         'reporting. Toll fraud monitoring. All measured against your current state baseline. '
         'You see the ROI before you sign a contract.'),
        ('Step 3', 'Production Deployment (4–6 Hours)',
         'Zero-downtime deployment. All protocols protected. All agents covered. '
         'All 9 compliance frameworks active. Your team receives full documentation, '
         'runbooks, and dedicated support for the first 90 days.'),
    ]

    for label, title, body in steps:
        add_scenario_box(doc, label, title, body)

    # Contact block
    contact_tbl = doc.add_table(rows=1, cols=3)
    contact_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblEl = contact_tbl._tbl
    tblPr = tblEl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tblEl.insert(0, tblPr)
    tblBorders = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'nil')
        tblBorders.append(node)
    tblPr.append(tblBorders)

    contacts = [
        ('✉', 'enterprise@qbitel.com', 'Email'),
        ('⊕', 'bridge.qbitel.com', 'Website'),
        ('◉', 'Contact your account team', 'Schedule a call'),
    ]
    for i, (icon, val, label) in enumerate(contacts):
        cell = contact_tbl.cell(0, i)
        set_cell_bg(cell, NAVY)
        set_cell_margins(cell, 160, 160, 180, 180)
        p = cell.paragraphs[0]
        r = p.add_run(f'{icon}  {val}')
        r.font.size = Pt(10)
        r.font.color.rgb = WHITE
        r.bold = True
        r.font.name = 'Calibri'
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p, 0, 40)
        p2 = cell.add_paragraph()
        r2 = p2.add_run(label)
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = TEAL
        r2.font.name = 'Calibri'
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p2, 0, 0)

    doc.add_paragraph()

    # Closing tagline
    p_close = doc.add_paragraph()
    r_close = p_close.add_run('QBITEL Bridge — Because the Human API Deserves Quantum-Safe Protection.')
    r_close.font.size = Pt(12)
    r_close.font.color.rgb = NAVY
    r_close.bold = True
    r_close.italic = True
    r_close.font.name = 'Calibri'
    p_close.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p_close, 200, 60)

    p_ver = doc.add_paragraph()
    r_ver = p_ver.add_run(
        'Document Version 1.0  |  February 2026  |  Confidential — For Authorized Recipients Only'
    )
    r_ver.font.size = Pt(8)
    r_ver.font.color.rgb = MID_GREY
    r_ver.font.name = 'Calibri'
    p_ver.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_footer(doc)

    out_path = 'docs/brochures/QBITEL_Bridge_BPO_Marketing_Pitch.docx'
    doc.save(out_path)
    print(f'DOCX saved: {out_path}')
    return out_path


if __name__ == '__main__':
    build_document()
