"""
Build QBITEL Bridge x TCN Partnership Brief — Google-Slides-safe variant.

Same deck as build_tcn_partnership_pptx.py but uses ONLY shapes that Google
Slides' server-side OOXML→Slides converter accepts without throwing a 500:
    Rectangles, Rounded rectangles, Ovals, Block arrows, Line connectors.
No rotated pentagons, no diamonds, no rotated right-triangles, no raw OOXML
shadow effects, no rotation transforms.

Upload to drive.google.com → right-click → Open with Google Slides.
"""
from __future__ import annotations

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# ─── Brand palette ───────────────────────────────────────────────────────────
NAVY = RGBColor(0x0D, 0x1B, 0x3E)
NAVY_LIGHT = RGBColor(0x1A, 0x2D, 0x5A)
TEAL = RGBColor(0x00, 0x8B, 0x9A)
TEAL_DARK = RGBColor(0x00, 0x6B, 0x7A)
TEAL_LIGHT = RGBColor(0xCD, 0xE8, 0xEC)
GOLD = RGBColor(0xF0, 0xA5, 0x00)
GOLD_LIGHT = RGBColor(0xFD, 0xE9, 0xBE)
LIGHT_BG = RGBColor(0xF4, 0xF7, 0xFA)
MID_GREY = RGBColor(0x5A, 0x6A, 0x7A)
LIGHT_GREY = RGBColor(0xE6, 0xEA, 0xEF)
DARK_TEXT = RGBColor(0x1A, 0x1A, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)

# ─── Slide geometry (16:9) ───────────────────────────────────────────────────
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN_L = Inches(0.5)
CONTENT_W = Inches(12.333)
TITLE_H = Inches(0.7)
FOOTER_H = Inches(0.35)
BODY_TOP = Inches(0.95)
BODY_H = Inches(6.2)

FONT = 'Calibri'


# ─── Low-level helpers ───────────────────────────────────────────────────────

def _apply_shadow(shape, blur=6, distance=2, alpha=40000):
    """No-op: Google Slides' converter chokes on outerShdw effectLst blocks."""
    return


def set_fill(shape, rgb):
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb


def set_line(shape, rgb, width_pt=0.75):
    shape.line.color.rgb = rgb
    shape.line.width = Pt(width_pt)


def no_line(shape):
    shape.line.fill.background()


def set_text(shape, text, *, size=12, bold=False, color=DARK_TEXT,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=FONT,
             italic=False, line_spacing=1.15):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(4)
    tf.margin_right = Pt(4)
    tf.margin_top = Pt(2)
    tf.margin_bottom = Pt(2)
    tf.vertical_anchor = anchor

    lines = text.split('\n')
    tf.text = lines[0]
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    for run in p.runs:
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
    for extra in lines[1:]:
        p = tf.add_paragraph()
        p.text = extra
        p.alignment = align
        p.line_spacing = line_spacing
        for run in p.runs:
            run.font.name = font
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.color.rgb = color


def rounded_box(slide, x, y, w, h, *, fill=WHITE, line=TEAL, line_w=0.75,
                text='', font_size=11, bold=False, font_color=DARK_TEXT,
                shadow=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.adjustments[0] = 0.12
    set_fill(shp, fill)
    if line is None:
        no_line(shp)
    else:
        set_line(shp, line, line_w)
    if text:
        set_text(shp, text, size=font_size, bold=bold, color=font_color,
                 align=align, anchor=anchor)
    if shadow:
        _apply_shadow(shp)
    return shp


def plain_box(slide, x, y, w, h, *, fill=WHITE, line=None, text='',
              font_size=11, bold=False, font_color=DARK_TEXT,
              align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    set_fill(shp, fill)
    if line is None:
        no_line(shp)
    else:
        set_line(shp, line, 0.75)
    if text:
        set_text(shp, text, size=font_size, bold=bold, color=font_color,
                 align=align, anchor=anchor)
    return shp


def text_box(slide, x, y, w, h, text, *, size=11, bold=False, color=DARK_TEXT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False,
             line_spacing=1.2):
    tb = slide.shapes.add_textbox(x, y, w, h)
    set_text(tb, text, size=size, bold=bold, color=color, align=align,
             anchor=anchor, italic=italic, line_spacing=line_spacing)
    return tb


def add_bullets(shape, items, *, size=11, color=DARK_TEXT, bold_lead=False,
                line_spacing=1.25):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Pt(6)
    tf.margin_right = Pt(6)
    tf.margin_top = Pt(4)
    tf.margin_bottom = Pt(4)
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run()
            r1.text = '▸ ' + lead
            r1.font.name = FONT
            r1.font.size = Pt(size)
            r1.font.bold = True
            r1.font.color.rgb = color
            if rest:
                r2 = p.add_run()
                r2.text = '  ' + rest
                r2.font.name = FONT
                r2.font.size = Pt(size)
                r2.font.color.rgb = color
        else:
            r = p.add_run()
            r.text = '▸  ' + item
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold_lead
            r.font.color.rgb = color


def arrow(slide, x1, y1, x2, y2, *, color=TEAL_DARK, weight=1.75):
    """Straight connector with an arrowhead."""
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    line = conn.line
    line.color.rgb = color
    line.width = Pt(weight)
    # Add arrow-head on the end via OOXML
    ln = line._get_or_add_ln()
    for tail in ln.findall(qn('a:tailEnd')):
        ln.remove(tail)
    etree.SubElement(ln, qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'})
    return conn


def block_arrow(slide, x, y, w, h, *, direction='right', fill=TEAL, line=None,
                text='', font_size=10, font_color=WHITE, bold=True):
    shape_type = {
        'right': MSO_SHAPE.RIGHT_ARROW,
        'left': MSO_SHAPE.LEFT_ARROW,
        'down': MSO_SHAPE.DOWN_ARROW,
        'up': MSO_SHAPE.UP_ARROW,
    }[direction]
    shp = slide.shapes.add_shape(shape_type, x, y, w, h)
    set_fill(shp, fill)
    if line is None:
        no_line(shp)
    else:
        set_line(shp, line, 0.5)
    if text:
        set_text(shp, text, size=font_size, bold=bold, color=font_color)
    return shp


def step_circle(slide, x, y, diameter, number, *, fill=GOLD, color=NAVY):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, diameter, diameter)
    set_fill(shp, fill)
    no_line(shp)
    set_text(shp, str(number), size=14, bold=True, color=color,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return shp


def label_chip(slide, x, y, w, h, text, *, fill=NAVY, color=WHITE,
               font_size=9):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.adjustments[0] = 0.5
    set_fill(shp, fill)
    no_line(shp)
    set_text(shp, text, size=font_size, bold=True, color=color,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return shp


# ─── Page furniture ──────────────────────────────────────────────────────────

def add_title_bar(slide, title, subtitle=None):
    # Top navy band
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, TITLE_H)
    set_fill(bar, NAVY)
    no_line(bar)
    # Gold accent on the left
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.08), TITLE_H)
    set_fill(accent, GOLD)
    no_line(accent)
    # Teal accent right of gold
    accent2 = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.08), 0, Inches(0.04), TITLE_H)
    set_fill(accent2, TEAL)
    no_line(accent2)
    # Title
    title_tb = slide.shapes.add_textbox(
        Inches(0.35), Inches(0.05), Inches(11), Inches(0.4))
    set_text(title_tb, title.upper(), size=18, bold=True, color=WHITE,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        sub_tb = slide.shapes.add_textbox(
            Inches(0.35), Inches(0.42), Inches(11), Inches(0.25))
        set_text(sub_tb, subtitle, size=10.5, bold=False, color=TEAL_LIGHT,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=True)
    # Right-side QBITEL × TCN tag
    rt = slide.shapes.add_textbox(
        Inches(10), Inches(0.18), Inches(3.1), Inches(0.3))
    set_text(rt, 'QBITEL  ×  TCN', size=10, bold=True, color=GOLD,
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def add_footer(slide, page_num):
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, SLIDE_H - FOOTER_H, SLIDE_W, FOOTER_H)
    set_fill(bar, NAVY)
    no_line(bar)
    # Teal stripe
    stripe = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, SLIDE_H - FOOTER_H - Inches(0.02),
        SLIDE_W, Inches(0.02))
    set_fill(stripe, TEAL)
    no_line(stripe)
    # Left text
    left = slide.shapes.add_textbox(
        Inches(0.35), SLIDE_H - FOOTER_H, Inches(8), FOOTER_H)
    set_text(left,
             'Confidential — For Authorized Recipients Only  •  © 2026 QBITEL',
             size=8, color=WHITE, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.MIDDLE)
    # Right text
    right = slide.shapes.add_textbox(
        Inches(10.5), SLIDE_H - FOOTER_H, Inches(2.4), FOOTER_H)
    set_text(right, f'{page_num}', size=9, color=WHITE,
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, bold=True)


# ─── Slide builders ──────────────────────────────────────────────────────────

def slide_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    set_fill(bg, NAVY)
    no_line(bg)

    # Top-right gold accent band (Google-Slides-safe: no rotated triangle)
    gold_band = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(8.5), 0, Inches(4.833), Inches(0.45))
    set_fill(gold_band, GOLD)
    no_line(gold_band)

    # Teal accent rectangle below the gold band
    teal_acc = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(10.0), Inches(0.45),
        Inches(3.333), Inches(0.35))
    set_fill(teal_acc, TEAL)
    no_line(teal_acc)

    # Bottom teal strip
    bot = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, SLIDE_H - Inches(1.4), SLIDE_W, Inches(1.4))
    set_fill(bot, TEAL_DARK)
    no_line(bot)
    bot_gold = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, SLIDE_H - Inches(1.4),
        SLIDE_W, Inches(0.06))
    set_fill(bot_gold, GOLD)
    no_line(bot_gold)

    # Title
    title_tb = slide.shapes.add_textbox(
        Inches(0.6), Inches(2.0), Inches(10), Inches(1.5))
    tf = title_tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r1 = p.add_run()
    r1.text = 'QBITEL  '
    r1.font.name = FONT
    r1.font.size = Pt(64)
    r1.font.bold = True
    r1.font.color.rgb = WHITE
    r2 = p.add_run()
    r2.text = '×  TCN'
    r2.font.name = FONT
    r2.font.size = Pt(64)
    r2.font.bold = True
    r2.font.color.rgb = TEAL

    # Gold underline
    ul = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(3.3),
        Inches(4.0), Inches(0.08))
    set_fill(ul, GOLD)
    no_line(ul)

    # Subtitle
    sub = slide.shapes.add_textbox(
        Inches(0.6), Inches(3.45), Inches(11), Inches(0.6))
    set_text(sub, 'Partnership Brief for the VP of Technology',
             size=24, bold=True, color=WHITE,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

    # Tagline
    tag = slide.shapes.add_textbox(
        Inches(0.6), Inches(4.05), Inches(11), Inches(0.5))
    set_text(tag,
             'Embedded Security & Compliance Layer for TCN Operator',
             size=15, italic=True, color=TEAL_LIGHT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

    # Metric boxes row
    metrics = [
        ('5 SEAMS', 'SIP • Webhooks • REST\nCRM Hooks • SSO'),
        ('4 WEEKS', 'Joint POC to a\nWorking Demo'),
        ('3 MODELS', 'Embedded OEM\nMarketplace • Referral'),
        ('<10 MIN', 'Per-Tenant\nCompliance Pack'),
    ]
    bx_w = Inches(2.85)
    bx_h = Inches(1.05)
    bx_gap = Inches(0.15)
    total_w = bx_w * 4 + bx_gap * 3
    bx_start = (SLIDE_W - total_w) / 2
    by = Inches(4.85)
    for i, (big, small) in enumerate(metrics):
        bx = bx_start + i * (bx_w + bx_gap)
        bg_c = TEAL if i % 2 == 1 else NAVY_LIGHT
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, bx_w, bx_h)
        box.adjustments[0] = 0.1
        set_fill(box, bg_c)
        no_line(box)
        tb = slide.shapes.add_textbox(bx, by + Inches(0.08),
                                       bx_w, Inches(0.45))
        set_text(tb, big, size=22, bold=True,
                 color=GOLD if bg_c == TEAL else GOLD,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        tb2 = slide.shapes.add_textbox(bx, by + Inches(0.55),
                                        bx_w, Inches(0.45))
        set_text(tb2, small, size=10, color=WHITE,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP)

    # Bottom strip text
    bot_tb = slide.shapes.add_textbox(
        Inches(0.6), SLIDE_H - Inches(1.1), Inches(12), Inches(0.5))
    set_text(bot_tb,
             'Integration Architecture  •  Flow Diagrams  •  Commercial Models  •  POC Plan',
             size=12, bold=True, color=WHITE,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
    ver_tb = slide.shapes.add_textbox(
        Inches(0.6), SLIDE_H - Inches(0.6), Inches(12), Inches(0.35))
    set_text(ver_tb,
             'Version 1.0  •  February 2026  •  Confidential',
             size=10, color=RGBColor(0xAA, 0xBB, 0xCC),
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=True)


def slide_agenda(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Agenda',
                  'What we will cover in this conversation')

    items = [
        ('1', 'Meeting objective and what we are asking from TCN'),
        ('2', 'What we understand about TCN Operator (please correct)'),
        ('3', 'QBITEL Bridge in one slide'),
        ('4', 'Why a TCN × QBITEL partnership — overlap and complementarity'),
        ('5', 'Integration architecture and the five seams'),
        ('6', 'Three flow diagrams: inbound, outbound, payment'),
        ('7', 'Multi-tenant isolation for TCN\'s BPO customers'),
        ('8', 'Commercial models — three options'),
        ('9', 'Technical risks and mitigations'),
        ('10', 'Joint POC proposal — four weeks'),
        ('11', 'Next steps'),
    ]
    y = Inches(1.1)
    line_h = Inches(0.46)
    for num, text in items:
        step_circle(slide, Inches(0.7), y, Inches(0.38), num,
                    fill=GOLD, color=NAVY)
        tb = slide.shapes.add_textbox(
            Inches(1.25), y + Inches(0.02), Inches(11), Inches(0.42))
        set_text(tb, text, size=14, color=DARK_TEXT,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
        y += line_h


def slide_objective(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Meeting Objective',
                  'Why we asked for this conversation')

    # Big callout box
    box = rounded_box(
        slide, Inches(0.7), Inches(1.2), Inches(11.9), Inches(2.0),
        fill=NAVY, line=None, shadow=True,
        text='Explore a technology partnership in which QBITEL Bridge becomes '
             'an embedded or marketplace-available security & compliance layer '
             'inside TCN Operator — giving TCN\'s BPO and enterprise customers '
             'a single answer to toll fraud, PCI/HIPAA scope, post-quantum '
             'readiness, and multi-tenant compliance, without any change to '
             'the agent experience or TCN\'s product surface.',
        font_size=14, font_color=WHITE, bold=False)

    # Two-up framing
    left = rounded_box(
        slide, Inches(0.7), Inches(3.5), Inches(5.8), Inches(2.6),
        fill=LIGHT_BG, line=TEAL, shadow=False, text='')
    set_text(left,
             'TCN handles the contact center.',
             size=15, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    left_body = slide.shapes.add_textbox(
        Inches(0.9), Inches(4.3), Inches(5.4), Inches(1.7))
    set_text(left_body,
             'Dialers • IVR/IVM • Agent workspace\nRecording • Native CRM/ITSM • Analytics',
             size=12, color=DARK_TEXT,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    right = rounded_box(
        slide, Inches(6.85), Inches(3.5), Inches(5.8), Inches(2.6),
        fill=TEAL, line=None, shadow=False, text='')
    set_text(right,
             'QBITEL handles what the regulators ask about it.',
             size=15, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    right_body = slide.shapes.add_textbox(
        Inches(7.0), Inches(4.3), Inches(5.5), Inches(1.7))
    set_text(right_body,
             'PQC encryption • DTMF masking • Toll-fraud detection\n'
             'Agent DLP • Multi-tenant compliance evidence',
             size=12, color=WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    tag = slide.shapes.add_textbox(
        Inches(0.7), Inches(6.3), Inches(11.9), Inches(0.5))
    set_text(tag,
             'Together you sell one platform. Separately, you both leave money on the table.',
             size=14, italic=True, color=GOLD,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, bold=True)


def slide_tcn_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'What We Understand About TCN Operator',
                  'From public materials — please correct anything that does not match')

    # Two-column layout
    rows = [
        ('Product', 'TCN Operator — unified cloud contact center platform'),
        ('Core capabilities',
         'Autodialer • Manual dialer • Preview dialer • IVR • IVM • '
         'Voicemail • Speech analytics • Call recording • Agent scripting • '
         'Dashboards • Live monitoring'),
        ('Channels',
         'Inbound • Outbound • Blended voice • Email • SMS'),
        ('Native integrations',
         'Salesforce • Zendesk • ServiceNow • Finvi • Zoho • Freshworks • '
         'Leadsquared  (close to 2,000 logos total)'),
        ('Integration framework',
         'Advanced REST API; Synapse engine for outbound webhooks and '
         'integration actions (2026 enhancement)'),
        ('Telephony',
         'SIP-based voice infrastructure with IVR/IVM logic'),
        ('Customer base',
         'BPOs • Collections • Healthcare • Financial services • CX operations'),
        ('2026 direction',
         'AI-powered features • Centralized automation • Omnichannel throughput'),
    ]
    y = Inches(1.05)
    row_h = Inches(0.6)
    for label, value in rows:
        # Label cell
        lbl = rounded_box(slide, Inches(0.7), y, Inches(2.6), row_h,
                          fill=NAVY, line=None, text=label,
                          font_size=11, bold=True, font_color=WHITE,
                          align=PP_ALIGN.LEFT)
        lbl.text_frame.margin_left = Pt(10)
        # Value cell
        val = rounded_box(slide, Inches(3.35), y, Inches(9.3), row_h,
                          fill=LIGHT_BG, line=LIGHT_GREY, text=value,
                          font_size=10.5, font_color=DARK_TEXT,
                          align=PP_ALIGN.LEFT)
        val.text_frame.margin_left = Pt(10)
        y += row_h + Inches(0.06)

    # Caption strip
    cap = slide.shapes.add_textbox(
        Inches(0.7), Inches(6.55), Inches(11.9), Inches(0.4))
    set_text(cap,
             'The gap your BPO customers raise loudest: the security & compliance layer '
             'across all of these components per-client. That is what QBITEL Bridge does.',
             size=11, italic=True, color=TEAL_DARK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)


def slide_qbitel_modules(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'QBITEL Bridge — At a Glance',
                  'AI-powered, network-overlay security & compliance for contact centers')

    # Five module cards across the top
    modules = [
        ('DISCOVER', 'AI protocol &\nasset graph',
         '2–4 hour scan\n89%+ accuracy'),
        ('UNDERSTAND', 'LLM-generated\nrisk narratives',
         'Per-tenant\nrisk explained'),
        ('MODERNIZE', 'Auto REST APIs\n+ 6 SDKs',
         'Translation Studio\nfor legacy/custom'),
        ('PROTECT', 'PQC overlay, DLP,\ntoll-fraud, DTMF',
         '78% autonomous\n<2ms voice'),
        ('PROVE', 'Per-tenant evidence\npacks',
         '9 frameworks\n<10 min per pack'),
    ]
    cw = Inches(2.4)
    ch = Inches(2.4)
    gap = Inches(0.13)
    total = cw * 5 + gap * 4
    start_x = (SLIDE_W - total) / 2
    y = Inches(1.05)
    for i, (head, sub, foot) in enumerate(modules):
        x = start_x + i * (cw + gap)
        # Card body
        card = rounded_box(slide, x, y, cw, ch, fill=LIGHT_BG, line=TEAL,
                           line_w=1.0, text='', shadow=True)
        # Header band
        band = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, x, y, cw, Inches(0.55))
        set_fill(band, NAVY)
        no_line(band)
        set_text(band, head, size=13, bold=True, color=GOLD,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # Gold accent
        acc = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, x, y + Inches(0.55),
            cw, Inches(0.04))
        set_fill(acc, GOLD)
        no_line(acc)
        # Sub text
        sub_tb = slide.shapes.add_textbox(
            x, y + Inches(0.7), cw, Inches(0.9))
        set_text(sub_tb, sub, size=12, bold=True, color=NAVY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # Foot text
        foot_tb = slide.shapes.add_textbox(
            x, y + Inches(1.65), cw, Inches(0.7))
        set_text(foot_tb, foot, size=10, color=MID_GREY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, italic=True)

    # Key specs strip
    specs_y = Inches(3.85)
    specs_h = Inches(0.85)
    label_chip(slide, Inches(0.7), specs_y, Inches(2.6), specs_h,
               '<2 ms\nPQC voice overhead', fill=NAVY, font_size=12)
    label_chip(slide, Inches(3.5), specs_y, Inches(2.6), specs_h,
               '<5 ms\nDTMF masking', fill=TEAL, font_size=12)
    label_chip(slide, Inches(6.3), specs_y, Inches(2.6), specs_h,
               '<1 sec\ntoll fraud block', fill=NAVY, font_size=12)
    label_chip(slide, Inches(9.1), specs_y, Inches(3.55), specs_h,
               '4–6 hours\nfull deployment, zero downtime',
               fill=TEAL, font_size=12)

    # Complement note
    note = rounded_box(
        slide, Inches(0.7), Inches(4.95), Inches(11.95), Inches(1.95),
        fill=GOLD_LIGHT, line=GOLD, line_w=1.0,
        text='QBITEL is a complement, not a competitor to a contact-center platform.\n\n'
             'We do not run the dialer, the IVR, the agent desktop, or the recording engine.\n'
             'We sit on the wire and at the API boundary, observe what those systems do, '
             'and add the controls and evidence that auditors and CISOs require.',
        font_size=13, bold=False, font_color=NAVY,
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide_partnership_thesis(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Why TCN × QBITEL',
                  'Customer overlap + capability complementarity = obvious joint motion')

    # Section header — overlap
    h1 = plain_box(slide, Inches(0.7), Inches(1.05), Inches(11.95), Inches(0.4),
                   fill=NAVY, text='1.  CUSTOMER OVERLAP',
                   font_size=12, bold=True, font_color=WHITE,
                   align=PP_ALIGN.LEFT)
    h1.text_frame.margin_left = Pt(10)

    overlap_text = (
        'Every TCN customer that operates as a BPO — and every TCN customer with '
        'PCI, HIPAA, SOX, or GDPR exposure — already has a budget line for the '
        'problems QBITEL solves. Today that budget either goes unspent (and the '
        'BPO absorbs the loss) or it goes to point tools that do not integrate '
        'with TCN. Embedded in TCN Operator, those budgets flow through TCN.'
    )
    text_box(slide, Inches(0.8), Inches(1.55), Inches(11.7), Inches(0.7),
             overlap_text, size=11, color=DARK_TEXT, line_spacing=1.3)

    # Section header — complementarity
    h2 = plain_box(slide, Inches(0.7), Inches(2.35), Inches(11.95), Inches(0.4),
                   fill=NAVY, text='2.  CAPABILITY COMPLEMENTARITY  (no overlap)',
                   font_size=12, bold=True, font_color=WHITE,
                   align=PP_ALIGN.LEFT)
    h2.text_frame.margin_left = Pt(10)

    # Two-column comparison
    layers = [
        ('Voice signaling (SIP)',
         'Trunking, routing, IVR logic',
         'PQC-TLS overlay, toll-fraud detection'),
        ('Voice media (RTP)',
         'Codec, mixing, recording',
         'PQC-SRTP encryption, DTMF masking'),
        ('Agent workspace',
         'Scripting, dashboards',
         'DLP across 6 vectors (clipboard, screen, USB, etc.)'),
        ('CRM/ITSM integration',
         'Salesforce/Zendesk/ServiceNow/Zoho native',
         'PII/PHI/PAN masking at API boundary'),
        ('Recording storage',
         'Recording engine + retention',
         'Quantum-safe encryption (ML-KEM-1024)'),
        ('Reporting',
         'Operational + analytics',
         'Compliance evidence packs (9 frameworks)'),
        ('Multi-tenancy',
         'Account/campaign isolation',
         'Cryptographic + policy + audit isolation per BPO client'),
    ]
    # Header row
    yh = Inches(2.85)
    rh = Inches(0.5)
    plain_box(slide, Inches(0.7), yh, Inches(3.5), rh,
              fill=NAVY_LIGHT, text='LAYER', font_size=11, bold=True,
              font_color=WHITE, align=PP_ALIGN.LEFT)
    plain_box(slide, Inches(4.25), yh, Inches(4.2), rh,
              fill=NAVY_LIGHT, text='TCN OWNS', font_size=11, bold=True,
              font_color=WHITE, align=PP_ALIGN.LEFT)
    plain_box(slide, Inches(8.5), yh, Inches(4.15), rh,
              fill=TEAL_DARK, text='QBITEL ADDS', font_size=11, bold=True,
              font_color=WHITE, align=PP_ALIGN.LEFT)
    for sh in slide.shapes:
        if sh.has_text_frame:
            sh.text_frame.margin_left = Pt(10)
    y = yh + rh
    for i, (layer, tcn, qb) in enumerate(layers):
        bg = LIGHT_BG if i % 2 == 0 else WHITE
        plain_box(slide, Inches(0.7), y, Inches(3.5), Inches(0.5),
                  fill=bg, text=layer, font_size=10.5, bold=True,
                  font_color=NAVY, align=PP_ALIGN.LEFT).text_frame.margin_left = Pt(10)
        plain_box(slide, Inches(4.25), y, Inches(4.2), Inches(0.5),
                  fill=bg, text=tcn, font_size=10, font_color=DARK_TEXT,
                  align=PP_ALIGN.LEFT).text_frame.margin_left = Pt(10)
        plain_box(slide, Inches(8.5), y, Inches(4.15), Inches(0.5),
                  fill=bg, text=qb, font_size=10, font_color=TEAL_DARK,
                  align=PP_ALIGN.LEFT, bold=True).text_frame.margin_left = Pt(10)
        y += Inches(0.5)


# ─── The architecture diagram (Slide 7) ──────────────────────────────────────

def slide_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Integration Architecture',
                  'Five integration seams — all use existing TCN surfaces')

    # ── TCN zone (upper) ─────────────────────────────────────────────────
    tcn_x = Inches(0.6)
    tcn_y = Inches(1.1)
    tcn_w = Inches(12.13)
    tcn_h = Inches(2.55)
    tcn_zone = rounded_box(slide, tcn_x, tcn_y, tcn_w, tcn_h,
                           fill=RGBColor(0xEB, 0xF1, 0xF7),
                           line=NAVY_LIGHT, line_w=1.5, text='')
    # Zone label
    zone_lbl = label_chip(slide, tcn_x + Inches(0.18), tcn_y + Inches(0.08),
                          Inches(2.2), Inches(0.3),
                          'TCN OPERATOR (CLOUD)',
                          fill=NAVY, font_size=10)

    # Top row of 4 boxes — TCN components
    comp_w = Inches(2.7)
    comp_h = Inches(0.85)
    comp_y = tcn_y + Inches(0.5)
    comps = [
        ('DIALERS', 'Auto • Preview • Manual'),
        ('IVR / IVM', 'Voice menus • Messaging'),
        ('AGENT WORKSPACE', 'Scripting • Dashboards'),
        ('RECORDING', 'Capture • Analytics'),
    ]
    comp_xs = [tcn_x + Inches(0.3) + i * (comp_w + Inches(0.13))
               for i in range(4)]
    for (head, sub), cx in zip(comps, comp_xs):
        box = rounded_box(slide, cx, comp_y, comp_w, comp_h,
                          fill=WHITE, line=NAVY_LIGHT, line_w=1.0,
                          text='', shadow=True)
        tb1 = slide.shapes.add_textbox(cx, comp_y + Inches(0.1),
                                        comp_w, Inches(0.35))
        set_text(tb1, head, size=11, bold=True, color=NAVY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        tb2 = slide.shapes.add_textbox(cx, comp_y + Inches(0.42),
                                        comp_w, Inches(0.4))
        set_text(tb2, sub, size=9.5, color=MID_GREY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # Second row: Synapse + REST API
    api_y = comp_y + comp_h + Inches(0.12)
    api_h = Inches(0.6)
    # Synapse
    syn_w = Inches(5.5)
    syn_x = tcn_x + Inches(0.3)
    rounded_box(slide, syn_x, api_y, syn_w, api_h,
                fill=GOLD, line=None,
                text='SYNAPSE WEBHOOK ENGINE   (2026 enhancement)',
                font_size=11, bold=True, font_color=NAVY)
    # REST API
    rest_x = syn_x + syn_w + Inches(0.13)
    rest_w = Inches(5.93)
    rounded_box(slide, rest_x, api_y, rest_w, api_h,
                fill=TEAL, line=None,
                text='REST API FRAMEWORK   +   Native CRM/ITSM Integrations',
                font_size=11, bold=True, font_color=WHITE)

    # ── Connectors zone (mid) ────────────────────────────────────────────
    conn_y = tcn_y + tcn_h + Inches(0.15)
    conn_h = Inches(0.85)

    # Five labeled connectors
    seam_labels = [
        ('1', 'SIP/RTP\nmirror', NAVY),
        ('2', 'Synapse\nwebhooks', GOLD),
        ('3', 'REST API\n(pull+push)', TEAL),
        ('4', 'CRM/ITSM\nhooks', NAVY),
        ('5', 'Admin SSO\n+ console', GOLD),
    ]
    seam_w = Inches(2.0)
    seam_gap = (CONTENT_W - seam_w * 5) / 4
    for i, (num, lbl, c) in enumerate(seam_labels):
        x = MARGIN_L + i * (seam_w + seam_gap)
        # Google-safe: rounded rectangle body + standalone DOWN_ARROW below.
        body_h = conn_h - Inches(0.25)
        body = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, conn_y, seam_w, body_h)
        body.adjustments[0] = 0.18
        set_fill(body, c)
        no_line(body)
        # Number + label inside the body
        text_color = WHITE if c != GOLD else NAVY
        lblbox = slide.shapes.add_textbox(
            x, conn_y, seam_w, body_h)
        tf = lblbox.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p1 = tf.paragraphs[0]
        p1.alignment = PP_ALIGN.CENTER
        r1 = p1.add_run()
        r1.text = num
        r1.font.name = FONT
        r1.font.size = Pt(15)
        r1.font.bold = True
        r1.font.color.rgb = text_color
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        for ln in lbl.split('\n'):
            run = p2.add_run()
            run.text = ln + '\n'
            run.font.name = FONT
            run.font.size = Pt(9)
            run.font.color.rgb = text_color
            run.font.bold = True
        # Small down arrow below the body to indicate flow direction
        arrow_w = Inches(0.4)
        arrow_h = Inches(0.2)
        arrow_x = x + (seam_w - arrow_w) / 2
        arrow_y = conn_y + body_h + Inches(0.02)
        block_arrow(slide, arrow_x, arrow_y, arrow_w, arrow_h,
                    direction='down', fill=c)

    # ── QBITEL zone (lower) ──────────────────────────────────────────────
    qb_y = conn_y + conn_h + Inches(0.18)
    qb_h = Inches(2.0)
    qb_zone = rounded_box(slide, tcn_x, qb_y, tcn_w, qb_h,
                          fill=NAVY, line=GOLD, line_w=1.5, text='')
    # Zone label
    label_chip(slide, tcn_x + Inches(0.18), qb_y + Inches(0.08),
               Inches(3.4), Inches(0.3),
               'QBITEL BRIDGE  (co-resident / cloud / on-prem)',
               fill=GOLD, color=NAVY, font_size=10)

    # 5 QBITEL components in row
    qb_comp_w = Inches(2.25)
    qb_comp_h = Inches(1.2)
    qb_comp_y = qb_y + Inches(0.5)
    qb_comp_gap = Inches(0.15)
    total_qb_w = qb_comp_w * 5 + qb_comp_gap * 4
    qb_start = tcn_x + (tcn_w - total_qb_w) / 2
    qb_comps = [
        ('AI Discovery', 'Protocol graph\n89%+ accuracy', TEAL),
        ('PQC Data Plane', 'Rust • Wire speed\n<2ms overhead', TEAL),
        ('Policy Engine', 'OPA • Per-tenant\nscoped policies', GOLD),
        ('Fraud / DLP', 'Toll fraud <1s\n6-vector DLP', TEAL),
        ('Compliance', 'Evidence packs\n9 frameworks', GOLD),
    ]
    for i, (head, sub, c) in enumerate(qb_comps):
        cx = qb_start + i * (qb_comp_w + qb_comp_gap)
        rounded_box(slide, cx, qb_comp_y, qb_comp_w, qb_comp_h,
                    fill=NAVY_LIGHT, line=c, line_w=1.5, text='',
                    shadow=False)
        tb = slide.shapes.add_textbox(cx, qb_comp_y + Inches(0.12),
                                       qb_comp_w, Inches(0.4))
        set_text(tb, head, size=12, bold=True, color=c,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        tb2 = slide.shapes.add_textbox(cx, qb_comp_y + Inches(0.5),
                                        qb_comp_w, Inches(0.65))
        set_text(tb2, sub, size=9.5, color=WHITE,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # Caption
    cap = slide.shapes.add_textbox(
        Inches(0.6), Inches(7.05), Inches(12.2), Inches(0.3))
    set_text(cap,
             'Default voice-path topology is passive-mirror — QBITEL failure '
             '= no voice impact. Inline PQC is opt-in per tenant.',
             size=10, italic=True, color=MID_GREY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide_five_seams(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'The Five Integration Seams',
                  'All use existing TCN surfaces — no TCN roadmap changes required')

    headers = ['#', 'SEAM', 'TCN SURFACE', 'QBITEL ROLE']
    col_x = [Inches(0.6), Inches(1.2), Inches(4.0), Inches(8.2)]
    col_w = [Inches(0.6), Inches(2.8), Inches(4.2), Inches(4.55)]
    rows = [
        ('1', 'SIP / RTP signal & media',
         'SBC mirror (or customer-edge SBC)',
         'Passive tap; PQC overlay; DTMF masking; toll-fraud detection'),
        ('2', 'Synapse webhooks',
         'Existing outbound webhook engine (2026)',
         'Subscribes to call.started / .ended / agent.login / payment events'),
        ('3', 'REST API',
         'Existing TCN public API framework',
         'Pulls recording metadata; pushes campaign pause on fraud blocks'),
        ('4', 'CRM / ITSM hooks',
         'Native Salesforce / Zendesk / ServiceNow / Zoho / Finvi / Freshworks',
         'Field-level PII/PHI/PAN masking at API boundary; DLP correlation'),
        ('5', 'Admin console + SSO',
         'TCN admin UI surface',
         'Embedded "Security & Compliance" tab via iframe + SAML/OIDC SSO'),
    ]
    # Header row
    yh = Inches(1.1)
    rh = Inches(0.55)
    for x, w, h in zip(col_x, col_w, headers):
        b = plain_box(slide, x, yh, w, rh, fill=NAVY, text=h,
                      font_size=11, bold=True, font_color=WHITE,
                      align=PP_ALIGN.LEFT)
        b.text_frame.margin_left = Pt(10)
    y = yh + rh
    for i, row in enumerate(rows):
        bg = LIGHT_BG if i % 2 == 0 else WHITE
        rh_i = Inches(0.95)
        for j, (x, w, val) in enumerate(zip(col_x, col_w, row)):
            color = GOLD if j == 0 else (NAVY if j == 1 else DARK_TEXT)
            bold = j <= 1
            b = plain_box(slide, x, y, w, rh_i, fill=bg, text=val,
                          font_size=11 if j <= 1 else 10.5, bold=bold,
                          font_color=color, align=PP_ALIGN.LEFT,
                          anchor=MSO_ANCHOR.MIDDLE)
            b.text_frame.margin_left = Pt(10)
            if j == 0:
                set_text(b, val, size=20, bold=True, color=GOLD,
                         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        y += rh_i

    cap = slide.shapes.add_textbox(
        Inches(0.6), Inches(6.6), Inches(12.2), Inches(0.4))
    set_text(cap,
             'Seam #1 is passive by default. Seams #2–#5 are read-only from TCN\'s perspective '
             'until the customer explicitly opts into write-back actions (e.g., campaign pause on fraud).',
             size=10, italic=True, color=TEAL_DARK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)


# ─── Flow diagrams ───────────────────────────────────────────────────────────

def slide_flow_inbound(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Flow 1 — Inbound Call With PQC Overlay & Per-Tenant Policy',
                  'Passive in default mode. PQC inline is opt-in per trunk.')

    # Top row: 5 sequential boxes left to right
    row_y = Inches(1.4)
    box_w = Inches(2.0)
    box_h = Inches(1.1)
    gap = Inches(0.35)
    start_x = Inches(0.5)
    nodes = [
        ('PSTN /\nCarrier', NAVY_LIGHT, WHITE),
        ('TCN\nSBC', NAVY, WHITE),
        ('TCN\nIVR/IVM', NAVY, WHITE),
        ('TCN\nAgent Workspace', NAVY, WHITE),
        ('Customer\nCRM', NAVY_LIGHT, WHITE),
    ]
    positions = []
    for i, (label, fill, fg) in enumerate(nodes):
        x = start_x + i * (box_w + gap)
        positions.append(x)
        rounded_box(slide, x, row_y, box_w, box_h, fill=fill, line=None,
                    text=label, font_size=12, bold=True, font_color=fg,
                    shadow=True)

    # Arrows between top-row nodes
    for i in range(4):
        x1 = start_x + i * (box_w + gap) + box_w
        x2 = start_x + (i + 1) * (box_w + gap)
        cy = row_y + box_h / 2
        block_arrow(slide, x1 + Inches(0.03), cy - Inches(0.12),
                    gap - Inches(0.06), Inches(0.24),
                    direction='right', fill=TEAL,
                    text='SIP/RTP' if i < 3 else 'CRM API',
                    font_size=8)

    # QBITEL band (lower) — receives webhooks + applies controls
    qb_y = Inches(3.6)
    qb_h = Inches(2.0)
    rounded_box(slide, Inches(0.5), qb_y, Inches(12.33), qb_h,
                fill=NAVY, line=GOLD, line_w=1.5, text='')
    label_chip(slide, Inches(0.7), qb_y + Inches(0.1),
               Inches(2.8), Inches(0.3),
               'QBITEL BRIDGE  (tenantA scope)',
               fill=GOLD, color=NAVY, font_size=10)

    # 4 capability chips inside
    qb_chips = [
        ('Discover\nprotocol', TEAL),
        ('Apply per-tenant\npolicy', GOLD),
        ('PQC overlay\n+ DTMF mask', TEAL),
        ('Sign + log\nevidence', GOLD),
    ]
    chip_w = Inches(2.6)
    chip_h = Inches(1.0)
    chip_y = qb_y + Inches(0.55)
    chip_gap = Inches(0.4)
    chip_total = chip_w * 4 + chip_gap * 3
    chip_start = Inches(0.5) + (Inches(12.33) - chip_total) / 2
    for i, (lbl, c) in enumerate(qb_chips):
        cx = chip_start + i * (chip_w + chip_gap)
        rounded_box(slide, cx, chip_y, chip_w, chip_h,
                    fill=NAVY_LIGHT, line=c, line_w=1.5,
                    text=lbl, font_size=11, bold=True, font_color=WHITE)

    # Connectors from top row down to QBITEL band
    cy_top = row_y + box_h
    cy_bot = qb_y
    # From TCN SBC
    arrow(slide, positions[1] + box_w / 2, cy_top,
          positions[1] + box_w / 2, cy_bot, color=NAVY_LIGHT)
    # label "(1) SPAN mirror"
    lbl = slide.shapes.add_textbox(
        positions[1] + box_w / 2 + Inches(0.05), cy_top + Inches(0.05),
        Inches(1.6), Inches(0.3))
    set_text(lbl, '(1) SPAN mirror', size=9, italic=True, color=NAVY_LIGHT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bold=True)
    # From IVR
    arrow(slide, positions[2] + box_w / 2, cy_top,
          positions[2] + box_w / 2, cy_bot, color=GOLD)
    lbl = slide.shapes.add_textbox(
        positions[2] + box_w / 2 + Inches(0.05), cy_top + Inches(0.05),
        Inches(1.8), Inches(0.3))
    set_text(lbl, '(2) Synapse webhook', size=9, italic=True, color=GOLD,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bold=True)
    # From Agent
    arrow(slide, positions[3] + box_w / 2, cy_top,
          positions[3] + box_w / 2, cy_bot, color=TEAL)
    lbl = slide.shapes.add_textbox(
        positions[3] + box_w / 2 + Inches(0.05), cy_top + Inches(0.05),
        Inches(1.6), Inches(0.3))
    set_text(lbl, '(3) Agent state', size=9, italic=True, color=TEAL,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bold=True)
    # From CRM
    arrow(slide, positions[4] + box_w / 2, cy_top,
          positions[4] + box_w / 2, cy_bot, color=NAVY_LIGHT)
    lbl = slide.shapes.add_textbox(
        positions[4] + box_w / 2 - Inches(1.5), cy_top + Inches(0.05),
        Inches(1.6), Inches(0.3))
    set_text(lbl, '(4) CRM hook (PII mask)', size=9, italic=True,
             color=NAVY_LIGHT,
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.TOP, bold=True)

    # Bottom: Numbered beats
    beats_y = Inches(5.8)
    beats = [
        ('1', 'Carrier delivers SIP INVITE to TCN SBC; TCN mirrors the signaling to QBITEL.'),
        ('2', 'TCN Synapse fires call.started; QBITEL identifies tenant from campaign / DID and loads policy.'),
        ('3', 'If tenant is opted into PQC trunks, QBITEL wraps the media in PQC-SRTP.'),
        ('4', 'Agent writes to CRM; QBITEL\'s mask hook tokenizes PAN / SSN / PHI fields before persistence.'),
        ('5', 'Every decision is signed and logged into the per-tenant evidence stream.'),
    ]
    for i, (num, t) in enumerate(beats):
        x = Inches(0.5) + (i * Inches(2.55))
        step_circle(slide, x, beats_y, Inches(0.3), num,
                    fill=GOLD, color=NAVY)
        tb = slide.shapes.add_textbox(
            x + Inches(0.35), beats_y - Inches(0.05),
            Inches(2.4), Inches(0.85))
        set_text(tb, t, size=8.5, color=DARK_TEXT,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)


def slide_flow_outbound(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Flow 2 — Outbound Call With Toll-Fraud Detection',
                  'Pattern detection in <1 second • Auto-block via TCN REST API (opt-in)')

    # Top row: TCN dialer → TCN SBC → Carrier
    top_y = Inches(1.3)
    box_w = Inches(2.4)
    box_h = Inches(1.0)
    gap = Inches(0.4)
    nodes = [
        ('TCN\nAutodialer', NAVY),
        ('TCN\nSBC (outbound)', NAVY),
        ('PSTN\nCarrier', NAVY_LIGHT),
    ]
    start_x = Inches(0.5)
    positions = []
    for i, (lbl, fill) in enumerate(nodes):
        x = start_x + i * (box_w + gap)
        positions.append(x)
        rounded_box(slide, x, top_y, box_w, box_h, fill=fill, line=None,
                    text=lbl, font_size=12, bold=True, font_color=WHITE,
                    shadow=True)
        if i < 2:
            # Block arrow between
            ax = x + box_w + Inches(0.05)
            block_arrow(slide, ax, top_y + Inches(0.3),
                        gap - Inches(0.1), Inches(0.4),
                        direction='right', fill=TEAL,
                        text='SIP INVITE', font_size=9)

    # Webhook arrow from autodialer down to QBITEL
    qb_x = Inches(8.4)
    qb_y = Inches(2.7)
    qb_w = Inches(4.5)
    qb_h = Inches(2.5)
    # QBITEL fraud-engine box
    rounded_box(slide, qb_x, qb_y, qb_w, qb_h,
                fill=NAVY, line=GOLD, line_w=2.0, text='', shadow=True)
    label_chip(slide, qb_x + Inches(0.15), qb_y + Inches(0.1),
               Inches(3.2), Inches(0.3),
               'QBITEL TOLL-FRAUD ENGINE',
               fill=GOLD, color=NAVY, font_size=10)
    # 10 pattern types listed inside
    patterns_tb = slide.shapes.add_textbox(
        qb_x + Inches(0.2), qb_y + Inches(0.55),
        qb_w - Inches(0.4), qb_h - Inches(0.7))
    tf = patterns_tb.text_frame
    tf.word_wrap = True
    items = [
        'IRSF (premium-rate destinations)',
        'PBX hacking (unauth trunk access)',
        'Wangiri callback manipulation',
        'Call transfer fraud',
        'Call pumping / revenue share abuse',
        'Bypass fraud (SIM box, CLI manip)',
        'Off-hours volume anomalies',
        'Geographic anomaly clusters',
        'Trunk credential brute force',
        'Premium-rate cluster bursts',
    ]
    tf.text = ''
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.15
        r = p.add_run()
        r.text = f'•  {it}'
        r.font.name = FONT
        r.font.size = Pt(10)
        r.font.color.rgb = WHITE

    # Arrow from autodialer to QBITEL (webhook)
    arrow(slide, positions[0] + box_w / 2, top_y + box_h,
          qb_x + Inches(0.5), qb_y, color=GOLD, weight=2.0)
    lbl = slide.shapes.add_textbox(
        positions[0] + box_w / 2 + Inches(0.1), top_y + box_h + Inches(0.05),
        Inches(2.6), Inches(0.3))
    set_text(lbl, '(1) call.started webhook', size=10, italic=True, color=GOLD,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bold=True)

    # Arrow from SBC mirror to QBITEL
    arrow(slide, positions[1] + box_w / 2, top_y + box_h,
          qb_x, qb_y + Inches(1.2), color=NAVY_LIGHT, weight=1.5)

    # Google-safe decision box (no DIAMOND — converter rejects it)
    decision_x = Inches(2.6)
    decision_y = Inches(3.3)
    decision_w = Inches(2.6)
    decision_h = Inches(1.4)
    diamond = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, decision_x, decision_y,
        decision_w, decision_h)
    diamond.adjustments[0] = 0.25
    set_fill(diamond, GOLD)
    no_line(diamond)
    # "DECISION" mini-banner inside
    banner = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, decision_x + Inches(0.2),
        decision_y + Inches(0.12), decision_w - Inches(0.4), Inches(0.28))
    set_fill(banner, NAVY)
    no_line(banner)
    set_text(banner, '? DECISION', size=10, bold=True, color=GOLD,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # Question text
    q_tb = slide.shapes.add_textbox(
        decision_x, decision_y + Inches(0.45),
        decision_w, decision_h - Inches(0.5))
    set_text(q_tb, 'Pattern detected\nin <1 sec?',
             size=14, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # Arrow from QBITEL engine to decision
    arrow(slide, qb_x, qb_y + qb_h / 2,
          Inches(5.0), Inches(4.0), color=GOLD, weight=2.0)

    # YES path → action box
    action = rounded_box(slide, Inches(0.5), Inches(5.3),
                         Inches(5.5), Inches(1.5),
                         fill=RED, line=None, text='', shadow=True)
    label_chip(slide, Inches(0.65), Inches(5.4),
               Inches(1.3), Inches(0.3),
               'BLOCK', fill=WHITE, color=RED, font_size=10)
    action_body = slide.shapes.add_textbox(
        Inches(0.65), Inches(5.75), Inches(5.2), Inches(1.0))
    tf = action_body.text_frame
    tf.word_wrap = True
    actions = [
        '(a)  Pause campaign via TCN REST API',
        '(b)  Push noc.alert webhook',
        '(c)  Log forensic evidence in tenant audit stream',
    ]
    for i, a in enumerate(actions):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.2
        r = p.add_run()
        r.text = a
        r.font.name = FONT
        r.font.size = Pt(10.5)
        r.font.color.rgb = WHITE
        r.font.bold = True

    # YES arrow (down to BLOCK action)
    arrow(slide, decision_x + Inches(0.7), decision_y + decision_h,
          Inches(3.0), Inches(5.3), color=RED, weight=2.5)
    yes_lbl = slide.shapes.add_textbox(
        Inches(2.5), Inches(4.75), Inches(0.6), Inches(0.3))
    set_text(yes_lbl, 'YES', size=11, bold=True, color=RED,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

    # NO path → proceeds
    no_box = rounded_box(slide, Inches(6.3), Inches(5.3),
                         Inches(2.0), Inches(0.7),
                         fill=GREEN, line=None,
                         text='Call proceeds',
                         font_size=12, bold=True, font_color=WHITE,
                         shadow=True)
    arrow(slide, decision_x + decision_w, decision_y + Inches(0.7),
          Inches(7.3), Inches(5.3), color=GREEN, weight=2.5)
    no_lbl = slide.shapes.add_textbox(
        Inches(5.6), Inches(4.3), Inches(0.6), Inches(0.3))
    set_text(no_lbl, 'NO', size=11, bold=True, color=GREEN,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)


def slide_flow_payment(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Flow 3 — Payment Call With DTMF Masking & PCI Evidence',
                  'Up to 80% PCI scope reduction • Card digits never reach agent, screen, or recording')

    # Left column: sequential steps (vertical flow)
    steps = [
        ('Agent presses\n"Take Payment"', NAVY),
        ('TCN agent workspace\nfires payment.start\nvia Synapse webhook', NAVY),
        ('QBITEL applies\ntenant PCI-DSS profile', GOLD),
        ('QBITEL signals TCN:\n• PAUSE recording\n• CLAMP DTMF tones\n• MASK agent screen', TEAL),
        ('Customer enters card\nvia DTMF', NAVY_LIGHT),
        ('Card → payment processor\n(direct path)', NAVY_LIGHT),
        ('payment.end webhook;\nQBITEL resumes &\ngenerates PCI evidence', GOLD),
    ]
    x = Inches(0.6)
    sw = Inches(3.5)
    sh = Inches(0.8)
    sy = Inches(1.15)
    sg = Inches(0.07)
    for i, (text, fill) in enumerate(steps):
        bx = x
        by = sy + i * (sh + sg)
        # Step number circle
        step_circle(slide, bx, by + Inches(0.2), Inches(0.4), str(i + 1),
                    fill=fill, color=WHITE if fill != GOLD else NAVY)
        # Step text box
        rounded_box(slide, bx + Inches(0.55), by, sw, sh,
                    fill=LIGHT_BG if fill not in (GOLD, TEAL) else
                         (GOLD_LIGHT if fill == GOLD else TEAL_LIGHT),
                    line=fill, line_w=1.5,
                    text=text, font_size=10.5, bold=False,
                    font_color=DARK_TEXT)
        # Down arrow between steps
        if i < len(steps) - 1:
            arrow(slide, bx + Inches(0.55) + sw / 2, by + sh,
                  bx + Inches(0.55) + sw / 2, by + sh + sg,
                  color=fill, weight=2.0)

    # Right panel: What the agent / recording / screen sees
    panel_x = Inches(5.0)
    panel_y = Inches(1.15)
    panel_w = Inches(7.85)
    panel_h = Inches(3.0)
    rounded_box(slide, panel_x, panel_y, panel_w, panel_h,
                fill=NAVY, line=GOLD, line_w=2.0, text='', shadow=True)
    label_chip(slide, panel_x + Inches(0.15), panel_y + Inches(0.1),
               Inches(3.0), Inches(0.3),
               'WHAT GETS PROTECTED',
               fill=GOLD, color=NAVY, font_size=10)
    # 3 protections
    prot = [
        ('Agent headset', 'Hears CLAMPed tone — no digits audible'),
        ('TCN recording', 'Paused for payment window'),
        ('Agent screen', 'Shows last-4 only on cardholder fields'),
    ]
    py = panel_y + Inches(0.6)
    for i, (k, v) in enumerate(prot):
        rounded_box(slide, panel_x + Inches(0.2),
                    py + i * Inches(0.75),
                    Inches(2.5), Inches(0.6),
                    fill=NAVY_LIGHT, line=None,
                    text=k, font_size=12, bold=True, font_color=GOLD)
        rounded_box(slide, panel_x + Inches(2.8),
                    py + i * Inches(0.75),
                    Inches(4.85), Inches(0.6),
                    fill=NAVY_LIGHT, line=TEAL, line_w=1.0,
                    text=v, font_size=11, bold=False, font_color=WHITE,
                    align=PP_ALIGN.LEFT)

    # PCI claim band
    band = rounded_box(slide, panel_x, Inches(4.35), panel_w, Inches(0.7),
                       fill=GOLD, line=None,
                       text='PCI scope reduction up to 80%   •   Evidence pack on demand in <10 minutes',
                       font_size=13, bold=True, font_color=NAVY)

    # Caption
    cap = rounded_box(slide, panel_x, Inches(5.25), panel_w, Inches(1.5),
                      fill=LIGHT_BG, line=TEAL, line_w=1.0, text='')
    cap_tb = slide.shapes.add_textbox(
        panel_x + Inches(0.2), Inches(5.35), panel_w - Inches(0.4),
        Inches(1.3))
    set_text(cap_tb,
             'With this flow active, the agent, the agent\'s screen, and the TCN '
             'recording engine never see the cardholder data. The card digits '
             'travel directly from the customer\'s telephone keypad to the '
             'payment processor — TCN and QBITEL are out of scope for that span.',
             size=11, italic=True, color=DARK_TEXT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.3)


def slide_multi_tenant(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Multi-Tenant Isolation — TCN\'s BPO Customers',
                  'The single most differentiating capability for the BPO segment')

    # Top: TCN Operator (single BPO tenant)
    tcn_x = Inches(3.0)
    tcn_y = Inches(1.1)
    tcn_w = Inches(7.3)
    tcn_h = Inches(0.8)
    rounded_box(slide, tcn_x, tcn_y, tcn_w, tcn_h,
                fill=NAVY, line=GOLD, line_w=1.5,
                text='TCN Operator  (one tenant = the BPO)',
                font_size=14, bold=True, font_color=WHITE, shadow=True)

    # Tagging row
    tag_tb = slide.shapes.add_textbox(
        tcn_x, tcn_y + tcn_h + Inches(0.05), tcn_w, Inches(0.35))
    set_text(tag_tb,
             'Campaign- and account-level tagging carries downstream client identity',
             size=10, italic=True, color=MID_GREY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # 3 down-arrows
    arr_y = Inches(2.4)
    for i, x_off in enumerate([Inches(2.1), Inches(6.66), Inches(11.2)]):
        block_arrow(slide, x_off - Inches(0.2), arr_y,
                    Inches(0.4), Inches(0.4),
                    direction='down', fill=GOLD)

    # 3 QBITEL tenant boxes
    tenants = [
        ('TENANT A',
         'NorthBank\n(Banking)',
         'PCI-DSS 4.0 + SOX + FCA\nML-KEM keys: 0x4A...\nOPA policy: bank-strict\nAudit log: append-only',
         NAVY_LIGHT, TEAL),
        ('TENANT B',
         'CareFirst\n(Healthcare)',
         'HIPAA + HITECH\nML-KEM keys: 0xC2...\nOPA policy: phi-strict\nAudit log: append-only',
         NAVY_LIGHT, GOLD),
        ('TENANT C',
         'ShopRight\n(Retail)',
         'PCI-DSS + GDPR + TCPA\nML-KEM keys: 0xE9...\nOPA policy: retail-strict\nAudit log: append-only',
         NAVY_LIGHT, TEAL),
    ]
    tw = Inches(3.9)
    th = Inches(3.0)
    ty = Inches(2.9)
    t_gap = Inches(0.3)
    t_start = (SLIDE_W - (tw * 3 + t_gap * 2)) / 2
    for i, (label, client, body, fill, accent) in enumerate(tenants):
        tx = t_start + i * (tw + t_gap)
        # Box
        rounded_box(slide, tx, ty, tw, th,
                    fill=fill, line=accent, line_w=2.5,
                    text='', shadow=True)
        # Header band
        label_chip(slide, tx + Inches(0.2), ty + Inches(0.15),
                   tw - Inches(0.4), Inches(0.4),
                   label, fill=accent, color=NAVY if accent == GOLD else WHITE,
                   font_size=12)
        # Client
        client_tb = slide.shapes.add_textbox(
            tx, ty + Inches(0.65), tw, Inches(0.7))
        set_text(client_tb, client, size=14, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # Divider
        div = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, tx + Inches(0.3),
            ty + Inches(1.4), tw - Inches(0.6), Inches(0.02))
        set_fill(div, accent)
        no_line(div)
        # Body
        body_tb = slide.shapes.add_textbox(
            tx + Inches(0.2), ty + Inches(1.55),
            tw - Inches(0.4), Inches(1.4))
        set_text(body_tb, body, size=10.5, color=WHITE,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
                 line_spacing=1.4)

    # Bottom callout
    callout = rounded_box(
        slide, Inches(0.5), Inches(6.1), Inches(12.33), Inches(0.85),
        fill=GOLD, line=None,
        text='Cross-tenant data access is structurally impossible.  '
             'Each tenant\'s evidence pack contains only that tenant\'s data.  '
             'Generation time: <10 minutes per framework, per tenant.',
        font_size=13, bold=True, font_color=NAVY)


# ─── Remaining content slides ────────────────────────────────────────────────

def slide_value_tcn(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Joint Value — For TCN',
                  'New revenue line • Higher ACV • Defensive moat • Win-rate lift')

    benefits = [
        ('New revenue line per existing customer',
         'Embedded QBITEL SKU billable per concurrent seat. Revenue share with TCN.'),
        ('Higher ACV at quota-bag level',
         'TCN AEs sell one platform; QBITEL adds materially to ACV — typical '
         '15–30% ARR uplift on a BPO account.'),
        ('Defensive moat',
         '"TCN is the only CCaaS with embedded NIST Level 5 PQC and multi-tenant '
         'compliance automation" — a category-of-one positioning.'),
        ('RFP win-rate lift',
         'BPO and regulated-enterprise RFPs increasingly include PQC, '
         'multi-tenant compliance, and toll-fraud SLAs. TCN + QBITEL answers '
         'them; TCN alone has to caveat.'),
        ('Reduced churn on regulated accounts',
         'Compliance pack delivery removes a recurring pain point that often '
         'becomes a contract-renewal risk.'),
    ]
    y = Inches(1.1)
    for i, (head, body) in enumerate(benefits):
        # Number circle
        step_circle(slide, Inches(0.6), y + Inches(0.15),
                    Inches(0.5), str(i + 1), fill=GOLD, color=NAVY)
        # Title
        head_tb = slide.shapes.add_textbox(
            Inches(1.3), y, Inches(11), Inches(0.4))
        set_text(head_tb, head, size=14, bold=True, color=NAVY,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
        # Body
        body_tb = slide.shapes.add_textbox(
            Inches(1.3), y + Inches(0.38),
            Inches(11), Inches(0.7))
        set_text(body_tb, body, size=11, color=DARK_TEXT,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
                 line_spacing=1.25)
        y += Inches(1.1)


def slide_value_customer(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Joint Value — For TCN\'s Customers',
                  'Where each customer pain meets a measurable QBITEL outcome')

    pairs = [
        ('Toll-fraud surprises on monthly carrier invoice',
         'Real-time blocking, <1s detection, 30–60 day ROI on the QBITEL SKU'),
        ('Annual PCI-DSS audit costing $500K–$2M+',
         'Up to 80% scope reduction; evidence pack in <10 minutes'),
        ('Three enterprise clients = three separate audits',
         'Per-tenant evidence packs; structural isolation; one platform'),
        ('Quantum harvest against 7-year recording retention',
         'ML-KEM-1024 recording encryption from day one'),
        ('Insider exfiltration via clipboard, USB, screen, voice',
         'Kernel-level DLP across 6 vectors'),
        ('Remote agents on consumer-grade home networks',
         'VPN-less PQC tunnels with continuous posture checks'),
    ]
    # Header
    plain_box(slide, Inches(0.5), Inches(1.1), Inches(6.0), Inches(0.45),
              fill=RED, text='CUSTOMER PAIN', font_size=12, bold=True,
              font_color=WHITE, align=PP_ALIGN.LEFT,
              anchor=MSO_ANCHOR.MIDDLE).text_frame.margin_left = Pt(10)
    plain_box(slide, Inches(6.6), Inches(1.1), Inches(6.23), Inches(0.45),
              fill=GREEN, text='WHAT THEY GET WITH QBITEL', font_size=12,
              bold=True, font_color=WHITE, align=PP_ALIGN.LEFT,
              anchor=MSO_ANCHOR.MIDDLE).text_frame.margin_left = Pt(10)
    y = Inches(1.6)
    rh = Inches(0.78)
    for i, (pain, gain) in enumerate(pairs):
        bg = LIGHT_BG if i % 2 == 0 else WHITE
        pain_box = plain_box(slide, Inches(0.5), y, Inches(6.0), rh,
                              fill=bg, text=pain, font_size=11,
                              font_color=DARK_TEXT, align=PP_ALIGN.LEFT,
                              anchor=MSO_ANCHOR.MIDDLE)
        pain_box.text_frame.margin_left = Pt(12)
        gain_box = plain_box(slide, Inches(6.6), y, Inches(6.23), rh,
                              fill=bg, text=gain, font_size=11, bold=True,
                              font_color=TEAL_DARK, align=PP_ALIGN.LEFT,
                              anchor=MSO_ANCHOR.MIDDLE)
        gain_box.text_frame.margin_left = Pt(12)
        # Arrow between
        arr = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW, Inches(6.3), y + Inches(0.2),
            Inches(0.3), Inches(0.38))
        set_fill(arr, GOLD)
        no_line(arr)
        y += rh + Inches(0.05)


def slide_commercial(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Commercial Models — Three Options',
                  'We are flexible — pick the structure that matches your existing partner motion')

    options = [
        ('A', 'EMBEDDED OEM',
         'QBITEL ships as a tier of TCN Operator',
         ['TCN bills customer; QBITEL revenue share',
          'Strongest joint positioning',
          'Fastest adoption velocity',
          'Best for: long-term strategic partnership'], NAVY),
        ('B', 'MARKETPLACE',
         'Listed alongside Salesforce, Zendesk, ServiceNow…',
         ['Customer opts in per account',
          'Direct QBITEL ↔ customer contract',
          'TCN receives platform / referral fee',
          'Best for: incremental rollout, low TCN commitment'], TEAL),
        ('C', 'REFERRAL / CO-SELL',
         'TCN sales refers qualified accounts to QBITEL',
         ['Separate contracts, joint motions',
          'Referral fee or rev-share per deal',
          'Shared collateral, co-marketing',
          'Best for: a fast first proof-point'], GOLD),
    ]
    col_w = Inches(4.0)
    col_h = Inches(5.4)
    gap = Inches(0.17)
    start_x = (SLIDE_W - (col_w * 3 + gap * 2)) / 2
    y = Inches(1.1)
    for i, (letter, name, sub, bullets, color) in enumerate(options):
        cx = start_x + i * (col_w + gap)
        # Card
        card_fill = NAVY if color == NAVY else (LIGHT_BG)
        text_color = WHITE if color == NAVY else DARK_TEXT
        rounded_box(slide, cx, y, col_w, col_h,
                    fill=card_fill, line=color, line_w=2.5,
                    text='', shadow=True)
        # Letter badge
        badge = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, cx + Inches(0.2), y + Inches(0.2),
            Inches(0.85), Inches(0.85))
        set_fill(badge, color)
        no_line(badge)
        set_text(badge, letter, size=36, bold=True,
                 color=WHITE if color != GOLD else NAVY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # Name
        name_tb = slide.shapes.add_textbox(
            cx + Inches(1.15), y + Inches(0.2),
            col_w - Inches(1.3), Inches(0.4))
        set_text(name_tb, name, size=15, bold=True, color=color,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
        # Sub
        sub_tb = slide.shapes.add_textbox(
            cx + Inches(1.15), y + Inches(0.6),
            col_w - Inches(1.3), Inches(0.5))
        set_text(sub_tb, sub, size=10, italic=True, color=text_color,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
        # Divider
        div = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, cx + Inches(0.3), y + Inches(1.3),
            col_w - Inches(0.6), Inches(0.025))
        set_fill(div, color)
        no_line(div)
        # Bullets
        bullets_tb = slide.shapes.add_textbox(
            cx + Inches(0.3), y + Inches(1.5),
            col_w - Inches(0.6), col_h - Inches(1.8))
        tf = bullets_tb.text_frame
        tf.word_wrap = True
        for bi, b in enumerate(bullets):
            p = tf.paragraphs[0] if bi == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT
            p.line_spacing = 1.35
            r = p.add_run()
            r.text = ('★ ' if b.startswith('Best for') else '•  ') + b
            r.font.name = FONT
            r.font.size = Pt(11)
            r.font.color.rgb = (color if b.startswith('Best for')
                                else text_color)
            r.font.bold = b.startswith('Best for')

    # Recommendation strip
    rec = rounded_box(
        slide, Inches(0.5), Inches(6.6), Inches(12.33), Inches(0.55),
        fill=GOLD_LIGHT, line=GOLD, line_w=1.5,
        text='Our recommendation: start with Option C or B for the first two quarters '
             '(validate demand + integration quality), graduate to Option A for the long term.',
        font_size=12, bold=True, font_color=NAVY)


def slide_tech_risks(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Technical Risks & Mitigations',
                  'The questions a VP of Technology should ask — with our honest answers')

    risks = [
        ('Voice quality impact?',
         'PQC overhead is <2ms — within ITU-T G.114 budget. MOS measured before/after on every deployment. Passive by default.'),
        ('New failure mode for voice path?',
         'Default mirror topology = QBITEL down has zero voice impact. Inline PQC fails open by default; fail-closed is opt-in per tenant.'),
        ('Latency on Synapse webhook engine?',
         'QBITEL subscribes asynchronously. Webhooks fire at TCN\'s normal latency; QBITEL processes out-of-band. No back-pressure.'),
        ('CSM team has to learn a new product?',
         '"Security & Compliance" tab is iframe + SSO inside TCN Operator. CSM training is a 1-hour session.'),
        ('Data residency / regulated geos?',
         'Three topologies: TCN-hosted, customer-hosted (on-prem), hybrid. On-prem uses Ollama LLM — zero cloud egress.'),
        ('What if a tenant wants to leave?',
         'Opt-out is a flag on the tenant; QBITEL stops processing within minutes. Evidence pack exported per contract.'),
        ('Liability if QBITEL incorrectly flags fraud?',
         'Tenant-configurable: block+alert / alert-only / alert+recommend. Default in early rollout is alert-only; auto-block is opt-in.'),
        ('GA-stable? Production references?',
         'In production at BPO and FS accounts (references under NDA). Multi-tenant hardened with audit-grade evidence.'),
    ]
    # Two-column grid
    cw = Inches(6.05)
    ch = Inches(1.25)
    y = Inches(1.1)
    for i, (q, a) in enumerate(risks):
        col = i % 2
        row = i // 2
        x = Inches(0.5) + col * (cw + Inches(0.15))
        yy = y + row * (ch + Inches(0.1))
        # Card
        rounded_box(slide, x, yy, cw, ch, fill=LIGHT_BG, line=TEAL,
                    line_w=1.0, text='', shadow=False)
        # Q
        q_tb = slide.shapes.add_textbox(
            x + Inches(0.2), yy + Inches(0.08),
            cw - Inches(0.4), Inches(0.35))
        set_text(q_tb, '?  ' + q, size=11.5, bold=True, color=NAVY,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
        # A
        a_tb = slide.shapes.add_textbox(
            x + Inches(0.2), yy + Inches(0.42),
            cw - Inches(0.4), Inches(0.85))
        set_text(a_tb, a, size=10, color=DARK_TEXT,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
                 line_spacing=1.25)


def slide_poc_plan(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'POC Plan — 4-Week Joint Sprint',
                  'One engineer from each side + one pilot customer = working demo')

    weeks = [
        ('WEEK 1', 'INTEGRATION SPRINT',
         ['Synapse webhook subscription set up',
          'REST API auth handshake validated',
          'One CRM hook (e.g., Salesforce) integrated',
          'SSO configured for console tab'], NAVY),
        ('WEEK 2', 'CAPABILITY ACTIVATION',
         ['Toll-fraud detection on one outbound campaign',
          'DTMF masking on one payment-handling skill',
          'Agent DLP rolled out to 10 pilot endpoints',
          'PQC overlay verified on test trunk'], TEAL),
        ('WEEK 3', 'MULTI-TENANT DEMO',
         ['Two logical tenants configured in QBITEL',
          'Per-tenant policy + key + audit isolation verified',
          'Compliance evidence pack generated per tenant',
          'Cross-tenant access negative test'], GOLD),
        ('WEEK 4', 'JOINT REVIEW',
         ['Technical readout with TCN eng / product / security',
          'Customer feedback session with pilot BPO',
          'Decision: proceed to commercial structure',
          'Joint go-to-market plan drafted'], NAVY),
    ]
    cw = Inches(2.95)
    ch = Inches(4.0)
    gap = Inches(0.16)
    start_x = (SLIDE_W - (cw * 4 + gap * 3)) / 2
    y = Inches(1.1)
    for i, (label, title, items, color) in enumerate(weeks):
        cx = start_x + i * (cw + gap)
        # Card
        rounded_box(slide, cx, y, cw, ch, fill=LIGHT_BG,
                    line=color, line_w=2.0, text='', shadow=True)
        # Top band
        band = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, cx, y, cw, Inches(0.6))
        set_fill(band, color)
        no_line(band)
        set_text(band, label, size=14, bold=True,
                 color=WHITE if color != GOLD else NAVY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # Title
        tt = slide.shapes.add_textbox(cx, y + Inches(0.7),
                                       cw, Inches(0.4))
        set_text(tt, title, size=12, bold=True, color=NAVY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # Bullets
        bt = slide.shapes.add_textbox(
            cx + Inches(0.2), y + Inches(1.2),
            cw - Inches(0.4), ch - Inches(1.4))
        tf = bt.text_frame
        tf.word_wrap = True
        for bi, b in enumerate(items):
            p = tf.paragraphs[0] if bi == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT
            p.line_spacing = 1.4
            r = p.add_run()
            r.text = '✓  ' + b
            r.font.name = FONT
            r.font.size = Pt(10.5)
            r.font.color.rgb = DARK_TEXT
        # Arrow to next
        if i < 3:
            ax = cx + cw + Inches(0.01)
            block_arrow(slide, ax, y + ch / 2 - Inches(0.15),
                        gap - Inches(0.02), Inches(0.3),
                        direction='right', fill=GOLD)

    # Success criteria strip
    sc = rounded_box(
        slide, Inches(0.5), Inches(5.3), Inches(12.33), Inches(1.9),
        fill=NAVY, line=GOLD, line_w=1.5, text='', shadow=True)
    label_chip(slide, Inches(0.7), Inches(5.4),
               Inches(2.5), Inches(0.35),
               'SUCCESS CRITERIA',
               fill=GOLD, color=NAVY, font_size=10)
    sc_tb = slide.shapes.add_textbox(
        Inches(0.7), Inches(5.8), Inches(12), Inches(1.4))
    tf = sc_tb.text_frame
    tf.word_wrap = True
    sc_items = [
        'Integration non-disruptive to TCN data plane (verified by latency + MOS measurement)',
        'Pilot customer attests the console tab is usable inside TCN Operator with no training',
        'Demonstrated end-to-end: one toll-fraud block + one DTMF-masked payment + one per-tenant compliance pack',
        'Joint commercial path identified and agreed',
    ]
    for i, item in enumerate(sc_items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.3
        r = p.add_run()
        r.text = '◆  ' + item
        r.font.name = FONT
        r.font.size = Pt(11)
        r.font.color.rgb = WHITE


def slide_next_steps(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, 'Next Steps',
                  'If today\'s conversation is productive, here is the path forward')

    steps = [
        ('1 WEEK', 'MUTUAL NDA + REFERENCES',
         'Mutual NDA in place. QBITEL shares technical reference architecture under NDA, plus customer references.'),
        ('2 WEEKS', 'TECHNICAL SCOPING CALL',
         'Joint scoping call: TCN engineering + QBITEL integration team. Confirm seams; pick pilot customer.'),
        ('4 WEEKS', 'POC KICKOFF',
         'Four-week joint sprint per the prior slide.'),
        ('12 WEEKS', 'JOINT GO/NO-GO',
         'Decision on commercial structure (Option A / B / C) and roadmap to GA partnership.'),
    ]
    timeline_y = Inches(1.3)
    timeline_h = Inches(2.5)
    # Horizontal timeline
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), timeline_y + Inches(1.2),
        Inches(11.7), Inches(0.06))
    set_fill(line, GOLD)
    no_line(line)

    for i, (when, title, body) in enumerate(steps):
        cx = Inches(0.8 + i * 3.0)
        # Bubble
        bub = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, cx, timeline_y + Inches(0.85), Inches(0.8), Inches(0.8))
        set_fill(bub, NAVY)
        no_line(bub)
        set_text(bub, str(i + 1), size=22, bold=True, color=GOLD,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # When chip
        when_chip = label_chip(slide, cx - Inches(0.3),
                                timeline_y + Inches(0.4),
                                Inches(1.4), Inches(0.35),
                                when, fill=GOLD, color=NAVY, font_size=10)
        # Title
        title_tb = slide.shapes.add_textbox(
            cx - Inches(0.6), timeline_y + Inches(1.75),
            Inches(2.5), Inches(0.4))
        set_text(title_tb, title, size=12, bold=True, color=NAVY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # Body
        body_tb = slide.shapes.add_textbox(
            cx - Inches(0.6), timeline_y + Inches(2.15),
            Inches(2.6), Inches(1.3))
        set_text(body_tb, body, size=10, color=DARK_TEXT,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP,
                 line_spacing=1.3)

    # What we'd like from this meeting
    ask = rounded_box(
        slide, Inches(0.5), Inches(5.2), Inches(12.33), Inches(2.0),
        fill=NAVY, line=GOLD, line_w=1.5, text='', shadow=True)
    label_chip(slide, Inches(0.7), Inches(5.3),
               Inches(3.5), Inches(0.35),
               'WHAT WE\'D LIKE FROM THIS MEETING',
               fill=GOLD, color=NAVY, font_size=10)
    asks = [
        'Your initial reaction to the integration architecture',
        'Confirmation (or correction) of our understanding of TCN Operator',
        'Pointer to the right TCN counterpart for the Week-1 scoping call',
        'Indication of which commercial model is most natural for TCN today',
    ]
    ask_tb = slide.shapes.add_textbox(
        Inches(0.7), Inches(5.7), Inches(12), Inches(1.5))
    tf = ask_tb.text_frame
    tf.word_wrap = True
    for i, a in enumerate(asks):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.35
        r = p.add_run()
        r.text = '▸  ' + a
        r.font.name = FONT
        r.font.size = Pt(12)
        r.font.color.rgb = WHITE


def slide_contact(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    set_fill(bg, NAVY)
    no_line(bg)

    # Accent strip
    acc = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, Inches(3.5), SLIDE_W, Inches(0.06))
    set_fill(acc, GOLD)
    no_line(acc)

    # Thank you
    ty_tb = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.5), Inches(12.33), Inches(1.5))
    set_text(ty_tb, 'Thank You', size=56, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    sub_tb = slide.shapes.add_textbox(
        Inches(0.5), Inches(2.7), Inches(12.33), Inches(0.5))
    set_text(sub_tb,
             'Let us build the joint motion that makes TCN the only CCaaS with embedded quantum-safe compliance.',
             size=15, italic=True, color=TEAL_LIGHT,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # Contact strip
    c_y = Inches(4.0)
    c_h = Inches(1.6)
    contacts = [
        ('✉  ENTERPRISE', 'enterprise@qbitel.com'),
        ('⊕  WEBSITE', 'bridge.qbitel.com'),
        ('◉  SCHEDULE', 'Contact your account team'),
    ]
    cw = Inches(3.8)
    cg = Inches(0.3)
    start_x = (SLIDE_W - (cw * 3 + cg * 2)) / 2
    for i, (head, val) in enumerate(contacts):
        cx = start_x + i * (cw + cg)
        rounded_box(slide, cx, c_y, cw, c_h, fill=NAVY_LIGHT,
                    line=TEAL, line_w=1.5, text='', shadow=True)
        h_tb = slide.shapes.add_textbox(cx, c_y + Inches(0.25),
                                         cw, Inches(0.4))
        set_text(h_tb, head, size=13, bold=True, color=GOLD,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        v_tb = slide.shapes.add_textbox(cx, c_y + Inches(0.75),
                                         cw, Inches(0.7))
        set_text(v_tb, val, size=14, color=WHITE,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # Footer
    f_tb = slide.shapes.add_textbox(
        Inches(0.5), Inches(6.7), Inches(12.33), Inches(0.4))
    set_text(f_tb,
             'QBITEL Bridge — Quantum-Safe Security for the Human API   |   © 2026 QBITEL   |   Confidential',
             size=10, color=RGBColor(0xAA, 0xBB, 0xCC),
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, italic=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def build_pptx():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_cover(prs)            # 1
    slide_agenda(prs)           # 2
    slide_objective(prs)        # 3
    slide_tcn_overview(prs)     # 4
    slide_qbitel_modules(prs)   # 5
    slide_partnership_thesis(prs)  # 6
    slide_architecture(prs)     # 7  ← BIG architecture diagram
    slide_five_seams(prs)       # 8
    slide_flow_inbound(prs)     # 9  ← Flow 1
    slide_flow_outbound(prs)    # 10 ← Flow 2
    slide_flow_payment(prs)     # 11 ← Flow 3
    slide_multi_tenant(prs)     # 12 ← Multi-tenant diagram
    slide_value_tcn(prs)        # 13
    slide_value_customer(prs)   # 14
    slide_commercial(prs)       # 15
    slide_tech_risks(prs)       # 16
    slide_poc_plan(prs)         # 17
    slide_next_steps(prs)       # 18
    slide_contact(prs)          # 19

    # Add footers to all slides except cover and contact
    for idx, slide in enumerate(prs.slides):
        if idx == 0 or idx == len(prs.slides) - 1:
            continue
        add_footer(slide, idx + 1)

    out_path = 'docs/brochures/QBITEL_TCN_Partnership_Brief_Safe.pptx'
    prs.save(out_path)
    print(f'PPTX saved: {out_path}')
    return out_path


if __name__ == '__main__':
    build_pptx()
