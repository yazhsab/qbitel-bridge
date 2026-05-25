#!/usr/bin/env python3
"""
Build QBITEL Bridge - Mainframe Modernization PowerPoint Presentation
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Brand Colors ──
BG_DARK    = RGBColor(0x0A, 0x0B, 0x10)
BG_CARD    = RGBColor(0x11, 0x18, 0x27)
ACCENT     = RGBColor(0x00, 0xD9, 0xFF)
GREEN      = RGBColor(0x00, 0xFF, 0x88)
RED        = RGBColor(0xFF, 0x47, 0x57)
ORANGE     = RGBColor(0xFF, 0x9F, 0x43)
YELLOW     = RGBColor(0xFF, 0xD4, 0x3B)
WHITE      = RGBColor(0xE2, 0xE8, 0xF0)
DIM        = RGBColor(0x94, 0xA3, 0xB8)
DARK_TEXT   = RGBColor(0x64, 0x74, 0x8B)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Helper Functions ──

def set_slide_bg(slide, color=BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_shape(slide, left, top, width, height, fill_color=BG_CARD, border_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    # Subtle corner radius
    return shape

def add_text(slide, left, top, width, height, text, size=18, color=WHITE, bold=False, align=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = align
    return txBox

def add_para(text_frame, text, size=16, color=WHITE, bold=False, space_before=Pt(4), space_after=Pt(4), align=PP_ALIGN.LEFT):
    p = text_frame.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = "Calibri"
    p.space_before = space_before
    p.space_after = space_after
    p.alignment = align
    return p

def add_bullet_box(slide, left, top, width, height, title, bullets, title_color=ACCENT, bullet_color=WHITE, title_size=20, bullet_size=16):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(title_size)
    p.font.color.rgb = title_color
    p.font.bold = True
    p.font.name = "Calibri"
    p.space_after = Pt(8)
    for b in bullets:
        bp = tf.add_paragraph()
        bp.text = f"  {b}"
        bp.font.size = Pt(bullet_size)
        bp.font.color.rgb = bullet_color
        bp.font.name = "Calibri"
        bp.space_before = Pt(4)
        bp.space_after = Pt(4)
        bp.level = 0
    return txBox

def add_metric_card(slide, left, top, width, height, value, label, value_color=GREEN):
    shape = add_shape(slide, left, top, width, height, BG_CARD)
    add_text(slide, left, top + Inches(0.25), width, Inches(0.6), value, size=36, color=value_color, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, left, top + Inches(0.9), width, Inches(0.4), label, size=13, color=DIM, align=PP_ALIGN.CENTER)

def add_label(slide, left, top, text, color=ACCENT, size=11):
    add_text(slide, left, top, Inches(4), Inches(0.3), text.upper(), size=size, color=color, bold=True)


# ═══════════════════════════════════════════════════════════
# SLIDE 1: TITLE
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_slide_bg(slide)

# Accent line
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Pt(4))
shape.fill.solid(); shape.fill.fore_color.rgb = ACCENT; shape.line.fill.background()

add_label(slide, Inches(1.5), Inches(1.8), "UC1 — Legacy Mainframe Modernization")
add_text(slide, Inches(1.5), Inches(2.2), Inches(10), Inches(1.2), "QBITEL Bridge", size=60, color=ACCENT, bold=True)
add_text(slide, Inches(1.5), Inches(3.4), Inches(10), Inches(0.7), "AI-Powered Quantum-Safe Security for Legacy Systems", size=26, color=DIM)

# Stats row
stats = [("$3T", "Daily COBOL Transactions"), ("60%", "Fortune 500 on Legacy"), ("38yr", "Average System Age"), ("4.5M", "Lines of COBOL")]
for i, (val, lbl) in enumerate(stats):
    x = Inches(1.5 + i * 2.7)
    add_metric_card(slide, x, Inches(4.8), Inches(2.2), Inches(1.3), val, lbl)

add_text(slide, Inches(1.5), Inches(6.5), Inches(10), Inches(0.4), "100% Open Source  |  Apache 2.0 License  |  Air-Gapped Capable", size=14, color=DARK_TEXT, align=PP_ALIGN.LEFT)


# ═══════════════════════════════════════════════════════════
# SLIDE 2: THE PROBLEM
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "The Problem")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "Three Converging Crises", size=40, color=WHITE, bold=True)

crises = [
    ("Legacy Crisis", "$2-10M", "Cost to reverse-engineer ONE legacy system. Original developers retired. No documentation. 6-12 months per system.", RED),
    ("Quantum Threat", "5-10 years", "Until RSA/ECC breaks. Nation-states are harvesting encrypted data TODAY for future quantum decryption.", ORANGE),
    ("Speed Gap", "65 minutes", "Average SOC response time. Machine-speed attacks happen in seconds. Humans can't keep up.", YELLOW),
]

for i, (title, stat, desc, color) in enumerate(crises):
    x = Inches(0.8 + i * 4.0)
    card = add_shape(slide, x, Inches(2.2), Inches(3.6), Inches(4.5), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    add_text(slide, x + Inches(0.3), Inches(2.5), Inches(3), Inches(0.5), title, size=22, color=ACCENT, bold=True)
    add_text(slide, x + Inches(0.3), Inches(3.1), Inches(3), Inches(0.7), stat, size=42, color=color, bold=True)
    add_text(slide, x + Inches(0.3), Inches(4.0), Inches(3), Inches(2.2), desc, size=15, color=DIM)


# ═══════════════════════════════════════════════════════════
# SLIDE 3: THE SOLUTION — FIVE-STAGE JOURNEY
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "The Solution")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "Five-Stage Modernization Journey", size=40, color=WHITE, bold=True)

stages = [
    ("1", "Discover", "AI learns unknown protocols from raw traffic", "2-4 hours vs 6-12 months"),
    ("2", "Protect", "NIST Level 5 PQC encryption wrapping", "Zero code changes, <1ms overhead"),
    ("3", "Translate", "Auto-generate REST APIs + SDKs", "6 languages in minutes"),
    ("4", "Comply", "9 compliance frameworks automated", "Reports in <10 minutes"),
    ("5", "Operate", "Autonomous threat detection & response", "78% autonomous, <1s decisions"),
]

for i, (num, title, desc, metric) in enumerate(stages):
    x = Inches(0.5 + i * 2.5)
    card = add_shape(slide, x, Inches(2.2), Inches(2.2), Inches(4.5), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    # Number circle
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.75), Inches(2.5), Inches(0.6), Inches(0.6))
    circle.fill.solid(); circle.fill.fore_color.rgb = ACCENT; circle.line.fill.background()
    add_text(slide, x + Inches(0.75), Inches(2.55), Inches(0.6), Inches(0.5), num, size=22, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.2), Inches(3.3), Inches(1.8), Inches(0.5), title, size=22, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.15), Inches(3.8), Inches(1.9), Inches(1.2), desc, size=14, color=DIM, align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.15), Inches(5.2), Inches(1.9), Inches(1.0), metric, size=13, color=GREEN, bold=True, align=PP_ALIGN.CENTER)

    # Arrow between cards
    if i < 4:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + Inches(2.25), Inches(4.2), Inches(0.25), Inches(0.2))
        arrow.fill.solid(); arrow.fill.fore_color.rgb = ACCENT; arrow.line.fill.background()


# ═══════════════════════════════════════════════════════════
# SLIDE 4: PLATFORM ARCHITECTURE
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "Platform Architecture")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "Four-Layer Polyglot Design", size=40, color=WHITE, bold=True)

layers = [
    ("React / TypeScript", "UI Console", "Admin Dashboard  •  Protocol Copilot  •  Marketplace  •  Real-Time Monitoring", RGBColor(0x61, 0xDA, 0xFB)),
    ("Go", "Control Plane", "Service Orchestration  •  OPA Policies  •  Vault Secrets  •  gRPC Gateway", RGBColor(0x00, 0xAD, 0xD8)),
    ("Python / FastAPI", "AI Engine", "Protocol Discovery  •  Multi-Agent System  •  LLM  •  RAG  •  Compliance Automation", RGBColor(0xFF, 0xD4, 0x3B)),
    ("Rust", "Data Plane", "PQC-TLS Termination  •  DPDK Packet Processing  •  DPI  •  Protocol Adapters  •  <1ms Latency", RGBColor(0xDE, 0xA5, 0x84)),
]

for i, (lang, name, desc, color) in enumerate(layers):
    y = Inches(2.0 + i * 1.25)
    card = add_shape(slide, Inches(0.8), y, Inches(11.5), Inches(1.05), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    # Language badge
    badge = add_shape(slide, Inches(1.1), y + Inches(0.22), Inches(2.0), Inches(0.55), color)
    add_text(slide, Inches(1.1), y + Inches(0.25), Inches(2.0), Inches(0.5), lang, size=14, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(3.4), y + Inches(0.15), Inches(2.0), Inches(0.4), name, size=20, color=WHITE, bold=True)
    add_text(slide, Inches(3.4), y + Inches(0.55), Inches(8.5), Inches(0.4), desc, size=13, color=DIM)

add_text(slide, Inches(0.8), Inches(7.0), Inches(11), Inches(0.3), "100% Open Source  •  Apache 2.0  •  Air-Gapped On-Premise LLM (Ollama)  •  No Cloud Dependency", size=13, color=DARK_TEXT, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════
# SLIDE 5: PQC — POST-QUANTUM CRYPTOGRAPHY
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "Post-Quantum Cryptography")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "NIST-Standardized Quantum-Safe Algorithms", size=40, color=WHITE, bold=True)

# Algorithm cards
algos = [
    ("ML-KEM-768", "FIPS 203", "Key Encapsulation", "1184B pub key\n1088B ciphertext\n32B shared secret", "TLS 1.3 key exchange, session keys"),
    ("ML-DSA-65", "FIPS 204", "Digital Signatures", "1952B pub key\n3293B signature\n<1ms sign/verify", "SWIFT message signing, audit trails"),
    ("SLH-DSA", "FIPS 205", "Hash-Based Signatures", "Stateless design\nSPHINCS+ based\nUltra-conservative", "Long-term document signing"),
    ("Hybrid KEM", "X25519 + ML-KEM", "Backward Compatible", "Classical + PQC\nTLS 1.3 ready\nCNSA 2.0 compliant", "Production TLS, gradual migration"),
]

for i, (name, standard, category, specs, use_case) in enumerate(algos):
    x = Inches(0.5 + i * 3.1)
    card = add_shape(slide, x, Inches(2.0), Inches(2.9), Inches(5.0), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    add_text(slide, x + Inches(0.2), Inches(2.2), Inches(2.5), Inches(0.4), name, size=20, color=ACCENT, bold=True)
    add_text(slide, x + Inches(0.2), Inches(2.65), Inches(2.5), Inches(0.3), standard, size=12, color=GREEN, bold=True)
    add_text(slide, x + Inches(0.2), Inches(3.0), Inches(2.5), Inches(0.3), category, size=14, color=WHITE, bold=True)
    add_text(slide, x + Inches(0.2), Inches(3.5), Inches(2.5), Inches(1.5), specs, size=12, color=DIM)
    add_text(slide, x + Inches(0.2), Inches(5.2), Inches(2.5), Inches(1.2), f"Use: {use_case}", size=12, color=GREEN)


# ═══════════════════════════════════════════════════════════
# SLIDE 6: WHY QBITEL IS BEST — COMPETITIVE DIFFERENTIATION
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "Competitive Advantage")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "Why QBITEL Bridge is the Best Choice", size=40, color=WHITE, bold=True)

advantages = [
    ("Only Platform Combining All Three", "AI protocol discovery + quantum-safe crypto + autonomous security — no other vendor offers all three in a single solution.", ACCENT),
    ("No Rip-and-Replace", "Legacy systems remain operational. We protect at the network layer — zero code changes, zero downtime. Competitors require full rewrites.", GREEN),
    ("100% Open Source (Apache 2.0)", "No open-core trap. No feature gating. Full enterprise capability in the free release. Same license as Kubernetes and Kafka.", GREEN),
    ("Air-Gapped / On-Premise LLM", "Runs entirely on-premise with local Ollama models. No cloud dependency. Critical for defense, banking, and critical infrastructure.", ACCENT),
    ("2-4 Hour Protocol Discovery", "What takes consultants 6-12 months and $2-10M, our AI does in hours from raw network traffic — fully automated.", GREEN),
    ("<1 Second Autonomous Response", "78% of security incidents handled autonomously in under 1 second. SOC teams average 65 minutes. 900x faster.", ACCENT),
]

for i, (title, desc, color) in enumerate(advantages):
    col = i % 2
    row = i // 2
    x = Inches(0.6 + col * 6.2)
    y = Inches(2.0 + row * 1.7)
    card = add_shape(slide, x, y, Inches(5.9), Inches(1.5), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    add_text(slide, x + Inches(0.25), y + Inches(0.15), Inches(5.4), Inches(0.4), title, size=17, color=color, bold=True)
    add_text(slide, x + Inches(0.25), y + Inches(0.6), Inches(5.4), Inches(0.8), desc, size=13, color=DIM)


# ═══════════════════════════════════════════════════════════
# SLIDE 7: HEAD-TO-HEAD COMPARISON
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "Market Comparison")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "QBITEL vs. The Competition", size=40, color=WHITE, bold=True)

# Table header
headers = ["Capability", "QBITEL Bridge", "Traditional\nSecurity Vendors", "Legacy\nModernization\nConsultants", "PQC-Only\nVendors"]
col_widths = [Inches(3.0), Inches(2.5), Inches(2.5), Inches(2.5), Inches(2.5)]
col_starts = [Inches(0.5)]
for w in col_widths[:-1]:
    col_starts.append(col_starts[-1] + w)

# Header row
for i, (hdr, x, w) in enumerate(zip(headers, col_starts, col_widths)):
    bg_c = ACCENT if i == 1 else BG_CARD
    card = add_shape(slide, x, Inches(2.0), w - Inches(0.08), Inches(0.7), bg_c, border_color=RGBColor(0x1E, 0x29, 0x3B))
    tc = BG_DARK if i == 1 else WHITE
    add_text(slide, x + Inches(0.1), Inches(2.05), w - Inches(0.2), Inches(0.65), hdr, size=12, color=tc, bold=True, align=PP_ALIGN.CENTER)

# Data rows
rows = [
    ("AI Protocol Discovery", "YES  (2-4 hrs)", "NO", "Manual  (6-12 mo)", "NO"),
    ("Post-Quantum Crypto", "NIST Level 5", "Classical only", "NO", "YES"),
    ("Legacy System Protection", "Zero downtime", "Requires changes", "Rip & replace", "NO"),
    ("Autonomous Response", "78% auto, <1s", "Manual SOC", "NO", "NO"),
    ("Compliance Automation", "9 frameworks", "Partial (2-3)", "Manual audits", "NO"),
    ("Open Source", "Apache 2.0", "Proprietary", "Proprietary", "Mixed"),
    ("Air-Gapped Capability", "Full on-premise", "Cloud required", "N/A", "Partial"),
    ("Cost per System", "$200K-500K", "$1-5M", "$5-50M", "$500K-2M"),
]

for ri, (cap, *vals) in enumerate(rows):
    y = Inches(2.75 + ri * 0.55)
    # Row bg
    row_bg = BG_CARD if ri % 2 == 0 else BG_DARK
    for ci, (val, x, w) in enumerate(zip([cap] + list(vals), col_starts, col_widths)):
        card = add_shape(slide, x, y, w - Inches(0.08), Inches(0.48), row_bg)
        if ci == 0:
            add_text(slide, x + Inches(0.1), y + Inches(0.05), w - Inches(0.2), Inches(0.4), val, size=11, color=WHITE, bold=True, align=PP_ALIGN.LEFT)
        elif ci == 1:
            add_text(slide, x + Inches(0.1), y + Inches(0.05), w - Inches(0.2), Inches(0.4), val, size=11, color=GREEN, bold=True, align=PP_ALIGN.CENTER)
        else:
            c = RED if val in ("NO", "N/A") else ORANGE if "Manual" in val or "Partial" in val or "Classical" in val or "Requires" in val or "Rip" in val or "Cloud" in val else DIM
            add_text(slide, x + Inches(0.1), y + Inches(0.05), w - Inches(0.2), Inches(0.4), val, size=11, color=c, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════
# SLIDE 8: BUSINESS IMPACT METRICS
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "Business Impact")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "Measurable Results", size=40, color=WHITE, bold=True)

impacts = [
    ("6-12 months", "2-4 hours", "Protocol Discovery"),
    ("None", "NIST Level 5", "Quantum Readiness"),
    ("65 minutes", "<1 second", "Security Response"),
    ("2-4 weeks", "<10 minutes", "Compliance Reports"),
    ("$5-50M / system", "$200K-500K", "Integration Cost"),
    ("$10-50 / event", "<$0.01", "Security Cost / Event"),
]

for i, (before, after, label) in enumerate(impacts):
    col = i % 3
    row = i // 3
    x = Inches(0.6 + col * 4.1)
    y = Inches(2.0 + row * 2.6)
    card = add_shape(slide, x, y, Inches(3.7), Inches(2.3), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    add_text(slide, x + Inches(0.2), y + Inches(0.25), Inches(3.3), Inches(0.4), before, size=18, color=RED, align=PP_ALIGN.CENTER)
    # Strikethrough line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x + Inches(0.8), y + Inches(0.52), Inches(2.1), Pt(2))
    line.fill.solid(); line.fill.fore_color.rgb = RED; line.line.fill.background()
    add_text(slide, x + Inches(0.2), y + Inches(0.75), Inches(3.3), Inches(0.7), after, size=32, color=GREEN, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.2), y + Inches(1.6), Inches(3.3), Inches(0.4), label, size=14, color=DIM, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════
# SLIDE 9: COMPLIANCE AUTOMATION
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "Compliance")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "9 Regulatory Frameworks Automated", size=40, color=WHITE, bold=True)

frameworks = [
    ("PCI-DSS 4.0", "Payment card security", "94%"),
    ("DORA (EU)", "Digital resilience", "91%"),
    ("SOX", "Financial controls", "97%"),
    ("NIST 800-53", "Federal security", "89%"),
    ("HIPAA", "Healthcare data", "92%"),
    ("ISO 27001", "InfoSec management", "95%"),
    ("BASEL III/IV", "Banking risk", "90%"),
    ("NERC-CIP", "Critical infrastructure", "88%"),
    ("FDA 21 CFR 11", "Medical devices", "93%"),
]

for i, (name, desc, score) in enumerate(frameworks):
    col = i % 3
    row = i // 3
    x = Inches(0.6 + col * 4.1)
    y = Inches(2.0 + row * 1.6)
    card = add_shape(slide, x, y, Inches(3.7), Inches(1.35), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    add_text(slide, x + Inches(0.25), y + Inches(0.15), Inches(2.4), Inches(0.35), name, size=17, color=ACCENT, bold=True)
    add_text(slide, x + Inches(0.25), y + Inches(0.5), Inches(2.4), Inches(0.3), desc, size=12, color=DIM)
    add_text(slide, x + Inches(2.5), y + Inches(0.2), Inches(1.0), Inches(0.5), score, size=28, color=GREEN, bold=True, align=PP_ALIGN.RIGHT)
    # Bar
    bar_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x + Inches(0.25), y + Inches(0.95), Inches(3.2), Pt(6))
    bar_bg.fill.solid(); bar_bg.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B); bar_bg.line.fill.background()
    pct = int(score.replace('%', '')) / 100
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x + Inches(0.25), y + Inches(0.95), Inches(3.2 * pct), Pt(6))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background()

add_text(slide, Inches(0.8), Inches(7.0), Inches(11), Inches(0.3), "Reports generated in <10 minutes  •  98%+ audit pass rate  •  Continuous monitoring with agent AGT-003", size=13, color=DARK_TEXT, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════
# SLIDE 10: DOMAIN COVERAGE
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "Industry Coverage")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "Serving 9 Critical Sectors", size=40, color=WHITE, bold=True)

domains = [
    ("Banking & Finance", "ISO-8583, SWIFT, ACH, FedWire, FedNow", "10,000+ TPS, <50ms latency"),
    ("Healthcare", "HL7, DICOM, FHIR", "500K+ connected devices"),
    ("Critical Infrastructure", "Modbus, DNP3, IEC 61850", "100M+ population protected"),
    ("Automotive", "V2X, IEEE 1609.2, CAN", "<10ms real-time signatures"),
    ("Aviation", "ADS-B, ACARS, ARINC 429", "600bps-2.4kbps optimized"),
    ("Telecommunications", "SS7, Diameter, 5G Core", "Carrier-grade PQC"),
    ("Defense / Government", "CNSA 2.0, NSA Suite B", "Air-gapped TOP SECRET"),
    ("Insurance", "ACORD, ISO 20022", "Claims & policy protection"),
    ("BPO / Call Centers", "SIP, IVR, DTMF, CTI", "PCI voice security"),
]

for i, (name, protocols, metric) in enumerate(domains):
    col = i % 3
    row = i // 3
    x = Inches(0.5 + col * 4.15)
    y = Inches(2.0 + row * 1.7)
    card = add_shape(slide, x, y, Inches(3.8), Inches(1.5), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    add_text(slide, x + Inches(0.2), y + Inches(0.12), Inches(3.4), Inches(0.35), name, size=16, color=ACCENT, bold=True)
    add_text(slide, x + Inches(0.2), y + Inches(0.5), Inches(3.4), Inches(0.3), protocols, size=11, color=DIM)
    add_text(slide, x + Inches(0.2), y + Inches(0.9), Inches(3.4), Inches(0.3), metric, size=12, color=GREEN, bold=True)


# ═══════════════════════════════════════════════════════════
# SLIDE 11: LIVE DEMO OVERVIEW
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_label(slide, Inches(0.8), Inches(0.5), "Live Demo")
add_text(slide, Inches(0.8), Inches(0.9), Inches(11), Inches(0.8), "End-to-End Product Walkthrough", size=40, color=WHITE, bold=True)

demo_steps = [
    ("1", "Network Discovery", "Scan 10.1.50.0/24, discover 3 legacy mainframes, 7 unencrypted channels"),
    ("2", "Traffic Capture", "Capture EBCDIC packets, reverse-engineer field boundaries, detect PII exposure"),
    ("3", "COBOL Analysis", "Analyze CUSTMAST.cbl (1985) + ACCTPROC.cbl (1988) — complexity, patterns, opportunities"),
    ("4", "PQC Protection", "ML-KEM-768 keygen, encrypt SWIFT MT103 wire transfer, Dilithium-3 signing"),
    ("5", "Code Generation", "COBOL → Python dataclasses + FastAPI endpoints + SQL schemas"),
    ("6", "Security Monitor", "Live dashboard: 15M txn/day, threat timeline, 5-agent orchestration"),
    ("7", "Compliance Report", "PCI-DSS 94%, DORA 91%, SOX 97%, NIST 89%, HIPAA 92%"),
    ("8", "Modernization Plan", "Multi-phase roadmap, risk assessment, effort estimation, go-live date"),
]

for i, (num, title, desc) in enumerate(demo_steps):
    col = i % 2
    row = i // 2
    x = Inches(0.6 + col * 6.2)
    y = Inches(2.0 + row * 1.25)
    card = add_shape(slide, x, y, Inches(5.9), Inches(1.05), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.15), y + Inches(0.25), Inches(0.5), Inches(0.5))
    circle.fill.solid(); circle.fill.fore_color.rgb = ACCENT; circle.line.fill.background()
    add_text(slide, x + Inches(0.15), y + Inches(0.28), Inches(0.5), Inches(0.45), num, size=18, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.8), y + Inches(0.12), Inches(4.8), Inches(0.35), title, size=17, color=WHITE, bold=True)
    add_text(slide, x + Inches(0.8), y + Inches(0.5), Inches(4.8), Inches(0.45), desc, size=12, color=DIM)

add_text(slide, Inches(0.8), Inches(7.0), Inches(11), Inches(0.3), "Demo URL:  http://localhost:8001/e2e-demo   |   Start:  python run_demo.py --server", size=14, color=ACCENT, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════
# SLIDE 12: CALL TO ACTION
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

# Accent line
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.46), Inches(13.333), Pt(4))
shape.fill.solid(); shape.fill.fore_color.rgb = ACCENT; shape.line.fill.background()

add_text(slide, Inches(1.5), Inches(1.5), Inches(10), Inches(0.5), "NEXT STEPS", size=14, color=ACCENT, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, Inches(1.5), Inches(2.0), Inches(10), Inches(1.0), "Start Your Proof of Concept", size=48, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, Inches(2.5), Inches(3.2), Inches(8), Inches(0.8), "2-week PoC. Connect to your test environment.\nFull protocol discovery and modernization assessment.", size=20, color=DIM, align=PP_ALIGN.CENTER)

# Feature badges
features = ["100% Open Source", "Air-Gapped Ready", "Zero Code Changes", "Apache 2.0 License", "9 Compliance Frameworks"]
for i, f in enumerate(features):
    x = Inches(1.3 + i * 2.2)
    badge = add_shape(slide, x, Inches(4.5), Inches(2.0), Inches(0.5), BG_CARD, border_color=RGBColor(0x1E, 0x29, 0x3B))
    add_text(slide, x, Inches(4.53), Inches(2.0), Inches(0.45), f, size=11, color=DIM, align=PP_ALIGN.CENTER)

# CTA button
cta = add_shape(slide, Inches(4.5), Inches(5.5), Inches(4.3), Inches(0.8), ACCENT)
add_text(slide, Inches(4.5), Inches(5.55), Inches(4.3), Inches(0.7), "Contact Us", size=24, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)

add_text(slide, Inches(1.5), Inches(6.6), Inches(10), Inches(0.4), "enterprise@qbitel.com", size=18, color=ACCENT, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "QBITEL_Mainframe_Modernization_Presentation.pptx")
prs.save(output_path)
print(f"Presentation saved to: {output_path}")
print(f"Slides: {len(prs.slides)}")
