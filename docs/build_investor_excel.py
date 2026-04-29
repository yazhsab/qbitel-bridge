"""
QBITEL India Investor Pitch — Excel Generator
Enterprise Theme: Navy Blue (#0D1B40), Gold (#C9A84C), White (#FFFFFF), Light Grey (#F4F6FA)
"""

from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, LineChart, Reference, PieChart
from openpyxl.chart.series import DataPoint
from openpyxl.drawing.image import Image
from openpyxl.chart.label import DataLabelList
import os

wb = Workbook()

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
NAVY        = "0D1B40"   # primary dark background
NAVY_MID    = "16295E"   # secondary header
NAVY_LIGHT  = "1E3A73"   # section header rows
GOLD        = "C9A84C"   # accent / highlight
GOLD_LIGHT  = "F0D07A"   # light gold for alternates
WHITE       = "FFFFFF"
LIGHT_GREY  = "F4F6FA"
MID_GREY    = "D9DDE8"
DARK_GREY   = "6B7280"
GREEN       = "1A7F4B"   # positive numbers
RED         = "C0392B"   # negative numbers
GREEN_BG    = "D4EFDF"
RED_BG      = "FADBD8"
GOLD_BG     = "FEF9EC"

# ─── FILL HELPERS ────────────────────────────────────────────────────────────
def fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def border(style="thin", color="C9A84C"):
    s = Side(style=style, color=color)
    return Border(left=s, right=s, top=s, bottom=s)

def bottom_border(color=GOLD):
    s = Side(style="medium", color=color)
    return Border(bottom=s)

def thin_border(color=MID_GREY):
    s = Side(style="thin", color=color)
    return Border(left=s, right=s, top=s, bottom=s)

# ─── FONT HELPERS ────────────────────────────────────────────────────────────
def hdr_font(size=11, color=WHITE, bold=True):
    return Font(name="Calibri", size=size, color=color, bold=bold)

def body_font(size=10, color="1F2937", bold=False):
    return Font(name="Calibri", size=size, color=color, bold=bold)

def gold_font(size=10, bold=True):
    return Font(name="Calibri", size=size, color=GOLD, bold=bold)

def navy_font(size=10, bold=True):
    return Font(name="Calibri", size=size, color=NAVY, bold=bold)

# ─── ALIGNMENT HELPERS ───────────────────────────────────────────────────────
CENTER  = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT    = Alignment(horizontal="left",   vertical="center", wrap_text=True)
RIGHT   = Alignment(horizontal="right",  vertical="center")
CENTER_TOP = Alignment(horizontal="center", vertical="top", wrap_text=True)

# ─── UTILITY: write a styled cell ────────────────────────────────────────────
def wc(ws, row, col, value, font=None, fill_=None, align=None, border_=None, num_fmt=None):
    cell = ws.cell(row=row, column=col, value=value)
    if font:    cell.font      = font
    if fill_:   cell.fill      = fill_
    if align:   cell.alignment = align
    if border_: cell.border    = border_
    if num_fmt: cell.number_format = num_fmt
    return cell

def merge_title(ws, row, col_start, col_end, value, bg=NAVY, fg=WHITE, size=13, align=CENTER):
    ws.merge_cells(start_row=row, start_column=col_start, end_row=row, end_column=col_end)
    cell = ws.cell(row=row, column=col_start, value=value)
    cell.font      = Font(name="Calibri", size=size, color=fg, bold=True)
    cell.fill      = fill(bg)
    cell.alignment = align
    return cell

def shade_row(ws, row, cols, hex_color, font_=None):
    for c in range(cols[0], cols[1]+1):
        ws.cell(row=row, column=c).fill = fill(hex_color)
        if font_:
            ws.cell(row=row, column=c).font = font_

def set_col_widths(ws, widths):
    for col_idx, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = w

def freeze(ws, cell="A3"):
    ws.freeze_panes = cell

def add_section_gap(ws, row, ncols=10):
    ws.row_dimensions[row].height = 8
    for c in range(1, ncols+1):
        ws.cell(row=row, column=c).fill = fill(LIGHT_GREY)

# ─── SHEET 1: COVER PAGE ─────────────────────────────────────────────────────
ws1 = wb.active
ws1.title = "Cover"
ws1.sheet_view.showGridLines = False
ws1.sheet_properties.tabColor = NAVY

set_col_widths(ws1, [3, 18, 22, 22, 22, 22, 18, 3])
for r in range(1, 50):
    ws1.row_dimensions[r].height = 18

# Full background
for r in range(1, 50):
    for c in range(1, 9):
        ws1.cell(r, c).fill = fill(NAVY)

# Gold accent line top
ws1.row_dimensions[3].height = 5
for c in range(1, 9):
    ws1.cell(3, c).fill = fill(GOLD)

# Logo / company name block
ws1.merge_cells("B6:G6")
c = ws1["B6"]
c.value = "Q B I T E L"
c.font  = Font(name="Calibri", size=36, bold=True, color=GOLD)
c.alignment = CENTER

ws1.merge_cells("B7:G7")
c = ws1["B7"]
c.value = "B R I D G E"
c.font  = Font(name="Calibri", size=20, bold=False, color=WHITE)
c.alignment = CENTER

ws1.row_dimensions[8].height = 5
for c in range(2, 8):
    ws1.cell(8, c).fill = fill(GOLD)

ws1.merge_cells("B10:G10")
c = ws1["B10"]
c.value = "AI-Powered Quantum-Safe Security Platform"
c.font  = Font(name="Calibri", size=15, bold=False, color=GOLD_LIGHT, italic=True)
c.alignment = CENTER

ws1.merge_cells("B12:G12")
c = ws1["B12"]
c.value = "INDIA INVESTOR PITCH  |  SERIES A  |  MARCH 2026"
c.font  = Font(name="Calibri", size=12, bold=True, color=WHITE)
c.alignment = CENTER

# Key numbers box
kv_data = [
    ("INVESTMENT ASK",   "₹20 Crore",           "PRE-MONEY VALUATION", "₹60 Crore"),
    ("POST-MONEY",       "₹80 Crore",           "INVESTOR OWNERSHIP",   "25%"),
    ("RUNWAY",           "22 Months",            "BREAK-EVEN",          "Month 19"),
    ("YEAR 2 ARR",       "₹19.2 Crore",          "YEAR 2 EBITDA",       "+₹3.64 Crore"),
    ("BASE CASE EXIT",   "₹600 Crore",           "INVESTOR MOIC",       "5.4×"),
]
for i, (l1, v1, l2, v2) in enumerate(kv_data):
    r = 15 + i*3
    ws1.row_dimensions[r].height   = 13
    ws1.row_dimensions[r+1].height = 20
    ws1.row_dimensions[r+2].height = 6

    for col, label, val, bg in [(2, l1, v1, NAVY_MID), (5, l2, v2, NAVY_MID)]:
        ws1.merge_cells(start_row=r, start_column=col, end_row=r, end_column=col+2)
        lc = ws1.cell(r, col, label)
        lc.font = Font(name="Calibri", size=8, bold=True, color=GOLD_LIGHT)
        lc.alignment = CENTER
        lc.fill = fill(NAVY_LIGHT)

        ws1.merge_cells(start_row=r+1, start_column=col, end_row=r+1, end_column=col+2)
        vc = ws1.cell(r+1, col, val)
        vc.font = Font(name="Calibri", size=16, bold=True, color=WHITE)
        vc.alignment = CENTER
        vc.fill = fill(NAVY_MID)

# One-liner
ws1.row_dimensions[34].height = 30
ws1.merge_cells("B34:G34")
c = ws1["B34"]
c.value = (
    '"The only platform protecting India\'s banking mainframes, hospital networks,\n'
    'and power grids from quantum-era cyberattacks — using autonomous AI."'
)
c.font  = Font(name="Calibri", size=10, italic=True, color=GOLD_LIGHT)
c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
c.fill = fill(NAVY_LIGHT)

# Footer
ws1.row_dimensions[44].height = 5
for c in range(1, 9):
    ws1.cell(44, c).fill = fill(GOLD)

ws1.merge_cells("B46:G46")
c = ws1["B46"]
c.value = "CONFIDENTIAL  |  © 2026 QBITEL  |  enterprise@qbitel.com  |  Exchange Rate: 1 USD = ₹84"
c.font  = Font(name="Calibri", size=8, color=DARK_GREY)
c.alignment = CENTER
c.fill = fill(NAVY)

# ─── SHEET 2: EXECUTIVE SUMMARY ──────────────────────────────────────────────
ws2 = wb.create_sheet("Executive Summary")
ws2.sheet_view.showGridLines = False
ws2.sheet_properties.tabColor = NAVY_MID
set_col_widths(ws2, [2, 28, 20, 20, 20, 20, 2])
freeze(ws2, "B3")

merge_title(ws2, 1, 2, 6, "QBITEL BRIDGE — EXECUTIVE SUMMARY", NAVY, WHITE, 14)
merge_title(ws2, 2, 2, 6, "India Series A  |  ₹20 Crore  |  March 2026", NAVY_MID, GOLD_LIGHT, 10)

# Deal snapshot
r = 4
merge_title(ws2, r, 2, 6, "DEAL SNAPSHOT", NAVY_LIGHT, GOLD, 11)
snap = [
    ("Company",              "QBITEL"),
    ("Product",              "QBITEL Bridge — AI-Powered Quantum-Safe Security Platform"),
    ("Stage",                "Series A (India-First)"),
    ("Investment Ask",       "₹20,00,00,000  (₹20 Crore)"),
    ("Pre-Money Valuation",  "₹60,00,00,000  (₹60 Crore)"),
    ("Post-Money Valuation", "₹80,00,00,000  (₹80 Crore)"),
    ("Investor Ownership",   "25%"),
    ("Instrument",           "CCPS — Compulsorily Convertible Preference Shares"),
    ("Runway",               "22 months"),
    ("Break-even Month",     "Month 19  (Q1 FY2028)"),
    ("Primary Market",       "India — Banking, Healthcare, Critical Infrastructure, Telecom"),
    ("Expansion Markets",    "UAE, Saudi Arabia, Singapore (Year 2)"),
]
for i, (label, val) in enumerate(snap):
    row = r + 1 + i
    ws2.row_dimensions[row].height = 22
    bg = LIGHT_GREY if i % 2 == 0 else WHITE
    wc(ws2, row, 2, label, font=navy_font(10, True),  fill_=fill(bg), align=LEFT,   border_=thin_border())
    ws2.merge_cells(start_row=row, start_column=3, end_row=row, end_column=6)
    wc(ws2, row, 3, val,   font=body_font(10, "1F2937"), fill_=fill(bg), align=LEFT, border_=thin_border())

# Why Now
r2 = r + len(snap) + 2
add_section_gap(ws2, r2-1, 7)
merge_title(ws2, r2, 2, 6, "WHY NOW — INDIA MARKET CATALYSTS", NAVY_LIGHT, GOLD, 11)
catalysts = [
    ("RBI IT Master Direction 2024",      "All scheduled commercial banks",          "Quantum readiness assessments mandatory",   "April 2025"),
    ("CERT-In Incident Reporting Rules",  "All critical infrastructure operators",   "Auto-reporting <6 hours",                  "In effect"),
    ("SEBI Cybersecurity Circular 2023",  "Stock brokers, depositories, exchanges",  "Protocol-layer encryption mandated",        "Jan 2025"),
    ("IRDAI Cybersecurity Guidelines",    "All insurance companies",                 "Data encryption across policy systems",     "2025"),
    ("ABDM Security Standards",           "All hospitals and health-tech platforms", "PHI post-quantum encryption",               "2025"),
    ("DPDP Act (Data Privacy)",           "All digital enterprises",                 "Data residency + encryption requirements",  "2025-26"),
    ("TRAI Telecom Security Rules",       "All telecom operators",                   "SS7, 5G core security",                    "2025"),
]
hdr = ["Regulation / Mandate", "Target Organisations", "QBITEL Fit", "Deadline"]
for c_i, h in enumerate(hdr):
    ws2.row_dimensions[r2+1].height = 22
    wc(ws2, r2+1, c_i+2, h, font=hdr_font(10, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())
for i, row_data in enumerate(catalysts):
    row = r2 + 2 + i
    ws2.row_dimensions[row].height = 22
    bg = LIGHT_GREY if i % 2 == 0 else WHITE
    for c_i, val in enumerate(row_data):
        f = body_font(9)
        if c_i == 0: f = body_font(9, bold=True)
        if c_i == 3: f = Font(name="Calibri", size=9, color=RED, bold=True)
        wc(ws2, row, c_i+2, val, font=f, fill_=fill(bg), align=LEFT, border_=thin_border())

# The 3-year Financial Summary mini-table
r3 = r2 + len(catalysts) + 3
add_section_gap(ws2, r3-1, 7)
merge_title(ws2, r3, 2, 6, "3-YEAR FINANCIAL SNAPSHOT  (₹ Crore)", NAVY_LIGHT, GOLD, 11)
fin_hdr = ["Metric", "FY2026 (Year 1)", "FY2027 (Year 2)", "FY2028 (Year 3)"]
fin_rows = [
    ("Ending ARR",        "₹7.20 Crore",   "₹19.20 Crore",  "₹40.00 Crore"),
    ("Total Revenue",     "₹4.68 Crore",   "₹20.00 Crore",  "₹42.00 Crore"),
    ("Gross Margin",      "65.7%",          "69.7%",          "72.0%"),
    ("EBITDA",            "(₹7.08 Crore)", "+₹3.64 Crore",  "+₹11.00 Crore"),
    ("Customers",         "6",              "16",             "36"),
    ("NRR",               "110%",           "128%",           "135%"),
    ("ARR Growth YoY",    "—",              "167%",           "108%"),
]
for c_i, h in enumerate(fin_hdr):
    ws2.row_dimensions[r3+1].height = 22
    wc(ws2, r3+1, c_i+2, h, font=hdr_font(10, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())
for i, row_data in enumerate(fin_rows):
    row = r3 + 2 + i
    ws2.row_dimensions[row].height = 20
    bg = GOLD_BG if i % 2 == 0 else WHITE
    for c_i, val in enumerate(row_data):
        is_neg = "(" in str(val)
        is_pos = "+" in str(val)
        f = body_font(10)
        if c_i == 0: f = navy_font(10)
        if is_neg: f = Font(name="Calibri", size=10, color=RED, bold=True)
        if is_pos: f = Font(name="Calibri", size=10, color=GREEN, bold=True)
        wc(ws2, row, c_i+2, val, font=f, fill_=fill(bg), align=CENTER, border_=thin_border())

# ─── SHEET 3: INVESTMENT BREAKDOWN ───────────────────────────────────────────
ws3 = wb.create_sheet("Investment Breakdown")
ws3.sheet_view.showGridLines = False
ws3.sheet_properties.tabColor = GOLD
set_col_widths(ws3, [2, 36, 15, 15, 12, 15, 2])
freeze(ws3, "B4")

merge_title(ws3, 1, 2, 6, "INVESTMENT BREAKDOWN — ₹20 CRORE  (FULLY ITEMISED)", NAVY, WHITE, 14)
merge_title(ws3, 2, 2, 6, "Every Rupee Accounted For  |  22-Month Plan  |  FY2026 – FY2027", NAVY_MID, GOLD_LIGHT, 10)

col_hdrs = ["Line Item", "Year 1 (₹)", "Year 2 (₹)", "Total (₹)", "% of Total"]
ws3.row_dimensions[3].height = 24
for c_i, h in enumerate(col_hdrs):
    wc(ws3, 3, c_i+2, h, font=hdr_font(11, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())

buckets = [
    {
        "name": "BUCKET 1 — PRODUCT & ENGINEERING  (40% | ₹8 Crore)",
        "color": NAVY_LIGHT,
        "items": [
            ("Engineering team salaries — 15 engineers × 22 months (avg ₹22 LPA incl PF/gratuity)", 20000000, 15000000, 35000000),
            ("AI/ML model training compute — AWS India / Azure India GPU instances",                  5500000,  3500000,  9000000),
            ("Post-quantum cryptography engine (Kyber-1024, Dilithium-5, SPHINCS+)",                  3000000,  2000000,  5000000),
            ("Protocol Marketplace — 300 Indian protocols (UPI, RTGS, NEFT, HL7, DICOM, Modbus)",    3500000,  2500000,  6000000),
            ("Translation Studio — REST API auto-generation, 6 SDK targets",                          2000000,  1500000,  3500000),
            ("Domain modules — Banking COBOL, Healthcare ABDM, OT SCADA, Telecom SS7/5G",            4500000,  3500000,  8000000),
            ("QA, penetration testing, DevSecOps tooling (CI/CD, SAST, DAST)",                       2500000,  2500000,  5000000),
            ("Cloud infrastructure & co-location — Mumbai + Hyderabad (prod + DR)",                   6000000,  4000000, 10000000),
        ],
        "subtotal": (47000000, 34000000, 80000000, 40.0),
    },
    {
        "name": "BUCKET 2 — GO-TO-MARKET  (25% | ₹5 Crore)",
        "color": NAVY_MID,
        "items": [
            ("Enterprise Account Executives — 3 AEs in Mumbai / Delhi / Bengaluru (₹30L base + commission)", 10000000, 6500000, 16500000),
            ("Solution Engineers / Pre-Sales — 2 people (₹25 LPA)",                                          5500000,  3700000,  9200000),
            ("Sales Development Reps — 2 SDRs (₹15 LPA base)",                                               3300000,  2200000,  5500000),
            ("VP Sales / Sales Head — ex-Cisco / Palo Alto India profile (₹50 LPA)",                         5500000,  3700000,  9200000),
            ("Marketing: brand, content, digital, BFSI events (NBFC Summit, Fintech Conclave)",              3500000,  2500000,  6000000),
            ("Customer Success Manager — 1 person (₹18 LPA)",                                                2000000,  1300000,  3300000),
            ("Partner / channel development — MSSP onboarding, SI partnerships",                              1700000,  1100000,  2800000),
            ("CRM, sales tooling, LinkedIn Sales Nav, Bombora intent data",                                    900000,   600000,  1500000),
            ("Travel and customer entertainment — India field sales",                                         1800000,  1200000,  3000000),
            ("Free Quantum Risk Assessment delivery — 12 pilots (passive tap hardware + labour)",             1800000,  1200000,  3000000),
        ],
        "subtotal": (36000000, 24000000, 50000000, 25.0),
    },
    {
        "name": "BUCKET 3 — CERTIFICATIONS & COMPLIANCE  (20% | ₹4 Crore)",
        "color": NAVY_LIGHT,
        "items": [
            ("CERT-In Empanelment — Information Security Auditing Organisation",                  3000000, 2000000, 5000000),
            ("ISO 27001:2022 Certification — BIS-accredited audit + implementation",              1800000, 1200000, 3000000),
            ("SOC 2 Type I & II — Required for UAE and Singapore expansion",                      4500000, 3000000, 7500000),
            ("STQC / MeitY Government IT Product Registration",                                   2400000, 1600000, 4000000),
            ("RBI Cybersecurity Framework — Third-party audit for banking pilots",                1500000, 1000000, 2500000),
            ("SEBI Cybersecurity & Cyber Resilience Framework compliance audit",                  1200000,  800000, 2000000),
            ("IRDAI Cybersecurity Guidelines compliance audit",                                    900000,  600000, 1500000),
            ("NPCI integration certification — UPI, IMPS, NACH payment rail",                    1800000, 1200000, 3000000),
            ("GeM Government e-Marketplace product registration",                                  300000,  200000,  500000),
            ("HSM integration validation — Sify, NeST, Thales India",                            1500000, 1000000, 2500000),
            ("Legal: IP — 3 provisional patents (Indian Patents Act) + PCT filing",              2400000, 1600000, 4000000),
            ("Legal: DPDP Act compliance, customer contracts, NDA templates",                     1200000,  800000, 2000000),
            ("Legal: Company secretarial, board compliance, FEMA (foreign VC)",                   1500000, 1000000, 2500000),
            ("International compliance — UAE CBUAE, Singapore MAS (Year 2 reserve)",                    0, 8000000, 8000000),
        ],
        "subtotal": (24000000, 16000000, 40000000, 20.0),
    },
    {
        "name": "BUCKET 4 — OPERATIONS & WORKING CAPITAL  (15% | ₹3 Crore)",
        "color": NAVY_MID,
        "items": [
            ("Office rent — Bengaluru HQ 3,000 sq ft @ ₹45/sqft/month (Electronic City)",  3300000, 2200000, 5400000),
            ("Office rent — Mumbai sales office 1,000 sq ft @ ₹45/sqft/month (BKC)",       2000000, 1300000, 3300000),
            ("Office fit-out, workstations, secure PQC testing lab",                        3000000,       0, 3000000),
            ("Finance & accounting — CFO-as-a-service retainer (₹12K/month)",               1300000,  900000, 2200000),
            ("HR and payroll administration — HRMS + processing",                            900000,  600000, 1500000),
            ("Business insurance — D&O, cyber liability, professional indemnity",            750000,  450000, 1200000),
            ("IT procurement — 26-person team laptops + secure hardware",                   2500000,       0, 2500000),
            ("Recruitment fees — senior hires via agencies (8–10% of annual CTC)",          1200000,  800000, 2000000),
            ("Contingency reserve — 2% of total raise",                                     2400000,  500000, 3900000),
        ],
        "subtotal": (17350000, 6750000, 30000000, 15.0),
    },
]

current_row = 4
grand_total_y1 = grand_total_y2 = grand_total = 0

for bucket in buckets:
    ws3.row_dimensions[current_row].height = 26
    ws3.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=6)
    c = ws3.cell(current_row, 2, bucket["name"])
    c.font      = Font(name="Calibri", size=11, bold=True, color=GOLD)
    c.fill      = fill(bucket["color"])
    c.alignment = LEFT
    current_row += 1

    for i, (desc, y1, y2, total) in enumerate(bucket["items"]):
        ws3.row_dimensions[current_row].height = 20
        bg = LIGHT_GREY if i % 2 == 0 else WHITE
        wc(ws3, current_row, 2, desc,  font=body_font(9),   fill_=fill(bg), align=LEFT,   border_=thin_border())
        wc(ws3, current_row, 3, y1,    font=body_font(9),   fill_=fill(bg), align=RIGHT,  border_=thin_border(), num_fmt='₹#,##0')
        wc(ws3, current_row, 4, y2,    font=body_font(9),   fill_=fill(bg), align=RIGHT,  border_=thin_border(), num_fmt='₹#,##0')
        wc(ws3, current_row, 5, total, font=body_font(9),   fill_=fill(bg), align=RIGHT,  border_=thin_border(), num_fmt='₹#,##0')
        pct = total / 200000000 * 100
        wc(ws3, current_row, 6, f"{pct:.1f}%", font=body_font(9, DARK_GREY), fill_=fill(bg), align=CENTER, border_=thin_border())
        current_row += 1

    # Subtotal row
    st_y1, st_y2, st_tot, st_pct = bucket["subtotal"]
    ws3.row_dimensions[current_row].height = 24
    wc(ws3, current_row, 2, "Subtotal",  font=hdr_font(10, WHITE), fill_=fill(NAVY), align=LEFT,   border_=thin_border())
    wc(ws3, current_row, 3, st_y1,       font=hdr_font(10, GOLD),  fill_=fill(NAVY), align=RIGHT,  border_=thin_border(), num_fmt='₹#,##0')
    wc(ws3, current_row, 4, st_y2,       font=hdr_font(10, GOLD),  fill_=fill(NAVY), align=RIGHT,  border_=thin_border(), num_fmt='₹#,##0')
    wc(ws3, current_row, 5, st_tot,      font=hdr_font(10, GOLD),  fill_=fill(NAVY), align=RIGHT,  border_=thin_border(), num_fmt='₹#,##0')
    wc(ws3, current_row, 6, f"{st_pct:.0f}%", font=hdr_font(10, GOLD_LIGHT), fill_=fill(NAVY), align=CENTER, border_=thin_border())
    grand_total_y1 += st_y1; grand_total_y2 += st_y2; grand_total += st_tot
    current_row += 2  # gap row

# Grand Total
ws3.row_dimensions[current_row].height = 30
for c in range(2, 7):
    ws3.cell(current_row, c).fill = fill(GOLD)
wc(ws3, current_row, 2, "GRAND TOTAL",    font=Font(name="Calibri", size=13, bold=True, color=NAVY), fill_=fill(GOLD), align=LEFT,   border_=thin_border(GOLD))
wc(ws3, current_row, 3, grand_total_y1,   font=Font(name="Calibri", size=12, bold=True, color=NAVY), fill_=fill(GOLD), align=RIGHT,  border_=thin_border(GOLD), num_fmt='₹#,##0')
wc(ws3, current_row, 4, grand_total_y2,   font=Font(name="Calibri", size=12, bold=True, color=NAVY), fill_=fill(GOLD), align=RIGHT,  border_=thin_border(GOLD), num_fmt='₹#,##0')
wc(ws3, current_row, 5, grand_total,      font=Font(name="Calibri", size=12, bold=True, color=NAVY), fill_=fill(GOLD), align=RIGHT,  border_=thin_border(GOLD), num_fmt='₹#,##0')
wc(ws3, current_row, 6, "100%",           font=Font(name="Calibri", size=12, bold=True, color=NAVY), fill_=fill(GOLD), align=CENTER, border_=thin_border(GOLD))

# ─── SHEET 4: YEAR 1 P&L ─────────────────────────────────────────────────────
ws4 = wb.create_sheet("Year 1 — FY2026 P&L")
ws4.sheet_view.showGridLines = False
ws4.sheet_properties.tabColor = NAVY_LIGHT
set_col_widths(ws4, [2, 32, 14, 14, 14, 14, 14, 2])
freeze(ws4, "B4")

merge_title(ws4, 1, 2, 7, "YEAR 1  —  FY2026 PROFIT & LOSS  (April 2026 – March 2027)", NAVY, WHITE, 13)
merge_title(ws4, 2, 2, 7, "Target: 6 Enterprise Customers  |  ₹7.20 Crore Ending ARR  |  65.7% Gross Margin", NAVY_MID, GOLD_LIGHT, 10)

p_hdrs = ["Line Item", "Q1 (Apr–Jun)", "Q2 (Jul–Sep)", "Q3 (Oct–Dec)", "Q4 (Jan–Mar)", "FY2026 Total"]
ws4.row_dimensions[3].height = 24
for c_i, h in enumerate(p_hdrs):
    wc(ws4, 3, c_i+2, h, font=hdr_font(11, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())

sections = [
    {
        "title": "REVENUE",
        "color": NAVY_LIGHT,
        "rows": [
            ("New Customers in Quarter",        1,      2,      2,      1,      6),
            ("Cumulative Customers",            1,      3,      5,      6,      None),
            ("Subscription Revenue",            150000, 587500, 1112500, 1475000, None),
            ("Professional Services Revenue",   225000, 450000,  450000,  225000, None),
        ],
        "total": ("Total Revenue", 375000, 1037500, 1562500, 1700000, None),
        "total_color": NAVY_LIGHT,
    },
    {
        "title": "COST OF REVENUE",
        "color": NAVY_MID,
        "rows": [
            ("Subscription COGS (28%)",          42000,  164500,  311500,  413000, None),
            ("Professional Services COGS (50%)", 112500, 225000,  225000,  112500, None),
        ],
        "total": ("Total COGS", 154500, 389500, 536500, 525500, None),
        "total_color": NAVY_MID,
    },
    {
        "title": "GROSS PROFIT",
        "color": GREEN,
        "rows": [],
        "total": ("Gross Profit", 220500, 648000, 1026000, 1174500, None),
        "total_color": GREEN,
        "pct":   ("Gross Margin %", "58.8%", "62.5%", "65.7%", "69.1%", "65.7%"),
    },
    {
        "title": "OPERATING EXPENSES",
        "color": NAVY_LIGHT,
        "rows": [
            ("Engineering & R&D",              10000000, 10000000,  9500000,  9000000, None),
            ("Sales & Marketing",               6000000,  6500000,  6500000,  6000000, None),
            ("General & Administrative",        4000000,  4000000,  3800000,  3700000, None),
            ("Certifications & Legal",          7500000,  7500000,  5000000,  2500000, None),
        ],
        "total": ("Total Operating Expenses", 27500000, 28000000, 24800000, 21200000, None),
        "total_color": NAVY_LIGHT,
    },
]

y1_revenue    = [375000, 1037500, 1562500, 1700000]
y1_cogs       = [154500,  389500,  536500,  525500]
y1_gp         = [v1 - v2 for v1, v2 in zip(y1_revenue, y1_cogs)]
y1_opex       = [27500000, 28000000, 24800000, 21200000]
y1_ebitda     = [gp - opex for gp, opex in zip(y1_gp, y1_opex)]

current_row = 4
for sec in sections:
    ws4.row_dimensions[current_row].height = 22
    ws4.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=7)
    c = ws4.cell(current_row, 2, sec["title"])
    c.font = Font(name="Calibri", size=10, bold=True, color=GOLD)
    c.fill = fill(sec["color"])
    c.alignment = LEFT
    current_row += 1

    for i, row_data in enumerate(sec.get("rows", [])):
        ws4.row_dimensions[current_row].height = 20
        bg = LIGHT_GREY if i % 2 == 0 else WHITE
        desc = row_data[0]
        vals = list(row_data[1:])
        total_val = sum(v for v in vals[:4] if isinstance(v, (int, float))) if vals[4] is None else vals[4]
        vals[4] = total_val
        wc(ws4, current_row, 2, desc, font=body_font(9), fill_=fill(bg), align=LEFT, border_=thin_border())
        for c_i, v in enumerate(vals[:5]):
            is_money = isinstance(v, float) and abs(v) > 100
            is_int_money = isinstance(v, int) and abs(v) > 100
            fmt = '₹#,##0' if (is_money or is_int_money) else None
            wc(ws4, current_row, c_i+3, v, font=body_font(9), fill_=fill(bg), align=RIGHT if fmt else CENTER, border_=thin_border(), num_fmt=fmt)
        current_row += 1

    # total row
    ws4.row_dimensions[current_row].height = 22
    t = sec["total"]
    tc = sec["total_color"]
    vals = list(t[1:])
    total_val = sum(v for v in vals[:4] if isinstance(v, (int, float)))
    vals[4] = total_val
    wc(ws4, current_row, 2, t[0], font=hdr_font(10, WHITE), fill_=fill(tc), align=LEFT,  border_=thin_border())
    for c_i, v in enumerate(vals[:5]):
        is_neg = isinstance(v, (int, float)) and v < 0
        f_color = RED_BG if is_neg else (GREEN_BG if tc == GREEN else WHITE)
        f_font  = Font(name="Calibri", size=10, bold=True, color=RED if is_neg else (GREEN if tc == GREEN else WHITE))
        wc(ws4, current_row, c_i+3, v, font=f_font, fill_=fill(tc), align=RIGHT, border_=thin_border(), num_fmt='₹#,##0')
    current_row += 1

    if "pct" in sec:
        ws4.row_dimensions[current_row].height = 18
        p = sec["pct"]
        wc(ws4, current_row, 2, p[0], font=body_font(9, DARK_GREY), fill_=fill(LIGHT_GREY), align=LEFT, border_=thin_border())
        for c_i, v in enumerate(p[1:]):
            wc(ws4, current_row, c_i+3, v, font=body_font(9, GREEN, True), fill_=fill(LIGHT_GREY), align=CENTER, border_=thin_border())
        current_row += 1

    current_row += 1  # gap

# EBITDA row
ws4.row_dimensions[current_row].height = 28
ebitda_total = sum(y1_ebitda)
ebitda_row = ["EBITDA"] + y1_ebitda + [ebitda_total]
wc(ws4, current_row, 2, "EBITDA (Net Operating Loss)", font=Font(name="Calibri", size=12, bold=True, color=NAVY), fill_=fill(GOLD), align=LEFT, border_=thin_border(GOLD))
for c_i, v in enumerate(ebitda_row[1:]):
    wc(ws4, current_row, c_i+3, v, font=Font(name="Calibri", size=11, bold=True, color=RED), fill_=fill(GOLD), align=RIGHT, border_=thin_border(GOLD), num_fmt='₹#,##0')

current_row += 2
# KPI Summary
merge_title(ws4, current_row, 2, 7, "KEY METRICS — END OF YEAR 1", NAVY_LIGHT, GOLD, 11)
current_row += 1
kpis = [
    ("Ending ARR",                   "₹7.20 Crore"),
    ("Enterprise Customers",         "6"),
    ("Average ARR per Customer",     "₹1.20 Crore"),
    ("Gross Margin",                 "65.7%"),
    ("Net Cash Burn (Year 1)",       "₹7.08 Crore"),
    ("Cash Remaining from ₹20Cr Raise", "₹12.92 Crore"),
    ("CAC Payback Period",           "5.5 Months"),
    ("LTV:CAC Ratio",                "7.4×"),
]
for i, (k, v) in enumerate(kpis):
    ws4.row_dimensions[current_row].height = 22
    bg = LIGHT_GREY if i % 2 == 0 else WHITE
    ws4.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=4)
    wc(ws4, current_row, 2, k, font=navy_font(10), fill_=fill(bg), align=LEFT, border_=thin_border())
    ws4.merge_cells(start_row=current_row, start_column=5, end_row=current_row, end_column=7)
    f = Font(name="Calibri", size=10, bold=True, color=GREEN if "+" in str(v) or "Crore" in str(v) else NAVY)
    wc(ws4, current_row, 5, v, font=f, fill_=fill(bg), align=CENTER, border_=thin_border())
    current_row += 1

# ─── SHEET 5: YEAR 2 P&L ─────────────────────────────────────────────────────
ws5 = wb.create_sheet("Year 2 — FY2027 P&L")
ws5.sheet_view.showGridLines = False
ws5.sheet_properties.tabColor = GREEN
set_col_widths(ws5, [2, 36, 15, 15, 15, 15, 15, 2])
freeze(ws5, "B4")

merge_title(ws5, 1, 2, 7, "YEAR 2  —  FY2027 PROFIT & LOSS  (April 2027 – March 2028)", NAVY, WHITE, 13)
merge_title(ws5, 2, 2, 7, "Target: 16 Customers  |  ₹19.20 Crore ARR  |  PROFITABLE  |  +₹3.64 Crore EBITDA", GREEN, WHITE, 10)

p_hdrs = ["Line Item", "Q1 (Apr–Jun)", "Q2 (Jul–Sep)", "Q3 (Oct–Dec)", "Q4 (Jan–Mar)", "FY2027 Total"]
ws5.row_dimensions[3].height = 24
for c_i, h in enumerate(p_hdrs):
    wc(ws5, 3, c_i+2, h, font=hdr_font(11, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())

y2_sections = [
    {
        "title": "REVENUE",
        "color": NAVY_LIGHT,
        "rows": [
            ("New Customers in Quarter",         3,          3,          2,          2,          10),
            ("Cumulative Customers",             9,          12,         14,         16,         None),
            ("Subscription Revenue",             30000000,   38000000,   33000000,   34000000,   None),
            ("Expansion / Upsell Revenue",        3000000,    5000000,    5000000,    5000000,   None),
            ("Professional Services Revenue",    11250000,   11250000,    8750000,   13750000,   None),
            ("Protocol Marketplace Licensing",    500000,     500000,     500000,     500000,    None),
        ],
        "total": ("Total Revenue", 44750000, 54750000, 47250000, 53250000, None),
        "total_color": NAVY_LIGHT,
    },
    {
        "title": "COST OF REVENUE",
        "color": NAVY_MID,
        "rows": [
            ("Subscription COGS (28%)",           8400000,   10640000,   9240000,   9520000,    None),
            ("PS COGS (50%)",                     5625000,    5625000,   4375000,   6875000,    None),
            ("Marketplace COGS",                   50000,      50000,     50000,     50000,     None),
        ],
        "total": ("Total COGS", 14075000, 16315000, 13665000, 16445000, None),
        "total_color": NAVY_MID,
    },
    {
        "title": "OPERATING EXPENSES",
        "color": NAVY_LIGHT,
        "rows": [
            ("Engineering & R&D",               10500000,   10500000,   10500000,  10500000,   None),
            ("Sales & Marketing",                9000000,    9000000,    8000000,   9000000,   None),
            ("General & Administrative",         4500000,    4500000,    4500000,   4500000,   None),
            ("Certifications & Legal",           2000000,    2000000,    2000000,   2000000,   None),
        ],
        "total": ("Total Operating Expenses", 26000000, 26000000, 25000000, 26000000, None),
        "total_color": NAVY_LIGHT,
    },
]

current_row = 4
y2_rev  = [44750000, 54750000, 47250000, 53250000]
y2_cogs = [14075000, 16315000, 13665000, 16445000]
y2_gp   = [r - c for r, c in zip(y2_rev, y2_cogs)]
y2_opex = [26000000, 26000000, 25000000, 26000000]
y2_ebitda = [gp - opex for gp, opex in zip(y2_gp, y2_opex)]

for sec in y2_sections:
    ws5.row_dimensions[current_row].height = 22
    ws5.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=7)
    c = ws5.cell(current_row, 2, sec["title"])
    c.font = Font(name="Calibri", size=10, bold=True, color=GOLD)
    c.fill = fill(sec["color"])
    c.alignment = LEFT
    current_row += 1

    for i, row_data in enumerate(sec.get("rows", [])):
        ws5.row_dimensions[current_row].height = 20
        bg = LIGHT_GREY if i % 2 == 0 else WHITE
        desc = row_data[0]
        vals = list(row_data[1:])
        if vals[4] is None:
            vals[4] = sum(v for v in vals[:4] if isinstance(v, (int, float)))
        wc(ws5, current_row, 2, desc, font=body_font(9), fill_=fill(bg), align=LEFT, border_=thin_border())
        for c_i, v in enumerate(vals[:5]):
            fmt = '₹#,##0' if isinstance(v, (int, float)) and abs(v) > 100 else None
            wc(ws5, current_row, c_i+3, v, font=body_font(9), fill_=fill(bg), align=RIGHT if fmt else CENTER, border_=thin_border(), num_fmt=fmt)
        current_row += 1

    ws5.row_dimensions[current_row].height = 22
    t = sec["total"]
    tc = sec["total_color"]
    vals = list(t[1:])
    vals[4] = sum(v for v in vals[:4] if isinstance(v, (int, float)))
    wc(ws5, current_row, 2, t[0], font=hdr_font(10, WHITE), fill_=fill(tc), align=LEFT, border_=thin_border())
    for c_i, v in enumerate(vals[:5]):
        wc(ws5, current_row, c_i+3, v, font=hdr_font(10, GOLD), fill_=fill(tc), align=RIGHT, border_=thin_border(), num_fmt='₹#,##0')
    current_row += 2

# Gross Profit
ws5.row_dimensions[current_row].height = 22
ws5.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=7)
c = ws5.cell(current_row, 2, "GROSS PROFIT")
c.font = Font(name="Calibri", size=10, bold=True, color=GOLD); c.fill = fill(GREEN); c.alignment = LEFT
current_row += 1
ws5.row_dimensions[current_row].height = 22
gp_total = sum(y2_gp)
wc(ws5, current_row, 2, "Gross Profit", font=hdr_font(10, WHITE), fill_=fill(GREEN), align=LEFT, border_=thin_border())
for c_i, v in enumerate(y2_gp + [gp_total]):
    wc(ws5, current_row, c_i+3, v, font=Font(name="Calibri", size=10, bold=True, color=WHITE), fill_=fill(GREEN), align=RIGHT, border_=thin_border(), num_fmt='₹#,##0')
current_row += 1
ws5.row_dimensions[current_row].height = 18
gm_pcts = [f"{gp/rev*100:.1f}%" for gp, rev in zip(y2_gp, y2_rev)] + [f"{gp_total/sum(y2_rev)*100:.1f}%"]
wc(ws5, current_row, 2, "Gross Margin %", font=body_font(9, DARK_GREY), fill_=fill(LIGHT_GREY), align=LEFT, border_=thin_border())
for c_i, v in enumerate(gm_pcts):
    wc(ws5, current_row, c_i+3, v, font=body_font(9, GREEN, True), fill_=fill(LIGHT_GREY), align=CENTER, border_=thin_border())
current_row += 2

# EBITDA
ws5.row_dimensions[current_row].height = 30
ebitda_total = sum(y2_ebitda)
wc(ws5, current_row, 2, "EBITDA  ✓ PROFITABLE", font=Font(name="Calibri", size=12, bold=True, color=NAVY), fill_=fill(GOLD), align=LEFT, border_=thin_border(GOLD))
for c_i, v in enumerate(y2_ebitda + [ebitda_total]):
    wc(ws5, current_row, c_i+3, v, font=Font(name="Calibri", size=11, bold=True, color=GREEN if v > 0 else RED), fill_=fill(GOLD), align=RIGHT, border_=thin_border(GOLD), num_fmt='₹#,##0')

current_row += 2
merge_title(ws5, current_row, 2, 7, "KEY METRICS — END OF YEAR 2", NAVY_LIGHT, GOLD, 11)
current_row += 1
kpis2 = [
    ("Ending ARR",                 "₹19.20 Crore"),
    ("Enterprise Customers",       "16"),
    ("Average ARR per Customer",   "₹1.20 Cr (India) / ₹2 Cr (International)"),
    ("Gross Margin",               "69.7%"),
    ("EBITDA",                     "+₹3.64 Crore  (PROFITABLE)"),
    ("Net Revenue Retention",      "128%"),
    ("Rule of 40 Score",           "167% growth + 18.2% EBITDA = 185"),
    ("International Revenue %",    "15% (UAE, Singapore customers)"),
]
for i, (k, v) in enumerate(kpis2):
    ws5.row_dimensions[current_row].height = 22
    bg = LIGHT_GREY if i % 2 == 0 else WHITE
    ws5.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=4)
    wc(ws5, current_row, 2, k, font=navy_font(10), fill_=fill(bg), align=LEFT, border_=thin_border())
    ws5.merge_cells(start_row=current_row, start_column=5, end_row=current_row, end_column=7)
    is_positive = "+" in str(v) or "128%" in str(v) or "185" in str(v)
    wc(ws5, current_row, 5, v, font=Font(name="Calibri", size=10, bold=True, color=GREEN if is_positive else NAVY), fill_=fill(bg), align=CENTER, border_=thin_border())
    current_row += 1

# ─── SHEET 6: MONTHLY CASH FLOW ──────────────────────────────────────────────
ws6 = wb.create_sheet("Monthly Cash Flow")
ws6.sheet_view.showGridLines = False
ws6.sheet_properties.tabColor = NAVY_MID
set_col_widths(ws6, [2, 18, 10, 14, 14, 14, 14, 2])
freeze(ws6, "B4")

merge_title(ws6, 1, 2, 7, "MONTHLY CASH FLOW  —  YEAR 1  (₹ Lakh)", NAVY, WHITE, 13)
merge_title(ws6, 2, 2, 7, "Starting Cash: ₹20 Crore  |  All figures in ₹ Lakh  |  (₹1 Crore = ₹100 Lakh)", NAVY_MID, GOLD_LIGHT, 10)

cf_hdrs = ["Month", "Customers", "Revenue (₹L)", "Expenses (₹L)", "Net (₹L)", "Cumulative Cash (₹Cr)"]
ws6.row_dimensions[3].height = 24
for c_i, h in enumerate(cf_hdrs):
    wc(ws6, 3, c_i+2, h, font=hdr_font(11, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())

cf_data = [
    ("Apr 2026 (M1)",  "0→1",  27.5,  115,  -87.5,  19.13),
    ("May 2026 (M2)",  "1",    20.0,  105,  -85.0,  18.28),
    ("Jun 2026 (M3)",  "1",    20.0,  105,  -85.0,  17.43),
    ("Jul 2026 (M4)",  "1→2",  42.5,  105,  -62.5,  16.80),
    ("Aug 2026 (M5)",  "2→3",  47.5,  100,  -52.5,  16.28),
    ("Sep 2026 (M6)",  "3",    37.5,  100,  -62.5,  15.65),
    ("Oct 2026 (M7)",  "3→4",  52.5,  100,  -47.5,  15.18),
    ("Nov 2026 (M8)",  "4→5",  57.5,   95,  -37.5,  14.80),
    ("Dec 2026 (M9)",  "5",    45.0,   90,  -45.0,  14.35),
    ("Jan 2027 (M10)", "5→6",  60.0,   88,  -28.0,  14.07),
    ("Feb 2027 (M11)", "6",    50.0,   85,  -35.0,  13.72),
    ("Mar 2027 (M12)", "6",    50.0,   85,  -35.0,  13.37),
]

for i, (month, cust, rev, exp, net, cum) in enumerate(cf_data):
    row = 4 + i
    ws6.row_dimensions[row].height = 22
    bg = LIGHT_GREY if i % 2 == 0 else WHITE
    wc(ws6, row, 2, month, font=body_font(10, bold=True), fill_=fill(bg), align=LEFT, border_=thin_border())
    wc(ws6, row, 3, cust,  font=body_font(10),            fill_=fill(bg), align=CENTER, border_=thin_border())
    wc(ws6, row, 4, rev,   font=body_font(10, GREEN),     fill_=fill(bg), align=RIGHT, border_=thin_border(), num_fmt='#,##0.0')
    wc(ws6, row, 5, exp,   font=body_font(10, RED),       fill_=fill(bg), align=RIGHT, border_=thin_border(), num_fmt='#,##0.0')
    wc(ws6, row, 6, net,   font=Font(name="Calibri", size=10, bold=True, color=RED),   fill_=fill(RED_BG), align=RIGHT, border_=thin_border(), num_fmt='#,##0.0')
    wc(ws6, row, 7, cum,   font=Font(name="Calibri", size=10, bold=True, color=NAVY),  fill_=fill(GOLD_BG), align=RIGHT, border_=thin_border(), num_fmt='#,##0.00')

# Totals row
total_row = 4 + len(cf_data)
ws6.row_dimensions[total_row].height = 26
totals = ["TOTAL YEAR 1", "—", sum(r[2] for r in cf_data), sum(r[3] for r in cf_data),
          sum(r[4] for r in cf_data), cf_data[-1][5]]
wc(ws6, total_row, 2, totals[0], font=hdr_font(11, NAVY), fill_=fill(GOLD), align=LEFT, border_=thin_border(GOLD))
wc(ws6, total_row, 3, totals[1], font=hdr_font(11, NAVY), fill_=fill(GOLD), align=CENTER, border_=thin_border(GOLD))
wc(ws6, total_row, 4, totals[2], font=Font(name="Calibri", size=11, bold=True, color=GREEN), fill_=fill(GOLD), align=RIGHT, border_=thin_border(GOLD), num_fmt='#,##0.0')
wc(ws6, total_row, 5, totals[3], font=Font(name="Calibri", size=11, bold=True, color=RED),   fill_=fill(GOLD), align=RIGHT, border_=thin_border(GOLD), num_fmt='#,##0.0')
wc(ws6, total_row, 6, totals[4], font=Font(name="Calibri", size=11, bold=True, color=RED),   fill_=fill(GOLD), align=RIGHT, border_=thin_border(GOLD), num_fmt='#,##0.0')
wc(ws6, total_row, 7, totals[5], font=Font(name="Calibri", size=13, bold=True, color=NAVY),  fill_=fill(GOLD), align=RIGHT, border_=thin_border(GOLD), num_fmt='#,##0.00')

# Note box
note_r = total_row + 2
ws6.row_dimensions[note_r].height = 40
ws6.merge_cells(start_row=note_r, start_column=2, end_row=note_r, end_column=7)
c = ws6.cell(note_r, 2)
c.value = ("KEY INSIGHT:  Cash remaining at end of Year 1 = ₹13.37 Crore — sufficient to fully fund Year 2 operations "
           "(budgeted at ₹10.30 Crore).  QBITEL becomes self-funded and profitable by Month 19 without requiring Series B.")
c.font = Font(name="Calibri", size=10, italic=True, bold=True, color=NAVY)
c.fill = fill(GOLD_BG)
c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
c.border = Border(
    left=Side(style="medium", color=GOLD),
    right=Side(style="medium", color=GOLD),
    top=Side(style="medium", color=GOLD),
    bottom=Side(style="medium", color=GOLD)
)

# Cash flow chart
chart = LineChart()
chart.title = "Cumulative Cash Balance (₹ Crore)"
chart.style = 10
chart.y_axis.title = "₹ Crore"
chart.x_axis.title = "Month"
chart.width = 22
chart.height = 12

cum_cash_vals = [20] + [r[5] for r in cf_data]
# write chart data to hidden rows
data_start = note_r + 3
ws6.cell(data_start, 2, "Month")
ws6.cell(data_start, 3, "Cash (₹Cr)")
for idx, val in enumerate(cum_cash_vals):
    ws6.cell(data_start + 1 + idx, 2, f"M{idx}")
    ws6.cell(data_start + 1 + idx, 3, val)

data_ref = Reference(ws6, min_col=3, min_row=data_start, max_row=data_start + len(cum_cash_vals))
chart.add_data(data_ref, titles_from_data=True)
chart.series[0].graphicalProperties.line.solidFill = GOLD
chart.series[0].graphicalProperties.line.width = 25000
ws6.add_chart(chart, f"B{note_r + 4}")

# ─── SHEET 7: 3-YEAR SUMMARY ─────────────────────────────────────────────────
ws7 = wb.create_sheet("3-Year Summary")
ws7.sheet_view.showGridLines = False
ws7.sheet_properties.tabColor = GOLD
set_col_widths(ws7, [2, 30, 18, 18, 18, 4])
freeze(ws7, "B4")

merge_title(ws7, 1, 2, 5, "3-YEAR FINANCIAL SUMMARY  (FY2026 – FY2028)", NAVY, WHITE, 14)
merge_title(ws7, 2, 2, 5, "India-First Strategy  |  Profitable by Year 2  |  ₹40 Crore ARR by Year 3", NAVY_MID, GOLD_LIGHT, 10)

col_hdrs7 = ["Metric", "FY2026 (Year 1)", "FY2027 (Year 2)", "FY2028 (Year 3)"]
ws7.row_dimensions[3].height = 24
for c_i, h in enumerate(col_hdrs7):
    wc(ws7, 3, c_i+2, h, font=hdr_font(12, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())

rows7 = [
    ("CUSTOMER METRICS",          None,           None,           None,           "header"),
    ("Enterprise Customers",      "6",            "16",           "36",           "data"),
    ("New Logos in Year",         "6",            "10",           "20",           "data"),
    ("Churn Rate",                "<5%",          "<5%",          "<5%",          "data"),
    ("Net Revenue Retention",     "110%",         "128%",         "135%",         "positive"),

    ("ARR METRICS",               None,           None,           None,           "header"),
    ("Ending ARR",                "₹7.20 Crore",  "₹19.20 Crore", "₹40.00 Crore","positive"),
    ("ARR Growth YoY",            "—",            "167%",         "108%",         "positive"),
    ("Average ARR per Customer",  "₹1.20 Crore",  "₹1.20 Crore",  "₹1.11 Crore", "data"),

    ("INCOME STATEMENT",          None,           None,           None,           "header"),
    ("Total Revenue",             "₹4.68 Crore",  "₹20.00 Crore", "₹42.00 Crore","positive"),
    ("Gross Profit",              "₹3.07 Crore",  "₹13.94 Crore", "₹30.24 Crore","positive"),
    ("Gross Margin %",            "65.7%",        "69.7%",        "72.0%",        "positive"),
    ("Total Operating Expenses",  "₹10.15 Crore", "₹10.30 Crore", "₹13.00 Crore","data"),
    ("EBITDA",                    "(₹7.08 Crore)","+ ₹3.64 Crore","+ ₹11.00 Crore","ebitda"),
    ("EBITDA Margin",             "—",            "18.2%",        "26.2%",        "positive"),

    ("UNIT ECONOMICS",            None,           None,           None,           "header"),
    ("Customer Acquisition Cost", "₹35 Lakh",     "₹30 Lakh",     "₹25 Lakh",    "data"),
    ("CAC Payback Period",        "5.5 months",   "5.0 months",   "4.5 months",   "data"),
    ("LTV (3-year)",              "₹2.59 Crore",  "₹2.59 Crore",  "₹2.80 Crore", "data"),
    ("LTV:CAC Ratio",             "7.4×",         "8.6×",         "11.2×",        "positive"),

    ("RULE OF 40",                None,           None,           None,           "header"),
    ("Revenue Growth %",          "—",            "327%",         "110%",         "positive"),
    ("EBITDA Margin %",           "—",            "18.2%",        "26.2%",        "positive"),
    ("Rule of 40 Score",          "—",            "185",          "136",          "positive"),
]

current_row = 4
for i, row_data in enumerate(rows7):
    label, v1, v2, v3, row_type = row_data
    ws7.row_dimensions[current_row].height = 23 if row_type == "header" else 21

    if row_type == "header":
        ws7.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=5)
        c = ws7.cell(current_row, 2, label)
        c.font = Font(name="Calibri", size=10, bold=True, color=GOLD)
        c.fill = fill(NAVY_LIGHT); c.alignment = LEFT
    else:
        bg = LIGHT_GREY if i % 2 == 0 else WHITE
        is_neg = v1 and "(" in str(v1)
        is_pos_ebitda = row_type == "ebitda"

        wc(ws7, current_row, 2, label, font=navy_font(10), fill_=fill(bg), align=LEFT, border_=thin_border())
        for c_i, val in enumerate([v1, v2, v3]):
            neg = val and "(" in str(val)
            pos = val and ("+" in str(val) or row_type == "positive")
            fc = RED if neg else (GREEN if pos else NAVY)
            f = Font(name="Calibri", size=10, bold=(neg or pos), color=fc)
            wc(ws7, current_row, c_i+3, val or "—", font=f, fill_=fill(bg), align=CENTER, border_=thin_border())
    current_row += 1

# ─── SHEET 8: UNIT ECONOMICS ─────────────────────────────────────────────────
ws8 = wb.create_sheet("Unit Economics")
ws8.sheet_view.showGridLines = False
ws8.sheet_properties.tabColor = NAVY_LIGHT
set_col_widths(ws8, [2, 32, 20, 20, 4])
freeze(ws8, "B3")

merge_title(ws8, 1, 2, 4, "UNIT ECONOMICS  &  CUSTOMER ECONOMICS", NAVY, WHITE, 13)
merge_title(ws8, 2, 2, 4, "Why QBITEL Has Best-in-Class SaaS Metrics", NAVY_MID, GOLD_LIGHT, 10)

ue_sections = [
    ("DEAL STRUCTURE — TYPICAL ENTERPRISE CUSTOMER", [
        ("Revenue Stream",           "Amount (₹)",    "Notes"),
        ("Core Platform Subscription (annual)", "₹1,20,00,000", "Per site, SaaS license"),
        ("LLM Autonomous AI Bundle (annual)",   "₹30,00,000",  "Agentic security ops add-on"),
        ("Quantum Readiness Sprint (one-time)", "₹50,00,000",  "Onboarding + protocol discovery"),
        ("Total Year 1 Contract Value",         "₹2,00,00,000","TCV including onboarding"),
        ("Annual Recurring Value",              "₹1,50,00,000","ARR post-onboarding"),
    ]),
    ("CUSTOMER ACQUISITION ECONOMICS", [
        ("Metric",                              "Year 1",       "Year 2"),
        ("Sales & Marketing Spend",             "₹5 Crore",     "₹3.5 Crore"),
        ("New Customers Won",                   "6",            "10"),
        ("Customer Acquisition Cost (CAC)",     "₹35 Lakh",     "₹30 Lakh"),
        ("Average ARR per Customer",            "₹1.20 Crore",  "₹1.20 Crore"),
        ("Gross Margin on Subscription",        "72%",          "72%"),
        ("Annual Gross Profit per Customer",    "₹86.4 Lakh",   "₹86.4 Lakh"),
        ("CAC Payback Period",                  "5.5 months",   "5.0 months"),
        ("Industry Benchmark (Payback)",        "12–18 months", "12–18 months"),
    ]),
    ("LIFETIME VALUE ANALYSIS", [
        ("Metric",                              "Value",         "Notes"),
        ("Average Customer Lifetime",           "6+ years",     "Compliance lock-in prevents churn"),
        ("Annual Gross Profit per Customer",    "₹86.4 Lakh",   "72% × ₹1.20 Crore ARR"),
        ("LTV — 3-Year Conservative",           "₹2.59 Crore",  "Minimum viable LTV"),
        ("LTV — 5-Year Base Case",              "₹4.32 Crore",  "Includes multi-site upsell"),
        ("LTV — 6-Year (full lifecycle)",       "₹5.18 Crore",  "Full NRR compounding at 128%"),
        ("LTV:CAC Ratio (Year 1)",              "7.4×",         "Excellent: >3× is good"),
        ("LTV:CAC Ratio (Year 2)",              "8.6×",         "Improving with CAC efficiency"),
        ("Industry LTV:CAC Benchmark",          ">3×",          "QBITEL is 2.5× the benchmark"),
    ]),
    ("NET REVENUE RETENTION — WHY CUSTOMERS EXPAND", [
        ("Retention Driver",                    "Mechanism",     "Revenue Impact"),
        ("Compliance lock-in",                  "Removing QBITEL means rebuilding months of RBI / CERT-In audit evidence", "0% churn"),
        ("Multi-site expansion",                "First site success drives rollout to all bank branches / hospital network sites", "+50–100% ARR"),
        ("LLM bundle upsell",                   "Core customer upgrades to Autonomous AI ops module", "+25% ARR"),
        ("Protocol Marketplace",                "Custom adapter packs for proprietary systems",         "+₹5–15L one-time"),
        ("Regulatory broadening",               "SEBI + RBI + IRDAI scope increases over time",        "+10–20% ARR"),
        ("International expansion with customer","Bank with UAE ops buys QBITEL for international sites","New site ARR"),
        ("Resulting NRR",                       "Year 1: 110%  |  Year 2: 128%  |  Year 3: 135%",     "World-class retention"),
    ]),
]

current_row = 3
for sec_title, sec_rows in ue_sections:
    ws8.row_dimensions[current_row].height = 26
    ws8.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=4)
    c = ws8.cell(current_row, 2, sec_title)
    c.font = Font(name="Calibri", size=11, bold=True, color=GOLD)
    c.fill = fill(NAVY_LIGHT); c.alignment = LEFT
    current_row += 1

    for i, row_data in enumerate(sec_rows):
        ws8.row_dimensions[current_row].height = 22
        is_hdr = i == 0
        bg = NAVY_MID if is_hdr else (LIGHT_GREY if i % 2 == 1 else WHITE)
        for c_i, val in enumerate(row_data):
            f = hdr_font(10, GOLD_LIGHT) if is_hdr else body_font(9)
            if not is_hdr and c_i == 0: f = navy_font(9)
            is_money = not is_hdr and "₹" in str(val) and ("Crore" in str(val) or "Lakh" in str(val))
            if is_money: f = Font(name="Calibri", size=9, bold=True, color=GREEN)
            wc(ws8, current_row, c_i+2, val, font=f, fill_=fill(bg), align=LEFT, border_=thin_border())
        current_row += 1
    current_row += 1

# ─── SHEET 9: INVESTOR RETURNS ───────────────────────────────────────────────
ws9 = wb.create_sheet("Investor Returns")
ws9.sheet_view.showGridLines = False
ws9.sheet_properties.tabColor = GOLD
set_col_widths(ws9, [2, 30, 20, 20, 20, 4])
freeze(ws9, "B3")

merge_title(ws9, 1, 2, 5, "INVESTOR RETURNS  &  EXIT ANALYSIS", NAVY, WHITE, 13)
merge_title(ws9, 2, 2, 5, "Series A: ₹20 Crore at ₹80 Crore Post-Money  |  25% Ownership", NAVY_MID, GOLD_LIGHT, 10)

current_row = 3
# Valuation Journey
merge_title(ws9, current_row, 2, 5, "VALUATION JOURNEY", NAVY_LIGHT, GOLD, 11)
current_row += 1
vj_hdrs = ["Round", "Timing", "ARR at Close", "Valuation Multiple", "Implied Valuation"]
ws9.row_dimensions[current_row].height = 22
for c_i, h in enumerate(vj_hdrs):
    wc(ws9, current_row, c_i+2, h, font=hdr_font(10, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())
current_row += 1
vj_rows = [
    ("Series A  ◀ THIS ROUND", "March 2026",  "₹2 Crore trailing", "—",       "₹80 Crore post-money"),
    ("Series B  (projected)",  "April 2027",  "₹12 Crore trailing","12–15×",  "₹144–₹180 Crore"),
    ("Exit / IPO  (base case)","FY2028–2029", "₹40 Crore ARR",     "15–20×",  "₹600–₹800 Crore"),
]
for i, row_data in enumerate(vj_rows):
    ws9.row_dimensions[current_row].height = 24
    bg = GOLD_BG if i == 0 else (LIGHT_GREY if i % 2 == 0 else WHITE)
    for c_i, val in enumerate(row_data):
        f = navy_font(10) if i == 0 else body_font(10)
        if "₹" in str(val) and ("Crore" in str(val)):
            f = Font(name="Calibri", size=10, bold=True, color=GREEN)
        wc(ws9, current_row, c_i+2, val, font=f, fill_=fill(bg), align=CENTER, border_=thin_border())
    current_row += 1

current_row += 1
# Return Scenarios
merge_title(ws9, current_row, 2, 5, "RETURN SCENARIOS FOR SERIES A INVESTOR", NAVY_LIGHT, GOLD, 11)
current_row += 1
ws9.row_dimensions[current_row].height = 22
sc_hdrs = ["Scenario", "Exit Valuation", "Investor Share (post-dilution)", "Return (₹ Crore)", "MOIC"]
for c_i, h in enumerate(sc_hdrs):
    wc(ws9, current_row, c_i+2, h, font=hdr_font(10, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())
current_row += 1
scenarios = [
    ("Conservative  (SME IPO / Regional M&A)", "₹300 Crore\n(7.5× ARR)",  "17% (after Series B dilution)", "₹51 Crore",  "2.5×"),
    ("Base Case  ★  (Strategic Acquisition)",  "₹600 Crore\n(15× ARR)",   "18%",                           "₹108 Crore", "5.4×"),
    ("Bull Case  (NSE Main Board / Global M&A)","₹900 Crore\n(22.5× ARR)","18%",                           "₹162 Crore", "8.1×"),
]
scen_colors = [LIGHT_GREY, GOLD_BG, GREEN_BG]
for i, (sc, ev, share, ret, moic) in enumerate(scenarios):
    ws9.row_dimensions[current_row].height = 36
    bg = scen_colors[i]
    is_base = i == 1
    wc(ws9, current_row, 2, sc,    font=Font(name="Calibri", size=10, bold=True, color=NAVY),   fill_=fill(bg), align=LEFT,   border_=thin_border())
    wc(ws9, current_row, 3, ev,    font=Font(name="Calibri", size=10, bold=True, color=NAVY),   fill_=fill(bg), align=CENTER, border_=thin_border())
    wc(ws9, current_row, 4, share, font=body_font(10),                                           fill_=fill(bg), align=CENTER, border_=thin_border())
    wc(ws9, current_row, 5, ret,   font=Font(name="Calibri", size=11, bold=True, color=GREEN),  fill_=fill(bg), align=CENTER, border_=thin_border())
    wc(ws9, current_row, 6, moic,  font=Font(name="Calibri", size=14, bold=True, color=NAVY if not is_base else GREEN), fill_=fill(GOLD if is_base else bg), align=CENTER, border_=thin_border())
    current_row += 1

current_row += 1
# Acquirers
merge_title(ws9, current_row, 2, 5, "STRATEGIC ACQUIRER LANDSCAPE", NAVY_LIGHT, GOLD, 11)
current_row += 1
ws9.row_dimensions[current_row].height = 22
for c_i, h in enumerate(["Acquirer", "Strategic Rationale", "Likely Valuation", "Probability"]):
    wc(ws9, current_row, c_i+2, h, font=hdr_font(10, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())
current_row += 1
acquirers = [
    ("TCS / Wipro / Infosys",     "Embed QBITEL platform into banking & healthcare security practice", "₹300–500 Crore", "High"),
    ("HCL Technologies",           "HCL security services + QBITEL = full-stack offering",              "₹400–600 Crore", "High"),
    ("Palo Alto Networks",         "India market entry + legacy + PQC fills critical white space",      "₹600–900 Crore", "Medium"),
    ("IBM",                        "IBM mainframe + QBITEL COBOL shield is a natural fit",              "₹500–800 Crore", "Medium"),
    ("Cisco",                      "Networking giant adds autonomous AI + PQC to India routing stack",  "₹500–700 Crore", "Medium"),
    ("NSE / BSE IPO (Main Board)", "Deep-tech BFSI SaaS at 20–30× ARR on public markets",             "₹400–800 Crore", "High"),
]
for i, (acq, rat, val, prob) in enumerate(acquirers):
    ws9.row_dimensions[current_row].height = 22
    bg = LIGHT_GREY if i % 2 == 0 else WHITE
    prob_color = GREEN if prob == "High" else GOLD
    wc(ws9, current_row, 2, acq,  font=navy_font(10),    fill_=fill(bg), align=LEFT,   border_=thin_border())
    wc(ws9, current_row, 3, rat,  font=body_font(9),     fill_=fill(bg), align=LEFT,   border_=thin_border())
    wc(ws9, current_row, 4, val,  font=Font(name="Calibri", size=9, bold=True, color=GREEN), fill_=fill(bg), align=CENTER, border_=thin_border())
    wc(ws9, current_row, 5, prob, font=Font(name="Calibri", size=9, bold=True, color=prob_color), fill_=fill(bg), align=CENTER, border_=thin_border())
    current_row += 1

current_row += 1
# Terms
merge_title(ws9, current_row, 2, 5, "TERM SHEET SUMMARY", NAVY_LIGHT, GOLD, 11)
current_row += 1
terms = [
    ("Round Size",              "₹20,00,00,000  (₹20 Crore)"),
    ("Instrument",              "CCPS — Compulsorily Convertible Preference Shares"),
    ("Pre-Money Valuation",     "₹60,00,00,000  (₹60 Crore)"),
    ("Post-Money Valuation",    "₹80,00,00,000  (₹80 Crore)"),
    ("Investor Ownership",      "25%  (post-money, fully diluted)"),
    ("ESOP Pool",               "15%  (created pre-money, included in fully diluted cap table)"),
    ("Liquidation Preference",  "1× non-participating"),
    ("Anti-Dilution",           "Broad-based weighted average"),
    ("Board Structure",         "2 investor seats + 2 founder seats + 1 independent director"),
    ("Pro-Rata Rights",         "Yes — for investors holding >5%"),
    ("Founder Lock-Up",         "4-year vest, 1-year cliff"),
    ("Lead Investor Target",    "Tier-1 Indian VC — Sequoia India / Elevation / Matrix / Nexus"),
    ("Co-Investors Welcome",    "SIDBI, NaBFID, Corporate VC (HDFC, Axis, Bajaj Allianz)"),
]
for i, (k, v) in enumerate(terms):
    ws9.row_dimensions[current_row].height = 22
    bg = LIGHT_GREY if i % 2 == 0 else WHITE
    ws9.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=2)
    wc(ws9, current_row, 2, k, font=navy_font(10),  fill_=fill(bg), align=LEFT, border_=thin_border())
    ws9.merge_cells(start_row=current_row, start_column=3, end_row=current_row, end_column=5)
    wc(ws9, current_row, 3, v, font=body_font(10),  fill_=fill(bg), align=LEFT, border_=thin_border())
    current_row += 1

# ─── SHEET 10: MARKET & COMPETITION ──────────────────────────────────────────
ws10 = wb.create_sheet("Market & Competition")
ws10.sheet_view.showGridLines = False
ws10.sheet_properties.tabColor = NAVY
set_col_widths(ws10, [2, 24, 14, 14, 14, 14, 14, 2])
freeze(ws10, "B4")

merge_title(ws10, 1, 2, 7, "MARKET OPPORTUNITY  &  COMPETITIVE LANDSCAPE", NAVY, WHITE, 13)
merge_title(ws10, 2, 2, 7, "TAM ₹4,200 Crore  |  No Direct Incumbent  |  Made-in-India Advantage", NAVY_MID, GOLD_LIGHT, 10)

# TAM
r = 3
merge_title(ws10, r, 2, 7, "INDIA MARKET SIZING  (₹ Crore)", NAVY_LIGHT, GOLD, 11)
tam_rows = [
    ("TAM — India Total Cybersecurity", "₹18,500 Crore", "NASSCOM 2025; 18% CAGR", "All cybersecurity spend in India"),
    ("TAM — Quantum-Safe Segment",      "₹4,200 Crore",  "Post-quantum remediation across regulated industries", "Our primary addressable segment"),
    ("SAM — QBITEL Target Verticals",   "₹1,800 Crore",  "Banking, healthcare, infra, telecom", "Serviceable with current platform"),
    ("SOM — Year 1–3",                  "₹120 Crore",    "24 enterprise customers × ₹5 Crore ACV", "Conservative bottom-up estimate"),
]
ws10.row_dimensions[r+1].height = 22
for c_i, h in enumerate(["Market Level", "Size (₹ Crore)", "Definition", "Notes"]):
    wc(ws10, r+1, c_i+2, h, font=hdr_font(10, WHITE), fill_=fill(NAVY), align=CENTER, border_=thin_border())
for i, row_data in enumerate(tam_rows):
    ws10.row_dimensions[r+2+i].height = 30
    bg = GOLD_BG if i < 2 else (LIGHT_GREY if i % 2 == 0 else WHITE)
    for c_i, val in enumerate(row_data):
        f = navy_font(10) if c_i == 0 else (Font(name="Calibri", size=11, bold=True, color=GREEN) if c_i == 1 else body_font(9))
        wc(ws10, r+2+i, c_i+2, val, font=f, fill_=fill(bg), align=LEFT, border_=thin_border())

# Competitive matrix
r2 = r + len(tam_rows) + 3
merge_title(ws10, r2, 2, 7, "COMPETITIVE CAPABILITY MATRIX  (India Market)", NAVY_LIGHT, GOLD, 11)
cap_hdrs = ["Capability", "QBITEL", "Cisco / Palo Alto", "TCS/Wipro Security", "Quick Heal / eScan", "IBM Security"]
ws10.row_dimensions[r2+1].height = 22
for c_i, h in enumerate(cap_hdrs):
    bg = NAVY if c_i != 1 else GREEN
    wc(ws10, r2+1, c_i+2, h, font=hdr_font(10, WHITE if c_i != 1 else WHITE), fill_=fill(bg), align=CENTER, border_=thin_border())
cap_rows = [
    ("Legacy Mainframe / COBOL Protection",    "YES ✓","No ✗",   "Partial (services only)","No ✗","No ✗"),
    ("Post-Quantum Cryptography (NIST)",        "YES ✓","No ✗",   "No ✗",                  "No ✗","Roadmap only"),
    ("AI Protocol Discovery (2–4 hrs)",         "YES ✓","No ✗",   "No ✗",                  "No ✗","No ✗"),
    ("Autonomous Response (<10 sec, 78%)",      "YES ✓","Playbooks","No ✗",                "No ✗","No ✗"),
    ("Air-Gapped On-Premise AI",               "YES ✓","No ✗",   "No ✗",                  "No ✗","No ✗"),
    ("SCADA / OT / Industrial Protection",      "YES ✓","Basic","No ✗",                    "No ✗","No ✗"),
    ("Medical Device PQC (64KB RAM)",           "YES ✓","No ✗",   "No ✗",                  "No ✗","No ✗"),
    ("CERT-In Empanelled (India)",              "YES ✓","No ✗",   "YES ✓",                 "YES ✓","YES ✓"),
    ("Made in India (GeM / Defence eligible)",  "YES ✓","No ✗",   "YES ✓",                 "YES ✓","No ✗"),
    ("9-Framework Compliance Automation",       "YES ✓","Partial","No ✗",                  "No ✗","Partial"),
    ("RBI IT Master Direction 2024 ready",      "YES ✓","Partial","Manual only",           "No ✗","Partial"),
    ("Pricing (India mid-market affordable)",   "YES ✓","No (expensive)","No (services)","YES ✓","No (expensive)"),
]
for i, row_data in enumerate(cap_rows):
    ws10.row_dimensions[r2+2+i].height = 22
    bg = LIGHT_GREY if i % 2 == 0 else WHITE
    for c_i, val in enumerate(row_data):
        is_yes = "YES" in str(val)
        is_no  = "No ✗" == str(val).strip()
        f = (Font(name="Calibri", size=9, bold=True, color=GREEN) if is_yes else
             Font(name="Calibri", size=9, bold=False, color=RED)   if is_no  else
             body_font(9))
        if c_i == 0: f = navy_font(9)
        cell_bg = GREEN_BG if (is_yes and c_i == 1) else (RED_BG if is_no and c_i == 1 else bg)
        wc(ws10, r2+2+i, c_i+2, val, font=f, fill_=fill(cell_bg), align=CENTER if c_i > 0 else LEFT, border_=thin_border())

# ─── SET SHEET ORDER & PRINT SETTINGS ────────────────────────────────────────
for ws in [ws1, ws2, ws3, ws4, ws5, ws6, ws7, ws8, ws9, ws10]:
    ws.page_setup.orientation = ws.ORIENTATION_LANDSCAPE
    ws.page_setup.fitToPage   = True
    ws.page_setup.fitToWidth  = 1
    ws.page_setup.fitToHeight = 0
    ws.print_area = None

# ─── SAVE ─────────────────────────────────────────────────────────────────────
output_path = "/Users/prabakarankannan/qbitel/docs/QBITEL_India_Investor_Pitch_2026.xlsx"
wb.save(output_path)
print(f"Saved: {output_path}")
