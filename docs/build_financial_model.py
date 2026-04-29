"""
QBITEL Bridge — Financial Model & Forecasting (Excel)
Source: QBITEL_Bridge_Series_A_Pitch.pdf
Financials only: P&L, Revenue Model, Use of Funds, Cash Flow, ARR Bridge, Unit Economics
Theme: Navy #1B3A6B | Sky Blue #2E86C1 | Gold #F0A500 | White | Light Grey
"""

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.series import SeriesLabel

wb = Workbook()

# ── PALETTE ──────────────────────────────────────────────────────────────────
NAVY     = "1B3A6B"
NAVY2    = "243F7A"
BLUE     = "2E86C1"
BLUE2    = "1A5276"
GOLD     = "F0A500"
GOLD_LT  = "FDE8A0"
WHITE    = "FFFFFF"
LT_GREY  = "F2F4F8"
MID_GREY = "D5D8DC"
DK_GREY  = "626567"
GREEN    = "1E8449"
GREEN_BG = "D5F5E3"
RED      = "C0392B"
RED_BG   = "FADBD8"
AMBER    = "D4AC0D"

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fl(c): return PatternFill("solid", fgColor=c)
def fn(size=10, color=WHITE, bold=True, italic=False):
    return Font(name="Calibri", size=size, color=color, bold=bold, italic=italic)
def aln(h="center", v="center", wrap=True):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)
def bdr(style="thin", color=MID_GREY):
    s = Side(style=style, color=color)
    return Border(left=s, right=s, top=s, bottom=s)
def bdr_gold():
    s = Side(style="medium", color=GOLD)
    return Border(left=s, right=s, top=s, bottom=s)
def bdr_navy():
    s = Side(style="medium", color=NAVY)
    return Border(left=s, right=s, top=s, bottom=s)

def wc(ws, r, c, val, font=None, fill=None, align=None, border=None, fmt=None):
    cell = ws.cell(row=r, column=c, value=val)
    if font:   cell.font      = font
    if fill:   cell.fill      = fl(fill)
    if align:  cell.alignment = align
    if border: cell.border    = border
    if fmt:    cell.number_format = fmt
    return cell

def mc(ws, r1, c1, r2, c2, val, font=None, fill=None, align=None):
    ws.merge_cells(start_row=r1, start_column=c1, end_row=r2, end_column=c2)
    cell = ws.cell(r1, c1, val)
    if font:  cell.font      = font
    if fill:  cell.fill      = fl(fill)
    if align: cell.alignment = align
    return cell

def set_cols(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

def banner(ws, r, c1, c2, txt, bg=NAVY, fg=WHITE, size=12):
    mc(ws, r, c1, r, c2, txt,
       font=Font(name="Calibri", size=size, bold=True, color=fg),
       fill=bg, align=aln())
    ws.row_dimensions[r].height = 26

def sub_hdr(ws, r, c1, c2, txt, bg=BLUE2, fg=WHITE):
    mc(ws, r, c1, r, c2, txt,
       font=Font(name="Calibri", size=10, bold=True, color=fg),
       fill=bg, align=aln("left"))
    ws.row_dimensions[r].height = 22

def col_hdr(ws, r, cols, labels, bg=NAVY2, fg=GOLD):
    ws.row_dimensions[r].height = 22
    for c, lbl in zip(cols, labels):
        wc(ws, r, c, lbl,
           font=Font(name="Calibri", size=9, bold=True, color=fg),
           fill=bg, align=aln(), border=bdr())

def data_row(ws, r, c_start, values, alt=False, bold=False, number_cols=None, fmt='$#,##0', color=None):
    ws.row_dimensions[r].height = 19
    bg = LT_GREY if alt else WHITE
    for i, v in enumerate(values):
        c = c_start + i
        is_num = isinstance(v, (int, float)) and (number_cols is None or i in number_cols)
        cell_fmt = fmt if is_num else None
        fc = color if color else ("1F2937")
        bold_cell = bold or (i == 0)
        wc(ws, r, c, v,
           font=Font(name="Calibri", size=9, bold=bold_cell if i==0 else bold, color=fc),
           fill=bg, align=aln("right" if is_num else "left"), border=bdr(), fmt=cell_fmt)

def total_row(ws, r, c_start, values, fmt='$#,##0', bg=NAVY, fg=GOLD):
    ws.row_dimensions[r].height = 22
    for i, v in enumerate(values):
        c = c_start + i
        is_num = isinstance(v, (int, float))
        wc(ws, r, c, v,
           font=Font(name="Calibri", size=10, bold=True, color=fg),
           fill=bg, align=aln("right" if is_num and i > 0 else ("left" if i==0 else "center")),
           border=bdr_gold(), fmt=fmt if is_num and i > 0 else None)

def gap(ws, r, c1, c2, h=8, color=LT_GREY):
    ws.row_dimensions[r].height = h
    ws.merge_cells(start_row=r, start_column=c1, end_row=r, end_column=c2)
    ws.cell(r, c1).fill = fl(color)


# ════════════════════════════════════════════════════════════════════════════
# SHEET 1 — FINANCIAL SUMMARY (3-YEAR SNAPSHOT)
# ════════════════════════════════════════════════════════════════════════════
ws1 = wb.active
ws1.title = "Financial Summary"
ws1.sheet_view.showGridLines = False
ws1.sheet_properties.tabColor = NAVY
set_cols(ws1, [2, 32, 18, 18, 18, 2])

# Header
mc(ws1,1,2,1,5,"QBITEL BRIDGE  —  FINANCIAL SUMMARY",
   font=Font(name="Calibri",size=15,bold=True,color=WHITE), fill=NAVY, align=aln())
ws1.row_dimensions[1].height = 32
mc(ws1,2,2,2,5,"Series A  |  $18,000,000  |  Three-Year Financial Model  (FY2026 – FY2028)",
   font=Font(name="Calibri",size=10,bold=False,color=GOLD_LT,italic=True), fill=NAVY2, align=aln())
ws1.row_dimensions[2].height = 20

# Col headers
col_hdr(ws1, 3, [2,3,4,5], ["METRIC","FY2026  (Year 1)","FY2027  (Year 2)","FY2028  (Year 3)"])

rows_summary = [
    ("GROWTH METRICS",         None,      None,      None),
    ("Enterprise Customers",   6,         16,        32),
    ("New Logos in Year",       6,         10,        16),
    ("Ending ARR",             5100000,   14800000,  32400000),
    ("ARR Growth YoY",         "—",       "190%",    "119%"),
    ("Net Revenue Retention",  "110%",    "130%+",   "135%+"),
    ("Churn Rate",             "<5%",     "<5%",     "<5%"),

    ("INCOME STATEMENT",       None,      None,      None),
    ("Subscription Revenue",   3060000,   9830000,   21200000),
    ("Professional Services",  1500000,   2500000,   4000000),
    ("Marketplace Revenue",    0,         1008000,   6300000),
    ("Total Revenue",          4560000,   13338000,  31500000),
    ("Cost of Revenue",        1687200,   4268160,   8820000),
    ("Gross Profit",           2872800,   9069840,   22680000),
    ("Gross Margin %",         "63%",     "68%",     "72%"),
    ("R&D / Engineering",      4200000,   4500000,   5500000),
    ("Sales & Marketing",      2500000,   3500000,   5000000),
    ("General & Admin",        1500000,   1800000,   2500000),
    ("Total OpEx",             8200000,   9800000,   13000000),
    ("EBITDA",                 -5327200,  -730160,   9680000),
    ("EBITDA Margin",          "—",       "—",       "30.7%"),

    ("CASH & FUNDING",         None,      None,      None),
    ("Series A Raise",         18000000,  0,         0),
    ("Net Cash Burn / (Gen.)", -7014400,  -4998320,  0),
    ("Cumulative Cash (est.)", 10985600,  5987280,   "Self-Funded"),

    ("MARKETPLACE",            None,      None,      None),
    ("Protocol Adapters",      "500+",    "1,000+",  "3,000+"),
    ("Marketplace GMV",        0,         12000000,  70000000),
    ("Platform Fee (30%)",     0,         3600000,   21000000),
]

r = 4
for i, (label, v1, v2, v3) in enumerate(rows_summary):
    is_section = v1 is None
    ws1.row_dimensions[r].height = 22 if is_section else 19

    if is_section:
        gap(ws1, r, 2, 5, 6, LT_GREY)
        r += 1
        sub_hdr(ws1, r, 2, 5, f"  {label}", BLUE2, GOLD)
        r += 1
        continue

    is_total = label in ("Total Revenue","Gross Profit","EBITDA","Total OpEx")
    is_neg   = isinstance(v1, (int,float)) and v1 < 0
    is_pos_big = label == "EBITDA" and isinstance(v3,(int,float)) and v3 > 0

    bg_row = LT_GREY if i % 2 == 0 else WHITE
    for ci, (col, val) in enumerate([(2,label),(3,v1),(4,v2),(5,v3)]):
        is_num = isinstance(val,(int,float)) and ci > 0 and not isinstance(val, bool)
        neg    = isinstance(val,(int,float)) and val < 0
        pos    = is_pos_big and col == 5

        f = Font(name="Calibri", size=9,
                 bold=is_total or ci==0,
                 color=(RED if neg else (GREEN if (isinstance(val,(int,float)) and val>0 and label=="EBITDA" and ci==4) else (NAVY if ci==0 else "1F2937"))))

        if is_total:
            wc(ws1,r,col,val, font=Font(name="Calibri",size=10,bold=True,
               color=(RED if neg else NAVY)), fill=MID_GREY, align=aln("right" if is_num else "left"),
               border=bdr(color=NAVY), fmt='$#,##0' if is_num else None)
        else:
            wc(ws1,r,col,val, font=f, fill=bg_row,
               align=aln("right" if is_num else ("center" if ci>0 else "left")),
               border=bdr(), fmt='$#,##0' if is_num else None)
    r += 1

# Highlight EBITDA positive in Y3
ws1.cell(r-1, 5).font  = Font(name="Calibri", size=9, bold=True, color=GREEN)
ws1.cell(r-1, 5).fill  = fl(GREEN_BG)

# ── KEY CALLOUT BOXES ─────────────────────────────────────────────────────
r += 2
boxes = [
    (3, "$5.1M",    "Year 1 Ending\nARR",          NAVY),
    (4, "$14.8M",   "Year 2 Ending\nARR (Series B)",BLUE),
    (5, "$32.4M",   "Year 3 Ending\nARR",          GREEN),
]
ws1.row_dimensions[r].height   = 18
ws1.row_dimensions[r+1].height = 30
ws1.row_dimensions[r+2].height = 18
mc(ws1,r,2,r+2,2,"HEADLINE METRICS",
   font=Font(name="Calibri",size=10,bold=True,color=WHITE), fill=NAVY, align=aln())
for col, big, sub, clr in boxes:
    ws1.cell(r,   col).fill = fl(clr);  ws1.cell(r,   col).border = bdr_navy()
    ws1.cell(r+2, col).fill = fl(clr);  ws1.cell(r+2, col).border = bdr_navy()
    mc(ws1,r,col,r,col,   "",font=None,fill=clr)
    mc(ws1,r+1,col,r+1,col, big,
       font=Font(name="Calibri",size=20,bold=True,color=WHITE), fill=clr, align=aln())
    mc(ws1,r+2,col,r+2,col, sub,
       font=Font(name="Calibri",size=8,color=WHITE,italic=True), fill=clr, align=aln())
    for rr in [r, r+1, r+2]:
        ws1.cell(rr,col).border = bdr_navy()

r += 4
boxes2 = [
    (3,"63% → 72%","Gross Margin\nY1 to Y3",   BLUE2),
    (4, "24 Months","Runway on\n$18M Raise",    AMBER),
    (5,   "$830K",  "ARR per\nEnterprise Site", BLUE2),
]
ws1.row_dimensions[r].height   = 18
ws1.row_dimensions[r+1].height = 30
ws1.row_dimensions[r+2].height = 18
mc(ws1,r,2,r+2,2,"KEY METRICS",
   font=Font(name="Calibri",size=10,bold=True,color=WHITE), fill=NAVY2, align=aln())
for col, big, sub, clr in boxes2:
    for rr in [r,r+1,r+2]:
        ws1.cell(rr,col).fill   = fl(clr)
        ws1.cell(rr,col).border = bdr_gold()
    mc(ws1,r+1,col,r+1,col, big,
       font=Font(name="Calibri",size=16,bold=True,color=WHITE), fill=clr, align=aln())
    mc(ws1,r+2,col,r+2,col, sub,
       font=Font(name="Calibri",size=8,color=GOLD_LT,italic=True), fill=clr, align=aln())


# ════════════════════════════════════════════════════════════════════════════
# SHEET 2 — REVENUE MODEL (detailed build-up)
# ════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("Revenue Model")
ws2.sheet_view.showGridLines = False
ws2.sheet_properties.tabColor = BLUE
set_cols(ws2, [2, 30, 16, 16, 16, 16, 16, 2])

banner(ws2,1,2,7,"QBITEL BRIDGE  —  REVENUE MODEL  (FY2026 – FY2028)", NAVY, WHITE, 13)
mc(ws2,2,2,2,7,"Source: Series A Pitch Deck  |  Avg ARR/site: $830K  |  PS: $250K/engagement  |  Marketplace take: 30%",
   font=Font(name="Calibri",size=9,italic=True,color=GOLD_LT,bold=False), fill=NAVY2, align=aln())
ws2.row_dimensions[2].height = 18

col_hdr(ws2,3,[2,3,4,5,6,7],["LINE ITEM","FY2026 (Y1)","FY2027 (Y2)","FY2028 (Y3)","CAGR Y1→Y3","NOTES"])

# ── SUBSCRIPTION REVENUE ────────────────────────────────────────────────────
r = 4
sub_hdr(ws2,r,2,7,"  A.  SUBSCRIPTION REVENUE  (ARR-Based)")
r += 1

sub_rows = [
    ("Ending ARR (stated in pitch)",              5100000,  14800000, 32400000, "From pitch deck — primary metric"),
    ("New Customers Added in Year",               6,        10,       16,       "Y1: 6 from beta+GA  |  Y2: 10  |  Y3: 16"),
    ("Cumulative Active Customers",               6,        16,       32,       "Stated in pitch: 6 → 16 → 32"),
    ("Average ARR per Site (Core + LLM Bundle)",  830000,   830000,   1012500,  "Y3 expansion via multi-site NRR"),
    ("Avg Months Active — New Cohort",            6.5,      6.5,      6.5,      "Conservative: customers sign mid-year on avg"),
    ("New Cohort Sub Revenue (ramp-adj.)",        2145000,  5395000,  10920000, "New logos × $830K × 6.5/12"),
    ("Prior Cohort Full-Year Revenue",            0,        4980000,  13280000, "Existing base × $830K full year"),
    ("Upsell / Expansion Revenue",                0,        415000,   1245000,  "NRR 130%+ drives expansion ARR"),
]
for i,(label,v1,v2,v3,note) in enumerate(sub_rows):
    is_num = isinstance(v1, float) and v1 > 99
    is_int_m = isinstance(v1, int) and v1 > 99
    fmt = '$#,##0' if (is_num or is_int_m) else None
    bg = LT_GREY if i%2==0 else WHITE
    for col,val in [(2,label),(3,v1),(4,v2),(5,v3),(6,"—"),(7,note)]:
        is_n = isinstance(val,(int,float)) and col in [3,4,5] and abs(val)>99
        wc(ws2,r,col,val,
           font=Font(name="Calibri",size=9,bold=(col==2),color="1F2937"),
           fill=bg, align=aln("right" if is_n else "left"), border=bdr(),
           fmt='$#,##0' if is_n else None)
    r += 1

total_row(ws2,r,2,["TOTAL SUBSCRIPTION REVENUE",3060000,9830000,21200000,"—","79% CAGR"],fmt='$#,##0')
ws2.cell(r,6).value="79% CAGR"; ws2.cell(r,6).font=Font(name="Calibri",size=10,bold=True,color=GOLD); ws2.cell(r,6).fill=fl(NAVY); ws2.cell(r,6).alignment=aln()
r += 1

# ── PROFESSIONAL SERVICES ────────────────────────────────────────────────────
gap(ws2,r,2,7,6); r+=1
sub_hdr(ws2,r,2,7,"  B.  PROFESSIONAL SERVICES"); r+=1

ps_rows = [
    ("Quantum Readiness Sprint — Engagements",  6,       10,      16,      "1 per new customer onboarding"),
    ("Revenue per Engagement",                  250000,  250000,  250000,  "Fixed fee per pitch deck"),
    ("PS Revenue",                              1500000, 2500000, 4000000, "$250K × new customers"),
    ("PS Gross Margin",                         "50%",   "50%",   "50%",   "High-margin, repeatable engagement"),
    ("PS Gross Profit",                          750000, 1250000, 2000000, ""),
]
for i,(label,v1,v2,v3,note) in enumerate(ps_rows):
    is_n1 = isinstance(v1,(int,float)) and v1>99
    bg = LT_GREY if i%2==0 else WHITE
    for col,val in [(2,label),(3,v1),(4,v2),(5,v3),(6,"—"),(7,note)]:
        is_n = isinstance(val,(int,float)) and col in [3,4,5] and abs(val)>99
        wc(ws2,r,col,val,
           font=Font(name="Calibri",size=9,bold=(col==2),color="1F2937"),
           fill=bg, align=aln("right" if is_n else "left"), border=bdr(),
           fmt='$#,##0' if is_n else None)
    r+=1
total_row(ws2,r,2,["TOTAL PROFESSIONAL SERVICES",1500000,2500000,4000000,"—","63% CAGR"],fmt='$#,##0')
ws2.cell(r,6).value="63% CAGR"; ws2.cell(r,6).font=Font(name="Calibri",size=10,bold=True,color=GOLD); ws2.cell(r,6).fill=fl(NAVY); ws2.cell(r,6).alignment=aln()
r+=1

# ── MARKETPLACE ─────────────────────────────────────────────────────────────
gap(ws2,r,2,7,6); r+=1
sub_hdr(ws2,r,2,7,"  C.  MARKETPLACE  (Protocol Adapters — 30% Platform Fee)"); r+=1

mkt_rows = [
    ("Protocol Adapters Live",             "500+",     "1,000+",   "3,000+",   "From pitch deck roadmap"),
    ("Marketplace GMV",                    0,          12000000,   70000000,   "From pitch deck: $12M Y2 → $70M Y3"),
    ("QBITEL Platform Fee (30%)",          0,          3600000,    21000000,   "30% of GMV — from pitch deck"),
    ("Revenue Recognition (partial yr)",   0,          1008000,    6300000,    "Conservative: 28% recognized Y2, 30% Y3"),
    ("Marketplace Gross Margin",           "90%",      "90%",      "90%",      "Near-zero marginal cost per transaction"),
]
for i,(label,v1,v2,v3,note) in enumerate(mkt_rows):
    bg = LT_GREY if i%2==0 else WHITE
    for col,val in [(2,label),(3,v1),(4,v2),(5,v3),(6,"—"),(7,note)]:
        is_n = isinstance(val,(int,float)) and col in [3,4,5] and abs(val)>99
        wc(ws2,r,col,val,
           font=Font(name="Calibri",size=9,bold=(col==2),color="1F2937"),
           fill=bg, align=aln("right" if is_n else "left"), border=bdr(),
           fmt='$#,##0' if is_n else None)
    r+=1
total_row(ws2,r,2,["TOTAL MARKETPLACE REVENUE",0,1008000,6300000,"—","N/M → $6.3M"],fmt='$#,##0')
ws2.cell(r,6).value="N/M → $6.3M"; ws2.cell(r,6).font=Font(name="Calibri",size=10,bold=True,color=GOLD); ws2.cell(r,6).fill=fl(NAVY); ws2.cell(r,6).alignment=aln()
r+=2

# ── GRAND TOTAL REVENUE ──────────────────────────────────────────────────────
ws2.row_dimensions[r].height=28
for c in range(2,8): ws2.cell(r,c).fill=fl(GOLD)
wc(ws2,r,2,"TOTAL REVENUE  (All Streams)",
   font=Font(name="Calibri",size=12,bold=True,color=NAVY), fill=GOLD,
   align=aln("left"), border=bdr_gold())
for col,val in [(3,4560000),(4,13338000),(5,31500000)]:
    wc(ws2,r,col,val,
       font=Font(name="Calibri",size=12,bold=True,color=NAVY),
       fill=GOLD, align=aln("right"), border=bdr_gold(), fmt='$#,##0')
wc(ws2,r,6,"163% CAGR",
   font=Font(name="Calibri",size=11,bold=True,color=NAVY), fill=GOLD, align=aln(), border=bdr_gold())
r+=2

# ── COGS & GROSS PROFIT ──────────────────────────────────────────────────────
sub_hdr(ws2,r,2,7,"  D.  COST OF REVENUE  &  GROSS PROFIT"); r+=1
cogs_rows = [
    ("Subscription COGS (28% of sub rev)",    856800,   2752400, 5936000,  "Server infra, support, AI compute"),
    ("PS COGS (50% of PS rev)",               750000,  1250000,  2000000,  "Delivery engineers, travel"),
    ("Marketplace COGS (10% of mkt rev)",       0,      100800,   630000,  "Payment processing, QA"),
    ("Total COGS",                           1606800,  4103200,  8566000,  ""),
    ("Gross Profit",                         2953200,  9234800, 22934000,  ""),
    ("Gross Margin %",                          "64.8%", "69.2%", "72.8%", "Expanding with scale and automation"),
]
for i,(label,v1,v2,v3,note) in enumerate(cogs_rows):
    is_tot = "Total" in label or "Gross P" in label
    bg = MID_GREY if is_tot else (LT_GREY if i%2==0 else WHITE)
    fc = NAVY if is_tot else "1F2937"
    for col,val in [(2,label),(3,v1),(4,v2),(5,v3),(6,"—"),(7,note)]:
        is_n = isinstance(val,(int,float)) and col in [3,4,5] and abs(val)>99
        wc(ws2,r,col,val,
           font=Font(name="Calibri",size=9,bold=(col==2 or is_tot),color=fc),
           fill=bg, align=aln("right" if is_n else "left"), border=bdr(),
           fmt='$#,##0' if is_n else None)
    r+=1

# Revenue Bar Chart
chart = BarChart()
chart.type = "col"; chart.grouping = "clustered"
chart.title = "Annual Revenue by Stream ($)"
chart.style = 10; chart.width = 20; chart.height = 11
chart.y_axis.title = "USD ($)"
# write data
dr = r+2
ws2.cell(dr,2,"Stream"); ws2.cell(dr,3,"FY2026"); ws2.cell(dr,4,"FY2027"); ws2.cell(dr,5,"FY2028")
streams = [("Subscription",3060000,9830000,21200000),
           ("Prof. Services",1500000,2500000,4000000),
           ("Marketplace",0,1008000,6300000)]
for si,(sname,y1,y2,y3) in enumerate(streams):
    ws2.cell(dr+1+si,2,sname)
    ws2.cell(dr+1+si,3,y1); ws2.cell(dr+1+si,4,y2); ws2.cell(dr+1+si,5,y3)

data = Reference(ws2, min_col=3, max_col=5, min_row=dr, max_row=dr+3)
cats = Reference(ws2, min_col=2, min_row=dr+1, max_row=dr+3)
chart.add_data(data, titles_from_data=True)
chart.set_categories(cats)
chart.series[0].graphicalProperties.solidFill = NAVY
chart.series[1].graphicalProperties.solidFill = BLUE
chart.series[2].graphicalProperties.solidFill = GOLD
ws2.add_chart(chart, f"B{r+2}")


# ════════════════════════════════════════════════════════════════════════════
# SHEET 3 — USE OF FUNDS ($18M)
# ════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("Use of Funds  —  $18M")
ws3.sheet_view.showGridLines = False
ws3.sheet_properties.tabColor = GOLD
set_cols(ws3, [2, 34, 14, 14, 12, 14, 2])

banner(ws3,1,2,6,"USE OF FUNDS  —  $18,000,000 SERIES A", NAVY, WHITE, 13)
mc(ws3,2,2,2,6,"Source: Pitch Deck Page 11  |  45% Product  |  25% GTM  |  20% Certs  |  10% Working Capital",
   font=Font(name="Calibri",size=9,italic=True,color=GOLD_LT), fill=NAVY2, align=aln())
ws3.row_dimensions[2].height=18

col_hdr(ws3,3,[2,3,4,5,6],["INVESTMENT LINE ITEM","YEAR 1","YEAR 2","TOTAL ALLOCATED","% OF RAISE"])

buckets = [
    {
        "title":"BUCKET 1 — PRODUCT & AI COMPLETION  (45%  |  $8,100,000)",
        "color": NAVY,
        "items":[
            ("Autonomous AI engine — accuracy expansion to 95%+",            2800000, 1400000, 4200000),
            ("Protocol discovery SDK — all 6 languages (Python, Java, Go, C#, Rust, TS)", 1200000, 600000, 1800000),
            ("Air-gap deployment hardening (Ollama/vLLM pipeline)",           600000,  300000,  900000),
            ("Post-quantum crypto engine (ML-KEM Kyber-1024, ML-DSA Dilithium-5)",500000, 250000, 750000),
            ("Compliance automation — 9 framework evidence engine",            400000,  200000,  600000),
            ("QA, security testing, DevSecOps infrastructure",                 300000,  300000,  600000),
            ("Cloud infra / co-location (prod + DR)",                          150000,  100000,  250000),
        ],
        "subtotal":(5950000,3150000,8100000,45.0),
    },
    {
        "title":"BUCKET 2 — GO-TO-MARKET EXPANSION  (25%  |  $4,500,000)",
        "color": BLUE2,
        "items":[
            ("Enterprise Account Executives — 10 AEs  (pitch deck stated)",   2200000, 1100000, 3300000),
            ("Vertical-specific marketing — banking, healthcare, energy",       400000,  200000,  600000),
            ("Partner channel development (Accenture, Deloitte, IBM — from deck)",200000, 100000, 300000),
            ("Customer success & implementation engineers",                     200000,  100000,  300000),
        ],
        "subtotal":(3000000,1500000,4500000,25.0),
    },
    {
        "title":"BUCKET 3 — CERTIFICATIONS & PARTNERSHIPS  (20%  |  $3,600,000)",
        "color": NAVY,
        "items":[
            ("FedRAMP Moderate Authorization  (3PAO audit + remediation)",      900000,  700000, 1600000),
            ("SOC 2 Type II certification",                                      400000,  200000,  600000),
            ("FIPS 140-3 validation  (NIST CMVP programme)",                    400000,  200000,  600000),
            ("Strategic integrator partnerships — IBM, Accenture, Deloitte",    200000,  100000,  300000),
            ("Legal, IP protection, regulatory counsel",                         350000,  150000,  500000),
        ],
        "subtotal":(2250000,1350000,3600000,20.0),
    },
    {
        "title":"BUCKET 4 — WORKING CAPITAL  (10%  |  $1,800,000)",
        "color": BLUE2,
        "items":[
            ("Legal, finance infrastructure",                                    500000,  200000,  700000),
            ("Customer success headcount for Year 1 deployments",               450000,  250000,  700000),
            ("Office, IT equipment, operations",                                 200000,  100000,  300000),
            ("Contingency reserve",                                              100000,       0,  100000),
        ],
        "subtotal":(1250000,550000,1800000,10.0),
    },
]

r = 4
grand_y1=grand_y2=grand_tot=0
for bkt in buckets:
    ws3.row_dimensions[r].height=24
    mc(ws3,r,2,r,6, bkt["title"],
       font=Font(name="Calibri",size=11,bold=True,color=GOLD),
       fill=bkt["color"], align=aln("left"))
    r+=1
    for ii,(desc,y1,y2,tot) in enumerate(bkt["items"]):
        ws3.row_dimensions[r].height=19
        bg = LT_GREY if ii%2==0 else WHITE
        wc(ws3,r,2,desc,font=Font(name="Calibri",size=9,color="1F2937"),fill=bg,align=aln("left"),border=bdr())
        wc(ws3,r,3,y1, font=Font(name="Calibri",size=9,color="1F2937"),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
        wc(ws3,r,4,y2, font=Font(name="Calibri",size=9,color="1F2937"),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
        wc(ws3,r,5,tot,font=Font(name="Calibri",size=9,color="1F2937"),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
        pct=tot/18000000*100
        wc(ws3,r,6,f"{pct:.1f}%",font=Font(name="Calibri",size=9,color=DK_GREY),fill=bg,align=aln(),border=bdr())
        r+=1
    sy1,sy2,stot,spct=bkt["subtotal"]
    ws3.row_dimensions[r].height=22
    wc(ws3,r,2,f"Subtotal — {spct:.0f}%",font=Font(name="Calibri",size=10,bold=True,color=WHITE),fill=bkt["color"],align=aln("left"),border=bdr_gold())
    wc(ws3,r,3,sy1,font=Font(name="Calibri",size=10,bold=True,color=GOLD),fill=bkt["color"],align=aln("right"),border=bdr_gold(),fmt='$#,##0')
    wc(ws3,r,4,sy2,font=Font(name="Calibri",size=10,bold=True,color=GOLD),fill=bkt["color"],align=aln("right"),border=bdr_gold(),fmt='$#,##0')
    wc(ws3,r,5,stot,font=Font(name="Calibri",size=10,bold=True,color=GOLD),fill=bkt["color"],align=aln("right"),border=bdr_gold(),fmt='$#,##0')
    wc(ws3,r,6,f"{spct:.0f}%",font=Font(name="Calibri",size=10,bold=True,color=GOLD_LT),fill=bkt["color"],align=aln(),border=bdr_gold())
    grand_y1+=sy1; grand_y2+=sy2; grand_tot+=stot
    r+=2

ws3.row_dimensions[r].height=28
for c in range(2,7): ws3.cell(r,c).fill=fl(GOLD)
wc(ws3,r,2,"TOTAL  —  SERIES A  $18M",font=Font(name="Calibri",size=13,bold=True,color=NAVY),fill=GOLD,align=aln("left"),border=bdr_gold())
wc(ws3,r,3,grand_y1,font=Font(name="Calibri",size=12,bold=True,color=NAVY),fill=GOLD,align=aln("right"),border=bdr_gold(),fmt='$#,##0')
wc(ws3,r,4,grand_y2,font=Font(name="Calibri",size=12,bold=True,color=NAVY),fill=GOLD,align=aln("right"),border=bdr_gold(),fmt='$#,##0')
wc(ws3,r,5,grand_tot,font=Font(name="Calibri",size=12,bold=True,color=NAVY),fill=GOLD,align=aln("right"),border=bdr_gold(),fmt='$#,##0')
wc(ws3,r,6,"100%",font=Font(name="Calibri",size=12,bold=True,color=NAVY),fill=GOLD,align=aln(),border=bdr_gold())

# Burn vs Revenue chart
r+=3
ws3.cell(r,2,"Month"); ws3.cell(r,3,"Cumul. Spend ($)"); ws3.cell(r,4,"Cumul. Revenue ($)")
monthly_spend = 18000000/24
monthly_rev_y1 = [380000,380000,380000,760000,760000,760000,1140000,1140000,1140000,1520000,1520000,1520000]
monthly_rev_y2 = [1112000]*12
cumspend=0; cumrev=0
for mo in range(1,25):
    ws3.cell(r+mo,2,f"M{mo}")
    cumspend+=monthly_spend
    if mo<=12: cumrev+=monthly_rev_y1[mo-1]
    else:      cumrev+=monthly_rev_y2[mo-13]
    ws3.cell(r+mo,3,round(cumspend))
    ws3.cell(r+mo,4,round(cumrev))

ch2=LineChart(); ch2.title="Cumulative Spend vs Revenue — 24 Months"
ch2.style=10; ch2.width=20; ch2.height=10
ch2.y_axis.title="USD ($)"
d1=Reference(ws3,min_col=3,max_col=4,min_row=r,max_row=r+24)
ch2.add_data(d1,titles_from_data=True)
ch2.series[0].graphicalProperties.line.solidFill=RED
ch2.series[1].graphicalProperties.line.solidFill=GREEN
ch2.series[0].graphicalProperties.line.width=20000
ch2.series[1].graphicalProperties.line.width=20000
ws3.add_chart(ch2,f"B{r+26}")


# ════════════════════════════════════════════════════════════════════════════
# SHEET 4 — QUARTERLY P&L  (12 quarters, Y1–Y3)
# ════════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("Quarterly P&L")
ws4.sheet_view.showGridLines = False
ws4.sheet_properties.tabColor = NAVY2
set_cols(ws4,[2,26]+[11]*12+[2])

banner(ws4,1,2,15,"QUARTERLY P&L  |  FY2026 – FY2028  (12 Quarters)", NAVY, WHITE, 13)
mc(ws4,2,2,2,15,"All figures in USD  |  Subscription revenue ramp-adjusted  |  Professional services = $250K per new customer",
   font=Font(name="Calibri",size=9,italic=True,color=GOLD_LT), fill=NAVY2, align=aln())
ws4.row_dimensions[2].height=18

# Year banners
r=3; ws4.row_dimensions[r].height=18
mc(ws4,r,2,r,2,"",fill=NAVY);
mc(ws4,r,3,r,6,"◀  FY2026 (Year 1)  ▶",font=Font(name="Calibri",size=10,bold=True,color=WHITE),fill=NAVY,align=aln())
mc(ws4,r,7,r,10,"◀  FY2027 (Year 2)  ▶",font=Font(name="Calibri",size=10,bold=True,color=GOLD),fill=BLUE2,align=aln())
mc(ws4,r,11,r,14,"◀  FY2028 (Year 3)  ▶",font=Font(name="Calibri",size=10,bold=True,color=WHITE),fill=GREEN,align=aln())
r+=1

qs=["Q1'26","Q2'26","Q3'26","Q4'26","Q1'27","Q2'27","Q3'27","Q4'27","Q1'28","Q2'28","Q3'28","Q4'28"]
col_hdr(ws4,r,[2]+list(range(3,15)),["METRIC"]+qs, bg=NAVY, fg=GOLD)
r+=1

# DATA — quarterly breakdown (consistent with annual totals)
# Y1 quarterly: ramp. Y2: flat growth. Y3: flat growth.
sub_q = [255000,637500,1020000,1147500, 2081250,2406250,2587500,2755000, 4462500,5050000,5512500,6175000]
ps_q  = [375000,375000,375000,375000,  625000,625000,625000,625000,   1000000,1000000,1000000,1000000]
mkt_q = [0,0,0,0,                      0,168000,336000,504000,        472500,1260000,2047500,2520000]
rev_q = [s+p+m for s,p,m in zip(sub_q,ps_q,mkt_q)]
cogs_q= [round(r2*0.37) for r2 in rev_q]
gp_q  = [r2-c for r2,c in zip(rev_q,cogs_q)]

# Y1 OpEx quarterly (from $18M budget: Y1=~$9.9M total)
rd_q  = [1050000,1050000,1050000,1050000, 1125000,1125000,1125000,1125000, 1375000,1375000,1375000,1375000]
sm_q  = [625000,625000,625000,625000,    875000,875000,875000,875000,   1250000,1250000,1250000,1250000]
ga_q  = [375000,375000,375000,375000,    450000,450000,450000,450000,    625000,625000,625000,625000]
opex_q= [r2+s+g for r2,s,g in zip(rd_q,sm_q,ga_q)]
ebitda_q=[gp-opex for gp,opex in zip(gp_q,opex_q)]

pnl_rows=[
    ("New Customers",         [1,2,2,1, 3,3,2,2, 4,4,4,4]),
    ("Cumul. Customers",      [1,3,5,6, 9,12,14,16, 20,24,28,32]),
    ("",None),
    ("Subscription Revenue",  sub_q),
    ("Professional Services", ps_q),
    ("Marketplace Revenue",   mkt_q),
    ("TOTAL REVENUE",         rev_q),
    ("",None),
    ("Cost of Revenue",       cogs_q),
    ("GROSS PROFIT",          gp_q),
    ("Gross Margin %",        [f"{gp/rv*100:.0f}%" for gp,rv in zip(gp_q,rev_q)]),
    ("",None),
    ("R&D / Engineering",     rd_q),
    ("Sales & Marketing",     sm_q),
    ("General & Admin",       ga_q),
    ("TOTAL OPEX",            opex_q),
    ("",None),
    ("EBITDA",                ebitda_q),
]

for label, vals in pnl_rows:
    if not label:
        gap(ws4,r,2,14,5); r+=1; continue

    is_total = label.startswith("TOTAL") or label=="GROSS PROFIT" or label=="EBITDA"
    is_gm    = label == "Gross Margin %"
    ws4.row_dimensions[r].height = 22 if is_total else 18

    bg_fn = NAVY if is_total else (LT_GREY if r%2==0 else WHITE)
    fg_fn = GOLD if is_total else NAVY

    wc(ws4,r,2,label,
       font=Font(name="Calibri",size=9,bold=(is_total or label in ("R&D / Engineering","Sales & Marketing","General & Admin")),
                 color=WHITE if is_total else NAVY),
       fill=bg_fn, align=aln("left"), border=bdr())

    for qi,v in enumerate(vals):
        col=3+qi
        is_neg = isinstance(v,(int,float)) and v<0
        is_pos_ebitda = label=="EBITDA" and isinstance(v,(int,float)) and v>0
        fc = (RED if is_neg else (GREEN if is_pos_ebitda else (GOLD if is_total else "1F2937")))
        bg = GREEN_BG if is_pos_ebitda else (RED_BG if is_neg and label=="EBITDA" else bg_fn)
        fmt_v = '$#,##0' if isinstance(v,(int,float)) and abs(v)>99 else None
        wc(ws4,r,col,v,
           font=Font(name="Calibri",size=9,bold=is_total,color=fc),
           fill=bg, align=aln("right" if fmt_v else "center"), border=bdr(), fmt=fmt_v)
    r+=1

# ════════════════════════════════════════════════════════════════════════════
# SHEET 5 — ARR BRIDGE
# ════════════════════════════════════════════════════════════════════════════
ws5 = wb.create_sheet("ARR Bridge")
ws5.sheet_view.showGridLines = False
ws5.sheet_properties.tabColor = BLUE
set_cols(ws5,[2,30,16,16,16,2])

banner(ws5,1,2,5,"ARR BRIDGE  —  $5.1M  ▶  $14.8M  ▶  $32.4M", NAVY, WHITE, 13)
mc(ws5,2,2,2,5,"Annual Recurring Revenue waterfall  |  FY2026 → FY2027 → FY2028",
   font=Font(name="Calibri",size=9,italic=True,color=GOLD_LT), fill=NAVY2, align=aln())
ws5.row_dimensions[2].height=18

col_hdr(ws5,3,[2,3,4,5],["ARR BRIDGE COMPONENT","FY2026","FY2027","FY2028"])

bridge_rows=[
    ("Opening ARR (Start of Year)",           0,          5100000,   14800000,  NAVY),
    ("  + New Logo ARR  (6 → 10 → 16 logos × $830K)",4980000,8300000,13280000, BLUE),
    ("  + Expansion / Upsell ARR (NRR >130%)", 120000,   1600000,   5200000,   GREEN),
    ("  − Churned ARR  (<5% churn rate)",     -0,         -200000,   -880000,   RED),
    ("Closing ARR (End of Year)",             5100000,   14800000,  32400000,  NAVY),
    ("",None,None,None,None),
    ("YoY ARR Growth",                        "—",        "190%",    "119%",   BLUE2),
    ("New ARR Added in Year",                 5100000,    9700000,   17600000, GREEN),
    ("Net Revenue Retention",                 "~110%",    "~130%",   "~135%",  NAVY),
    ("Avg ARR per Customer (End of Year)",    850000,     925000,    1012500,  BLUE2),
]

r=4
for i,(label,v1,v2,v3,clr) in enumerate(bridge_rows):
    if not label: gap(ws5,r,2,5,6); r+=1; continue
    ws5.row_dimensions[r].height=22
    is_closing = "Closing" in label or "Opening" in label
    bg=MID_GREY if is_closing else (LT_GREY if i%2==0 else WHITE)
    wc(ws5,r,2,label,
       font=Font(name="Calibri",size=10,bold=is_closing,color=clr if clr else NAVY),
       fill=bg,align=aln("left"),border=bdr())
    for col,val in [(3,v1),(4,v2),(5,v3)]:
        is_neg=isinstance(val,(int,float)) and val<0
        is_n=isinstance(val,(int,float)) and abs(val)>99
        fc=RED if is_neg else (GREEN if (isinstance(val,(int,float)) and val>0 and "+" in label) else NAVY)
        wc(ws5,r,col,val,
           font=Font(name="Calibri",size=10,bold=is_closing,color=fc),
           fill=bg,align=aln("right" if is_n else "center"),border=bdr(),fmt='$#,##0' if is_n else None)
    r+=1

# ARR Line chart
r+=2
ws5.cell(r,2,"Year"); ws5.cell(r,3,"ARR ($)")
for yr,arr in [("Y1 End",5100000),("Y2 End",14800000),("Y3 End",32400000)]:
    ws5.cell(r+1+(["Y1 End","Y2 End","Y3 End"].index(yr)),2,yr)
    ws5.cell(r+1+(["Y1 End","Y2 End","Y3 End"].index(yr)),3,arr)
ch3=LineChart(); ch3.title="ARR Growth: $5.1M → $14.8M → $32.4M"
ch3.style=10; ch3.width=18; ch3.height=10
ch3.y_axis.title="ARR (USD)"
d=Reference(ws5,min_col=3,min_row=r,max_row=r+3)
cats=Reference(ws5,min_col=2,min_row=r+1,max_row=r+3)
ch3.add_data(d,titles_from_data=True)
ch3.set_categories(cats)
ch3.series[0].graphicalProperties.line.solidFill=BLUE
ch3.series[0].graphicalProperties.line.width=30000
marker=ch3.series[0].marker
marker.symbol="circle"; marker.size=8
ws5.add_chart(ch3,f"B{r+5}")


# ════════════════════════════════════════════════════════════════════════════
# SHEET 6 — UNIT ECONOMICS
# ════════════════════════════════════════════════════════════════════════════
ws6 = wb.create_sheet("Unit Economics")
ws6.sheet_view.showGridLines = False
ws6.sheet_properties.tabColor = NAVY
set_cols(ws6,[2,32,18,18,18,2])

banner(ws6,1,2,5,"UNIT ECONOMICS  |  CAC  ·  LTV  ·  Payback  ·  ACV", NAVY, WHITE, 13)
mc(ws6,2,2,2,5,"Per-customer economics driving the SaaS model",
   font=Font(name="Calibri",size=9,italic=True,color=GOLD_LT), fill=NAVY2, align=aln())
ws6.row_dimensions[2].height=18

col_hdr(ws6,3,[2,3,4,5],["METRIC","FY2026","FY2027","FY2028"])

ue_sections=[
    ("DEAL ECONOMICS",[
        ("Core Platform ARR / Site",                  650000,  650000,  650000,  '$#,##0'),
        ("LLM Feature Bundle ARR / Site",             180000,  180000,  180000,  '$#,##0'),
        ("Total ARR per Enterprise Site",             830000,  830000,  830000,  '$#,##0'),
        ("Quantum Readiness Sprint (PS, one-time)",   250000,  250000,  250000,  '$#,##0'),
        ("Year 1 Total Contract Value",              1080000, 1080000, 1080000,  '$#,##0'),
        ("Gross Margin — Subscription",               "72%",   "72%",  "72%",   None),
        ("Annual Gross Profit per Customer",          597600,  597600,  597600,  '$#,##0'),
    ]),
    ("CUSTOMER ACQUISITION COST",[
        ("Sales & Marketing Spend",                  2500000, 3500000, 5000000,  '$#,##0'),
        ("New Customers Won",                              6,      10,      16,  None),
        ("Blended CAC",                               416667,  350000,  312500,  '$#,##0'),
        ("Industry Benchmark CAC",                  "12–18mo payback","12–18mo","12–18mo", None),
    ]),
    ("LIFETIME VALUE  (LTV)",[
        ("Customer Lifetime (compliance lock-in)",  "6+ yrs", "6+ yrs","6+ yrs", None),
        ("Annual Gross Profit per Customer",          597600,  597600,  597600,  '$#,##0'),
        ("LTV — 3-Year Base",                        1792800, 1792800, 1792800,  '$#,##0'),
        ("LTV — 6-Year Full Lifecycle",              3585600, 3585600, 3585600,  '$#,##0'),
    ]),
    ("PAYBACK & EFFICIENCY",[
        ("CAC Payback Period",                       "8.4 mo","7.0 mo","6.3 mo", None),
        ("LTV : CAC Ratio (3-Year LTV)",                "4.3×",  "5.1×",  "5.7×", None),
        ("Net Revenue Retention",                    "~110%", "~130%", "~135%",  None),
        ("Churn Rate",                               "<5%",   "<5%",   "<5%",    None),
        ("Annual Cost per Autonomous Security Event","$0.01", "$0.01", "$0.01",  None),
        ("Equivalent Manual SOC Cost per Event",     "$10–$50","$10–$50","$10–$50",None),
        ("Autonomous Handling Rate",                 "78%",   "78%+",  "78%+",   None),
        ("Response Time (autonomous)",               "<10 sec","<10 sec","<10 sec",None),
    ]),
    ("MARKETPLACE UNIT ECONOMICS",[
        ("Marketplace GMV",                              0, 12000000, 70000000, '$#,##0'),
        ("Platform Take Rate",                        "30%",   "30%",   "30%",   None),
        ("Gross Marketplace Revenue",                    0,  3600000, 21000000, '$#,##0'),
        ("Marketplace Gross Margin",                  "90%",  "90%",   "90%",   None),
        ("Creator Revenue Share (70%)",                  0,  2520000, 14700000, '$#,##0'),
        ("Target Adapters Live",                    "500+", "1,000+","3,000+",  None),
    ]),
]

r=4
for sec_title, sec_rows in ue_sections:
    gap(ws6,r,2,5,6); r+=1
    sub_hdr(ws6,r,2,5,f"  {sec_title}"); r+=1
    for ii,(label,v1,v2,v3,fmt_v) in enumerate(sec_rows):
        ws6.row_dimensions[r].height=20
        bg=LT_GREY if ii%2==0 else WHITE
        wc(ws6,r,2,label,font=Font(name="Calibri",size=9,bold=True,color=NAVY),fill=bg,align=aln("left"),border=bdr())
        for col,val in [(3,v1),(4,v2),(5,v3)]:
            is_n=isinstance(val,(int,float)) and abs(val)>99
            green_it = (isinstance(val,(int,float)) and val>0 and "LTV" in label) or ("%" in label and "Margin" in label) or ("Ratio" in label)
            fc=GREEN if green_it else "1F2937"
            wc(ws6,r,col,val,font=Font(name="Calibri",size=9,color=fc),fill=bg,
               align=aln("right" if is_n else "center"),border=bdr(),
               fmt=fmt_v if is_n else None)
        r+=1


# ════════════════════════════════════════════════════════════════════════════
# SHEET 7 — MONTHLY CASH FLOW (Year 1) + Quarterly (Year 2–3)
# ════════════════════════════════════════════════════════════════════════════
ws7 = wb.create_sheet("Cash Flow")
ws7.sheet_view.showGridLines = False
ws7.sheet_properties.tabColor = GREEN
set_cols(ws7,[2,22,14,14,14,14,2])

banner(ws7,1,2,6,"CASH FLOW MODEL  —  MONTHLY (Y1)  +  QUARTERLY (Y2–Y3)", NAVY, WHITE, 13)
mc(ws7,2,2,2,6,"Opening cash: $18M (Series A close)  |  All figures in USD",
   font=Font(name="Calibri",size=9,italic=True,color=GOLD_LT), fill=NAVY2, align=aln())
ws7.row_dimensions[2].height=18

r=3
sub_hdr(ws7,r,2,6,"  YEAR 1 — MONTHLY (FY2026)  |  Starting Cash: $18,000,000")
r+=1
col_hdr(ws7,r,[2,3,4,5,6],["PERIOD","REVENUE IN ($)","EXPENSES OUT ($)","NET CASH ($)","CUMUL. CASH ($)"])
r+=1

# Monthly Y1: $750K/mo avg expenses, revenue ramps
monthly_y1=[
    ("Apr 2026  (M1)",   380000,  900000),
    ("May 2026  (M2)",   380000,  875000),
    ("Jun 2026  (M3)",   380000,  875000),
    ("Jul 2026  (M4)",   637500,  875000),
    ("Aug 2026  (M5)",   637500,  850000),
    ("Sep 2026  (M6)",   637500,  850000),
    ("Oct 2026  (M7)",  1020000,  840000),
    ("Nov 2026  (M8)",  1020000,  825000),
    ("Dec 2026  (M9)",  1020000,  825000),
    ("Jan 2027  (M10)", 1147500,  810000),
    ("Feb 2027  (M11)", 1147500,  800000),
    ("Mar 2027  (M12)", 1147500,  800000),
]
cum=18000000
for ii,(period,rev,exp) in enumerate(monthly_y1):
    net=rev-exp; cum+=net
    is_neg=net<0; is_pos=net>0
    bg=LT_GREY if ii%2==0 else WHITE
    ws7.row_dimensions[r].height=19
    wc(ws7,r,2,period,font=Font(name="Calibri",size=9,bold=True,color=NAVY),fill=bg,align=aln("left"),border=bdr())
    wc(ws7,r,3,rev,font=Font(name="Calibri",size=9,color=GREEN),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
    wc(ws7,r,4,exp,font=Font(name="Calibri",size=9,color=RED),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
    wc(ws7,r,5,net,font=Font(name="Calibri",size=9,bold=True,color=RED if is_neg else GREEN),
       fill=RED_BG if is_neg else GREEN_BG,align=aln("right"),border=bdr(),fmt='$#,##0')
    wc(ws7,r,6,cum,font=Font(name="Calibri",size=9,bold=True,color=NAVY),
       fill=LT_GREY if cum>5000000 else RED_BG,align=aln("right"),border=bdr(),fmt='$#,##0')
    r+=1

# Y1 Total
y1_rev=sum(r2[1] for r2 in monthly_y1); y1_exp=sum(r2[2] for r2 in monthly_y1)
total_row(ws7,r,2,["YEAR 1 TOTAL",y1_rev,y1_exp,y1_rev-y1_exp,cum],fmt='$#,##0')
r+=2

# Year 2 quarterly
sub_hdr(ws7,r,2,6,f"  YEAR 2 — QUARTERLY (FY2027)  |  Opening Cash: ${cum:,.0f}"); r+=1
col_hdr(ws7,r,[2,3,4,5,6],["QUARTER","REVENUE IN ($)","EXPENSES OUT ($)","NET CASH ($)","CUMUL. CASH ($)"])
r+=1
y2q=[("Q1 FY2027",3325000,2450000),("Q2 FY2027",3325000,2450000),
     ("Q3 FY2027",3344000,2450000),("Q4 FY2027",3344000,2450000)]
for ii,(q,rev,exp) in enumerate(y2q):
    net=rev-exp; cum+=net
    bg=LT_GREY if ii%2==0 else WHITE
    ws7.row_dimensions[r].height=19
    wc(ws7,r,2,q,font=Font(name="Calibri",size=9,bold=True,color=NAVY),fill=bg,align=aln("left"),border=bdr())
    wc(ws7,r,3,rev,font=Font(name="Calibri",size=9,color=GREEN),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
    wc(ws7,r,4,exp,font=Font(name="Calibri",size=9,color=RED),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
    net_c= GREEN if net>0 else RED
    wc(ws7,r,5,net,font=Font(name="Calibri",size=9,bold=True,color=net_c),
       fill=GREEN_BG if net>0 else RED_BG,align=aln("right"),border=bdr(),fmt='$#,##0')
    wc(ws7,r,6,cum,font=Font(name="Calibri",size=9,bold=True,color=NAVY),fill=LT_GREY,align=aln("right"),border=bdr(),fmt='$#,##0')
    r+=1
y2_rev=sum(q[1] for q in y2q); y2_exp=sum(q[2] for q in y2q)
total_row(ws7,r,2,["YEAR 2 TOTAL",y2_rev,y2_exp,y2_rev-y2_exp,cum],fmt='$#,##0')
r+=2

# Year 3 quarterly
sub_hdr(ws7,r,2,6,f"  YEAR 3 — QUARTERLY (FY2028)  |  Opening Cash: ${cum:,.0f}"); r+=1
col_hdr(ws7,r,[2,3,4,5,6],["QUARTER","REVENUE IN ($)","EXPENSES OUT ($)","NET CASH ($)","CUMUL. CASH ($)"])
r+=1
y3q=[("Q1 FY2028",7000000,3250000),("Q2 FY2028",7875000,3250000),
     ("Q3 FY2028",8312500,3250000),("Q4 FY2028",8312500,3250000)]
for ii,(q,rev,exp) in enumerate(y3q):
    net=rev-exp; cum+=net
    bg=LT_GREY if ii%2==0 else WHITE
    ws7.row_dimensions[r].height=19
    wc(ws7,r,2,q,font=Font(name="Calibri",size=9,bold=True,color=NAVY),fill=bg,align=aln("left"),border=bdr())
    wc(ws7,r,3,rev,font=Font(name="Calibri",size=9,color=GREEN),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
    wc(ws7,r,4,exp,font=Font(name="Calibri",size=9,color=RED),fill=bg,align=aln("right"),border=bdr(),fmt='$#,##0')
    wc(ws7,r,5,net,font=Font(name="Calibri",size=9,bold=True,color=GREEN),fill=GREEN_BG,align=aln("right"),border=bdr(),fmt='$#,##0')
    wc(ws7,r,6,cum,font=Font(name="Calibri",size=9,bold=True,color=NAVY),fill=LT_GREY,align=aln("right"),border=bdr(),fmt='$#,##0')
    r+=1
y3_rev=sum(q[1] for q in y3q); y3_exp=sum(q[2] for q in y3q)
total_row(ws7,r,2,["YEAR 3 TOTAL",y3_rev,y3_exp,y3_rev-y3_exp,cum],fmt='$#,##0')
r+=2
# Final callout
ws7.merge_cells(start_row=r,start_column=2,end_row=r,end_column=6)
ws7.row_dimensions[r].height=36
c=ws7.cell(r,2)
c.value=(f"CASH POSITION AT END OF YEAR 3 (FY2028):   ${cum:,.0f}   |   "
         "Self-funded from Year 2 onwards  |  Series B is optional, not existential")
c.font=Font(name="Calibri",size=10,bold=True,color=NAVY)
c.fill=fl(GOLD)
c.alignment=aln()
c.border=bdr_gold()


# ── PRINT SETTINGS ───────────────────────────────────────────────────────────
for ws in [ws1,ws2,ws3,ws4,ws5,ws6,ws7]:
    ws.page_setup.orientation="landscape"
    ws.page_setup.fitToPage=True
    ws.page_setup.fitToWidth=1
    ws.page_setup.fitToHeight=0
    ws.freeze_panes="B4"

out = "/Users/prabakarankannan/qbitel/docs/QBITEL_Bridge_Financial_Model.xlsx"
wb.save(out)
print(f"Saved → {out}")
