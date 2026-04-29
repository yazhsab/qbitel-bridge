const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType, TabStopPosition,
} = require("docx");

// ── Colors ──
const BRAND_DARK = "1B2A4A";
const BRAND_BLUE = "2E75B6";
const BRAND_ACCENT = "E8792F";
const BRAND_GREEN = "2E8B57";
const GRAY_LIGHT = "F2F2F2";
const WHITE = "FFFFFF";
const RED = "CC0000";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}
function para(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, alignment: opts.align || AlignmentType.LEFT, children: [new TextRun({ text, font: "Arial", size: 22, ...opts })] });
}
function boldPara(text, opts = {}) { return para(text, { bold: true, ...opts }); }
function spacer() { return new Paragraph({ spacing: { after: 60 }, children: [] }); }

function headerCell(text, width, opts = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: { fill: opts.fill || BRAND_DARK, type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT, children: [new TextRun({ text, bold: true, color: WHITE, font: "Arial", size: 18 })] })],
  });
}
function dataCell(text, width, opts = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT, children: [new TextRun({ text: String(text), font: "Arial", size: 18, bold: opts.bold || false, color: opts.color || "000000" })] })],
  });
}
function totalCell(text, width, opts = {}) {
  return dataCell(text, width, { bold: true, shading: "E8E8E8", ...opts });
}

function makeTable(headers, rows, colWidths, opts = {}) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const tRows = [
    new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i], { align: i > 0 ? AlignmentType.RIGHT : AlignmentType.LEFT })) }),
    ...rows.map((row, ri) =>
      new TableRow({
        children: row.map((cell, ci) => {
          const isTotal = cell.toString().startsWith("TOTAL") || row[0]?.toString().startsWith("TOTAL") || row[0]?.toString().startsWith("Net") || row[0]?.toString().startsWith("EBITDA") || row[0]?.toString().startsWith("Closing");
          const fn = isTotal ? totalCell : dataCell;
          return fn(cell, colWidths[ci], {
            shading: isTotal ? "E8E8E8" : (ri % 2 === 1 ? GRAY_LIGHT : undefined),
            align: ci > 0 ? AlignmentType.RIGHT : AlignmentType.LEFT,
            color: cell.toString().startsWith("-") || cell.toString().startsWith("(") ? RED : "000000",
          });
        }),
      })
    ),
  ];
  return new Table({ width: { size: totalWidth, type: WidthType.DXA }, columnWidths: colWidths, rows: tRows });
}

const numbering = {
  config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
    ]},
    { reference: "numbers", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
    ]},
  ],
};
function bullet(text, level = 0) {
  return new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 60 }, children: [new TextRun({ text, font: "Arial", size: 22 })] });
}

// ══════════════════════════════════════════════════════════
// COVER PAGE
// ══════════════════════════════════════════════════════════
function coverPage() {
  return [
    spacer(), spacer(), spacer(), spacer(), spacer(), spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "QBITEL BRIDGE", font: "Arial", size: 72, bold: true, color: BRAND_DARK })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "AI-Powered Quantum-Safe Legacy Modernisation Platform", font: "Arial", size: 28, color: BRAND_BLUE })] }),
    spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BRAND_ACCENT, space: 1 } }, children: [] }),
    spacer(), spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "FINANCIAL PLAN & PROPOSAL", font: "Arial", size: 40, bold: true, color: BRAND_DARK })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Series A: \u20B920 Crore | FY2026-FY2029", font: "Arial", size: 28, color: BRAND_BLUE })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [new TextRun({ text: "All figures in Indian Rupees (\u20B9) | India-First Capital Efficiency", font: "Arial", size: 24, color: BRAND_ACCENT, bold: true })] }),
    spacer(), spacer(), spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "STRICTLY CONFIDENTIAL", font: "Arial", size: 20, bold: true, color: RED })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "March 2026 | QBITEL Technologies Private Limited", font: "Arial", size: 20, color: "666666" })] }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ══════════════════════════════════════════════════════════
// MAIN CONTENT
// ══════════════════════════════════════════════════════════
const children = [
  ...coverPage(),

  // ═══ 1. FINANCIAL OVERVIEW ═══
  heading("1. Financial Overview & Key Assumptions"),
  spacer(),
  makeTable(
    ["Parameter", "Assumption"],
    [
      ["Exchange Rate", "1 USD = \u20B984 (RBI Reference Rate, March 2026)"],
      ["Financial Year", "April - March (Indian Standard)"],
      ["Revenue Recognition", "Subscription: Ratable over contract | PS: As delivered"],
      ["Accounting Standard", "Ind AS (Indian Accounting Standards)"],
      ["Currency", "All projections in INR unless stated"],
      ["Tax Regime", "25.17% effective corporate tax (new regime)"],
      ["GST", "18% on software services (pass-through to customer)"],
      ["Inflation Assumption", "5% annual salary inflation, 3% operational"],
      ["Discount Rate (WACC)", "18% for DCF valuations"],
      ["Terminal Growth Rate", "5% (conservative)"],
    ],
    [3500, 5860]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 2. REVENUE MODEL ═══
  heading("2. Revenue Model & Pricing Architecture"),
  spacer(),
  boldPara("2.1 Revenue Streams"),
  makeTable(
    ["Revenue Stream", "Description", "Gross Margin", "% of Revenue (Yr 1)"],
    [
      ["Platform Subscription", "Annual licence per site/deployment", "78%", "65%"],
      ["LLM AI Bundle", "AI-powered discovery + autonomous response add-on", "82%", "15%"],
      ["Professional Services", "Onboarding, training, architecture review, custom adapters", "50%", "15%"],
      ["Managed Detection (MDR)", "24x7 SOC-as-a-service with QBITEL platform", "60%", "5%"],
    ],
    [2200, 3200, 1500, 2460]
  ),
  spacer(),

  boldPara("2.2 Pricing Tiers (Per Site, Annual)"),
  makeTable(
    ["Tier", "Target Customer", "Subscription", "LLM Bundle", "PS (Year 1)", "Total Year 1 TCV"],
    [
      ["Enterprise", "Tier-1 Banks, Large Hospitals, Refineries", "\u20B91,20,00,000", "\u20B930,00,000", "\u20B950,00,000", "\u20B92,00,00,000"],
      ["Mid-Market", "PSBs, Mid-size Hospitals, Regional Utilities", "\u20B980,00,000", "\u20B920,00,000", "\u20B930,00,000", "\u20B91,30,00,000"],
      ["SME / Starter", "Small Banks, Clinics, BPO Centres", "\u20B950,00,000", "\u20B915,00,000", "\u20B920,00,000", "\u20B985,00,000"],
      ["Government / PSU", "Defence, ISRO, DRDO, PSUs (GeM)", "\u20B91,00,00,000", "\u20B925,00,000", "\u20B940,00,000", "\u20B91,65,00,000"],
    ],
    [1400, 2000, 1500, 1300, 1300, 1860]
  ),
  spacer(),

  boldPara("2.3 Contract Structure"),
  bullet("Preferred: 3-year contracts with annual billing (10% discount for 3-year commitment)"),
  bullet("Standard: 1-year annual subscription with auto-renewal"),
  bullet("Payment Terms: Net 30 days (enterprise), Net 45 days (government/PSU)"),
  bullet("Price Escalation: 5-7% annual increase built into multi-year contracts"),
  bullet("Multi-site Discount: 15% for 3+ sites, 25% for 10+ sites"),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 3. THREE-YEAR P&L ═══
  heading("3. Three-Year Profit & Loss Statement"),
  spacer(),

  boldPara("3.1 Revenue Projections"),
  makeTable(
    ["Revenue Line Item", "FY2026-27 (Year 1)", "FY2027-28 (Year 2)", "FY2028-29 (Year 3)"],
    [
      ["New Customers", "6", "10", "20"],
      ["Cumulative Customers", "6", "16", "36"],
      ["Avg. ARR per Customer", "\u20B91,20,00,000", "\u20B91,20,00,000", "\u20B91,11,00,000"],
      ["", "", "", ""],
      ["Subscription Revenue", "\u20B93,06,00,000", "\u20B912,48,00,000", "\u20B926,64,00,000"],
      ["LLM AI Bundle Revenue", "\u20B972,00,000", "\u20B93,36,00,000", "\u20B97,56,00,000"],
      ["Professional Services", "\u20B972,00,000", "\u20B93,20,00,000", "\u20B95,40,00,000"],
      ["Managed Detection (MDR)", "\u20B918,00,000", "\u20B996,00,000", "\u20B92,40,00,000"],
      ["TOTAL REVENUE", "\u20B94,68,00,000", "\u20B920,00,00,000", "\u20B942,00,00,000"],
    ],
    [3000, 2120, 2120, 2120]
  ),
  spacer(),

  boldPara("3.2 Cost of Revenue (COGS)"),
  makeTable(
    ["COGS Line Item", "FY2026-27", "FY2027-28", "FY2028-29"],
    [
      ["Cloud Infrastructure (AWS/Azure)", "\u20B936,00,000", "\u20B91,20,00,000", "\u20B92,10,00,000"],
      ["LLM Compute (GPU instances)", "\u20B924,00,000", "\u20B996,00,000", "\u20B91,80,00,000"],
      ["PS Delivery Cost (Engineers)", "\u20B948,00,000", "\u20B92,00,00,000", "\u20B93,00,00,000"],
      ["Third-Party Licences (HSM, etc.)", "\u20B912,00,000", "\u20B948,00,000", "\u20B984,00,000"],
      ["Customer Support (L1/L2)", "\u20B940,00,000", "\u20B91,20,00,000", "\u20B92,10,00,000"],
      ["TOTAL COGS", "\u20B91,60,00,000", "\u20B95,84,00,000", "\u20B99,84,00,000"],
      ["GROSS PROFIT", "\u20B93,08,00,000", "\u20B914,16,00,000", "\u20B932,16,00,000"],
      ["Gross Margin %", "65.8%", "70.8%", "76.6%"],
    ],
    [3500, 1953, 1953, 1954]
  ),
  spacer(),

  boldPara("3.3 Operating Expenses"),
  makeTable(
    ["OpEx Category", "FY2026-27", "FY2027-28", "FY2028-29"],
    [
      ["Engineering Salaries (22 people)", "\u20B94,40,00,000", "\u20B95,28,00,000", "\u20B96,60,00,000"],
      ["Sales & Marketing Salaries (9 ppl)", "\u20B92,16,00,000", "\u20B92,88,00,000", "\u20B93,60,00,000"],
      ["Marketing & Events", "\u20B960,00,000", "\u20B91,20,00,000", "\u20B91,80,00,000"],
      ["Office & Infrastructure", "\u20B948,00,000", "\u20B960,00,000", "\u20B984,00,000"],
      ["Legal & Compliance", "\u20B924,00,000", "\u20B936,00,000", "\u20B948,00,000"],
      ["Certifications (CERT-In, SOC2, ISO)", "\u20B960,00,000", "\u20B940,00,000", "\u20B924,00,000"],
      ["Travel & Business Development", "\u20B936,00,000", "\u20B960,00,000", "\u20B984,00,000"],
      ["Insurance & Miscellaneous", "\u20B924,00,000", "\u20B936,00,000", "\u20B948,00,000"],
      ["ESOP Cost (Non-Cash)", "\u20B948,00,000", "\u20B960,00,000", "\u20B972,00,000"],
      ["TOTAL OPERATING EXPENSES", "\u20B99,56,00,000", "\u20B912,28,00,000", "\u20B915,60,00,000"],
    ],
    [3500, 1953, 1953, 1954]
  ),
  spacer(),

  boldPara("3.4 EBITDA & Net Profit"),
  makeTable(
    ["Profitability Metric", "FY2026-27", "FY2027-28", "FY2028-29"],
    [
      ["Revenue", "\u20B94,68,00,000", "\u20B920,00,00,000", "\u20B942,00,00,000"],
      ["Gross Profit", "\u20B93,08,00,000", "\u20B914,16,00,000", "\u20B932,16,00,000"],
      ["Total Operating Expenses", "(\u20B99,56,00,000)", "(\u20B912,28,00,000)", "(\u20B915,60,00,000)"],
      ["EBITDA", "(\u20B96,48,00,000)", "\u20B91,88,00,000", "\u20B916,56,00,000"],
      ["EBITDA Margin", "-138.5%", "9.4%", "39.4%"],
      ["Depreciation & Amortisation", "(\u20B924,00,000)", "(\u20B936,00,000)", "(\u20B948,00,000)"],
      ["Interest Income", "\u20B940,00,000", "\u20B920,00,000", "\u20B930,00,000"],
      ["Profit Before Tax (PBT)", "(\u20B96,32,00,000)", "\u20B91,72,00,000", "\u20B916,38,00,000"],
      ["Tax (25.17%)", "\u20B90", "(\u20B943,29,240)", "(\u20B94,12,28,460)"],
      ["Net Profit After Tax", "(\u20B96,32,00,000)", "\u20B91,28,70,760", "\u20B912,25,71,540"],
      ["Net Margin", "-135.0%", "6.4%", "29.2%"],
    ],
    [3500, 1953, 1953, 1954]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 4. CASH FLOW ═══
  heading("4. Cash Flow Projections"),
  spacer(),
  makeTable(
    ["Cash Flow Item", "FY2026-27", "FY2027-28", "FY2028-29"],
    [
      ["Opening Cash Balance", "\u20B920,00,00,000", "\u20B913,37,00,000", "\u20B914,93,70,760"],
      ["", "", "", ""],
      ["Cash from Operations:", "", "", ""],
      ["Net Profit / (Loss)", "(\u20B96,32,00,000)", "\u20B91,28,70,760", "\u20B912,25,71,540"],
      ["Add: Depreciation", "\u20B924,00,000", "\u20B936,00,000", "\u20B948,00,000"],
      ["Add: ESOP (Non-Cash)", "\u20B948,00,000", "\u20B960,00,000", "\u20B972,00,000"],
      ["Working Capital Changes", "(\u20B960,00,000)", "(\u20B940,00,000)", "(\u20B960,00,000)"],
      ["Net Cash from Operations", "(\u20B96,20,00,000)", "\u20B91,84,70,760", "\u20B912,85,71,540"],
      ["", "", "", ""],
      ["Cash from Investing:", "", "", ""],
      ["CapEx (IT, Office)", "(\u20B936,00,000)", "(\u20B924,00,000)", "(\u20B936,00,000)"],
      ["Patent Filing", "(\u20B97,00,000)", "(\u20B94,00,000)", "(\u20B94,00,000)"],
      ["Net Cash from Investing", "(\u20B943,00,000)", "(\u20B928,00,000)", "(\u20B940,00,000)"],
      ["", "", "", ""],
      ["Cash from Financing:", "", "", ""],
      ["Series A Proceeds", "\u20B920,00,00,000", "\u20B90", "\u20B90"],
      ["Series B Proceeds (if needed)", "\u20B90", "\u20B90", "\u20B90"],
      ["Net Cash from Financing", "\u20B920,00,00,000", "\u20B90", "\u20B90"],
      ["", "", "", ""],
      ["Closing Cash Balance", "\u20B913,37,00,000", "\u20B914,93,70,760", "\u20B927,39,42,300"],
      ["Monthly Burn Rate", "\u20B955,50,000", "Cash Positive", "Cash Positive"],
    ],
    [3500, 1953, 1953, 1954]
  ),
  spacer(),
  boldPara("Key Takeaway: The company reaches cash-flow positive in Q2 FY2027-28 (Month 19), and accumulates \u20B927.4 Crore cash by FY2028-29 end. Series B funding is optional."),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 5. UNIT ECONOMICS ═══
  heading("5. Unit Economics"),
  spacer(),
  makeTable(
    ["Metric", "FY2026-27 (Year 1)", "FY2027-28 (Year 2)", "FY2028-29 (Year 3)", "Benchmark"],
    [
      ["Customer Acquisition Cost (CAC)", "\u20B935,00,000", "\u20B930,00,000", "\u20B925,00,000", "<\u20B950L"],
      ["CAC Payback Period", "5.5 months", "4.8 months", "4.2 months", "<12 months"],
      ["Customer Lifetime Value (LTV, 5yr)", "\u20B92,60,00,000", "\u20B92,80,00,000", "\u20B93,00,00,000", ">3x CAC"],
      ["LTV : CAC Ratio", "7.4x", "9.3x", "12.0x", ">3.0x"],
      ["Gross Margin", "65.8%", "70.8%", "76.6%", ">60%"],
      ["Net Revenue Retention (NRR)", "120%", "128%", "135%", ">110%"],
      ["Annual Churn Rate", "<5%", "<4%", "<3%", "<10%"],
      ["Revenue per Employee", "\u20B918,72,000", "\u20B955,55,556", "\u20B993,33,333", "Growing"],
      ["Cost per Security Event", "\u20B90.84", "\u20B90.63", "\u20B90.42", "<\u20B95"],
      ["ACV (Avg. Contract Value)", "\u20B91,20,00,000", "\u20B91,25,00,000", "\u20B91,17,00,000", "Growing"],
      ["Rule of 40 Score", "-73 (pre-revenue)", "165", "82", ">40"],
    ],
    [2800, 1640, 1640, 1640, 1640]
  ),
  spacer(),
  boldPara("Why These Metrics Are World-Class"),
  bullet("LTV:CAC of 7.4x-12x is 2-4x better than the industry benchmark of 3x"),
  bullet("CAC payback of 4-6 months is 3x faster than the 12-18 month SaaS benchmark"),
  bullet("NRR of 128-135% means every cohort grows 28-35% annually without new sales"),
  bullet("Gross margin expansion from 66% to 77% reflects platform leverage and scale"),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 6. HEADCOUNT & SALARY PLAN ═══
  heading("6. Headcount & Compensation Plan"),
  spacer(),
  boldPara("6.1 Headcount Ramp"),
  makeTable(
    ["Department", "Current", "FY2026-27", "FY2027-28", "FY2028-29"],
    [
      ["Engineering (AI/ML, PQC, Platform)", "15", "22", "28", "35"],
      ["Sales (AE, SE, SDR)", "5", "9", "14", "18"],
      ["Customer Success & Support", "1", "2", "4", "6"],
      ["Marketing", "1", "2", "3", "4"],
      ["Operations (Finance, Legal, HR, Admin)", "2", "3", "4", "5"],
      ["Leadership (CXO)", "2", "3", "3", "4"],
      ["TOTAL HEADCOUNT", "26", "41", "56", "72"],
    ],
    [3000, 1590, 1590, 1590, 1590]
  ),
  spacer(),

  boldPara("6.2 Compensation Benchmarks (Annual CTC in LPA)"),
  makeTable(
    ["Role", "CTC Range (LPA)", "Avg. CTC (LPA)", "Headcount (Yr 1)", "Annual Cost"],
    [
      ["CTO / Co-Founder", "\u20B980-100 LPA", "\u20B980 LPA", "1", "\u20B980,00,000"],
      ["CEO / Co-Founder", "\u20B960-80 LPA", "\u20B960 LPA", "1", "\u20B960,00,000"],
      ["VP Engineering / VP Sales", "\u20B950-70 LPA", "\u20B960 LPA", "2", "\u20B91,20,00,000"],
      ["Senior AI/ML Engineers", "\u20B925-35 LPA", "\u20B928 LPA", "4", "\u20B91,12,00,000"],
      ["PQC / Security Engineers", "\u20B922-30 LPA", "\u20B925 LPA", "3", "\u20B975,00,000"],
      ["Platform Engineers (Rust/Go)", "\u20B920-28 LPA", "\u20B924 LPA", "4", "\u20B996,00,000"],
      ["Account Executives", "\u20B925-35 LPA", "\u20B930 LPA", "6", "\u20B91,80,00,000"],
      ["Solution Engineers (Pre-sales)", "\u20B918-25 LPA", "\u20B920 LPA", "2", "\u20B940,00,000"],
      ["QA / DevSecOps", "\u20B915-22 LPA", "\u20B918 LPA", "3", "\u20B954,00,000"],
      ["Customer Success / Support", "\u20B912-18 LPA", "\u20B915 LPA", "2", "\u20B930,00,000"],
      ["Marketing", "\u20B915-22 LPA", "\u20B918 LPA", "2", "\u20B936,00,000"],
      ["Operations (Finance, Legal, Admin)", "\u20B910-15 LPA", "\u20B912 LPA", "3", "\u20B936,00,000"],
      ["TOTAL", "", "", "33+", "\u20B98,19,00,000"],
    ],
    [2400, 1500, 1300, 1560, 2500]
  ),
  spacer(),
  bullet("5% annual salary inflation factored into Year 2 and Year 3 projections"),
  bullet("Sales roles include base + 30-40% variable component (commission on ARR closed)"),
  bullet("ESOP pool: 15% of post-money equity, 4-year vesting with 1-year cliff"),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 7. CUSTOMER ACQUISITION ═══
  heading("7. Customer Acquisition Plan"),
  spacer(),
  boldPara("7.1 Year 1 Pipeline (6 Target Customers)"),
  makeTable(
    ["#", "Target Segment", "Target Customer Profile", "Expected ARR", "Sales Stage"],
    [
      ["1", "Banking (Pvt.)", "Top-10 Private Bank (ICICI, HDFC, Kotak)", "\u20B91,50,00,000", "Pilot Planning"],
      ["2", "Banking (PSB)", "SBI / Bank of Baroda / PNB", "\u20B91,20,00,000", "POV Proposal"],
      ["3", "Healthcare", "Large Hospital Chain (Apollo, Fortis, Max)", "\u20B91,00,00,000", "Initial Discussion"],
      ["4", "Critical Infra", "Power Utility (NTPC, PowerGrid, Adani Power)", "\u20B91,50,00,000", "RFP Response"],
      ["5", "Government/PSU", "Defence / DRDO / ISRO (via GeM)", "\u20B91,00,00,000", "GeM Registration"],
      ["6", "BPO / Telecom", "Large BPO (TCS BPS, Wipro BPS, Genpact)", "\u20B91,00,00,000", "Proof of Concept"],
    ],
    [500, 1500, 3000, 2000, 2360]
  ),
  spacer(),

  boldPara("7.2 Year 2-3 Expansion Strategy"),
  bullet("Year 2: 10 new customers (banking expansion + healthcare + government/defence)"),
  bullet("Year 2: International pilots in UAE (banking) and Singapore (telecom)"),
  bullet("Year 3: 20 new customers including 6 international (UAE, Singapore, ASEAN)"),
  bullet("Multi-site expansion: Upsell existing customers from 1 site to 3-5 sites (NRR driver)"),
  bullet("MSSP channel: Tata Communications, Wipro Cyber Defence resell to their customer base"),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 8. VALUATION ═══
  heading("8. Valuation Analysis"),
  spacer(),
  boldPara("8.1 Comparable Transaction Analysis"),
  makeTable(
    ["Company / Transaction", "Revenue Multiple", "Implied Valuation", "Notes"],
    [
      ["CrowdStrike (at similar ARR stage)", "25-35x ARR", "\u20B91,000-1,400 Cr", "Cloud-native security, high NRR"],
      ["SentinelOne (at similar stage)", "18-25x ARR", "\u20B9720-1,000 Cr", "AI-first endpoint security"],
      ["Palo Alto acquisition of Demisto", "20x ARR", "\u20B9800 Cr", "SOAR platform, $560M deal"],
      ["Cisco acquisition of Splunk", "12x ARR", "\u20B9480 Cr", "Mature revenue, large customer base"],
      ["India SaaS (Zoho/Freshworks stage)", "12-20x ARR", "\u20B9480-800 Cr", "India-based SaaS comparables"],
      ["Applied QBITEL Multiple (Base)", "15-20x ARR", "\u20B9600-800 Cr", "PQC premium + deep tech moat"],
    ],
    [3000, 1500, 2000, 2860]
  ),
  spacer(),

  boldPara("8.2 DCF Valuation (5-Year Horizon)"),
  makeTable(
    ["Year", "Revenue", "Free Cash Flow", "Discount Factor (18%)", "Present Value"],
    [
      ["FY2026-27", "\u20B94.68 Cr", "(\u20B96.63 Cr)", "0.847", "(\u20B95.62 Cr)"],
      ["FY2027-28", "\u20B920.00 Cr", "\u20B91.57 Cr", "0.718", "\u20B91.13 Cr"],
      ["FY2028-29", "\u20B942.00 Cr", "\u20B912.46 Cr", "0.609", "\u20B97.59 Cr"],
      ["FY2029-30", "\u20B972.00 Cr", "\u20B924.00 Cr", "0.516", "\u20B912.38 Cr"],
      ["FY2030-31", "\u20B9110.00 Cr", "\u20B938.50 Cr", "0.437", "\u20B916.82 Cr"],
      ["Terminal Value (5% growth)", "", "\u20B9310.38 Cr", "0.437", "\u20B9135.64 Cr"],
      ["Enterprise Value (DCF)", "", "", "", "\u20B9167.94 Cr"],
    ],
    [1500, 1800, 1800, 2000, 2260]
  ),
  spacer(),
  para("The DCF analysis yields an enterprise value of approximately \u20B9168 Crore, validating the \u20B960 Crore pre-money Series A valuation as attractive for investors with 2.8x upside to fundamental value."),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 9. SENSITIVITY ANALYSIS ═══
  heading("9. Sensitivity Analysis"),
  spacer(),
  boldPara("9.1 Revenue Sensitivity to Customer Count & ACV"),
  makeTable(
    ["Scenario", "Yr 3 Customers", "Avg. ACV", "Yr 3 Revenue", "Yr 3 EBITDA"],
    [
      ["Bear Case (-30%)", "25", "\u20B91,00,00,000", "\u20B929.4 Cr", "\u20B98.8 Cr"],
      ["Base Case", "36", "\u20B91,17,00,000", "\u20B942.0 Cr", "\u20B916.6 Cr"],
      ["Bull Case (+30%)", "47", "\u20B91,30,00,000", "\u20B958.8 Cr", "\u20B926.0 Cr"],
      ["Optimistic (+50%)", "54", "\u20B91,40,00,000", "\u20B972.0 Cr", "\u20B936.0 Cr"],
    ],
    [2000, 1500, 1800, 2000, 2060]
  ),
  spacer(),

  boldPara("9.2 Breakeven Sensitivity"),
  makeTable(
    ["Variable Changed", "Bear Case", "Base Case", "Bull Case"],
    [
      ["Sales Cycle Length", "9 months (+2)", "7 months", "5 months (-2)"],
      ["Breakeven Month", "Month 24", "Month 19", "Month 15"],
      ["Cash at Breakeven", "\u20B96.2 Cr", "\u20B913.4 Cr", "\u20B916.1 Cr"],
      ["Additional Funding Needed?", "Possibly \u20B95 Cr bridge", "No", "No"],
    ],
    [3000, 2120, 2120, 2120]
  ),
  spacer(),

  boldPara("9.3 Exit Valuation Sensitivity"),
  makeTable(
    ["Revenue Multiple", "Yr 3 Revenue \u20B929 Cr", "Yr 3 Revenue \u20B942 Cr", "Yr 3 Revenue \u20B959 Cr"],
    [
      ["10x (Conservative)", "\u20B9294 Cr (2.0x)", "\u20B9420 Cr (3.5x)", "\u20B9588 Cr (5.2x)"],
      ["15x (Base)", "\u20B9441 Cr (3.5x)", "\u20B9630 Cr (5.6x)", "\u20B9882 Cr (7.8x)"],
      ["20x (Growth Premium)", "\u20B9588 Cr (5.2x)", "\u20B9840 Cr (7.5x)", "\u20B91,176 Cr (10.5x)"],
      ["25x (Strategic Acq.)", "\u20B9735 Cr (6.5x)", "\u20B91,050 Cr (9.4x)", "\u20B91,470 Cr (13.1x)"],
    ],
    [2500, 2287, 2286, 2287]
  ),
  spacer(),
  para("Note: Values in parentheses represent MOIC (Multiple on Invested Capital) for Series A investors."),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 10. FUND DEPLOYMENT ═══
  heading("10. Detailed Fund Deployment Schedule"),
  spacer(),
  boldPara("10.1 Quarter-by-Quarter Deployment (\u20B920 Crore)"),
  makeTable(
    ["Category", "Q1 (Apr-Jun)", "Q2 (Jul-Sep)", "Q3 (Oct-Dec)", "Q4 (Jan-Mar)", "FY Total"],
    [
      ["Engineering Salaries", "\u20B980L", "\u20B91.0 Cr", "\u20B91.2 Cr", "\u20B91.4 Cr", "\u20B94.4 Cr"],
      ["Sales & Marketing Salaries", "\u20B940L", "\u20B950L", "\u20B960L", "\u20B966L", "\u20B92.16 Cr"],
      ["Marketing Campaigns & Events", "\u20B910L", "\u20B915L", "\u20B920L", "\u20B915L", "\u20B960L"],
      ["Cloud Infrastructure", "\u20B96L", "\u20B98L", "\u20B910L", "\u20B912L", "\u20B936L"],
      ["Certifications (CERT-In, SOC2)", "\u20B925L", "\u20B915L", "\u20B910L", "\u20B910L", "\u20B960L"],
      ["Office & IT", "\u20B915L", "\u20B910L", "\u20B912L", "\u20B911L", "\u20B948L"],
      ["Legal & Compliance", "\u20B98L", "\u20B96L", "\u20B95L", "\u20B95L", "\u20B924L"],
      ["Travel & BD", "\u20B96L", "\u20B98L", "\u20B910L", "\u20B912L", "\u20B936L"],
      ["Insurance & Misc", "\u20B96L", "\u20B96L", "\u20B96L", "\u20B96L", "\u20B924L"],
      ["TOTAL QUARTERLY SPEND", "\u20B91.96 Cr", "\u20B92.18 Cr", "\u20B92.53 Cr", "\u20B92.77 Cr", "\u20B99.44 Cr"],
    ],
    [2500, 1372, 1372, 1372, 1372, 1372]
  ),
  spacer(),
  para("Note: Year 1 total deployment of \u20B99.44 Crore against \u20B920 Crore raised, leaving \u20B910.56 Crore cash reserve entering Year 2 (before considering revenue). With \u20B94.68 Crore revenue in Year 1, closing cash is approximately \u20B913.37 Crore."),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 11. KEY FINANCIAL RATIOS ═══
  heading("11. Key Financial Ratios & KPIs"),
  spacer(),
  makeTable(
    ["Ratio / KPI", "FY2026-27", "FY2027-28", "FY2028-29", "Target"],
    [
      ["Revenue Growth Rate", "N/A (Year 1)", "327%", "110%", ">100% Y2"],
      ["Gross Margin", "65.8%", "70.8%", "76.6%", ">70%"],
      ["EBITDA Margin", "-138.5%", "9.4%", "39.4%", ">25% Y3"],
      ["Net Margin", "-135.0%", "6.4%", "29.2%", ">20% Y3"],
      ["ARR per Employee", "\u20B917.6L", "\u20B934.3L", "\u20B955.6L", "Growing"],
      ["Burn Multiple", "1.3x", "N/A (profitable)", "N/A", "<2x"],
      ["Cash Runway (months)", "22+", "Infinite", "Infinite", ">18"],
      ["Quick Ratio (Liquid)", "8.5x", "12.0x", "15.0x", ">1.0x"],
      ["Debt-to-Equity", "0.0x", "0.0x", "0.0x", "<0.5x"],
      ["Return on Equity (ROE)", "-31.6%", "5.7%", "29.5%", ">20% Y3"],
      ["Magic Number", "0.35", "1.2", "1.8", ">1.0"],
    ],
    [3000, 1590, 1590, 1590, 1590]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 12. INVESTOR RETURN ANALYSIS ═══
  heading("12. Investor Return Analysis"),
  spacer(),
  boldPara("12.1 Series A Returns (25% Ownership at \u20B920 Crore)"),
  makeTable(
    ["Exit Scenario", "Timeline", "Exit Valuation", "Investor Proceeds", "MOIC", "IRR"],
    [
      ["Conservative Acquisition", "Year 3 (FY2029)", "\u20B9300 Cr", "\u20B975 Cr", "3.75x", "~55%"],
      ["Base Case Acquisition", "Year 3-4", "\u20B9600 Cr", "\u20B9150 Cr", "7.5x", "~85%"],
      ["Strategic Premium Acq.", "Year 3-4", "\u20B9900 Cr", "\u20B9225 Cr", "11.25x", "~105%"],
      ["NSE IPO (Bull)", "Year 4-5", "\u20B91,200 Cr", "\u20B9300 Cr", "15.0x", "~115%"],
    ],
    [2300, 1200, 1500, 1560, 1200, 1600]
  ),
  spacer(),

  boldPara("12.2 Value Creation Bridge"),
  makeTable(
    ["Milestone", "Valuation Impact", "Cumulative Valuation"],
    [
      ["Series A Investment", "\u20B980 Cr (post-money)", "\u20B980 Cr"],
      ["6 Customers + \u20B97.2 Cr ARR", "+\u20B940 Cr (ARR validation)", "\u20B9120 Cr"],
      ["EBITDA Breakeven (Month 19)", "+\u20B960 Cr (de-risked)", "\u20B9180 Cr"],
      ["16 Customers + \u20B919.2 Cr ARR", "+\u20B9120 Cr (scale proof)", "\u20B9300 Cr"],
      ["International Revenue (Year 3)", "+\u20B9150 Cr (global potential)", "\u20B9450 Cr"],
      ["36 Customers + \u20B940 Cr ARR", "+\u20B9150-550 Cr (exit premium)", "\u20B9600-1,000 Cr"],
    ],
    [3000, 3360, 3000]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 13. APPENDIX ═══
  heading("13. Appendices"),
  spacer(),

  heading("Appendix A: Monthly Cash Flow (Year 1 Detail)", HeadingLevel.HEADING_2),
  makeTable(
    ["Month", "Revenue", "Expenses", "Net Cash Flow", "Closing Balance"],
    [
      ["Apr 2026", "\u20B90", "\u20B955L", "(\u20B955L)", "\u20B919.45 Cr"],
      ["May 2026", "\u20B90", "\u20B958L", "(\u20B958L)", "\u20B918.87 Cr"],
      ["Jun 2026", "\u20B930L", "\u20B960L", "(\u20B930L)", "\u20B918.57 Cr"],
      ["Jul 2026", "\u20B935L", "\u20B965L", "(\u20B930L)", "\u20B918.27 Cr"],
      ["Aug 2026", "\u20B935L", "\u20B968L", "(\u20B933L)", "\u20B917.94 Cr"],
      ["Sep 2026", "\u20B940L", "\u20B970L", "(\u20B930L)", "\u20B917.64 Cr"],
      ["Oct 2026", "\u20B945L", "\u20B972L", "(\u20B927L)", "\u20B917.37 Cr"],
      ["Nov 2026", "\u20B950L", "\u20B975L", "(\u20B925L)", "\u20B917.12 Cr"],
      ["Dec 2026", "\u20B960L", "\u20B978L", "(\u20B918L)", "\u20B916.94 Cr"],
      ["Jan 2027", "\u20B965L", "\u20B980L", "(\u20B915L)", "\u20B916.79 Cr"],
      ["Feb 2027", "\u20B970L", "\u20B982L", "(\u20B912L)", "\u20B916.67 Cr"],
      ["Mar 2027", "\u20B980L", "\u20B985L", "(\u20B95L)", "\u20B916.62 Cr"],
    ],
    [1200, 1600, 1600, 1800, 3160]
  ),
  spacer(),

  heading("Appendix B: Key Assumptions Summary", HeadingLevel.HEADING_2),
  bullet("Average deal size: \u20B91.2 Crore Year 1, growing to \u20B91.17 Crore as SME tier added"),
  bullet("Sales cycle: 7 months average (5-9 month range)"),
  bullet("Win rate: 25% Year 1, improving to 35% Year 3"),
  bullet("Quota per AE: \u20B92.5 Crore ARR annually"),
  bullet("Net Revenue Retention: 128% (cross-sell, multi-site, upsell)"),
  bullet("Annual churn: <5% (compliance lock-in reduces churn)"),
  bullet("Salary inflation: 5% annually"),
  bullet("Cloud cost: Scales sub-linearly with customers (shared infrastructure)"),
  bullet("Tax rate: 25.17% (new corporate tax regime, no tax holiday assumed)"),
  spacer(),

  heading("Appendix C: Regulatory Timeline", HeadingLevel.HEADING_2),
  makeTable(
    ["Quarter", "Regulation", "Impact"],
    [
      ["Q1 FY2026", "RBI IT Master Direction enforcement", "Banks must demonstrate quantum readiness"],
      ["Q2 FY2026", "SEBI Cybersecurity compliance", "Stock exchanges, brokers, depositories"],
      ["Q3 FY2026", "IRDAI cyber guidelines enforcement", "Insurance companies must comply"],
      ["Q4 FY2026", "TRAI telecom security rules", "Telecom operators mandate"],
      ["Q1 FY2027", "ABDM health data standards", "Hospitals and health systems"],
      ["Q2 FY2027", "DPDP Act enforcement", "Cross-sector data protection"],
      ["Q3 FY2027", "CERT-In advanced reporting", "Enhanced incident reporting"],
      ["FY2028+", "NIST PQC migration deadlines", "Global quantum-safe requirements"],
    ],
    [1500, 3500, 4360]
  ),
  spacer(), spacer(),

  new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BRAND_ACCENT, space: 1 } }, children: [] }),
  spacer(),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "End of Financial Plan & Proposal", font: "Arial", size: 24, bold: true, color: BRAND_DARK })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "QBITEL Technologies Private Limited | Strictly Confidential", font: "Arial", size: 20, color: "666666" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "March 2026", font: "Arial", size: 20, color: "666666" })] }),
];

// ── Build Document ──
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BRAND_DARK },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BRAND_BLUE, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: BRAND_BLUE },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
    ],
  },
  numbering,
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: BRAND_BLUE, space: 4 } },
            children: [
              new TextRun({ text: "QBITEL Bridge | Financial Plan & Proposal | FY2026-FY2029", font: "Arial", size: 16, color: "999999" }),
            ],
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          }),
        ],
      }),
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 2, color: BRAND_BLUE, space: 4 } },
            children: [
              new TextRun({ text: "STRICTLY CONFIDENTIAL | QBITEL Technologies Pvt. Ltd.", font: "Arial", size: 14, color: "999999" }),
              new TextRun({ text: "\tPage ", font: "Arial", size: 14, color: "999999" }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 14, color: "999999" }),
            ],
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          }),
        ],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/Users/prabakarankannan/qbitel/docs/QBITEL_Financial_Plan_Proposal_2026_INR.docx", buffer);
  console.log("Financial Plan document created successfully!");
});
