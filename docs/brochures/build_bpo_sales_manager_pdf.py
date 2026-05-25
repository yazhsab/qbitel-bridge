"""
Build QBITEL Bridge BPO Sales Manager Walkthrough — Professional PDF.

Thin wrapper over _walkthrough_render.build_walkthrough_pdf. All long-form
content lives in QBITEL_BPO_SALES_MANAGER_WALKTHROUGH.md; this script
supplies the cover-page values and output path.
"""
from _walkthrough_render import CoverSpec, build_walkthrough_pdf


def build_pdf():
    cover = CoverSpec(
        title_line_1='QBITEL',
        title_line_2='BRIDGE',
        subtitle='BPO SALES MANAGER WALKTHROUGH',
        tagline='Selling Into BPOs With Multiple Dialers, CRMs & Ticketing Systems',
        metric_boxes=[
            ('30–60 Days', 'Toll Fraud ROI\nPayback Window'),
            ('Up to 80%', 'PCI-DSS Audit\nScope Reduction'),
            ('4–6 Hours', 'Full Deployment\nZero Downtime'),
        ],
        metric_boxes_2=[
            ('$500K–$2M', 'Annual Audit Cost\nReduction per Client'),
            ('78%', 'Autonomous Threat\nResolution'),
            ('$4.8M', 'Avg Breach Cost\nAvoided (IBM 2024)'),
        ],
        footer_pillars='Discovery  •  Demo Walk  •  ROI Math  •  Objection Handling  •  Close',
        version_line='Sales Manager Walkthrough  |  Version 1.0  |  February 2026  |  Confidential',
        header_label='BPO Sales Manager Walkthrough',
    )

    out = build_walkthrough_pdf(
        markdown_path='docs/brochures/QBITEL_BPO_SALES_MANAGER_WALKTHROUGH.md',
        output_path='docs/brochures/QBITEL_BPO_Sales_Manager_Walkthrough.pdf',
        cover=cover,
    )
    print(f'PDF saved: {out}')
    return out


if __name__ == '__main__':
    build_pdf()
