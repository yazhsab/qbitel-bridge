"""
Build QBITEL Bridge x TCN Partnership Brief — Professional PDF.

Thin wrapper over _walkthrough_render.build_walkthrough_pdf. All long-form
content lives in QBITEL_TCN_PARTNERSHIP_BRIEF.md; this script supplies the
cover-page values and output path.
"""
from _walkthrough_render import CoverSpec, build_walkthrough_pdf


def build_pdf():
    cover = CoverSpec(
        title_line_1='QBITEL',
        title_line_2='× TCN',
        subtitle='PARTNERSHIP BRIEF — VP OF TECHNOLOGY',
        tagline='Embedded Security & Compliance Layer for TCN Operator',
        metric_boxes=[
            ('5 Seams', 'SIP / Webhooks / REST\nCRM Hooks / SSO'),
            ('3 Models', 'Embedded OEM\nMarketplace / Referral'),
            ('4 Weeks', 'Joint POC\nto Working Demo'),
        ],
        metric_boxes_2=[
            ('<2ms', 'PQC Voice Overhead\nWithin G.114 Budget'),
            ('<1 sec', 'Toll Fraud Detect\n& Block on Outbound'),
            ('<10 min', 'Per-Tenant Compliance\nEvidence Pack'),
        ],
        footer_pillars='Integration Architecture  •  Flow Diagrams  •  Commercial Options  •  POC Plan',
        version_line='Partnership Brief  |  Version 1.0  |  February 2026  |  Confidential',
        header_label='Partnership Brief — TCN VP of Technology',
    )

    out = build_walkthrough_pdf(
        markdown_path='docs/brochures/QBITEL_TCN_PARTNERSHIP_BRIEF.md',
        output_path='docs/brochures/QBITEL_TCN_Partnership_Brief.pdf',
        cover=cover,
    )
    print(f'PDF saved: {out}')
    return out


if __name__ == '__main__':
    build_pdf()
