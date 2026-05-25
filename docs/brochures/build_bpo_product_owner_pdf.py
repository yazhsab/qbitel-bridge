"""
Build QBITEL Bridge BPO Product Owner Walkthrough — Professional PDF.

Thin wrapper over _walkthrough_render.build_walkthrough_pdf. All long-form
content lives in QBITEL_BPO_PRODUCT_OWNER_WALKTHROUGH.md; this script
supplies the cover-page values and output path.
"""
from _walkthrough_render import CoverSpec, build_walkthrough_pdf


def build_pdf():
    cover = CoverSpec(
        title_line_1='QBITEL',
        title_line_2='BRIDGE',
        subtitle='BPO PRODUCT OWNER WALKTHROUGH',
        tagline='One Platform Across Every Client Stack — Dialer, CRM, Ticketing',
        metric_boxes=[
            ('5 Modules', 'Discover • Understand\nModernize • Protect • Prove'),
            ('1,000+', 'Pre-Built Protocols\nin Marketplace'),
            ('Per-Tenant', 'Cryptographic, Policy &\nAudit Isolation'),
        ],
        metric_boxes_2=[
            ('2–4 Hours', 'AI Protocol Discovery\nFirst-Pass 89%+ Accuracy'),
            ('<2ms', 'PQC Voice Overhead\nWithin G.114 Budget'),
            ('9 Frameworks', 'PCI / HIPAA / SOC 2\nGDPR / SOX / etc.'),
        ],
        footer_pillars='Network Overlay  •  AI Discovery  •  Quantum-Safe  •  Multi-Tenant Isolation',
        version_line='Product Owner Walkthrough  |  Version 1.0  |  February 2026  |  Confidential',
        header_label='BPO Product Owner Walkthrough',
    )

    out = build_walkthrough_pdf(
        markdown_path='docs/brochures/QBITEL_BPO_PRODUCT_OWNER_WALKTHROUGH.md',
        output_path='docs/brochures/QBITEL_BPO_Product_Owner_Walkthrough.pdf',
        cover=cover,
    )
    print(f'PDF saved: {out}')
    return out


if __name__ == '__main__':
    build_pdf()
