"""Generate high-resolution figures for OSFY article on QBITEL Bridge."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Shared styling ──────────────────────────────────────────────────
BLUE = "#2E5FA1"
DARK_BLUE = "#1A3A6B"
LIGHT_BLUE = "#D5E8F0"
GREEN = "#2E8B57"
LIGHT_GREEN = "#D5F0E0"
ORANGE = "#D4762C"
LIGHT_ORANGE = "#F5E0CC"
GRAY = "#666666"
LIGHT_GRAY = "#F0F0F0"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
})


def rounded_box(ax, x, y, w, h, text, fc, ec, fontsize=10, fontweight="normal", textcolor="black"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.5,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight=fontweight, color=textcolor,
        transform=ax.transAxes, zorder=3
    )


def arrow_between(ax, x1, y1, x2, y2, color=GRAY):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
        zorder=1
    )


# ═════════════════════════════════════════════════════════════════════
# Figure 1: Protocol discovery pipeline
# ═════════════════════════════════════════════════════════════════════
def figure1():
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, "QBITEL Bridge protocol discovery pipeline", ha="center", va="top",
            fontsize=14, fontweight="bold", color=DARK_BLUE)
    ax.text(0.5, 0.88, "From raw traffic to validated parsers", ha="center", va="top",
            fontsize=11, color=GRAY)

    # Pipeline stages
    stages = [
        ("Raw\nTraffic", LIGHT_GRAY, GRAY),
        ("Statistical\nAnalysis", LIGHT_BLUE, BLUE),
        ("PCFG Grammar\nLearning", LIGHT_BLUE, BLUE),
        ("Dynamic Parser\nGeneration", LIGHT_BLUE, BLUE),
        ("Ensemble ML\nClassification", LIGHT_GREEN, GREEN),
        ("Compliance\nValidation", LIGHT_ORANGE, ORANGE),
    ]

    box_w = 0.13
    box_h = 0.28
    start_x = 0.03
    gap = 0.035
    y_center = 0.45

    for i, (label, fc, ec) in enumerate(stages):
        x = start_x + i * (box_w + gap)
        rounded_box(ax, x, y_center, box_w, box_h, label, fc, ec, fontsize=9, fontweight="bold")
        if i > 0:
            arrow_between(ax, x - gap + 0.005, y_center + box_h / 2, x - 0.005, y_center + box_h / 2, color=ec)

    # Sub-labels beneath each box
    sublabels = [
        "Packet\ncaptures",
        "Entropy &\nbyte patterns",
        "EM algorithm\nproduction rules",
        "Runtime\nparsers",
        "CNN + LSTM\n+ Random Forest",
        "Anomaly\ndetection",
    ]
    for i, sub in enumerate(sublabels):
        x = start_x + i * (box_w + gap) + box_w / 2
        ax.text(x, y_center - 0.06, sub, ha="center", va="top", fontsize=8, color=GRAY,
                transform=ax.transAxes)

    # Output arrow
    final_x = start_x + 5 * (box_w + gap) + box_w + 0.02
    ax.annotate(
        "", xy=(0.97, y_center + box_h / 2), xytext=(final_x, y_center + box_h / 2),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5),
    )
    ax.text(0.97, y_center + box_h / 2 + 0.06, "Protected\nperimeter", ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=ORANGE, transform=ax.transAxes)

    # Confidence threshold annotation
    ax.text(start_x + 4 * (box_w + gap) + box_w / 2, y_center + box_h + 0.04,
            "Confidence threshold: 0.7", ha="center", va="bottom",
            fontsize=8, style="italic", color=GREEN, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig("/Users/prabakarankannan/qbitel/docs/figure1_discovery_pipeline.png",
                bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    print("Created figure1_discovery_pipeline.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 2: PQC algorithm selection by deployment domain
# ═════════════════════════════════════════════════════════════════════
def figure2():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "PQC algorithm selection by deployment domain", ha="center", va="top",
            fontsize=14, fontweight="bold", color=DARK_BLUE)

    # Central PQCEngine box
    rounded_box(ax, 0.35, 0.60, 0.30, 0.12, "PQCEngine\n(Domain-aware interface)", LIGHT_BLUE, BLUE,
                fontsize=11, fontweight="bold")

    # Algorithm boxes at bottom
    algorithms = [
        ("ML-KEM\n(FIPS 203)\nKey encapsulation", LIGHT_BLUE, BLUE),
        ("ML-DSA\n(FIPS 204)\nDigital signatures", LIGHT_BLUE, BLUE),
        ("Falcon\n(NIST alternate)\nCompact signatures", LIGHT_GREEN, GREEN),
        ("SLH-DSA\n(FIPS 205)\nHash-based signatures", LIGHT_BLUE, BLUE),
    ]

    alg_w = 0.18
    alg_h = 0.14
    alg_start_x = 0.07
    alg_gap = 0.05
    alg_y = 0.05

    for i, (label, fc, ec) in enumerate(algorithms):
        x = alg_start_x + i * (alg_w + alg_gap)
        rounded_box(ax, x, alg_y, alg_w, alg_h, label, fc, ec, fontsize=9, fontweight="bold")
        # Arrow from PQCEngine down to algorithm
        arrow_between(ax, 0.5, 0.60, x + alg_w / 2, alg_y + alg_h, color=ec)

    # Domain boxes at top
    domains = [
        ("Healthcare\nML-KEM-512\n(64KB RAM devices)", LIGHT_GREEN, GREEN),
        ("Automotive\nFalcon-512\n(<1ms real-time)", "#FFF3E0", ORANGE),
        ("Banking\nML-KEM-1024 + ML-DSA-87\n(Maximum security)", LIGHT_BLUE, BLUE),
        ("Aviation\nAggregate signatures\n(ARINC 429/629)", "#F3E5F5", "#7B1FA2"),
        ("Industrial\nIEC 61850 auth\n(Power grid)", LIGHT_ORANGE, ORANGE),
    ]

    dom_w = 0.16
    dom_h = 0.12
    dom_start_x = 0.04
    dom_gap = 0.025
    dom_y = 0.80

    for i, (label, fc, ec) in enumerate(domains):
        x = dom_start_x + i * (dom_w + dom_gap)
        rounded_box(ax, x, dom_y, dom_w, dom_h, label, fc, ec, fontsize=8, fontweight="bold")
        # Arrow from domain down to PQCEngine
        arrow_between(ax, x + dom_w / 2, dom_y, 0.5, 0.72, color=ec)

    # Hybrid mode annotation
    ax.text(0.5, 0.52, "Hybrid mode: X25519/P-384 ECDH + ML-KEM for TLS compatibility",
            ha="center", va="top", fontsize=9, style="italic", color=GRAY)

    fig.tight_layout()
    fig.savefig("/Users/prabakarankannan/qbitel/docs/figure2_pqc_algorithm_selection.png",
                bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    print("Created figure2_pqc_algorithm_selection.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 3: High-level system architecture
# ═════════════════════════════════════════════════════════════════════
def figure3():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.97, "QBITEL Bridge system architecture", ha="center", va="top",
            fontsize=14, fontweight="bold", color=DARK_BLUE)
    ax.text(0.5, 0.92, "Python AI engine (cold path) + Rust dataplane (hot path)", ha="center", va="top",
            fontsize=11, color=GRAY)

    # ── Left side: Python AI Engine ──────────────────────────────
    # Background box
    bg_left = FancyBboxPatch(
        (0.02, 0.08), 0.44, 0.76,
        boxstyle="round,pad=0.01", facecolor="#F5F8FF", edgecolor=BLUE,
        linewidth=2, linestyle="--", transform=ax.transAxes, zorder=0
    )
    ax.add_patch(bg_left)
    ax.text(0.24, 0.86, "Python AI engine (cold path)", ha="center", va="top",
            fontsize=12, fontweight="bold", color=BLUE, transform=ax.transAxes)

    py_components = [
        ("ML Protocol\nDiscovery", 0.06, 0.68, 0.17, 0.10),
        ("Agent\nFramework", 0.27, 0.68, 0.17, 0.10),
        ("LLM\nIntegration", 0.06, 0.50, 0.17, 0.10),
        ("PQC Crypto\nLayer", 0.27, 0.50, 0.17, 0.10),
        ("REST API\n(FastAPI)", 0.06, 0.32, 0.17, 0.10),
        ("Observability\n(Prometheus)", 0.27, 0.32, 0.17, 0.10),
    ]

    for label, x, y, w, h in py_components:
        rounded_box(ax, x, y, w, h, label, LIGHT_BLUE, BLUE, fontsize=9, fontweight="bold")

    # Tech stack label
    ax.text(0.24, 0.22, "FastAPI  |  PyTorch  |  LangGraph  |  Redis",
            ha="center", va="top", fontsize=8, color=GRAY, style="italic", transform=ax.transAxes)

    # ── Right side: Rust Dataplane ───────────────────────────────
    bg_right = FancyBboxPatch(
        (0.54, 0.08), 0.44, 0.76,
        boxstyle="round,pad=0.01", facecolor="#F5FFF5", edgecolor=GREEN,
        linewidth=2, linestyle="--", transform=ax.transAxes, zorder=0
    )
    ax.add_patch(bg_right)
    ax.text(0.76, 0.86, "Rust dataplane (hot path)", ha="center", va="top",
            fontsize=12, fontweight="bold", color=GREEN, transform=ax.transAxes)

    rust_components = [
        ("Packet\nProcessing", 0.58, 0.68, 0.17, 0.10),
        ("PQC-TLS\nTermination", 0.79, 0.68, 0.17, 0.10),
        ("Protocol\nProxying", 0.58, 0.50, 0.17, 0.10),
        ("Session\nManagement", 0.79, 0.50, 0.17, 0.10),
    ]

    for label, x, y, w, h in rust_components:
        rounded_box(ax, x, y, w, h, label, LIGHT_GREEN, GREEN, fontsize=9, fontweight="bold")

    # Performance label
    ax.text(0.76, 0.40, "10M+ packets/sec  |  <10ms latency",
            ha="center", va="top", fontsize=9, fontweight="bold", color=GREEN, transform=ax.transAxes)

    ax.text(0.76, 0.32, "Tokio async runtime  |  GPU acceleration",
            ha="center", va="top", fontsize=8, color=GRAY, style="italic", transform=ax.transAxes)

    # ── gRPC interface arrow ─────────────────────────────────────
    ax.annotate(
        "", xy=(0.54, 0.58), xytext=(0.46, 0.58),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=3),
    )
    ax.text(0.50, 0.62, "gRPC", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=ORANGE, transform=ax.transAxes)

    # ── External: Network traffic ────────────────────────────────
    rounded_box(ax, 0.60, 0.12, 0.32, 0.08, "Network traffic (legacy + modern protocols)",
                LIGHT_ORANGE, ORANGE, fontsize=9, fontweight="bold")

    arrow_between(ax, 0.76, 0.20, 0.76, 0.50, color=ORANGE)

    # ── External: Operators / dashboards ─────────────────────────
    rounded_box(ax, 0.06, 0.12, 0.32, 0.08, "Operators  |  Grafana  |  Alerting",
                LIGHT_GRAY, GRAY, fontsize=9, fontweight="bold")

    arrow_between(ax, 0.22, 0.20, 0.22, 0.32, color=GRAY)

    fig.tight_layout()
    fig.savefig("/Users/prabakarankannan/qbitel/docs/figure3_system_architecture.png",
                bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    print("Created figure3_system_architecture.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
