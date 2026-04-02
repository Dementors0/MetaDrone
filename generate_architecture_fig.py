#!/usr/bin/env python3
"""
Generate CVPR-style architecture figure for the Bilevel Meta-Learning framework.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np

# Set up the figure with CVPR style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'text.usetex': False,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Color scheme (CVPR friendly)
colors = {
    'env': '#E8F4FD',           # Light blue
    'worker': '#FFF3E0',        # Light orange
    'lgn': '#E8F5E9',           # Light green
    'loss': '#FCE4EC',          # Light pink
    'meta': '#F3E5F5',          # Light purple
    'gradient': '#FFEBEE',      # Light red
    'border_env': '#1976D2',    # Blue
    'border_worker': '#F57C00', # Orange
    'border_lgn': '#388E3C',    # Green
    'border_loss': '#C2185B',   # Pink
    'border_meta': '#7B1FA2',   # Purple
    'arrow': '#37474F',         # Dark gray
    'text': '#212121',          # Almost black
}

def draw_box(ax, x, y, w, h, label, color, border_color, fontsize=10, bold=False):
    """Draw a rounded box with label."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, color=colors['text'], weight=weight)
    return box

def draw_arrow(ax, start, end, color='#37474F', style='->', lw=1.5, connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle=style,
                            connectionstyle=connectionstyle,
                            color=color, linewidth=lw,
                            mutation_scale=12)
    ax.add_patch(arrow)
    return arrow

def draw_dashed_box(ax, x, y, w, h, label, color, fontsize=9):
    """Draw a dashed box for grouping."""
    rect = Rectangle((x, y), w, h, fill=True, facecolor=color,
                      edgecolor='#666666', linestyle='--', linewidth=1.5, alpha=0.3)
    ax.add_patch(rect)
    ax.text(x + 0.1, y + h - 0.15, label, ha='left', va='top',
            fontsize=fontsize, color='#444444', style='italic', weight='bold')

# ==================== MAIN ARCHITECTURE ====================

# Title
ax.text(6, 7.7, 'Bilevel Meta-Learning for Loss Weight Generation',
        ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])

# -------------------- Left Side: Environment & Networks --------------------

# Environment box
draw_dashed_box(ax, 0.3, 4.5, 3.5, 2.8, 'Environment', colors['env'])
draw_box(ax, 0.6, 6.3, 1.4, 0.6, 'Depth\nImage', colors['env'], colors['border_env'], fontsize=8)
draw_box(ax, 2.2, 6.3, 1.4, 0.6, 'State\nVector', colors['env'], colors['border_env'], fontsize=8)
draw_box(ax, 1.4, 5.0, 1.2, 0.6, 'Drone\nDynamics', colors['env'], colors['border_env'], fontsize=8)

# Networks
draw_box(ax, 0.5, 3.5, 1.5, 0.7, 'LGN\n(LossGenNet)', colors['lgn'], colors['border_lgn'], fontsize=9, bold=True)
draw_box(ax, 2.3, 3.5, 1.5, 0.7, 'Worker\nNetwork', colors['worker'], colors['border_worker'], fontsize=9, bold=True)

# Arrows from env to networks
draw_arrow(ax, (1.3, 6.3), (1.25, 4.25), colors['border_env'])
draw_arrow(ax, (2.9, 6.3), (2.9, 4.25), colors['border_env'])
draw_arrow(ax, (1.3, 6.3), (2.9, 4.25), colors['border_env'], connectionstyle="arc3,rad=-0.2")
draw_arrow(ax, (2.9, 6.3), (1.25, 4.25), colors['border_env'], connectionstyle="arc3,rad=0.2")

# Worker action to env
draw_arrow(ax, (3.05, 4.25), (2.2, 5.0), colors['border_worker'], connectionstyle="arc3,rad=-0.3")
ax.text(3.3, 4.8, 'action', fontsize=8, color=colors['border_worker'], style='italic')

# -------------------- Middle: Proxy Losses & Weights --------------------

# Proxy losses box
draw_dashed_box(ax, 4.2, 4.0, 2.8, 3.3, 'Proxy Losses', colors['loss'])

# Individual loss components
loss_y = 6.5
loss_names = ['$\\mathcal{L}_{speed}$', '$\\mathcal{L}_{dir}$', '$\\mathcal{L}_{avoid}$', '$\\mathcal{L}_{expl}$']
for i, name in enumerate(loss_names):
    draw_box(ax, 4.5, loss_y - i*0.65, 1.0, 0.5, name, colors['loss'], colors['border_loss'], fontsize=9)

# Weights from LGN
draw_box(ax, 5.7, 5.5, 1.1, 1.5, '$w_1$\n$w_2$\n$w_3$\n$w_4$', colors['lgn'], colors['border_lgn'], fontsize=9)
ax.text(6.25, 5.2, 'LGN\nWeights', fontsize=7, ha='center', color=colors['border_lgn'], style='italic')

# Arrow from LGN to weights
draw_arrow(ax, (2.0, 3.5), (5.7, 5.8), colors['border_lgn'], connectionstyle="arc3,rad=-0.3")

# Gradient symbols
draw_box(ax, 4.5, 4.2, 2.3, 0.6, '$\\nabla_{\\theta} \\mathcal{L}_i$', colors['gradient'], '#D32F2F', fontsize=10)

# -------------------- Right Side: Bilevel Optimization --------------------

# Inner loop box
draw_dashed_box(ax, 7.3, 4.5, 4.3, 2.8, 'Inner Loop (Differentiable)', colors['meta'])

# Combined gradients
draw_box(ax, 7.6, 6.3, 2.0, 0.7, '$g_{combined} = \\sum_i w_i \\nabla \\mathcal{L}_i$',
         colors['meta'], colors['border_meta'], fontsize=8)

# Fast params update
draw_box(ax, 7.6, 5.3, 2.0, 0.7, "$\\theta' = \\theta - \\alpha \\cdot g_{combined}$",
         colors['meta'], colors['border_meta'], fontsize=8)

# Fast worker
draw_box(ax, 10.0, 5.8, 1.4, 0.7, "Worker'\n(fast params)", colors['worker'], colors['border_worker'], fontsize=8)

# Arrows in inner loop
draw_arrow(ax, (6.8, 4.5), (7.6, 6.6), colors['border_loss'], connectionstyle="arc3,rad=-0.2")
draw_arrow(ax, (6.8, 6.0), (7.6, 6.6), colors['border_lgn'], connectionstyle="arc3,rad=0.1")
draw_arrow(ax, (8.6, 6.3), (8.6, 6.05), colors['border_meta'])
draw_arrow(ax, (9.6, 5.65), (10.0, 5.95), colors['border_meta'])

# -------------------- Bottom: Outer Loop / Meta Loss --------------------

# Outer loop box
draw_dashed_box(ax, 7.3, 1.0, 4.3, 3.2, 'Outer Loop (Meta Validation)', '#E3F2FD')

# Unrolled rollout
draw_box(ax, 7.6, 3.2, 2.0, 0.7, 'Unrolled\nRollout', '#E3F2FD', '#1565C0', fontsize=9)

# Meta loss components
draw_box(ax, 7.6, 2.2, 1.4, 0.6, '$\\mathcal{L}_{pos}$', '#E3F2FD', '#1565C0', fontsize=9)
draw_box(ax, 9.2, 2.2, 1.4, 0.6, '$\\mathcal{L}_{coll}$', '#E3F2FD', '#1565C0', fontsize=9)
draw_box(ax, 7.6, 1.4, 1.4, 0.6, '$\\mathcal{L}_{height}$', '#E3F2FD', '#1565C0', fontsize=9)
draw_box(ax, 9.2, 1.4, 1.4, 0.6, '$\\mathcal{L}_{guide}$', '#E3F2FD', '#1565C0', fontsize=9)

# Meta loss total
draw_box(ax, 10.0, 3.2, 1.4, 0.7, '$\\mathcal{L}_{meta}$', '#FFCDD2', '#C62828', fontsize=10, bold=True)

# Arrows in outer loop
draw_arrow(ax, (10.7, 5.8), (10.7, 3.95), colors['border_worker'])
draw_arrow(ax, (8.6, 3.2), (8.6, 2.85), '#1565C0')
draw_arrow(ax, (8.3, 2.2), (10.0, 3.3), '#1565C0', connectionstyle="arc3,rad=0.2")
draw_arrow(ax, (9.9, 2.2), (10.5, 3.2), '#1565C0', connectionstyle="arc3,rad=-0.1")
draw_arrow(ax, (8.3, 1.4), (10.0, 3.2), '#1565C0', connectionstyle="arc3,rad=0.3")
draw_arrow(ax, (9.9, 1.4), (10.7, 3.2), '#1565C0', connectionstyle="arc3,rad=-0.2")

# -------------------- Backward Gradient Flow --------------------

# Main backward arrow (meta loss to LGN)
ax.annotate('', xy=(1.25, 3.5), xytext=(10.0, 3.2),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5,
                          connectionstyle='arc3,rad=0.4', linestyle='--'))
ax.text(5.5, 1.5, '$\\frac{\\partial \\mathcal{L}_{meta}}{\\partial \\phi_{LGN}}$',
        fontsize=11, color='#C62828', weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#C62828', alpha=0.9))

# -------------------- Legend --------------------

legend_elements = [
    mpatches.Patch(facecolor=colors['lgn'], edgecolor=colors['border_lgn'], label='Loss Generation Network (LGN)'),
    mpatches.Patch(facecolor=colors['worker'], edgecolor=colors['border_worker'], label='Worker Network'),
    mpatches.Patch(facecolor=colors['loss'], edgecolor=colors['border_loss'], label='Proxy Loss Components'),
    mpatches.Patch(facecolor=colors['meta'], edgecolor=colors['border_meta'], label='Inner Loop (Differentiable)'),
    mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', label='Outer Loop (Meta Validation)'),
    Line2D([0], [0], color='#C62828', linewidth=2.5, linestyle='--', label='Gradient Flow to LGN'),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.35),
          fontsize=8, framealpha=0.95)

# -------------------- Key Insight Box --------------------

insight_text = (
    "Key Insight: LGN weights directly multiply gradients,\n"
    "creating differentiable path: $\\mathcal{L}_{meta} \\to w_i \\to \\phi_{LGN}$"
)
ax.text(0.5, 0.5, insight_text, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#FFFDE7', edgecolor='#FBC02D', alpha=0.9),
        color='#5D4037')

# Save figure
plt.tight_layout()
plt.savefig('/home/robot/transformer/multi_pub/architecture_cvpr.pdf',
            format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('/home/robot/transformer/multi_pub/architecture_cvpr.png',
            format='png', bbox_inches='tight', dpi=300)
print("Figure saved to architecture_cvpr.pdf and architecture_cvpr.png")

plt.show()
