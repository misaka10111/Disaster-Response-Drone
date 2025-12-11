import numpy as np
from matplotlib.patches import Patch, Circle, PathPatch
from matplotlib.path import Path
from detector import CATEGORY_COLORS


def draw_pin(ax, xy, color, text=None, size=1.0):
    x, y = xy
    r = size

    # Circular head
    circ = Circle((x, y + r * 0.5), radius=r * 0.6,
                  facecolor=color, edgecolor=(0, 0, 0, 0.3),
                  linewidth=0.8, zorder=5)
    ax.add_patch(circ)

    # Tip triangle
    verts = [(x, y), (x - r * 0.5, y + r * 0.4),
             (x + r * 0.5, y + r * 0.4), (x, y)]
    tip = PathPatch(Path(verts, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]),
                    facecolor=color, edgecolor=(0, 0, 0, 0.25),
                    linewidth=0.8, zorder=4)
    ax.add_patch(tip)

    if text:
        ax.text(x, y + r * 1.6, text,
                fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor=(0, 0, 0, 0.2), lw=0.7),
                zorder=6)


def draw_grid(ax, width_m, height_m, cell_size_m):
    xs = np.arange(0, width_m, cell_size_m)
    ys = np.arange(0, height_m, cell_size_m)
    for x in xs:
        ax.axvline(x, color=(0.85, 0.85, 0.85), linewidth=0.7)
    for y in ys:
        ax.axhline(y, color=(0.85, 0.85, 0.85), linewidth=0.7)


def draw_legend(ax):
    patches = [
        Patch(facecolor=np.array(rgb) / 255.0, edgecolor=(0, 0, 0, 0.3),
              label=cat.capitalize())
        for cat, rgb in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=patches, title="Semantic Classes",
              loc="upper right", framealpha=0.9)
