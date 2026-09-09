# -*- coding: utf-8 -*-
"""
Shared visual language for every figure and animation in this project.

The old scripts each set their own colours and fonts on top of matplotlib's
`dark_background`, so nothing matched anything else and everything carried
matplotlib's defaults. This module is the single source of truth: the palette
is the one the website uses, so a chart dropped onto the page looks like it
belongs there.

Import it before creating any figure:

    import f1_style
    f1_style.apply()
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# Palette - matches styles.css so figures sit on the page without clashing
# --------------------------------------------------------------------------

BG = '#0a0a0f'          # page background
PANEL = '#15151f'        # card background
PANEL_HI = '#1c1c28'     # raised panel
LINE = '#2a2a38'         # hairlines, grid
RED = '#e10600'          # F1 red, the primary accent
TEAL = '#00d2be'         # secondary accent
AMBER = '#ff8700'
YELLOW = '#fff200'
INK = '#ffffff'          # primary text
INK_2 = '#a0a0b0'        # secondary text
INK_3 = '#6a6a7a'        # muted text / captions

# Tyre compounds, using the FIA's own colours
TYRE = {
    'SOFT': '#ff2d2d', 'MEDIUM': '#ffd12e', 'HARD': '#efefef',
    'INTERMEDIATE': '#3fdb4a', 'WET': '#1f7bff', 'UNKNOWN': INK_3,
}

# 2024 constructor colours
TEAM = {
    'Red Bull Racing': '#3671C6', 'Ferrari': '#E8002D', 'McLaren': '#FF8000',
    'Mercedes': '#27F4D2', 'Aston Martin': '#229971', 'Alpine': '#FF87BC',
    'Williams': '#64C4FF', 'RB': '#6692FF', 'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD',
}

# Speed ramp: slow (deep blue) -> mid (teal) -> fast (white hot)
SPEED_CMAP = LinearSegmentedColormap.from_list('f1_speed', [
    '#12124a', '#1b4a8f', '#00a0c0', '#00d2be', '#c8f5ee', '#ffffff',
])

# Diverging ramp for deltas: driver A ahead (teal) <-> driver B ahead (red)
DELTA_CMAP = LinearSegmentedColormap.from_list('f1_delta', [
    TEAL, '#0d8a80', '#2a2a38', '#8f0400', RED,
])


def _pick_font() -> str:
    """Best available font from a preference list, falling back gracefully."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Inter', 'Roboto', 'Segoe UI', 'Helvetica Neue',
                 'Arial', 'DejaVu Sans'):
        if name in installed:
            return name
    return 'sans-serif'


FONT = _pick_font()


def apply(scale: float = 1.0) -> None:
    """Install the house style globally. `scale` multiplies every font size."""
    s = scale
    matplotlib.rcParams.update({
        # canvas
        'figure.facecolor': BG,
        'figure.edgecolor': BG,
        'savefig.facecolor': BG,
        'savefig.edgecolor': BG,
        'axes.facecolor': BG,

        # type
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT, 'DejaVu Sans'],
        'font.size': 11 * s,
        'axes.titlesize': 13 * s,
        'axes.labelsize': 10 * s,
        'xtick.labelsize': 9 * s,
        'ytick.labelsize': 9 * s,
        'legend.fontsize': 10 * s,

        # ink
        'text.color': INK,
        'axes.labelcolor': INK_2,
        'axes.titlecolor': INK,
        'xtick.color': INK_3,
        'ytick.color': INK_3,

        # frame: keep the two spines that carry meaning, drop the rest
        'axes.edgecolor': LINE,
        'axes.linewidth': 0.9,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # grid: present but never competing with the data
        'axes.grid': True,
        'grid.color': LINE,
        'grid.linewidth': 0.7,
        'grid.alpha': 0.55,
        'axes.axisbelow': True,

        # marks
        'lines.linewidth': 2.0,
        'lines.solid_capstyle': 'round',
        'patch.linewidth': 0,

        # legend
        'legend.frameon': False,
        'legend.labelcolor': INK_2,

        # output
        'figure.dpi': 110,
        'savefig.dpi': 160,
        'savefig.bbox': None,
        'animation.codec': 'h264',
    })


def panel(ax, title: str | None = None, pad: float = 0.0) -> None:
    """Turn an axes into a card: soft fill, hairline border, optional label."""
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(LINE)
        spine.set_linewidth(0.9)
    if title:
        ax.set_title(title, color=INK_3, fontsize=9, loc='left',
                     pad=6 + pad, fontweight='600')


def strip(ax) -> None:
    """A bare axes - no ticks, no grid, no frame. For maps and sparklines."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('none')


def title_block(fig, title: str, subtitle: str = '', x: float = 0.035,
                y: float = 0.955, size: int = 22) -> None:
    """Standard headline: heavy title with an accent rule and a subtitle."""
    fig.text(x, y, title, color=INK, fontsize=size, fontweight='bold',
             va='top', ha='left')
    fig.text(x, y - (size + 12) / fig.get_figheight() / 72,
             subtitle, color=INK_2, fontsize=size * 0.46, va='top', ha='left')


def accent_rule(fig, x: float = 0.035, y: float = 0.905,
                w: float = 0.055, lw: float = 3.0) -> None:
    """The short red underline used across the site."""
    fig.add_artist(plt.Line2D([x, x + w], [y, y], color=RED, lw=lw,
                              solid_capstyle='butt'))


def footer(fig, source: str, y: float = 0.022) -> None:
    """Provenance line. Every figure states where its numbers came from."""
    fig.text(0.035, y, source, color=INK_3, fontsize=8.5, ha='left', va='bottom')
    fig.text(0.965, y, 'eliherrera33.github.io/F1-Data-Analysis',
             color=INK_3, fontsize=8.5, ha='right', va='bottom')


def team_color(name: str, fallback: str = INK_2) -> str:
    return TEAM.get(name, fallback)


def tyre_color(compound: str) -> str:
    return TYRE.get(str(compound).upper(), TYRE['UNKNOWN'])


def fmt_laptime(seconds: float) -> str:
    """1:23.456 - the only lap time format anyone in the paddock reads."""
    if seconds is None or seconds != seconds:  # NaN
        return '--:--.---'
    m, s = divmod(float(seconds), 60)
    return f'{int(m)}:{s:06.3f}'


def fmt_delta(seconds: float) -> str:
    """Signed gap, always three decimals, always with a sign."""
    if seconds is None or seconds != seconds:
        return ' --.---'
    return f'{seconds:+.3f}'


def writer(fps: int = 50, crf: int = 19):
    """
    H.264 writer. GIF was the real ceiling on the old animations: 256 colours,
    no interframe compression, and in practice 8-12 fps. The same clip as MP4
    is smoother, sharper and several times smaller.

    Quality is controlled by CRF, so bitrate is left at -1 - passing both makes
    ffmpeg ignore one of them.
    """
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    from matplotlib.animation import FFMpegWriter
    return FFMpegWriter(
        fps=fps, codec='libx264', bitrate=-1,
        extra_args=[
            '-pix_fmt', 'yuv420p',   # required for browser/QuickTime playback
            '-preset', 'slow',
            '-crf', str(crf),
            '-profile:v', 'high',
            '-movflags', '+faststart',
        ],
    )
