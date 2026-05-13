#!/usr/bin/env python3
"""Generate field-plot images for the README."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Mock GUI deps (not needed for field computation)
from unittest.mock import MagicMock
for _mod in ['pyqtgraph', 'pyqtgraph.opengl', 'pyqtgraph.Qt', 'pyqtgraph.Qt.QtCore',
             'PySide6', 'OpenGL', 'OpenGL.GL']:
    sys.modules[_mod] = MagicMock()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from torc import RoundCoil, CoilPair, Z, cm, mm


def _add_coil_cross_sections(ax, inner_radius, outer_radius, height, separation):
    """Draw coil cross-sections as copper-coloured rectangles in the xz plane."""
    for z_center in [separation / 2, -separation / 2]:
        for x_lo, x_hi in [
            (inner_radius, outer_radius),
            (-outer_radius, -inner_radius),
        ]:
            rect = patches.Rectangle(
                (x_lo / cm, (z_center - height / 2) / cm),
                (x_hi - x_lo) / cm,
                height / cm,
                linewidth=0.8,
                edgecolor='#4a3010',
                facecolor='#b87333',
                alpha=0.9,
                zorder=5,
            )
            ax.add_patch(rect)


def anti_helmholtz_field():
    R = 5 * cm
    separation = R  # standard Helmholtz-ratio spacing
    height = 8 * mm
    inner_radius = R * 0.85
    outer_radius = R * 1.15

    coils = CoilPair(
        coiltype=RoundCoil,
        r0=(0, 0, 0),
        n=Z,
        separation=separation,
        height=height,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        num_turns=100,
        parity='anti-helmholtz',
    )

    extent = 4.5 * cm
    N = 40
    x = np.linspace(-extent, extent, N)
    z = np.linspace(-extent, extent, N)
    X_grid, Z_grid = np.meshgrid(x, z)  # both (N, N), X_grid[i,j]=x[j], Z_grid[i,j]=z[i]

    B = coils.B((X_grid, np.zeros_like(X_grid), Z_grid), I=1.0)
    Bx = B[0]  # (N, N)
    Bz = B[2]  # (N, N)
    Bmag = np.hypot(Bx, Bz)

    # Mask points inside/on coil cross-sections to avoid singularities there
    mask = np.zeros((N, N), dtype=bool)
    for z_center in [separation / 2, -separation / 2]:
        for x_lo, x_hi in [
            (inner_radius, outer_radius),
            (-outer_radius, -inner_radius),
        ]:
            in_coil = (
                (X_grid >= x_lo) & (X_grid <= x_hi)
                & (Z_grid >= z_center - height / 2) & (Z_grid <= z_center + height / 2)
            )
            mask |= in_coil

    Bx[mask] = np.nan
    Bz[mask] = np.nan
    Bmag[mask] = np.nan

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    log_mag = np.log1p(Bmag / np.nanpercentile(Bmag, 50))

    stream = ax.streamplot(
        x / cm,
        z / cm,
        Bx,
        Bz,
        color=log_mag,
        cmap='Blues',
        linewidth=1.2,
        density=1.2,
        arrowsize=1.0,
        norm=mcolors.Normalize(vmin=0, vmax=np.nanpercentile(log_mag, 98)),
    )

    _add_coil_cross_sections(ax, inner_radius, outer_radius, height, separation)

    ax.set_xlim(-extent / cm, extent / cm)
    ax.set_ylim(-extent / cm, extent / cm)
    ax.set_xlabel('x (cm)', fontsize=11)
    ax.set_ylabel('z (cm)', fontsize=11)
    ax.set_title('Anti-Helmholtz quadrupole field  (I = 1 A)', fontsize=11)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.4)

    cbar = fig.colorbar(stream.lines, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log(|B| / a.u.)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = Path(__file__).parent / 'anti_helmholtz_field.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def helmholtz_vs_antihelmholtz():
    R = 5 * cm
    separation = R
    height = 8 * mm
    inner_radius = R * 0.85
    outer_radius = R * 1.15

    common = dict(
        coiltype=RoundCoil,
        r0=(0, 0, 0),
        n=Z,
        separation=separation,
        height=height,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        num_turns=100,
    )
    helmholtz = CoilPair(parity='helmholtz', **common)
    anti = CoilPair(parity='anti-helmholtz', **common)

    z = np.linspace(-3 * R, 3 * R, 600)
    r = (np.zeros_like(z), np.zeros_like(z), z)

    B_h = helmholtz.B(r, I=1.0)
    B_a = anti.B(r, I=1.0)
    Bz_h = B_h[2]
    Bz_a = B_a[2]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)

    # --- Helmholtz ---
    ax = axes[0]
    ax.plot(z / cm, Bz_h * 1e4, color='#1f77b4', lw=2)
    for zc in [separation / 2, -separation / 2]:
        ax.axvline(zc / cm, color='#b87333', lw=1.8, ls='--', alpha=0.9)
    ax.set_xlabel('z (cm)', fontsize=11)
    ax.set_ylabel(r'$B_z$ (Gauss / A)', fontsize=11)
    ax.set_title('Helmholtz — uniform field', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color='#1f77b4', lw=2, label=r'$B_z$ on axis'),
            plt.Line2D([0], [0], color='#b87333', lw=1.8, ls='--', label='coil planes'),
        ],
        fontsize=9,
        loc='upper right',
    )

    # --- Anti-Helmholtz ---
    ax = axes[1]
    ax.plot(z / cm, Bz_a * 1e4, color='#d62728', lw=2)
    for zc in [separation / 2, -separation / 2]:
        ax.axvline(zc / cm, color='#b87333', lw=1.8, ls='--', alpha=0.9)
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xlabel('z (cm)', fontsize=11)
    ax.set_ylabel(r'$B_z$ (Gauss / A)', fontsize=11)
    ax.set_title('Anti-Helmholtz — linear gradient', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color='#d62728', lw=2, label=r'$B_z$ on axis'),
            plt.Line2D([0], [0], color='#b87333', lw=1.8, ls='--', label='coil planes'),
        ],
        fontsize=9,
        loc='upper right',
    )

    plt.tight_layout()
    out = Path(__file__).parent / 'helmholtz_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


if __name__ == '__main__':
    anti_helmholtz_field()
    helmholtz_vs_antihelmholtz()
