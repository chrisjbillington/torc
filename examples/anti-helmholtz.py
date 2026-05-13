# Path manipulation to ensure the example can run from the project directory:
import sys
from pathlib import Path
THIS_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = THIS_DIR.parent
if PROJECT_ROOT not in [Path(s).absolute() for s in sys.path]:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from torc import RoundCoil, CoilPair, Z, cm, gauss, gauss_per_cm

R_inner = 4 * cm
R_outer = 6 * cm
separation = 10 * cm
height = 1 * cm

# A pair of round coils in anti-Helmholtz configuration:
coils = CoilPair(
    coiltype=RoundCoil,
    r0=(0, 0, 0),  # centred at the origin
    n=Z,  # normal direction is +Z
    separation=separation,
    height=height,
    inner_radius=R_inner,
    outer_radius=R_outer,
    num_turns=100,
    parity='anti-helmholtz',
)

def plot_2d_field_and_gradient():
    # Compute the field and z gradient for current I = 1A in a 2D x-z plane:

    # Construct grid - first dimension will be z and second will be x, so that when we
    # imshow() the 2D results, the z direction will be the vertical direction:
    x = np.linspace(-10 * cm, 10 * cm, 256)[np.newaxis, :]
    y = 0
    z = np.linspace(-10 * cm, 10 * cm, 256)[:, np.newaxis]

    # Compute B field on the grid
    B = coils.B((x,y,z), I=1)

    # Compute z derivative of B on the grid
    dB_dz = coils.dB((x,y,z), s=Z, I=1)

    # Mask out areas inside or within 0.25cm of the coils:
    d_mask = 0.25 * cm
    mask = (
        (abs(x) > R_inner - d_mask)
        & (abs(x) < R_outer + d_mask)
        & (abs(z) > separation / 2 - height / 2 - d_mask)
        & (abs(z) < separation / 2 + height / 2 + d_mask)
    )
    B[:, mask] = 0
    dB_dz[:, mask] = 0

    # Extract components:
    Bx, By, Bz = B
    dBz_dz, dBy_dz, dBz_dz = dB_dz

    # Bz 2D plot
    plt.subplot(221)
    plt.imshow(
        Bz / gauss,
        extent=[x.min() / cm, x.max() / cm, z.min() / cm, z.max() / cm],
        origin='lower',
        cmap='seismic',
        vmin=-abs(Bz / gauss).max(),
        vmax=abs(Bz / gauss).max(),
    )
    plt.grid(True, color='k', linestyle=":", alpha=0.5)
    plt.xlabel('$x$ (cm)')
    plt.ylabel('$z$ (cm)')
    plt.colorbar(label='$B_z$ (G)')

    # dBz_dz 2D plot
    plt.subplot(222)
    plt.imshow(
        dBz_dz / gauss_per_cm,
        extent=[x.min() / cm, x.max() / cm, z.min() / cm, z.max() / cm],
        origin='lower',
        cmap='seismic',
        vmin=-abs(dBz_dz / gauss_per_cm).max(),
        vmax=abs(dBz_dz / gauss_per_cm).max(),
    )
    plt.grid(True, color='k', linestyle=":", alpha=0.5)
    plt.xlabel('$x$ (cm)')
    plt.ylabel('$z$ (cm)')
    plt.colorbar(label='$dB_z/dz$ (G/cm)')

def plot_1d_field_and_gradient():
    z = np.linspace(-10 * cm, 10 * cm, 1024)

    # Compute B field on the z axis for current I = 1A 
    B = coils.B((0, 0, z), I=1)

    # Compute z derivative of B on the z axis for current I = 1A 
    dB_dz = coils.dB((0, 0, z), s=Z, I=1)

    # Extract components:
    Bx, By, Bz = B
    dBz_dz, dBy_dz, dBz_dz = dB_dz

    # Plot Bz and dBz_dz:
    plt.subplot(223)
    plt.plot(z / cm, Bz / gauss)
    plt.grid(True, color='k', linestyle=":", alpha=0.5)
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$B_z$ (G)')

    plt.subplot(224)
    plt.plot(z / cm, dBz_dz / gauss_per_cm)
    plt.grid(True, color='k', linestyle=":", alpha=0.5)
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$dB_z/dz$ (G/cm)')


# Calculate fields/gradients and show plots:
plt.figure(figsize=(9, 6))
plot_2d_field_and_gradient()
plot_1d_field_and_gradient()
plt.tight_layout()
plt.savefig('anti-helmholtz-plots.png')
plt.show()

# Display a 3D rendering of the coils:
coils.show()
