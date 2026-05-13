# Path manipulation to ensure the example can run from the project directory:
import sys
from pathlib import Path
THIS_DIR = Path(__file__).absolute().parent

PROJECT_ROOT = THIS_DIR.parent
if PROJECT_ROOT not in [Path(s).absolute() for s in sys.path]:
    sys.path.insert(0, str(PROJECT_ROOT))


from torc import (
    RacetrackCoil,
    RoundCoil,
    Container,
    CoilPair,
    inch,
    mm,
    X,
    Y,
    Z,
    COPPER,
)
import numpy as np

"""This module creates magnetic coil objects with the specifications of the RbChip lab's
transport coils, which can then be used to compute fields and gradients"""

# One pair of these
MOT_R_INNER = 1.595 * inch
MOT_R_OUTER = 2.375 * inch
MOT_HEIGHT = 0.3 * inch
MOT_Z_POS = 28 * mm + MOT_HEIGHT / 2  # Inner edge is at 28 mm
MOT_N_TURNS = 33

# One pair of these:
SCIENCE_R_INNER = 1.470 * inch
SCIENCE_R_OUTER = 2.250 * inch
SCIENCE_HEIGHT = 0.3 * inch
SCIENCE_Z_POS = 28 * mm + SCIENCE_HEIGHT / 2  # Inner edge is at 28 mm
SCIENCE_N_TURNS = 32

# Four pairs of these
INNER_TRANS_R_INNER = 0.795 * inch
INNER_TRANS_R_OUTER = 1.575 * inch
INNER_TRANS_HEIGHT = 0.25 * inch
INNER_TRANS_Z_POS = 28 * mm + INNER_TRANS_HEIGHT / 2  # Inner edge is at 28 mm
INNER_TRANS_N_TURNS = 60

# Five pairs of these:
OUTER_TRANS_R_INNER = 0.795 * inch
OUTER_TRANS_R_OUTER = 1.575 * inch
OUTER_TRANS_HEIGHT = 0.5 * inch
OUTER_TRANS_Z_POS = 36 * mm + OUTER_TRANS_HEIGHT / 2  # Inner edge is at 36 mm
OUTER_TRANS_N_TURNS = 60

# One of these
PUSH_WIDTH = 37 * mm
PUSH_HEIGHT = 0.5 * inch
# I think the push coil's inner edge is 86.5mm below the MOT centre. This is PushY =
# 86.5 in Abby's code and it looks to be describing the edge and not the centre.
PUSH_Y_POS = -86.5 * mm - PUSH_HEIGHT / 2
PUSH_N_TURNS = 52
PUSH_R_INNER = 0
PUSH_R_OUTER = (
    PUSH_N_TURNS / OUTER_TRANS_N_TURNS * (OUTER_TRANS_R_OUTER - OUTER_TRANS_R_INNER)
)

def make_coils(
    MOT_coils_spacing_factor=1,
    science_coils_spacing_factor=1,
    inner_coils_spacing_factors=(1, 1, 1, 1),
    outer_coils_spacing_factors=(1, 1, 1, 1, 1),
):
    """Return a torc.Container containing all the coil (pairs). Includes two parameters
    to account for imperfect geometry of the coils: the deviation from design spacing
    between coilpairs at the MOT position, and the deviation at the science coil
    position. The spacing of the coils will be modelled as having a deviation from their
    design spacings that is a linear function of y, defined by the deviations at these
    two points."""

    coils = Container()

    # Push coil
    coils.add(
        RacetrackCoil(
            r0=(0, PUSH_Y_POS, 0),
            n=Y,
            u=X,
            inner_length=PUSH_WIDTH,
            inner_width=PUSH_WIDTH,
            height=PUSH_HEIGHT,
            inner_radius=PUSH_R_INNER,
            outer_radius=PUSH_R_OUTER,
            num_turns=PUSH_N_TURNS,
            name='push',
        )
    )

    # MOT coils
    coils.add(
        CoilPair(
            coiltype=RoundCoil,
            r0=(0, 0, 0),
            n=Z,
            separation=2 * MOT_Z_POS * MOT_coils_spacing_factor,
            height=MOT_HEIGHT,
            inner_radius=MOT_R_INNER,
            outer_radius=MOT_R_OUTER,
            num_turns=MOT_N_TURNS,
            parity='anti-helmholtz',
            name='MOT',
        )
    )

    # Outer transport coils
    first_y = MOT_R_OUTER
    for i, y in enumerate(np.linspace(first_y, first_y + 8 * OUTER_TRANS_R_OUTER, 5)):
        coils.add(
            CoilPair(
                coiltype=RoundCoil,
                r0=(0, y, 0),
                n=Z,
                separation=2 * OUTER_TRANS_Z_POS * outer_coils_spacing_factors[i],
                height=OUTER_TRANS_HEIGHT,
                inner_radius=OUTER_TRANS_R_INNER,
                outer_radius=OUTER_TRANS_R_OUTER,
                num_turns=OUTER_TRANS_N_TURNS,
                parity='anti-helmholtz',
                name=f'outer_{i}',
            )
        )

    # Inner transport coils
    first_y = MOT_R_OUTER + INNER_TRANS_R_OUTER
    for i, y in enumerate(np.linspace(first_y, first_y + 6 * INNER_TRANS_R_OUTER, 4)):
        coils.add(
            CoilPair(
                coiltype=RoundCoil,
                r0=(0, y, 0),
                n=Z,
                separation=2 * INNER_TRANS_Z_POS * inner_coils_spacing_factors[i],
                height=INNER_TRANS_HEIGHT,
                inner_radius=INNER_TRANS_R_INNER,
                outer_radius=INNER_TRANS_R_OUTER,
                num_turns=INNER_TRANS_N_TURNS,
                parity='anti-helmholtz',
                name=f'inner_{i}',
            )
        )

    # Science coils:
    science_y = MOT_R_OUTER + 8 * INNER_TRANS_R_OUTER + SCIENCE_R_OUTER
    coils.add(
        CoilPair(
            coiltype=RoundCoil,
            r0=(0, science_y, 0),
            n=Z,
            separation=2 * SCIENCE_Z_POS * science_coils_spacing_factor,
            height=SCIENCE_HEIGHT,
            inner_radius=SCIENCE_R_INNER,
            outer_radius=SCIENCE_R_OUTER,
            num_turns=SCIENCE_N_TURNS,
            parity='anti-helmholtz',
            name='science',
        )
    )

    # Sort by y positions
    coils.children.sort(key=lambda coil: coil.y)
    return coils


if __name__ == '__main__':
    # Show a 3D rendering of the coils
    coils = make_coils()
    coils.show(surfaces=True, color=COPPER)
