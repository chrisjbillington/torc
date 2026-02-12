try:
    from .__version__ import __version__
except ImportError:
    __version__ = None

COPPER = (0.722, 0.451, 0.200)
SILVER = (0.75, 0.75, 0.75)

# units
mm = 1e-3
inch = 25.4 * mm
cm = 1e-2
gauss = 1e-4
gauss_per_cm = gauss / cm

# Unit vectors:
X = (1, 0, 0)
Y = (0, 1, 0)
Z = (0, 0, 1)

from .torc import (
    CurrentObject,
    Container,
    Line,
    Arc,
    Loop,
    StraightSegment,
    CurvedSegment,
    RoundCoil,
    RacetrackCoil,
    CoilPair,
)

__all__ = [
    'COPPER',
    'SILVER',
    'mm',
    'inch',
    'cm',
    'gauss',
    'gauss_per_cm',
    'X',
    'Y',
    'Z',
    'CurrentObject',
    'Container',
    'Line',
    'Arc',
    'Loop',
    'StraightSegment',
    'CurvedSegment',
    'RoundCoil',
    'RacetrackCoil',
    'CoilPair',
]
