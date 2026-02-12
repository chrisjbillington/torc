try:
    from .__version__ import __version__
except ImportError:
    __version__ = None


from .torc import (
    COPPER,
    SILVER,
    mm,
    inch,
    cm,
    gauss,
    gauss_per_cm,
    X,
    Y,
    Z,
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
