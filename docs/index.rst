
==============
torc |release|
==============

`Chris Billington <mailto:chrisjbillington@gmail.com>`_, |today|


.. contents::
    :local:


``torc`` is a Python library for computing magnetic fields and gradients resulting
from current-carrying coils — loops, straight wires, round coils and racetrack coils of
rectangular cross-section — positioned and oriented arbitrarily in 3D space.

Finite cross-section conductors are approximated by distributing multiple idealised 1D
current elements through the cross-section.  Loop fields are computed analytically using
complete elliptic integrals; straight wire fields use the Biot–Savart result for a
finite wire; arcs and curved segments are approximated as sequences of straight
segments.

All quantities are in SI units: positions in metres, currents in amps, and fields in
tesla.

`View on PyPI <https://pypi.org/project/torc/>`_
| `View on GitHub <https://github.com/chrisjbillington/torc>`_
| `Read the docs <https://torc.readthedocs.io>`_

------------
Installation
------------

To install ``torc``, run:

.. code-block:: bash

    $ pip install torc

or to install from source, clone the repository and run:

.. code-block:: bash

    $ pip install .


-------------
Example usage
-------------

.. code-block:: python

    import numpy as np
    from torc import RoundCoil, CoilPair, Z, gauss_per_cm

    # A pair of round coils in anti-Helmholtz configuration:
    coils = CoilPair(
        coiltype=RoundCoil,
        r0=(0, 0, 0),
        n=Z,
        displacement=0.05,
        R_inner=0.04,
        R_outer=0.06,
        height=0.01,
        n_turns=100,
        parity='anti-helmholtz',
    )

    # Compute the field gradient at the centre:
    dBdz = coils.dB((0, 0, 0), I=1.0, s='z')
    print(f"dBz/dz = {dBdz[2] / gauss_per_cm:.1f} G/cm/A")

    # Display a 3D rendering of the coils:
    coils.show()


--------------
Unit constants
--------------

Multiply by these constants to convert to SI units.

.. autodata:: torc.torc.mm
.. autodata:: torc.torc.cm
.. autodata:: torc.torc.inch
.. autodata:: torc.torc.gauss
.. autodata:: torc.torc.gauss_per_cm


-------
Colours
-------

RGB colour tuples for use with :meth:`~torc.torc.CurrentObject.show`.

.. autodata:: torc.torc.COPPER
.. autodata:: torc.torc.SILVER


----------------------------
Unit vectors and coordinates
----------------------------

.. autodata:: torc.torc.X
.. autodata:: torc.torc.Y
.. autodata:: torc.torc.Z
.. autodata:: torc.torc.ORIGIN

----------------
Module reference
----------------

.. autoclass:: torc.torc.CurrentObject
    :members: r0, xprime, yprime, zprime, n_turns, name, x, y, z, B, dB, surfaces, lines, show

.. autoclass:: torc.torc.Container
    :members: add, index, B, surfaces, lines

.. autoclass:: torc.torc.Loop

.. autoclass:: torc.torc.Line

.. autoclass:: torc.torc.Arc

.. autoclass:: torc.torc.RoundCoil

.. autoclass:: torc.torc.StraightSegment

.. autoclass:: torc.torc.CurvedSegment

.. autoclass:: torc.torc.RacetrackCoil

.. autoclass:: torc.torc.CoilPair
