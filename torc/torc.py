"""Magnetic field and gradient calculations for current-carrying coils.

This module provides classes for computing the magnetic field produced by
current-carrying conductors of various geometries — loops, straight wires, arcs,
round coils, racetrack coils, and arbitrary combinations thereof — positioned and
oriented anywhere in 3D space.

Finite cross-section conductors (e.g. :class:`RoundCoil`, :class:`RacetrackCoil`)
are approximated by distributing multiple idealised 1D current elements through
the cross-section. Loop fields are computed analytically using complete elliptic
integrals; straight wire fields use the Biot–Savart result for a finite wire; arcs
and curved segments are approximated as sequences of straight segments.

All quantities are in SI units: positions in metres, currents in amps, and fields
in tesla. Convenience constants (:data:`mm`, :data:`cm`, :data:`inch`,
:data:`gauss`, :data:`gauss_per_cm`) are provided for unit conversions.

3D visualisation of coil geometry is available via :meth:`CurrentObject.show`,
which uses pyqtgraph/OpenGL.
"""

import numpy as np
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0

pi = np.pi

#: Millimetres — multiply by this to convert mm to metres.
mm = 1e-3
#: Inches — multiply by this to convert inches to metres.
inch = 25.4 * mm
#: Centimetres — multiply by this to convert cm to metres.
cm = 1e-2
#: Gauss — multiply by this to convert gauss to tesla.
gauss = 1e-4
#: Gauss per centimetre — multiply by this to convert gauss/cm to tesla/metre.
gauss_per_cm = gauss / cm

#: RGB colour tuple for copper, for use with :meth:`CurrentObject.show`.
COPPER = (0.722, 0.451, 0.200)
#: RGB colour tuple for silver, for use with :meth:`CurrentObject.show`.
SILVER = (0.75, 0.75, 0.75)

#: Unit vector in the x direction.
X = (1, 0, 0)
#: Unit vector in the y direction.
Y = (0, 1, 0)
#: Unit vector in the z direction.
Z = (0, 0, 1)


# Default discretisation parameters:
_DEFAULT_ARC_SEGS = 12
_DEFAULT_CROSS_SEC_SEGS = 12


def _formatobj(obj, *attrnames):
    """Format an object and some attributes for printing"""
    attrs = ", ".join(f"{name}={getattr(obj,  name, None)}" for name in attrnames)
    return f"<{obj.__class__.__name__}({attrs}) at {hex(id(obj))}>"


def _get_factors(n):
    """return all the factors of n"""
    factors = set()
    for i in range(1, int(n ** (0.5)) + 1):
        if not n % i:
            factors.update((i, n // i))
    return factors


def _segments(x_min, x_max, y_min, y_max, N_segments):
    """Find the optimal cartesian grid for splitting up a rectangle of spanning x_min to
    x_max and y_min to y_max into N_segments equal sized segments such that each segment
    is as close to square as possible. This is the same as minimising the surface area
    between segments. Return a list of the midpoints of each segment"""
    size_x = x_max - x_min
    size_y = y_max - y_min
    lowest_surface_area = None
    for n_x in _get_factors(N_segments):
        n_y = N_segments // n_x
        surface_area = n_x * size_y + n_y * size_x
        if lowest_surface_area is None or surface_area < lowest_surface_area:
            lowest_surface_area = surface_area
            best_n_x, best_n_y = n_x, n_y
    dx = size_x / best_n_x
    dy = size_y / best_n_y

    midpoints = []
    for x in np.linspace(x_min + dx / 2, x_max - dx / 2, best_n_x):
        for y in np.linspace(y_min + dy / 2, y_max - dy / 2, best_n_y):
            midpoints.append((x, y))
    return midpoints


def _rectangular_tube(x0, x1, y0, y1, z0, z1, nz=2, bevel=0.075):
    """Create 3 2D arrays x, y, z for the points on the surface of a tube with
    rectangular cross section. x0, x1, y0 and y1 are the transverse extent of the
    tube, z0 and z1 describe its longitudinal extent. nz may be specified, this is how
    many points will be created along the z direction. Although this is not necessary
    to describe a straight tube, a curved tube can be made by transforming the
    returned points, in which case more than 2 points is necessary for a smooth
    result. Bevel may be given, this is the fraction of the shorter side of the cross
    section that will be chopped off the corners of the cross section to create a 45
    degree bevel on each corner."""
    b = bevel * min((y1 - y0), (x1 - x0))
    # Four sides plus bevels plus duplicate final point to close the path
    n_transverse = 9
    # The shape of the cross section, with bevels:
    y = np.array([y1 - b, y1, y1, y1 - b, y0 + b, y0, y0, y0 + b, y1 - b])
    x = np.array([x0, x0 + b, x1 - b, x1, x1, x1 - b, x0 + b, x0, x0])
    z = np.linspace(z0, z1, nz)
    # Broadcasting
    z = np.broadcast_to(z[:, np.newaxis], (nz, n_transverse))
    x = np.broadcast_to(x, (nz, n_transverse))
    y = np.broadcast_to(y, (nz, n_transverse))
    return x, y, z


def _broadcast(r):
    """If r=(x, y, z) is a tuple or list of arrays or scalars, broadcast it to be a
    single array with the list/tuple index corresponding to the first dimension."""
    if not isinstance(r, np.ndarray):
        return np.array(np.broadcast_arrays(*r))
    return r


def field_of_current_loop(r, z, R, I):
    """Compute, in cylindrical coordinates, Br(r, z), Bz(r, z) of a current loop with
    current I and radius R, centred at the origin with normal vector pointing in the z
    direction"""
    k2 = 4 * r * R / (z ** 2 + (R + r) ** 2)
    E_k2 = ellipe(k2)
    K_k2 = ellipk(k2)
    rprime2 = z ** 2 + (r - R) ** 2

    B_r_num = mu_0 * z * I * ((R ** 2 + z ** 2 + r ** 2) / rprime2 * E_k2 - K_k2)
    B_r_denom = 2 * pi * r * np.sqrt(z ** 2 + (R + r) ** 2)

    # Some hoop jumping to set B_r = 0 when r = 0 despite the expression having a
    # division by zero in it in when r = 0:
    if isinstance(r, np.ndarray):
        B_r = np.zeros(B_r_denom.shape)
        B_r[r != 0] = B_r_num[r != 0] / B_r_denom[r != 0]
    elif r == 0:
        B_r = 0.0
    else:
        B_r = B_r_num / B_r_denom

    B_z_num = mu_0 * I * ((R ** 2 - z ** 2 - r ** 2) / rprime2 * E_k2 + K_k2)
    B_z_denom = 2 * pi * np.sqrt(z ** 2 + (R + r) ** 2)

    B_z = B_z_num / B_z_denom

    return B_r, B_z


def field_of_current_line(r, z, L, I):
    """compute, in cylindrical coordinates, B_phi(r, z) of a current-carrying straight
    wire of length L running from the origin to z = L with current flowing in the +z
    direction."""
    prefactor = mu_0 * I / (4 * pi * r)
    term1 = z / np.sqrt(r ** 2 + z ** 2)
    term2 = (L - z) / np.sqrt(r ** 2 + (L - z) ** 2)
    return prefactor * (term1 + term2)


def _cross(a, b):
    """Cross product of a and b. For some reason np.cross is very slow, so here we
    are."""
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return np.array([x, y, z])


def _srgb_to_linear(c):
    c = np.asarray(c)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.asarray(c)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _reinhard_luminance_tonemap(rgb_linear):
    """Compress HDR linear RGB to [0,1] via Reinhard, preserving hue."""
    BT709_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
    lum = rgb_linear @ BT709_LUMINANCE
    mapped = lum / (1 + lum)
    return rgb_linear * (mapped / (lum + 1e-12))[:, np.newaxis]


def _do_shading(verts, faces, color, r_light=(1,2,3), ambient=0.1):
    # Return shaded RGBA colours (as floats 0–1, with alpha always 1) for each face
    # based on angle to a fixed light source verts: n×3 array of points specifying x,y,z
    # coords of a list of vertices faces: m×3 array of indices into verts specifying a
    # list of triangles. Color should be a 3-tuple of floats 0–1.

    r_light = np.array(r_light, dtype=float)
    r_light /= np.linalg.norm(r_light)

    # Compute intensity for angle, such that average intensity over all angles is 1.0 +
    # ambient:
    AVG_ILLUMINATION = 2 / np.pi

    # face normals via cross product of triangle edges
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    intensity = np.abs(normals @ r_light) / AVG_ILLUMINATION

    # add some ambient so nothing is fully black
    intensity += ambient / AVG_ILLUMINATION

    # Convert desired colour to linear colour space, apply intensity factor, tone map to
    # displayable range, then convert back to SRGB:
    hdr_linear = intensity[:, np.newaxis] * _srgb_to_linear(color)
    ldr_linear = _reinhard_luminance_tonemap(hdr_linear)
    face_colors_srgb = _linear_to_srgb(ldr_linear)

    face_colors = np.column_stack([face_colors_srgb, np.ones(len(faces))])
    return face_colors


class CurrentObject(object):
    def __init__(self, r0, zprime, xprime=None, n_turns=1, name=None):
        """A current-carrying object with a coordinate system centred at position r0 =
        (x0, y0, z0), with primary axis pointing along zprime = (zprime_x, zprime_y,
        zprime_z) and secondary axis pointing along xprime = (xprime_x, xprime_y,
        xprime_z). These two axes define the orientation of a right handed coordinate
        system (xprime, yprime, zprime) for the object with respect to the lab
        coordinate directions (x, y, z). The two axes do not need to be normalised (they
        will be normalised automatically), but must be orthogonal. if xprime is None
        (perhaps if the object has rotational symmetry such that it doesn't matter), it
        will be chosen randomly. n_turns is an overall multiplier for the current."""
        self.r0 = np.array(r0)
        self.zprime = np.array(zprime) / np.sqrt(np.dot(zprime, zprime))
        if xprime is None:
            # A random vector that is orthogonal to zprime:
            xprime = _cross(np.random.randn(3), zprime)
        self.xprime = np.array(xprime) / np.sqrt(np.dot(xprime, xprime))

        if not abs(np.dot(self.xprime, self.zprime)) < 1e-10:
            raise ValueError("Primary and secondary axes of object not orthogonal")

        self.yprime = _cross(self.zprime, self.xprime)

        # Rotation matrix from local frame to lab frame:
        self.Q_rot = np.stack([self.xprime, self.yprime, self.zprime], axis=1)
        self.n_turns = n_turns
        self.name = name

    @property
    def x(self):
        """The x coordinate of the object's centre position in the lab frame."""
        return self.r0[0]

    @property
    def y(self):
        """The y coordinate of the object's centre position in the lab frame."""
        return self.r0[1]

    @property
    def z(self):
        """The z coordinate of the object's centre position in the lab frame."""
        return self.r0[2]

    def pos_to_local(self, r):
        """Take a point r = (x, y, z) in the lab frame and return rprime = (xprime,
        yprime, zprime) in the local frame of reference of the object."""
        r = _broadcast(r)
        return np.einsum('ij,j...->i...', self.Q_rot.T, (r.T - self.r0).T)

    def pos_to_lab(self, rprime):
        """Take a point rprime = (xprime, yprime, zprime) in the local frame of the
        object and  return r = (x, y, z) in the lab frame."""
        rprime = _broadcast(rprime)
        return (np.einsum('ij,j...->i...', self.Q_rot, rprime).T + self.r0).T

    def vector_to_local(self, v):
        """Take a vector v = (v_x, v_y, v_z) in the lab frame and return vprime =
        (v_xprime, v_yprime, v_zprime) in the local frame of reference of the object.
        This is different to transforming coordinates as it only rotates the vector, it
        does not translate it."""
        v = _broadcast(v)
        return np.einsum('ij,j...->i...', self.Q_rot.T, v)

    def vector_to_lab(self, vprime):
        """Take a vector vprime=(v_xprime, v_yprime, v_zprime) in the local frame of the
        object and return v = (v_x, v_y, v_z) in the lab frame. This is different to
        transforming coordinates as it only rotates the vector, it does not translate
        it."""
        vprime = _broadcast(vprime)
        return np.einsum('ij,j...->i...', self.Q_rot, vprime)

    def B(self, r, I):
        """Return the magnetic field at position r=(x, y, z)"""
        # r = _broadcast(r)
        rprime = self.pos_to_local(r)
        return self.vector_to_lab(self.B_local(rprime, I * self.n_turns))

    def B_local(self, rprime, I):
        """Compute the magnetic field in the local coordinate frame. Subclasses
        override this to provide the actual field calculation. The base class
        implementation returns zero."""
        return np.zeros_like(rprime)

    def dB(self, r, I, s, ds=10e-6):
        """Return a magnetic field derivative at position r=(x, y, z) for a given
        current. The derivative returned is that of the field vector in the direction s,
        which can be 'x', 'y', 'z', or an arbitrary vector whose direction will be used
        (magnitude ignored). Step size ds for numerical differentiation can be given,
        otherwise defaults to 10um. Derivative is evaluated with a 2nd order central
        finite difference."""
        if isinstance(s, str):
            try:
                s = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}[s]
            except KeyError:
                raise KeyError("s must be one of 'x', 'y', 'z' or a vector") from None
        s = np.array(s, dtype=float)
        s /= np.sqrt(np.dot(s, s))
        r = _broadcast(r)
        rp = ((r.T) + s * ds).T
        rm = ((r.T) - s * ds).T
        return (self.B(rp, I) - self.B(rm, I)) / (2 * ds)

    def surfaces(self):
        """Return a list of 3D surface meshes in lab coordinates for visualisation.
        Each element is a tuple ``(x, y, z)`` of 2D arrays suitable for mesh
        rendering."""
        return [self.pos_to_lab(pts) for pts in self.local_surfaces()]

    def lines(self):
        """Return a list of 3D line paths in lab coordinates for visualisation.
        Each element is a tuple ``(x, y, z)`` of 1D arrays tracing the path."""
        return [self.pos_to_lab(pts) for pts in self.local_lines()]

    def local_surfaces(self):
        """Return surface meshes in local coordinates. Subclasses override this to
        describe their geometry for visualisation. The base class returns an empty
        list."""
        return []

    def local_lines(self):
        """Return line paths in local coordinates. Subclasses override this to
        describe their geometry for visualisation. The base class returns an empty
        list."""
        return []

    # def show_mpl(
    #     self, surfaces=True, lines=False, color=COPPER, **kwargs
    # ):
    #     from mpl_toolkits import mplot3d
    #     import matplotlib.pyplot as plt

    #     ax = plt.axes(projection='3d')

    #     # Aspect ratio
    #     asp_x, asp_y, asp_z = 0, 0, 0

    #     if surfaces:
    #         surfaces = self.surfaces()
    #         for x, y, z in surfaces:
    #             ax.plot_surface(x, y, z, color=color, **kwargs)
    #             asp_x = max(asp_x, np.ptp(x))
    #             asp_y = max(asp_y, np.ptp(y))
    #             asp_z = max(asp_z, np.ptp(z))
    #     if lines:
    #         lines = self.lines()
    #         for x, y, z in lines:
    #             ax.plot3D(x, y, z, color=color, **kwargs)
    #             asp_x = max(asp_x, np.ptp(x))
    #             asp_y = max(asp_y, np.ptp(y))
    #             asp_z = max(asp_z, np.ptp(z))
                
    #     ax.set_box_aspect((asp_x, asp_y, asp_z))
    #     plt.show()

    def show(self, surfaces=True, lines=False, color=COPPER):
        """Open an interactive 3D pyqtgraph/OpenGL window displaying this object's
        geometry. The window blocks until closed."""
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtCore, QtGui
        
        VIEW_WIDTH = 800
        VIEW_HEIGHT = 600
        VIEW_FOV = 30

        app = pg.mkQApp()

        # antialiasing:
        fmt = QtGui.QSurfaceFormat()
        fmt.setSamples(16)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)

        view = gl.GLViewWidget()

        # Variable to store vertices for debugging
        all_verts = []
        
        if surfaces:
            surfaces_data = self.surfaces()
            for x, y, z in surfaces_data:
                # Print surface info for debugging
                # print(f"Surface shape: {x.shape}, Z range: {np.min(z)}-{np.max(z)}")
                
                # Convert meshgrid format to vertices and faces for GLMeshItem
                verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                all_verts.extend(verts)

                nrows, ncols = x.shape
                i, j = np.mgrid[: nrows - 1, : ncols - 1]
                idx = (i * ncols + j).ravel()
                faces = np.column_stack(
                    [
                        idx,
                        idx + 1,
                        idx + ncols,
                        idx + ncols,
                        idx + 1,
                        idx + ncols + 1,
                    ]
                ).reshape(-1, 3)
                
                mesh = gl.GLMeshItem(
                    vertexes=verts,
                    faces=faces,
                    faceColors=_do_shading(verts, faces, color),
                    smooth=False,
                    computeNormals=False,
                )
                view.addItem(mesh)
        
        if lines:
            lines_data = self.lines()
            for x, y, z in lines_data:
                # Print line info for debugging
                # print(f"Line shape: {x.shape}, Z range: {np.min(z)}-{np.max(z)}")
                
                # Create line path
                pts = np.vstack([x, y, z]).T
                all_verts.extend(pts)
                
                line = gl.GLLinePlotItem(
                    pos=pts,
                    color=(*color, 1.0),
                    antialias=True,
                )
                view.addItem(line)
        
        # Midpoint and extent of data:
        if all_verts:
            r = np.array(all_verts)
            r0 = (r.max(axis=0) + r.min(axis=0)) / 2
            rmax = np.sqrt(((r - r0) ** 2).sum(axis=1)).max()
        else:
            r0 = 0
            rmax = 1

        # Camera and scene params:
        view.resize(VIEW_WIDTH, VIEW_HEIGHT)
        view.setBackgroundColor('lightgrey')

        # axis guidelines
        axis = gl.GLAxisItem()
        axis.setSize(rmax / 2, rmax / 2, rmax / 2)
        view.addItem(axis)

        view.opts['fov'] = VIEW_FOV
        view.setCameraParams(elevation=30, azimuth=-60)

        view.opts['center'] = pg.Vector(*r0)
        theta_fov = VIEW_FOV * pi / 180
        view.opts['distance'] = (
            max(VIEW_WIDTH / VIEW_HEIGHT, 1) * rmax / np.tan(theta_fov / 2)
        )

        view.show()
        app.exec()
    
    def __str__(self):
        return _formatobj(self, 'name')

    def __repr__(self):
        return self.__str__()


class Container(CurrentObject):
    """A group of :class:`CurrentObject` instances whose fields are summed.

    Children can be passed at construction or added later with :meth:`add`.
    Individual children can be accessed by integer index, slice, or by name
    string."""

    def __init__(
        self,
        *children,
        r0=(0, 0, 0),
        zprime=(0, 0, 1),
        xprime=None,
        n_turns=1,
        name=None,
    ):
        super().__init__(
            r0=r0, zprime=zprime, xprime=xprime, n_turns=n_turns, name=name
        )
        self.children = list(children)

    def add(self, *children):
        """Add one or more :class:`CurrentObject` instances as children."""
        for child in children:
            self.children.append(child)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer, slice)):
            return self.children[key]
        elif isinstance(key, str):
            for child in self.children:
                if child.name == key:
                    return child
            raise KeyError(f"no object in container with name {key}")
        else:
            msg = f"""Can only look up objects in container by integer index or string
                name, not {type(key)} {key}"""
            raise TypeError(' '.join(msg.split()))

    def __delitem__(self, key):
        if isinstance(key, (int, np.integer, slice)):
            del self.children[key]
        elif isinstance(key, str):
            for child in self.children:
                if child.name == key:
                    self.children.remove(child)
            raise KeyError(f"no object in container with name {key}")
        else:
            msg = f"""Can only look up objects in container by integer index or string
                name, not {type(key)} {key}"""
            raise TypeError(' '.join(msg.split()))

    def __len__(self):
        return len(self.children)

    def index(self, item):
        """Return the index of a child object."""
        return self.children.index(item)

    def B(self, r, I):
        """Return the total magnetic field at position r due to all children."""
        Bs = []
        for child in self.children:
            Bs.append(child.B(r, I))
        return sum(Bs)

    def surfaces(self):
        """Return surface meshes from this object and all children, in lab
        coordinates."""
        surfaces = super().surfaces()
        for child in self.children:
            surfaces.extend(child.surfaces())
        return surfaces

    def lines(self):
        """Return line paths from this object and all children, in lab
        coordinates."""
        lines = super().lines()
        for child in self.children:
            lines.extend(child.lines())
        return lines


class Loop(CurrentObject):
    def __init__(self, r0, n, R, n_turns=1, name=None):
        """Counterclockwise current loop of radius R, centred at r0 = (x0, y0, z0) with
        normal vector n=(nx, ny, nz)"""
        super().__init__(r0=r0, zprime=n, n_turns=n_turns, name=name)
        self.R = R

    def B_local(self, rprime, I):
        """Field due to the loop at position rprime=(xprime, yprime, zprime) for current
        I"""
        xprime, yprime, zprime = rprime
        # Expression we need to call is in cylindrical coordinates:
        rho = np.sqrt(xprime ** 2 + yprime ** 2)
        B_rho, B_zprime = field_of_current_loop(rho, zprime, self.R, I)
        phi = np.arctan2(yprime, xprime)
        B_xprime = B_rho * np.cos(phi)
        B_yprime = B_rho * np.sin(phi)
        return np.array([B_xprime, B_yprime, B_zprime])

    def local_lines(self):
        theta = np.linspace(-pi, pi, 361)
        xprime = self.R * np.cos(theta)
        yprime = self.R * np.sin(theta)
        zprime = 0
        return [(xprime, yprime, zprime)]


class Line(CurrentObject):
    def __init__(self, r0, r1, n_turns=1, name=None):
        """Current line from r0 = (x0, y0, z0) to r1 = (x1, y1, z1) with current flowing
        from the former to the latter"""
        zprime = np.array(r1) - np.array(r0)
        super().__init__(r0=r0, zprime=zprime, n_turns=n_turns, name=name)
        self.L = np.sqrt(((np.array(r1) - np.array(r0)) ** 2).sum())

    def B_local(self, rprime, I):
        """Field due to the line at position rprime=(xprime, yprime, zprime) for current
        I"""
        xprime, yprime, zprime = rprime
        # Expression we need to call is in cylindrical coordinates:
        rho = np.sqrt(xprime ** 2 + yprime ** 2)
        B_phi = field_of_current_line(rho, zprime, self.L, I)
        phi = np.arctan2(yprime, xprime)
        B_xprime = -B_phi * np.sin(phi)
        B_yprime = B_phi * np.cos(phi)
        return np.array([B_xprime, B_yprime, np.zeros_like(B_xprime)])

    def local_lines(self):
        zprime = np.array([0, self.L], dtype=float)
        xprime = yprime = 0
        return [(xprime, yprime, zprime)]


class Arc(Container):
    def __init__(
        self,
        r0,
        n,
        n_perp,
        R,
        phi_0,
        phi_1,
        n_turns=1,
        n_segs=_DEFAULT_ARC_SEGS,
        name=None,
    ):
        """Current arc forming part of a loop centred at r0 with normal vector n, from
        angle phi_0 to phi_1 defined with respect to the direction n_perp, which
        should be a direction perpendicular to n. Current is flowing from phi_0 to
        phi_1, which if phi_0 < phi_1, is in the positive sense with respect to the
        normal direction n. This arc is constructed out of n_segs separate line
        segments, so the accuracy can be increased by increasing n_segs."""
        super().__init__(r0=r0, zprime=n, xprime=n_perp, n_turns=n_turns, name=name)
        self.R = R
        self.phi_0 = phi_0
        self.phi_1 = phi_1

        delta_phi = (phi_1 - phi_0) / n_segs
        for i in range(n_segs):
            phi_seg_start = phi_0 + i * delta_phi
            phi_seg_stop = phi_0 + (i + 1) * delta_phi
            xprime0 = R * np.cos(phi_seg_start)
            yprime0 = R * np.sin(phi_seg_start)
            xprime1 = R * np.cos(phi_seg_stop)
            yprime1 = R * np.sin(phi_seg_stop)

            r0_seg = self.pos_to_lab((xprime0, yprime0, 0))
            r1_seg = self.pos_to_lab((xprime1, yprime1, 0))
            self.add(Line(r0_seg, r1_seg, n_turns=n_turns))

    def local_lines(self):
        n_theta = int(round((self.phi_1 - self.phi_0) * 180 / pi)) + 1  # every 1 degree
        theta = np.linspace(self.phi_0, self.phi_1, n_theta)
        xprime = self.R * np.cos(theta)
        yprime = self.R * np.sin(theta)
        zprime = 0
        return [(xprime, yprime, zprime)]


class RoundCoil(Container):
    def __init__(
        self,
        r0,
        n,
        R_inner,
        R_outer,
        height,
        n_turns=1,
        cross_sec_segs=_DEFAULT_CROSS_SEC_SEGS,
        name=None,
    ):
        """A round loop of conductor with rectangular cross section, centred at r0 with
        normal vector n, inner radius R_inner, outer radius R_outer, and the given
        height (in the normal direction). The finite cross-section is approximated using
        a number cross_sec_segs of 1D current loops distributed evenly through the cross
        section. n_turns is an overall multiplier for the current used in field
        calculations"""
        super().__init__(r0=r0, zprime=n, n_turns=n_turns, name=name)
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.height = height

        n_turns_per_seg = self.n_turns / cross_sec_segs
        segs = _segments(R_inner, R_outer, -height / 2, height / 2, cross_sec_segs)
        for R, zprime in segs:
            r0_loop = self.pos_to_lab((0, 0, zprime))
            self.add(Loop(r0_loop, n, R, n_turns=n_turns_per_seg))

    def local_surfaces(self):
        # Create arrays (in local coordinates) describing surfaces of the coil for
        # plotting:
        n_theta = 361  # every 1 degree
        r, zprime, theta = _rectangular_tube(
            self.R_inner,
            self.R_outer,
            -self.height / 2,
            self.height / 2,
            -pi,
            pi,
            n_theta,
        )
        xprime = r * np.cos(theta)
        yprime = r * np.sin(theta)
        return [(xprime, yprime, zprime)]


class StraightSegment(Container):
    def __init__(
        self,
        r0,
        r1,
        n,
        width,
        height,
        n_turns=1,
        cross_sec_segs=_DEFAULT_CROSS_SEC_SEGS,
        name=None,
    ):
        """A straight segment of conductor, with current flowing in a rectangular cross
        section centred on the line from r0 to r1. A vector n normal to the direction of
        current flow determines which direction the 'width' refers to, the height refers
        to the size of the conductor in the remaining direction. The finite
        cross-section is approximated using a number cross_sec_segs of 1D current lines
        distributed evenly through the cross section. n_turns is an overall multiplier
        for the current used in field calculations"""
        r0 = np.array(r0, dtype=float)
        r1 = np.array(r1, dtype=float)
        super().__init__(r0=r0, zprime=r1 - r0, xprime=n, n_turns=n_turns, name=name)
        self.width = width
        self.height = height
        self.L = np.sqrt(((np.array(r1) - np.array(r0)) ** 2).sum())

        n_turns_per_seg = self.n_turns / cross_sec_segs
        segs = _segments(-width / 2, width / 2, -height / 2, height / 2, cross_sec_segs)
        for xprime, yprime in segs:
            r0_line = self.pos_to_lab((xprime, yprime, 0))
            r1_line = self.pos_to_lab((xprime, yprime, self.L))
            self.add(Line(r0_line, r1_line, n_turns=n_turns_per_seg))

    def local_surfaces(self):
        # Create arrays (in local coordinates) describing surfaces of the segment for
        # plotting:
        xprime, yprime, zprime = _rectangular_tube(
            -self.width / 2,
            self.width / 2,
            -self.height / 2,
            self.height / 2,
            0,
            self.L,
            2,
        )
        return [(xprime, yprime, zprime)]


class CurvedSegment(Container):
    def __init__(
        self,
        r0,
        n,
        n_perp,
        R_inner,
        R_outer,
        height,
        phi_0,
        phi_1,
        n_turns=1,
        cross_sec_segs=_DEFAULT_CROSS_SEC_SEGS,
        arc_segs=_DEFAULT_ARC_SEGS,
        name=None,
    ):

        """Rounded segment of conductor with rectangular cross section, forming part of
        a round coil centred at r0 with normal vector n, from angle phi_0 to phi_1
        defined with respect to the direction n_perp, which should be a direction
        perpendicular to n. Current is flowing from phi_0 to phi_1, which if phi_0 <
        phi_1, is in the positive sense with respect to the normal direction n. The
        finite cross-section is approximated using a number cross_sec_segs of 1D current
        arcs distributed evenly through the cross section, each itself approximated as
        arc_segs separate current lines. n_turns is an overall multiplier for the
        current used in field calculations"""
        super().__init__(r0=r0, zprime=n, xprime=n_perp, n_turns=n_turns, name=name)
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.height = height
        self.phi_0 = phi_0
        self.phi_1 = phi_1

        n_turns_per_seg = self.n_turns / cross_sec_segs
        segs = _segments(R_inner, R_outer, -height / 2, height / 2, cross_sec_segs)
        for R, zprime in segs:
            r0_arc = self.pos_to_lab((0, 0, zprime))
            self.add(Arc(r0_arc, n, n_perp, R, phi_0, phi_1, n_turns_per_seg, arc_segs))

    def local_surfaces(self):
        # Create arrays (in local coordinates) describing surfaces of the segment for
        # plotting:
        n_theta = int(round((self.phi_1 - self.phi_0) * 180 / pi)) + 1  # every 1 degree
        r, zprime, theta = _rectangular_tube(
            self.R_inner,
            self.R_outer,
            -self.height / 2,
            self.height / 2,
            self.phi_0,
            self.phi_1,
            n_theta,
        )
        xprime = r * np.cos(theta)
        yprime = r * np.sin(theta)
        return [(xprime, yprime, zprime)]


class RacetrackCoil(Container):
    def __init__(
        self,
        r0,
        n,
        n_perp,
        width,
        length,
        height,
        R_inner,
        R_outer,
        n_turns=1,
        arc_segs=_DEFAULT_ARC_SEGS,
        cross_sec_segs=_DEFAULT_CROSS_SEC_SEGS,
        name=None,
    ):
        """A rectangular cross section coil comprising four straight segments and four
        90-degree curved segments. The coil is centred at r0 with normal vector n, and
        has the given height in the normal direction. n_perp defines direction along
        which 'width' gives the distance between the inner surfaces of two straight
        segments. 'length' gives the distance between the inner surfaces of the other
        two straight segments. R_inner and R_outer are the inner and outer radii of
        curvature of the curved segments. The finite cross-section is approximated using
        a number cross_sec_segs of 1D current lines and arcs distributed evenly through
        the cross section, and each arc is further approximated as arc_segs separate
        current lines. n_turns is an overall multiplier for the current used in field
        calculations"""

        super().__init__(r0=r0, zprime=n, xprime=n_perp, n_turns=n_turns, name=name)
        self.width = width
        self.length = length
        self.height = height
        self.R_inner = R_inner
        self.R_outer = R_outer
        for xprime, yprime, phi_0, phi_1 in [
            [width / 2 - R_inner, length / 2 - R_inner, 0, pi / 2],
            [-width / 2 + R_inner, length / 2 - R_inner, pi / 2, pi],
            [-width / 2 + R_inner, -length / 2 + R_inner, pi, 3 * pi / 2],
            [width / 2 - R_inner, -length / 2 + R_inner, 3 * pi / 2, 2 * pi],
        ]:
            self.add(
                CurvedSegment(
                    self.pos_to_lab((xprime, yprime, 0)),
                    n,
                    n_perp,
                    R_inner,
                    R_outer,
                    height,
                    phi_0,
                    phi_1,
                    n_turns=self.n_turns,
                    cross_sec_segs=cross_sec_segs,
                    arc_segs=arc_segs,
                )
            )

        # Top and bottom bars:
        absxprime = width / 2 - R_inner
        absyprime = (length + R_outer - R_inner) / 2
        if absxprime != 0:  # Exclude this segment if its length is zero:
            for sign in [-1, +1]: # bottom, top
                xprime0 = sign * absxprime
                xprime1 = -sign * absxprime
                yprime = sign * absyprime
                self.add(
                    StraightSegment(
                        self.pos_to_lab((xprime0, yprime, 0)),
                        self.pos_to_lab((xprime1, yprime, 0)),
                        self.vector_to_lab((0, 1, 0)),
                        self.R_outer - self.R_inner,
                        self.height,
                        n_turns=n_turns,
                        cross_sec_segs=cross_sec_segs,
                    )
                )

        # Left and right bars
        absyprime = length / 2 - R_inner
        absxprime = (width + R_outer - R_inner) / 2
        if absyprime != 0:  # Exclude this segment if its length is zero:
            for sign in [-1, +1]: # Left, right
                yprime0 = -sign * absyprime
                yprime1 = sign * absyprime
                xprime = sign * absxprime
                self.add(
                    StraightSegment(
                        self.pos_to_lab((xprime, yprime0, 0)),
                        self.pos_to_lab((xprime, yprime1, 0)),
                        self.vector_to_lab((1, 0, 0)),
                        self.R_outer - self.R_inner,
                        self.height,
                        n_turns=n_turns,
                        cross_sec_segs=cross_sec_segs,
                    )
                )


class CoilPair(Container):
    def __init__(self, coiltype, r0, n, displacement, *args, **kwargs):
        """A pair of coils of the given type (any class accepting r0 and n as its first
        instantiation arguments) centred on r0. One coil is at (r0 + displacement * n) and
        has normal vector n, and the other is at (r0 - displacement * n). The second
        coil has normal vector n if parity is 1 or the string 'helmholtz', and  has
        normal vector -n if parity is -1 or the string 'anti-helmholtz'. Remaining
        arguments and keyword arguments will be passed to coiltype()."""
        name = kwargs.pop('name', None)
        super().__init__(r0=r0, zprime=n, name=name)
        parity = kwargs.pop('parity', 'helmholtz')
        if parity not in [+1, -1]:
            if parity == 'helmholtz':
                parity = +1
            elif parity == 'anti-helmholtz':
                parity = -1
            else:
                msg = "parity must be 'helmholtz' or 'anti-helmholtz' (or +/-1)."
                raise ValueError(msg)
        for unit_vec in [self.zprime, -self.zprime]:
            r0_coil = r0 + displacement * unit_vec
            n_coil = self.zprime if parity == +1 else unit_vec
            self.add(coiltype(r0_coil, n_coil, *args, **kwargs))
