import sys
sys.path.insert(0, '..')

from pyqtgraph.Qt.QtGui import QFont
from pyqtgraph.Qt.QtCore import Qt

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from torc import (
    Line,
    Arc,
    Loop,
    StraightSegment,
    CurvedSegment,
    RoundCoil,
    RacetrackCoil,
    CoilPair,
    X,
    Y,
    Z,
    ORIGIN,
)


ARROW_OFFSET = 0.05
ARROW_HEAD_LENGTH = 0.125
DISTANCE_LABEL_OFFSET = 0.25
POINT_LABEL_OFFSET = 0.125
TICK_LENGTH = 0.25
COORD_AXIS_SIZE = 0.5
CURRENT_INDICATOR_OFFSET = 1

FONT = QFont("Monospace")
FONT.setPointSize(13)

BOLDFONT = QFont(FONT)
BOLDFONT.setBold(True)

BLACK = pg.mkColor((0, 0, 0))
WHITE = pg.mkColor((255, 255, 255))
RED = pg.mkColor((255, 0, 0))
GREEN = pg.mkColor((0, 160, 0))
BLUE = pg.mkColor((0, 0, 255))
PURPLE = pg.mkColor((128, 0, 160))
ORANGEYELLOW = pg.mkColor((255, 160, 0))


AXIS_COLOR = pg.mkColor((0, 0, 0, 48))
TICK_COLOR = BLACK


def rotate_vector(v, axis, angle):
    """Rotate vector v by angle (radians) about axis using Rodrigues' formula."""
    n = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    return c * v + s * np.cross(n, v) + (1 - c) * np.dot(n, v) * n


def setup_scene(view):
    draw_coord_axes(view)


def draw_line(view, r_start, r_end, width=1, color=BLACK, always_on_top=True):
    pos = np.array([r_start, r_end])
    if always_on_top:
        kwargs = {}
    else:
        kwargs = {'glOptions': 'translucent'}
    line = gl.GLLinePlotItem(
        pos=pos,
        color=color,
        width=width,
        antialias=True,
        **kwargs,
    )
    view.addItem(line)

def draw_arc(view, r0, r, n, theta, radii=False, width=1, color=BLACK):
    # Arc from r0+r, sweeping angle theta about axis n
    npts = int(round(theta * 180 / np.pi)) + 1
    phi = np.linspace(0, theta, npts)
    c = np.cos(phi)[:, np.newaxis]
    s = np.sin(phi)[:, np.newaxis]
    pts = r0 + c * r + s * np.cross(n, r) + (1 - c) * np.dot(n, r) * n
    line = gl.GLLinePlotItem(
        pos=pts,
        color=color,
        width=width,
        antialias=True,
    )
    view.addItem(line)

    if radii:
        draw_line(view, r0, r0 + r, width=width, color=color)
        draw_line(view, r0, pts[-1], width=width, color=color)


def draw_arrowhead(view, tip, direction, head_direction, width=2, color=BLACK):
    """Draw an arrowhead at tip.

    direction: unit vector pointing in the direction of arrow travel at the tip.
    head_direction: vector perpendicular to travel direction; barbs spread along this.
    """
    h = head_direction / np.linalg.norm(head_direction)
    draw_line(view, tip, tip - ARROW_HEAD_LENGTH * (direction + h / 2), width=width, color=color)
    draw_line(view, tip, tip - ARROW_HEAD_LENGTH * (direction - h / 2), width=width, color=color)


def draw_arc_arrow(view, r0, r, n, theta, head='end', width=2, color=BLACK):
    """Draw an arc with arrowheads in the plane of the arc.

    Same arc convention as draw_arc: arc starts at r0+r, sweeps angle theta
    about axis n.  Arrowheads open perpendicular to the arc plane (along n)."""
    assert head in ['start', 'end', 'both']
    draw_arc(view, r0, r, n, theta, radii=False, width=width, color=color)

    n_hat = n / np.linalg.norm(n)
    nxr = np.cross(n_hat, r)  # tangent direction at phi=0 (unnormalized)

    if head in ['start', 'both']:
        p_start = r0 + r
        t = nxr / np.linalg.norm(nxr)
        draw_arrowhead(view, p_start, -t, r, width=width, color=color)

    if head in ['end', 'both']:
        p_end = r0 + rotate_vector(r, n_hat, theta)
        t = rotate_vector(nxr, n_hat, theta)
        t = t / np.linalg.norm(t)
        h = (p_end - r0)
        h = h / np.linalg.norm(h)
        draw_arrowhead(view, p_end, t, h, width=width, color=color)


def draw_sector(view, r0, r, n, theta, width=1, color=BLACK):
    draw_arc(view, r0, r, n, theta, radii=True, width=width, color=color)


def draw_arrow(view, r_start, r_end, head_direction, head='end', width=2, color=BLACK):
    assert head in ['start', 'end', 'both']
    r = r_end - r_start
    r /= np.linalg.norm(r)
    draw_line(view, r_start, r_end, width=width, color=color)
    if head in ['start', 'both']:
        draw_arrowhead(view, r_start, -r, head_direction, width=width, color=color)
    if head in ['end', 'both']:
        draw_arrowhead(view, r_end, r, head_direction, width=width, color=color)


def draw_coord_axes(
    view,
    origin=ORIGIN,
    x=X,
    y=Y,
    z=Z,
    x_name='u',
    y_name='v',
    z_name='n',
    guidelines=True,
    point_label='r0',
):
    """Draw coordinate axes at a given origin.

    x, y, z: direction vectors (may be negative to flip or None to skip).
    x_name, y_name, z_name: labels (None to skip a label).
    guidelines: whether to draw extended faint lines through the origin.
    point_label: label for the origin point (None to skip).
    """

    axes = [
        (x, x_name, RED, Y),
        (y, y_name, GREEN, X),
        (z, z_name, BLUE, Y - X),
    ]

    for direction, name, color, head_dir in axes:
        if direction is None:
            continue
        d = direction / np.linalg.norm(direction)
        if guidelines:
            draw_line(
                view, origin - 1000 * d, origin + 1000 * d,
                color=AXIS_COLOR, always_on_top=False,
            )
        draw_arrow(
            view, origin, origin + COORD_AXIS_SIZE * d,
            head_direction=head_dir, color=color,
        )
        if name is not None:
            label_pos = origin + (COORD_AXIS_SIZE + POINT_LABEL_OFFSET) * d
            ax = np.argmax(np.abs(d))
            if ax == 0:  # X-dominant
                alignment = 'left' if d[0] > 0 else 'right'
            elif ax == 1:  # Y-dominant
                alignment = 'right' if d[1] > 0 else 'left'
            else:  # Z-dominant
                alignment = 'bottom' if d[2] > 0 else 'top'
            draw_label(view, label_pos, name, alignment=alignment, color=color, font=BOLDFONT)

    # if point_label is not None:
    draw_point(view, origin, point_label)


def draw_label(view, pos, text, alignment='left', color=BLACK, font=None):
    if font is None:
        font = FONT
    if alignment == 'left':
        alignflags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    elif alignment == 'right':
        alignflags = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    elif alignment == 'top':
        alignflags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
    elif alignment == 'bottom':
        alignflags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
    else:
        raise ValueError(alignment)
    textitem = gl.GLTextItem(
        pos=pos,
        text=text,
        color=color,
        font=font,
        alignment=alignflags,
    )
    view.addItem(textitem)


def draw_point(view, pos, text=None, label_pos='below', size=6,color=BLACK):

    from OpenGL import GL

    assert label_pos in ['above', 'below']
    if label_pos == 'above':
        n_offset = Z
        alignment = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
    else:
        n_offset = -Z
        alignment = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop

    # Always on top no matter what
    GLOPTIONS = {
        GL.GL_DEPTH_TEST: False,
        GL.GL_BLEND: False,
        GL.GL_CULL_FACE: False,
    }

    point = gl.GLScatterPlotItem(
        pos=np.array([pos]),
        size=size,
        color=color,
        pxMode=True,
        glOptions=GLOPTIONS,
    )
    view.addItem(point)
    if text is not None:
        textitem = gl.GLTextItem(
            pos=pos + n_offset * POINT_LABEL_OFFSET,
            text=text,
            color=color,
            font=FONT,
            alignment=alignment,
        )
        view.addItem(textitem)


def draw_length_indicator(
    view, r_start, r_end, n_tick, tick_length, text, color, label_alignment='left'
):
    r = r_end - r_start
    r /= np.linalg.norm(r)
    n_tick = n_tick / np.linalg.norm(n_tick)
    arrow_head_direction = np.cross(r, n_tick)

    r_tick = n_tick * tick_length

    # draw ticks
    draw_line(view, r_start, r_start + r_tick, color=TICK_COLOR)
    draw_line(view, r_end, r_end + r_tick, color=TICK_COLOR)

    draw_arrow(
        view,
        r_start=r_start + r_tick - ARROW_HEAD_LENGTH * n_tick + ARROW_OFFSET * r,
        r_end=r_end + r_tick - ARROW_HEAD_LENGTH * n_tick - ARROW_OFFSET * r,
        head_direction=n_tick,
        head='both',
        color=color,
    )
    draw_label(
        view,
        pos=(r_start + r_end) / 2 + r_tick - TICK_LENGTH * n_tick / 2 + DISTANCE_LABEL_OFFSET * n_tick,
        text=text,
        alignment=label_alignment,
    )


def save_image(view, filename):
    app = pg.mkQApp()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save(filename)


def draw_StraightSegment():
    LENGTH = 7
    WIDTH = 2.5
    HEIGHT = 1.5

    HEIGHT = 1.75
    WIDTH = 1.5

    R_START = -LENGTH / 2 * X
    R_END = LENGTH / 2 * X

    obj = StraightSegment(
        r0=ORIGIN,
        n=Z,
        u=X,
        length=LENGTH,
        width=WIDTH,
        height=HEIGHT,
    )

    view = obj.show(blocking=False)

    setup_scene(view)

    WIDTH_GUIDELINE_START = -WIDTH / 2 * Y + HEIGHT / 2 * Z + LENGTH / 2 * X
    WIDTH_GUIDELINE_END = WIDTH / 2 * Y + HEIGHT / 2 * Z + LENGTH / 2 * X
    draw_length_indicator(
        view,
        WIDTH_GUIDELINE_START,
        WIDTH_GUIDELINE_END,
        n_tick=X,
        tick_length=TICK_LENGTH,
        text='width',
        color=GREEN,
    )

    LENGTH_GUIDELINE_START = R_START - WIDTH / 2 * Y - HEIGHT / 2 * Z
    LENGTH_GUIDELINE_END = R_END - WIDTH / 2 * Y - HEIGHT / 2 * Z
    draw_length_indicator(
        view,
        LENGTH_GUIDELINE_START,
        LENGTH_GUIDELINE_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='length',
        color=RED,
    )

    HEIGHT_GUIDELINE_START = -HEIGHT / 2 * Z + R_END - WIDTH / 2 * Y
    HEIGHT_GUIDELINE_END = HEIGHT / 2 * Z + R_END - WIDTH / 2 * Y

    draw_length_indicator(
        view,
        HEIGHT_GUIDELINE_START,
        HEIGHT_GUIDELINE_END,
        n_tick=X,
        tick_length=TICK_LENGTH,
        text='height',
        color=BLUE,
    )

    # Current direction
    I_EDGE = WIDTH / 2 * Y + HEIGHT / 2 * Z
    I_OFFSET = CURRENT_INDICATOR_OFFSET * (Z + Y) / np.sqrt(2)
    I_BASE = I_EDGE + I_OFFSET
    I_START = -LENGTH / 4 * X + I_BASE
    I_END = LENGTH / 4 * X + I_BASE
    draw_arrow(
        view, I_START, I_END, head_direction=(Y + Z) / np.sqrt(2), color=ORANGEYELLOW
    )
    draw_label(
        view,
        I_BASE + DISTANCE_LABEL_OFFSET * Y,
        "I",
        alignment='right',
        color=ORANGEYELLOW,
        font=BOLDFONT,
    )

    save_image(view, 'StraightSegment.png')


def draw_CurvedSegment():
    HEIGHT = 2
    R_INNER = 3
    R_OUTER = 4.5
    SWEPT_ANGLE = np.pi / 2

    DR = R_OUTER - R_INNER

    obj = CurvedSegment(
        r0=ORIGIN,
        n=Z,
        u=X,
        height=HEIGHT,
        inner_radius=R_INNER,
        outer_radius=R_OUTER,
        swept_angle=SWEPT_ANGLE,
    )

    view = obj.show(blocking=False)

    setup_scene(view)

    # height
    HEIGHT_GUIDELINE_START = -HEIGHT / 2 * Z + R_OUTER * X
    HEIGHT_GUIDELINE_END = HEIGHT / 2 * Z + R_OUTER * X
    draw_length_indicator(
        view,
        HEIGHT_GUIDELINE_START,
        HEIGHT_GUIDELINE_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='height',
        color=BLUE,
    )

    # Radii
    R_ARROWS_START = HEIGHT / 2 * Z
    draw_point(view, R_ARROWS_START)

    draw_sector(view, R_ARROWS_START, R_INNER * X, Z, np.pi / 2, color=TICK_COLOR)
    draw_sector(view, R_ARROWS_START, R_OUTER * X, Z, np.pi / 2, color=TICK_COLOR)

    R_INNER_ARROW_END = R_ARROWS_START + R_INNER * X
    R_OUTER_ARROW_END = R_ARROWS_START + R_OUTER * Y

    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_INNER_ARROW_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='inner_radius',
        color=PURPLE,
    )

    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_OUTER_ARROW_END,
        n_tick=-X,
        tick_length=TICK_LENGTH,
        text='outer_radius',
        color=PURPLE,
        label_alignment='right'
    )

    # swept angle
    SWEPT_ANGLE_ARC_RADIUS = 1
    draw_arc_arrow(
        view, HEIGHT / 2 * Z, SWEPT_ANGLE_ARC_RADIUS * X, Z, SWEPT_ANGLE, color=BLACK
    )
    draw_label(
        view,
        HEIGHT / 2 * Z
        + (SWEPT_ANGLE_ARC_RADIUS + DISTANCE_LABEL_OFFSET) * (X + Y) / np.sqrt(2),
        "swept_angle",
        alignment='bottom',
    )

    # Current direction
    I_CENTER = (HEIGHT / 2 + CURRENT_INDICATOR_OFFSET) * Z
    I_R = R_OUTER * rotate_vector(X, Z, np.pi / 8)
    draw_arc_arrow(view, I_CENTER, I_R, Z, np.pi / 4, color=ORANGEYELLOW)
    draw_label(
        view,
        I_CENTER + (R_OUTER + DISTANCE_LABEL_OFFSET) * (X + Y) / np.sqrt(2),
        "I",
        alignment='bottom',
        color=ORANGEYELLOW,
        font=BOLDFONT,
    )

    save_image(view, 'CurvedSegment.png')


def draw_RacetrackCoil():
    LENGTH = 6
    WIDTH = 4
    HEIGHT = 1

    R_INNER = 1.5
    R_OUTER = 2.25

    DR = R_OUTER - R_INNER

    obj = RacetrackCoil(
        r0=ORIGIN,
        n=Z,
        u=X,
        inner_length=LENGTH,
        inner_width=WIDTH,
        height=HEIGHT,
        inner_radius=R_INNER,
        outer_radius=R_OUTER,
    )

    view = obj.show(blocking=False)

    setup_scene(view)

    # width
    WIDTH_GUIDELINE_START = -LENGTH / 2 * X + HEIGHT / 2 * Z - (WIDTH / 2 - R_INNER) * Y
    WIDTH_GUIDELINE_END = LENGTH / 2 * X + HEIGHT / 2 * Z - (WIDTH / 2 - R_INNER) * Y
    draw_length_indicator(
        view,
        WIDTH_GUIDELINE_START,
        WIDTH_GUIDELINE_END,
        n_tick=-Y,
        tick_length=R_OUTER + TICK_LENGTH + HEIGHT,
        text='inner_length',
        color=RED,
    )

    # length
    LENGTH_GUIDELINE_START = -WIDTH / 2 * Y + (LENGTH / 2 - R_INNER) * X + HEIGHT / 2 * Z
    LENGTH_GUIDELINE_END = WIDTH / 2 * Y + (LENGTH / 2 - R_INNER) * X + HEIGHT / 2 * Z
    draw_length_indicator(
        view,
        LENGTH_GUIDELINE_START,
        LENGTH_GUIDELINE_END,
        n_tick=X,
        tick_length=R_OUTER + TICK_LENGTH,
        text='inner_width',
        color=GREEN,
    )

    # height
    HEIGHT_GUIDELINE_START = (
        -HEIGHT / 2 * Z - (LENGTH / 2 + DR) * X - (WIDTH / 2 - R_INNER) * Y
    )
    HEIGHT_GUIDELINE_END = (
        HEIGHT / 2 * Z - (LENGTH / 2 + DR) * X - (WIDTH / 2 - R_INNER) * Y
    )
    draw_length_indicator(
        view,
        HEIGHT_GUIDELINE_START,
        HEIGHT_GUIDELINE_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='height',
        color=BLUE,
    )

    # Radii
    R_ARROWS_START = HEIGHT / 2 * Z + WIDTH / 2 * Y - LENGTH / 2 * X - R_INNER * (Y - X)
    draw_point(view, R_ARROWS_START)

    draw_sector(view, R_ARROWS_START, R_INNER * Y, Z, np.pi / 2, color=TICK_COLOR)
    draw_sector(view, R_ARROWS_START, R_OUTER * Y, Z, np.pi / 2, color=TICK_COLOR)

    R_INNER_ARROW_END = R_ARROWS_START - R_INNER * X
    R_OUTER_ARROW_END = R_ARROWS_START + R_OUTER * Y

    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_INNER_ARROW_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='inner_radius',
        color=PURPLE,
    )

    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_OUTER_ARROW_END,
        n_tick=X,
        tick_length=TICK_LENGTH,
        text='outer_radius',
        color=PURPLE,
    )

    # Current direction
    I_CORNER = (
        (LENGTH / 2 - R_INNER) * X
        + (WIDTH / 2 - R_INNER) * Y
        + (HEIGHT / 2 + CURRENT_INDICATOR_OFFSET) * Z
    )
    I_R = R_OUTER * rotate_vector(X, Z, np.pi / 8)
    draw_arc(view, I_CORNER, I_R, Z, 3 * np.pi / 8, width=2, color=ORANGEYELLOW)
    I_ARC_END = I_CORNER + R_OUTER * Y
    I_ARROW_END = I_ARC_END - (LENGTH - 2 * R_INNER) * X
    draw_arrow(view, I_ARC_END, I_ARROW_END, head_direction=Y, color=ORANGEYELLOW)
    draw_label(
        view,
        I_ARC_END + DISTANCE_LABEL_OFFSET * Y,
        "I",
        alignment='right',
        color=ORANGEYELLOW,
        font=BOLDFONT,
    )

    save_image(view, 'RacetrackCoil.png')


def draw_RoundCoil():
    HEIGHT = 1.75
    R_INNER = 3
    R_OUTER = 4.5

    DR = R_OUTER - R_INNER

    obj = RoundCoil(
        r0=ORIGIN,
        n=Z,
        height=HEIGHT,
        inner_radius=R_INNER,
        outer_radius=R_OUTER,
    )

    view = obj.show(blocking=False)

    setup_scene(view)

    # height
    HEIGHT_GUIDELINE_START = -HEIGHT / 2 * Z - R_OUTER * X
    HEIGHT_GUIDELINE_END = HEIGHT / 2 * Z - R_OUTER * X
    draw_length_indicator(
        view,
        HEIGHT_GUIDELINE_START,
        HEIGHT_GUIDELINE_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='height',
        color=BLUE,
    )

    # Radii
    R_ARROWS_START = HEIGHT / 2 * Z
    draw_point(view, R_ARROWS_START)

    draw_sector(view, R_ARROWS_START, R_INNER * X, Z, np.pi / 2, color=TICK_COLOR)
    draw_sector(view, R_ARROWS_START, R_OUTER * X, Z, np.pi / 2, color=TICK_COLOR)

    R_INNER_ARROW_END = R_ARROWS_START + R_INNER * X
    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_INNER_ARROW_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='inner_radius',
        color=PURPLE,
    )

    R_OUTER_ARROW_END = R_ARROWS_START + R_OUTER * Y
    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_OUTER_ARROW_END,
        n_tick=-X,
        tick_length=TICK_LENGTH,
        text='outer_radius',
        color=PURPLE,
        label_alignment='right'
    )

    # Current direction
    I_CENTER = (HEIGHT / 2 + CURRENT_INDICATOR_OFFSET) * Z
    I_R = R_OUTER * rotate_vector(X, Z, np.pi / 8)
    draw_arc_arrow(view, I_CENTER, I_R, Z, np.pi / 4, color=ORANGEYELLOW)
    draw_label(
        view,
        I_CENTER + (R_OUTER + DISTANCE_LABEL_OFFSET) * (X + Y) / np.sqrt(2),
        "I",
        alignment='bottom',
        color=ORANGEYELLOW,
        font=BOLDFONT,
    )

    save_image(view, 'RoundCoil.png')


def draw_Line():
    LENGTH = 7

    obj = Line(r_start=-LENGTH / 2 * Z, r_end=LENGTH / 2 * Z)

    view = obj.show(surfaces=False, lines=True, blocking=False)

    setup_scene(view)

    # length
    LENGTH_GUIDELINE_START = -LENGTH / 2 * Z
    LENGTH_GUIDELINE_END = LENGTH / 2 * Z
    draw_length_indicator(
        view,
        LENGTH_GUIDELINE_START,
        LENGTH_GUIDELINE_END,
        n_tick=(X - Y) / np.sqrt(2),
        tick_length=1 + TICK_LENGTH,
        text='length',
        color=BLUE,
    )

    draw_point(view, -LENGTH / 2 * Z, "r_start")
    draw_point(view, LENGTH / 2 * Z, "r_end", label_pos='above')

    # Current direction
    I_OFFSET = (Y - X) / np.sqrt(2)
    I_START = -LENGTH / 4 * Z + I_OFFSET
    I_END = LENGTH / 4 * Z + I_OFFSET
    draw_arrow(view, I_START, I_END, head_direction=I_OFFSET, color=ORANGEYELLOW)
    draw_label(
        view,
        (1 + DISTANCE_LABEL_OFFSET) * I_OFFSET,
        "I",
        alignment='right',
        color=ORANGEYELLOW,
        font=BOLDFONT,
    )

    save_image(view, 'Line.png')


def draw_Arc():
    R = 4
    SWEPT_ANGLE = np.pi / 2

    obj = Arc(
        r0=ORIGIN,
        n=Z,
        u=X,
        radius=R,
        swept_angle=SWEPT_ANGLE,
    )

    view = obj.show(surfaces=False, lines=True, blocking=False)

    setup_scene(view)

    # Radius
    draw_sector(view, ORIGIN, R * X, Z, SWEPT_ANGLE, color=TICK_COLOR)

    R_END = R * X

    draw_length_indicator(
        view,
        ORIGIN,
        R_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='radius',
        color=PURPLE,
    )

    # swept angle
    SWEPT_ANGLE_ARC_RADIUS = 1
    draw_arc_arrow(
        view, ORIGIN, SWEPT_ANGLE_ARC_RADIUS * X, Z, SWEPT_ANGLE, color=BLACK
    )
    draw_label(
        view,
        (SWEPT_ANGLE_ARC_RADIUS + DISTANCE_LABEL_OFFSET) * (X + Y) / np.sqrt(2),
        "swept_angle",
        alignment='bottom',
    )

    # Current direction
    I_CENTER = CURRENT_INDICATOR_OFFSET * Z
    I_R = R * rotate_vector(X, Z, np.pi / 8)
    draw_arc_arrow(view, I_CENTER, I_R, Z, np.pi / 4, color=ORANGEYELLOW)
    draw_label(
        view,
        I_CENTER + (R + DISTANCE_LABEL_OFFSET) * (X + Y) / np.sqrt(2),
        "I",
        alignment='bottom',
        color=ORANGEYELLOW,
        font=BOLDFONT,
    )

    save_image(view, 'Arc.png')


def draw_Loop():
    R = 4

    obj = Loop(
        r0=ORIGIN,
        n=Z,
        radius=R,
    )

    view = obj.show(surfaces=False, lines=True, blocking=False)
    setup_scene(view)

    # Radius
    R_END = R * X
    draw_length_indicator(
        view,
        ORIGIN,
        R_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        text='radius',
        color=PURPLE,
    )

    # Current direction
    I_CENTER = CURRENT_INDICATOR_OFFSET * Z
    I_R = R * rotate_vector(X, Z, np.pi / 8)
    draw_arc_arrow(view, I_CENTER, I_R, Z, np.pi / 4, color=ORANGEYELLOW)
    draw_label(
        view,
        I_CENTER + (R + DISTANCE_LABEL_OFFSET) * (X + Y) / np.sqrt(2),
        "I",
        alignment='bottom',
        color=ORANGEYELLOW,
        font=BOLDFONT,
    )

    save_image(view, 'Loop.png')


def draw_CoilPair():
    SEPARATION = 8
    LENGTH = 6
    WIDTH = 4
    HEIGHT = 1
    R_INNER = 1.5
    R_OUTER = 2.25
    DR = R_OUTER - R_INNER

    obj = CoilPair(
        coiltype=RacetrackCoil,
        r0=ORIGIN,
        n=Z,
        separation=SEPARATION,
        u=X,
        inner_length=LENGTH,
        inner_width=WIDTH,
        height=HEIGHT,
        inner_radius=R_INNER,
        outer_radius=R_OUTER,
        parity='anti-helmholtz',
    )

    view = obj.show(blocking=False)
    setup_scene(view)

    # Separation indicator
    SEP_XY = (LENGTH / 2 - R_INNER) * X - (WIDTH / 2 + DR) * Y
    SEP_TOP = SEP_XY + SEPARATION / 2 * Z
    SEP_BOT = SEP_XY - SEPARATION / 2 * Z
    draw_length_indicator(
        view,
        SEP_BOT,
        SEP_TOP,
        n_tick=X,
        tick_length=TICK_LENGTH,
        text='separation',
        color=BLUE,
    )

    # Coordinate axes at each coil center
    TOP_CENTER = SEPARATION / 2 * Z
    draw_coord_axes(view, y=None, origin=TOP_CENTER, guidelines=False, point_label=None)

    BOTTOM_CENTER = -SEPARATION / 2 * Z
    draw_coord_axes(
        view,
        origin=BOTTOM_CENTER,
        x=X,
        y=None,
        z=-Z,
        x_name='u',
        z_name='-n',
        guidelines=False,
        point_label=None,
    )

    # Current direction indicators on both coils
    # Same geometry, offset by ±SEPARATION/2 in Z. Top coil has arrowhead on the
    # straight segment; bottom coil has arrowhead at the start of the arc.
    I_CORNER_XY = (LENGTH / 2 - R_INNER) * X + (WIDTH / 2 - R_INNER) * Y
    I_R = R_OUTER * rotate_vector(X, Z, np.pi / 8)
    I_STRAIGHT = (LENGTH - 2 * R_INNER) * X

    for sign, arrow_on_line, arc_head in [(+1, True, None), (-1, False, 'start')]:
        I_Z = (sign * SEPARATION / 2 + HEIGHT / 2 + CURRENT_INDICATOR_OFFSET / 2) * Z
        I_CORNER = I_CORNER_XY + I_Z

        if arc_head is not None:
            draw_arc_arrow(
                view, I_CORNER, I_R, Z, 3 * np.pi / 8,
                head=arc_head, width=2, color=ORANGEYELLOW,
            )
        else:
            draw_arc(
                view, I_CORNER, I_R, Z, 3 * np.pi / 8, width=2, color=ORANGEYELLOW
            )

        I_ARC_END = I_CORNER + R_OUTER * Y
        I_LINE_END = I_ARC_END - I_STRAIGHT
        if arrow_on_line:
            draw_arrow(
                view, I_ARC_END, I_LINE_END, head_direction=Y, color=ORANGEYELLOW
            )
        else:
            draw_line(view, I_ARC_END, I_LINE_END, width=2, color=ORANGEYELLOW)

        draw_label(
            view,
            I_ARC_END + DISTANCE_LABEL_OFFSET * Y,
            "I",
            alignment='right',
            color=ORANGEYELLOW,
            font=BOLDFONT,
        )

    # Parity label
    draw_label(
        view,
        -(SEPARATION / 2 + COORD_AXIS_SIZE + 3 * POINT_LABEL_OFFSET) * Z,
        "parity = 'anti-helmholtz'",
        alignment='top',
    )

    save_image(view, 'CoilPair.png')


if __name__ == '__main__':
    # Suppress spurious Qt warning:
    import os
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland.textinput=false"

    # draw_StraightSegment()
    # draw_RacetrackCoil()
    # draw_RoundCoil()
    # draw_CurvedSegment()
    # draw_Line()
    # draw_Arc()
    # draw_Loop()
    draw_CoilPair()
