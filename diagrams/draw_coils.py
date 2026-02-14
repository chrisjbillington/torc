import sys
sys.path.insert(0, '..')

from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

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

FONT = QFont("Monospace")
FONT.setPointSize(13)

BOLDFONT = QFont(FONT)
BOLDFONT.setBold(True)

BLACK = pg.mkColor((0, 0, 0))
WHITE = pg.mkColor((255, 255, 255))
RED = pg.mkColor((255, 0, 0))
GREEN = pg.mkColor((0, 160, 0))
BLUE = pg.mkColor((0, 0, 255))
PURPLE = pg.mkColor((192, 0, 192))


AXIS_COLOR = pg.mkColor((148, 148, 148, 128))
AXIS_COLOR = pg.mkColor((0, 0, 0, 48))
TICK_COLOR = BLACK


def setup_scene(view):
    # view.setCameraParams(elevation=35.264, azimuth=-135)
    # view.setBackgroundColor('lightgrey')
    pass


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
        # mode='lines',
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
        # mode='lines',
        antialias=True,
        # glOptions='translucent',
    )
    view.addItem(line)

    if radii:
        draw_line(view, r0, r0 + r, width=width, color=color)
        draw_line(view, r0, pts[-1], width=width, color=color)


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
        h = r / np.linalg.norm(r)  # radial (outward) at start
        draw_line(view, p_start, p_start + ARROW_HEAD_LENGTH * (t + h / 2), width=width, color=color)
        draw_line(view, p_start, p_start + ARROW_HEAD_LENGTH * (t - h / 2), width=width, color=color)

    if head in ['end', 'both']:
        c, s = np.cos(theta), np.sin(theta)
        p_end = r0 + c * r + s * nxr + (1 - c) * np.dot(n_hat, r) * n_hat
        t_unnorm = -s * r + c * nxr
        t = t_unnorm / np.linalg.norm(t_unnorm)
        h = (p_end - r0)
        h = h / np.linalg.norm(h)  # radial (outward) at end
        draw_line(view, p_end, p_end - ARROW_HEAD_LENGTH * (t + h / 2), width=width, color=color)
        draw_line(view, p_end, p_end - ARROW_HEAD_LENGTH * (t - h / 2), width=width, color=color)


def draw_sector(view, r0, r, n, theta, width=1, color=BLACK):
    draw_arc(view, r0, r, n, theta, radii=True, width=width, color=color)


def draw_arrow(view, r_start, r_end, head_direction, head='end', width=2, color=BLACK):
    assert head in ['start', 'end', 'both']
    h = head_direction / np.linalg.norm(head_direction)
    r = r_end - r_start
    r /= np.linalg.norm(r)
    draw_line(view, r_start, r_end, width=width, color=color)
    if head in ['start', 'both']:
        # start
        draw_line(view, r_start, r_start + ARROW_HEAD_LENGTH * (r + h / 2), width=width, color=color)
        draw_line(view, r_start, r_start + ARROW_HEAD_LENGTH * (r - h / 2), width=width, color=color)
    if head in ['end', 'both']:
        # end
        draw_line(view, r_end, r_end - ARROW_HEAD_LENGTH * (r + h / 2), width=width, color=color)
        draw_line(view, r_end, r_end - ARROW_HEAD_LENGTH * (r - h / 2), width=width, color=color)


def draw_coord_axes(view):

    draw_line(view, -1000 * X, 1000 * X, color=AXIS_COLOR, always_on_top=False)
    draw_line(view, -1000 * Y, 1000 * Y, color=AXIS_COLOR, always_on_top=False)
    draw_line(view, -1000 * Z, 1000 * Z, color=AXIS_COLOR, always_on_top=False)

    draw_arrow(view, ORIGIN, COORD_AXIS_SIZE*X , head_direction=Y, color=RED)
    draw_arrow(view, ORIGIN, COORD_AXIS_SIZE * Y, head_direction=X, color=GREEN)
    draw_arrow(view, ORIGIN, COORD_AXIS_SIZE * Z, head_direction=X - Y, color=BLUE)

    textitem = gl.GLTextItem(
        pos=(COORD_AXIS_SIZE + POINT_LABEL_OFFSET) * X,
        text='u',
        color=RED,
        font=BOLDFONT,
        alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    )
    view.addItem(textitem)

    textitem = gl.GLTextItem(
        pos=(COORD_AXIS_SIZE + POINT_LABEL_OFFSET) * Y,
        text='v',
        color=GREEN,
        font=BOLDFONT,
        alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    )
    view.addItem(textitem)

    textitem = gl.GLTextItem(
        pos=(COORD_AXIS_SIZE + POINT_LABEL_OFFSET) * Z,
        text='n',
        color=BLUE,
        font=BOLDFONT,
        alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
    )
    view.addItem(textitem)

    draw_point(view, ORIGIN, "r0")

def draw_label(view, pos, text, alignment='left', color=BLACK):
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
        font=FONT,
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

    app, view = obj.show()

    setup_scene(view)
    draw_coord_axes(view)

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

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('StraightSegment.png')


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

    app, view = obj.show()

    setup_scene(view)
    draw_coord_axes(view)


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
        # n_tick=Y,
        # tick_length=R_OUTER + TICK_LENGTH,
        text='inner_radius',
        color=PURPLE,
        # label_alignment='right'
    )

    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_OUTER_ARROW_END,
        n_tick=-X,
        tick_length=TICK_LENGTH,
        # n_tick=-X,
        # tick_length=R_OUTER + TICK_LENGTH,
        text='outer_radius',
        color=PURPLE,
        label_alignment='right'
    )

    # swept angle
    draw_arc_arrow(view, HEIGHT / 2 * Z, X, Z, SWEPT_ANGLE, color=BLACK)
    draw_label(
        view,
        HEIGHT / 2 * Z + (1 + DISTANCE_LABEL_OFFSET) * (X + Y) / np.sqrt(2),
        "swept_angle",
        alignment='bottom',
    )

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('CurvedSegment.png')


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

    app, view = obj.show()

    setup_scene(view)
    draw_coord_axes(view)

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
        # n_tick=Y,
        # tick_length=R_OUTER + TICK_LENGTH,
        text='inner_radius',
        color=PURPLE,
        # label_alignment='right'
    )

    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_OUTER_ARROW_END,
        n_tick=X,
        tick_length=TICK_LENGTH,
        # n_tick=-X,
        # tick_length=R_OUTER + TICK_LENGTH,
        text='outer_radius',
        color=PURPLE,
        # label_alignment='right'
    )

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('RacetrackCoil.png')


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

    app, view = obj.show()

    setup_scene(view)
    draw_coord_axes(view)

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
        # n_tick=Y,
        # tick_length=R_OUTER + TICK_LENGTH,
        text='inner_radius',
        color=PURPLE,
        # label_alignment='right'
    )

    R_OUTER_ARROW_END = R_ARROWS_START + R_OUTER * Y
    draw_length_indicator(
        view,
        R_ARROWS_START,
        R_OUTER_ARROW_END,
        n_tick=-X,
        tick_length=TICK_LENGTH,
        # n_tick=-X,
        # tick_length=R_OUTER + TICK_LENGTH,
        text='outer_radius',
        color=PURPLE,
        label_alignment='right'
    )

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('RoundCoil.png')


def draw_Line():
    LENGTH = 7

    obj = Line(r_start=-LENGTH / 2 * Z, r_end=LENGTH / 2 * Z)

    app, view = obj.show(surfaces=False, lines=True)

    setup_scene(view)
    draw_coord_axes(view)

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

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('Line.png')


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

    app, view = obj.show(surfaces=False, lines=True)

    setup_scene(view)
    draw_coord_axes(view)

    # Radius
    draw_sector(view, ORIGIN, R * X, Z, SWEPT_ANGLE, color=TICK_COLOR)

    R_END = R * X

    draw_length_indicator(
        view,
        ORIGIN,
        R_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        # n_tick=Y,
        # tick_length=R_OUTER + TICK_LENGTH,
        text='radius',
        color=PURPLE,
        # label_alignment='right'
    )

    # swept angle
    draw_arc_arrow(view, ORIGIN, X, Z, SWEPT_ANGLE, color=BLACK)
    draw_label(
        view,
        (1 + DISTANCE_LABEL_OFFSET) * (X + Y) / np.sqrt(2),
        "swept_angle",
        alignment='bottom',
    )

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('Arc.png')


def draw_Loop():
    R = 4

    obj = Loop(
        r0=ORIGIN,
        n=Z,
        radius=R,
    )

    app, view = obj.show(surfaces=False, lines=True)

    setup_scene(view)
    draw_coord_axes(view)

    # Radius
    # draw_sector(view, ORIGIN, R * X, Z, SWEPT_ANGLE, color=TICK_COLOR)

    R_END = R * X

    draw_length_indicator(
        view,
        ORIGIN,
        R_END,
        n_tick=-Y,
        tick_length=TICK_LENGTH,
        # n_tick=Y,
        # tick_length=R_OUTER + TICK_LENGTH,
        text='radius',
        color=PURPLE,
        # label_alignment='right'
    )

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('Loop.png')


# draw_StraightSegment()
# draw_RacetrackCoil()
# draw_RoundCoil()
# draw_CurvedSegment()
# draw_Line()
# draw_Arc()
# draw_Loop()
