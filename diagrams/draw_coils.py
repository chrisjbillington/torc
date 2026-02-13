import sys
sys.path.insert(0, '..')

from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from torc import StraightSegment, RacetrackCoil, RoundCoil, X, Y, Z, ORIGIN



ARROW_OFFSET = 0.1
ARROW_HEAD_LENGTH = 0.2
DISTANCE_LABEL_OFFSET = 0.5
POINT_LABEL_OFFSET = 0.2
TICK_LENGTH = 1

FONT = QFont("Monospace")
FONT.setPointSize(14)

BLACK = pg.mkColor((0, 0, 0))
RED = pg.mkColor((255, 0, 0))
GREEN = pg.mkColor((0, 160, 0))
BLUE = pg.mkColor((0, 0, 255))
PURPLE = pg.mkColor("purple")
BLACK_SEMITRANSPARENT = pg.mkColor((0, 0, 0, 255))

def setup_scene(view):
    view.setCameraParams(elevation=35.264, azimuth=-135)
    # g = gl.GLGridItem(color=pg.mkColor((0,0,0,48)))
    # g.setSize(10, 10, 10)
    # g.setSpacing(0.25, 0.25)
    # view.addItem(g)

def draw_line(view, r_start, r_end, width=2, color=BLACK):
    pos = np.array([r_start, r_end])
    line = gl.GLLinePlotItem(
        pos=pos,
        color=color,
        width=width,
        mode='lines',
        antialias=True,
    )
    view.addItem(line)


def draw_arrow(view, r_start, r_end, head_direction, head='end', width=2, color=BLACK):
    assert head in ['start', 'end', 'both']
    h = head_direction / np.linalg.norm(head_direction)
    r = r_end - r_start
    r /= np.linalg.norm(r)
    draw_line(view, r_start, r_end, width=width, color=color)
    if head in ['start', 'both']:
        # start
        draw_line(view, r_start, r_start + ARROW_HEAD_LENGTH * (r + h / 2), color=color)
        draw_line(view, r_start, r_start + ARROW_HEAD_LENGTH * (r - h / 2), color=color)
    if head in ['end', 'both']:
        # end
        draw_line(view, r_end, r_end - ARROW_HEAD_LENGTH * (r + h / 2), color=color)
        draw_line(view, r_end, r_end - ARROW_HEAD_LENGTH * (r - h / 2), color=color)


def draw_coord_axes(view):
    draw_arrow(view, ORIGIN, X, head_direction=Y, color=RED)
    draw_arrow(view, ORIGIN, Y, head_direction=X, color=GREEN)
    draw_arrow(view, ORIGIN, Z, head_direction=X - Y, color=BLUE)


def draw_label(view, pos, text, color=BLACK):
    textitem = gl.GLTextItem(
        pos=pos,
        text=text,
        color=color,
        font=FONT,
        alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    )
    view.addItem(textitem)


def draw_point(view, pos, text=None, n_offset=None, size=6,color=BLACK):
    point = gl.GLScatterPlotItem(
        pos=np.array([pos]),
        size=size,
        color=color,
        pxMode=True,
        glOptions='opaque',
    )
    view.addItem(point)
    if text is not None:
        draw_label(view, pos + POINT_LABEL_OFFSET *  n_offset, text)


def draw_length_indicator(
    view, r_start, r_end, n_tick, tick_length, text, color
):
    r = r_end - r_start
    r /= np.linalg.norm(r)
    n_tick = n_tick / np.linalg.norm(n_tick)
    arrow_head_direction = np.cross(r, n_tick)

    r_tick = n_tick * tick_length

    # draw ticks
    draw_line(view, r_start, r_start + r_tick, color=BLACK_SEMITRANSPARENT)
    draw_line(view, r_end, r_end + r_tick, color=BLACK_SEMITRANSPARENT)

    draw_arrow(
        view,
        r_start=r_start + r_tick - TICK_LENGTH * n_tick / 2 + ARROW_OFFSET * r,
        r_end=r_end + r_tick - TICK_LENGTH * n_tick / 2 - ARROW_OFFSET * r,
        head_direction=n_tick,
        head='both',
        color=color,
    )
    draw_label(
        view,
        pos=(r_start + r_end) / 2 + r_tick - TICK_LENGTH * n_tick / 2 + DISTANCE_LABEL_OFFSET * n_tick,
        text=text,
    )


def draw_straightsegment():
    LENGTH = 7
    WIDTH = 2
    HEIGHT = 1.5

    R_START = -LENGTH / 2 * X
    R_END = LENGTH / 2 * X

    obj = StraightSegment(
        n=Z,
        r_start=R_START,
        r_end=R_END,
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
        text='|r_end - r_start|',
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

    # Label start and end
    draw_point(view, R_START, "r_start", n_offset=-Y)
    draw_point(view, R_END, "r_end", n_offset=-Y)

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('StraightSegment.png')


def draw_racetrackcoil():
    LENGTH = 4
    WIDTH = 6
    HEIGHT = 1

    R_INNER = 1.25
    R_OUTER = 2.25

    DR = R_OUTER - R_INNER

    obj = RacetrackCoil(
        r0=ORIGIN,
        n=Z,
        n_perp=X,
        width=WIDTH,
        length=LENGTH,
        height=HEIGHT,
        R_inner=R_INNER,
        R_outer=R_OUTER,
    )

    app, view = obj.show()

    setup_scene(view)
    draw_coord_axes(view)

    # width
    WIDTH_GUIDELINE_START = -WIDTH / 2 * X + HEIGHT / 2 * Z - (LENGTH / 2 - DR) * Y
    WIDTH_GUIDELINE_END = WIDTH / 2 * X + HEIGHT / 2 * Z - (LENGTH / 2 - DR) * Y
    draw_length_indicator(
        view,
        WIDTH_GUIDELINE_START,
        WIDTH_GUIDELINE_END,
        n_tick=-Y,
        tick_length=R_OUTER + TICK_LENGTH + HEIGHT / np.sqrt(2),
        text='width',
        color=RED,
    )

    # length
    LENGTH_GUIDELINE_START = -LENGTH / 2 * Y + (WIDTH / 2 - DR) * X + HEIGHT / 2 * Z
    LENGTH_GUIDELINE_END = LENGTH / 2 * Y + (WIDTH / 2 - DR) * X + HEIGHT / 2 * Z
    draw_length_indicator(
        view,
        LENGTH_GUIDELINE_START,
        LENGTH_GUIDELINE_END,
        n_tick=X,
        tick_length=R_OUTER + TICK_LENGTH,
        text='length',
        color=GREEN,
    )

    # height
    HEIGHT_GUIDELINE_START = (
        -HEIGHT / 2 * Z - (WIDTH / 2 + DR) * X - (LENGTH / 2 - R_INNER) * Y
    )
    HEIGHT_GUIDELINE_END = (
        HEIGHT / 2 * Z - (WIDTH / 2 + DR) * X - (LENGTH / 2 - R_INNER) * Y
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

    R_ARROWS_START = HEIGHT / 2 * Z + LENGTH / 2 * Y - WIDTH / 2 * X - R_INNER * (Y - X)
    draw_point(view, R_ARROWS_START)

    R_INNER_ARROW_END = R_ARROWS_START + R_INNER * Y
    draw_arrow(
        view,
        R_ARROWS_START + ARROW_OFFSET * Y,
        R_INNER_ARROW_END - ARROW_OFFSET * Y,
        head_direction=X,
        head='both',
        color=PURPLE,
    )
    draw_label(
        view,
        pos=(R_ARROWS_START + R_INNER_ARROW_END) / 2 + DISTANCE_LABEL_OFFSET * X,
        text='R_inner',
    )

    R_OUTER_ARROW_END = R_ARROWS_START - R_OUTER * X
    draw_arrow(
        view,
        R_ARROWS_START - ARROW_OFFSET * X,
        R_OUTER_ARROW_END + ARROW_OFFSET * X,
        head_direction=Y,
        head='both',
        color=PURPLE,
    )
    draw_label(
        view,
        pos=(R_ARROWS_START + R_OUTER_ARROW_END) / 2 - DISTANCE_LABEL_OFFSET * Y,
        text='R_outer',
    )

    draw_point(view, ORIGIN, "r0", n_offset=-Y)

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('RacetrackCoil.png')


def draw_roundcoil():
    HEIGHT = 1
    R_INNER = 3
    R_OUTER = 4

    DR = R_OUTER - R_INNER

    obj = RoundCoil(
        r0=ORIGIN,
        n=Z,
        height=HEIGHT,
        R_inner=R_INNER,
        R_outer=R_OUTER,
    )

    app, view = obj.show()

    setup_scene(view)
    draw_coord_axes(view)


    # height
    HEIGHT_GUIDELINE_START = (
        -HEIGHT / 2 * Z - R_OUTER * X
    )
    HEIGHT_GUIDELINE_END = (
        HEIGHT / 2 * Z - R_OUTER * X
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

    R_ARROWS_START = HEIGHT / 2 * Z
    draw_point(view, R_ARROWS_START)

    R_INNER_ARROW_END = R_ARROWS_START + R_INNER * X
    draw_arrow(
        view,
        R_ARROWS_START + ARROW_OFFSET * X,
        R_INNER_ARROW_END - ARROW_OFFSET * X,
        head_direction=-Y,
        head='both',
        color=PURPLE,
    )
    draw_label(
        view,
        pos=(R_ARROWS_START + R_INNER_ARROW_END) / 2 - DISTANCE_LABEL_OFFSET * Y,
        text='R_inner',
    )

    R_OUTER_ARROW_END = R_ARROWS_START + R_OUTER * Y
    draw_arrow(
        view,
        R_ARROWS_START + ARROW_OFFSET * Y,
        R_OUTER_ARROW_END - ARROW_OFFSET * Y,
        head_direction=-X,
        head='both',
        color=PURPLE,
    )
    draw_label(
        view,
        pos=(R_ARROWS_START + R_OUTER_ARROW_END) / 2 + DISTANCE_LABEL_OFFSET * X,
        text='R_outer',
    )

    draw_point(view, ORIGIN, "r0", n_offset=-Y)

    view.show()
    app.processEvents()
    img = view.grabFramebuffer()
    img.save('RoundCoil.png')


# draw_straightsegment()
# draw_racetrackcoil()
draw_roundcoil()

