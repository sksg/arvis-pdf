import numpy as np


def main():
    canvas = tikz_canvas()

    line_pen = pencil()
    line_pen.moveto(0., 0.)
    line_pen.controlpoint(10., 10., relative="++")
    line_pen.controlpoint(10., -10., relative="++")
    line_pen.curveto(10., 10., relative="++")

    canvas.apply_path(line_pen.path, draw=True, color="red")

    arrowhead_pen = pencil()
    arrowhead_pen.moveto(0., 1.)
    arrowhead_pen.lineto(-0.5, 0.)
    arrowhead_pen.lineto(0., 0.1)
    arrowhead_pen.lineto(0.5, 0.)
    arrowhead_pen.lineto(0., 1.)

    arrow_opts = {"inline shift": 0}
    arrowhead_path = arrowhead_pen.decorate_position(line_pen.path, 1,
                                                     **arrow_opts)
    canvas.apply_path(arrowhead_path, fill=True, color="red")

    barb_pen = pencil()
    barb_pen.moveto(-0.5, -0.2)
    barb_pen.lineto(+0.5, +0.2)
    barb_pen.moveto(-0.5, +0.2)
    barb_pen.lineto(+0.5, -0.2)

    barb_path = barb_pen.decorate_equidistant(line_pen.path, 2)
    canvas.apply_path(barb_path, draw=True, color="red")

    canvas.standalone_latex("test.tex")


class axis:
    def __init__(self, mapper=None, painter=None, locator=None):
        self.data, self.min, self.max, self.ptp = None, None, None, None

    def setdata(self, data):
        self.data = data
        self.min = self.data.min()
        self.max = self.data.max()
        self.ptp = self.data.ptp()

    def adddata(self, data):
        self.data = data if self.data is None else np.append(self.data, data)
        self.min = self.data.min()
        self.max = self.data.max()
        self.ptp = self.data.ptp()


class cartesian_axis(axis):
    def __init__(self, unit_vector, mapper=None, painter=None):
        self.unit_vector = unit_vector

        def locator(axis, position, normalized=False):
            if position == "min":
                return self.min * self.unit_vector
            elif position == "max":
                return self.min * self.unit_vector
            elif normalized:
                position = self.min + position * self.ptp
            return position * self.unit_vector

        super(cartesian_axis, self).__init__(mapper, painter, locator)


class schoolbook_cartesian_axis_painter:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.arrowhead_pen = pencil()
        self.arrowhead_pen.moveto(0., 1.)
        self.arrowhead_pen.lineto(-0.5, 0.)
        self.arrowhead_pen.lineto(0., 0.1)
        self.arrowhead_pen.lineto(0.5, 0.)
        self.arrowhead_pen.lineto(0., 1.)

    def __call__(self, axis, canvas):
        min, max = axis.locate("min"), axis.locate("max")
        if "min_offset" in self.options:
            min += axis.locate(self.options["min_offset"])
        if "min_offset_relative" in self.options:
            min += axis.locate(self.options["min_offset_relative"],
                               normalized=True)
        if "max_offset" in self.options:
            max += axis.locate(self.options["max_offset"])
        if "max_offset_relative" in self.options:
            max += axis.locate(self.options["max_offset_relative"],
                               normalized=True)
        pen = pencil()
        pen.moveto(*min)
        pen.lineto(*max)

        axis_path_opts = {"color": "black", "linewidth": 0.1}
        if "axis_color" in self.options:
            axis_path_opts["color"] = self.options["axis_color"]
        if "axis_linewidth" in self.options:
            axis_path_opts["linewidth"] = self.options["axis_linewidth"]

        canvas.apply_path(pen.path, draw=True, **axis_path_opts)
        # canvas.apply_path(arrowhead, fill=True, **axis_path_opts)


class schoolbook_cartesian_axis(cartesian_axis):
    def __init__(self, unit_vector, mapper=None, **kwargs):
        painter = schoolbook_cartesian_axis_painter(**kwargs)
        super(schoolbook_cartesian_axis, self).__init__(unit_vector, mapper,
                                                        painter)


def schoolbook_2D_axes(unit_x=np.array((1, 0), dtype=np.float64),
                       unit_y=np.array((0, 1), dtype=np.float64),
                       **kwargs):
    def get_x(labels, data):
        return data[:, 0]

    def get_y(labels, data):
        return data[:, 1]
    return [schoolbook_cartesian_axis(u, m) for u, m in ((unit_x, get_x),
                                                         (unit_y, get_y))]


def schoolbook_3D_axes(unit_x=np.array((1, 0), dtype=np.float64),
                       unit_y=np.array((0, 1), dtype=np.float64),
                       unit_z=np.array((0.5, 0.5), dtype=np.float64),
                       **kwargs):
    def get_x(labels, data):
        return data[:, 0]

    def get_y(labels, data):
        return data[:, 1]

    def get_z(labels, data):
        return data[:, 2]
    return [schoolbook_cartesian_axis(u, m) for u, m in ((unit_x, get_x),
                                                         (unit_y, get_y),
                                                         (unit_z, get_z))]


class plot:
    def __init__(self, data=None, labels=None, axes=None,
                 painter=None, locator=None):
        self.data = data
        self.attributes = None
        self.labels = labels
        self.axes = axes
        self.painter = painter
        self.locator = locator

    def map(self):
        if self.axes is None:
            raise RuntimeError("No axes supplied!")
        self.attributes = np.empty((self.data.shape[1], len(self.axes)),
                                   dtype=np.float)
        for i, ax in enumerate(self.axes):
            self.attributes[:, i] = ax.map(self.labels, self.data)

    def paint(self, canvas):
        if self.painter is None:
            raise RuntimeError("No way to paint this axis is supplied!")
        if self.attributes is None:
            self.map()
        return self.painter(self)

    def locate(self, position, normalized=False):
        if self.locator is None:
            raise RuntimeError("No way to locate this axis is supplied!")
        if self.attributes is None:
            self.map()
        return self.locator(self, position, normalized)

if __name__ == '__main__':
    main()
