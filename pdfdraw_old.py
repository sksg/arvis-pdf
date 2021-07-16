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


def line(p0, p1):
    return pencil().moveto(p0).lineto(p1).path


def arrow(p0, p1, **arrow_opts):
    line = pencil().moveto(p0).lineto(p1).path
    pen = pencil().moveto(0., 1.)
    pen.lineto(-0.5, 0.).lineto(0., 0.1).lineto(0.5, 0.).lineto(0., 1.)
    line.add_path(pen.decorate_position(line, 1, **arrow_opts))
    return line


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


class pencil:
    class path(object):
        move_action = "m"
        line_action = "l"
        control_action = "c"
        curve_action = "C"
        relative_action = "++"
        relative_return_action = "+"

        relative_action_aliases = ["++", "+"]
        move_action_aliases = [r + a for a in ["m", " ", ""]
                               for r in ["++", "+"]]
        line_action_aliases = [r + a for a in ["l", "-", "--"]
                               for r in ["++", "+"]]
        control_action_aliases = [r + "c" for r in ["++", "+"]]
        curve_action_aliases = [r + "C" for r in ["++", "+"]]

        @staticmethod
        def is_relative(action, and_return=None):
            if action is None:
                return None
            if and_return is None:
                relative = pencil.path.relative_action_aliases
                return any([r in action for r in relative])
            elif not and_return:
                return pencil.path.relative_action in action
            else:
                not_rel_return = pencil.path.relative_action not in action
                return not_rel_return and pencil.path.relative_action in action

        @staticmethod
        def is_move(action):
            return pencil.path.move_action in action

        @staticmethod
        def is_line(action):
            return pencil.path.line_action in action

        @staticmethod
        def is_curve(action):
            return pencil.path.curve_action in action

        @staticmethod
        def is_control(action):
            return pencil.path.control_action in action

        def __init__(self):
            self.action_list = []  # list of strings
            self.coordinate_list = None
            self._interpolator_cache = None
            self._absolute_coordinates_cache = None

        def _next_previous(self, next_index, previous):
            next = self.coordinate_list[next_index]
            action = self.action_list[next_index]
            if self.is_relative(action, and_return=False):
                next = next + previous
                previous = np.array(next)
            elif self.is_relative(action, and_return=True):
                next = next + previous
            else:
                previous = np.array(next)
            return next, previous

        def absolute_coordinates(self, rewrite_cache=False):
            rewrite_cache |= self._absolute_coordinates_cache is None
            rewrite_cache &= self.coordinate_list is not None
            if rewrite_cache:
                coords = np.array(self.coordinate_list, dtype=np.float64)
                previous = coords[0] * 0.
                for i in range(len(coords)):
                    coords[i], previous = self._next_previous(i, previous)
                self._absolute_coordinates_cache = coords
            return self._absolute_coordinates_cache

        def interpolator(self, rewrite_cache=False, bezier_samples=None):
            rewrite_cache |= self._interpolator_cache is None
            rewrite_cache |= bezier_samples is not None
            if rewrite_cache:
                intrp = _approximate_linear_interpolator(self, bezier_samples)
                self._interpolator_cache = intrp
            return self._interpolator_cache

        def full_length(self, bezier_samples=None):
            return self.interpolator(bezier_samples=bezier_samples).full_length

        def rotate(self, *R):
            R = _rotation_matrix(R)
            path = pencil.path()
            path.action_list = self.action_list.copy()
            path.coordinate_list = R.dot(self.coordinate_list.T).T
            return path

        def translate(self, *T):
            T = np.array(T, dtype=np.float64)[np.newaxis, :]
            path = pencil.path()
            path.action_list = self.action_list.copy()
            path.coordinate_list = T + self.coordinate_list
            return path

        def transform(self, T):
            N = self.coordinate_list.shape[1]
            R, T = T[:N, :N], T[np.newaxis, :N, N]
            path = pencil.path()
            path.action_list = self.action_list.copy()
            path.coordinate_list = R.dot(self.coordinate_list.T).T + T
            return path

        def append_action(self, coordinates, action, relative=None):
            if self.is_relative(relative):
                action = relative + action
            elif relative is not None:
                msg = "Unknown relative option: {}".format(relative)
                raise RuntimeError(msg)
            self.action_list.append(action)
            coordinates = np.array(coordinates, dtype=np.float64)
            coordinates = coordinates[np.newaxis, :]
            if self.coordinate_list is not None:
                coordinates = np.vstack((self.coordinate_list, coordinates))
            self.coordinate_list = coordinates
            self._absolute_coordinates_cache = None
            self._interpolator_cache = None

        def append_path(self, path):
            self.action_list += path.action_list
            if self.coordinate_list is not None:
                self.coordinate_list = np.vstack((self.coordinate_list,
                                                  path.coordinate_list))
            else:
                self.coordinate_list = np.array(path.coordinate_list)
            self._absolute_coordinates_cache = None
            self._interpolator_cache = None

        def _positional_transform(self, position, **transform_options):
            sample_count = transform_options.pop("new_bezier_sample_count",
                                                 None)
            length = self.full_length(sample_count)
            inline_shift = transform_options.pop("inline_shift", 0)
            shift = transform_options.pop("shift", 0)
            position *= length
            position += inline_shift
            N = self.coordinate_list.shape[1]
            T = np.eye(N + 1, dtype=np.float64)
            intpr = self.interpolator()
            c, v = intpr.eval(position), intpr.eval_tangent(position)
            up = np.zeros((N,), dtype=np.float64)
            up[1] = 1.
            T[:N, N] = c
            T[:N, :N] = _rotation_matrix(_vector_vector_rotation(up, v)).T
            T[:N, N] += np.array(shift, dtype=np.float64)
            return T

    def __init__(self):
        self.path = pencil.path()

    def moveto(self, *coords, relative=None):
        self.path.append_action(coords, pencil.path.move_action, relative)
        return self

    def lineto(self, *coords, relative=None):
        if len(self.path.coordinate_list) == 0:
            msg = "You are trying to add a line to a pencil.path.\n\n"
            msg += "The pencil needs an origin!!---Use pencil.moveto() before "
            msg += "any pencil.lineto(), curveto(), or controlpoint()."
            raise RuntimeError(msg)
        self.path.append_action(coords, pencil.path.line_action, relative)
        return self

    def controlpoint(self, *coords, relative=None):
        if len(self.path.coordinate_list) == 0:
            msg = "You are trying to add a control point to a pencil.path.\n\n"
            msg += "The pencil needs an origin!!---Use pencil.moveto() before "
            msg += "any pencil.lineto(), curveto(), or controlpoint()."
            raise RuntimeError(msg)
        self.path.append_action(coords, pencil.path.control_action, relative)
        return self

    def curveto(self, *coords, relative=None):
        if len(self.path.coordinate_list) == 0:
            msg = "You are trying to add a control point to a pencil.path.\n\n"
            msg += "The pencil needs an origin!!---Use pencil.moveto() before "
            msg += "any pencil.lineto(), curveto(), or controlpoint()."
            raise RuntimeError(msg)
        self.path.append_action(coords, pencil.path.curve_action, relative)
        return self

    def add_path(self, *args):
        while len(args) > 0:
            action, args = args[0], args[1:]
            relative = "+" if "+" in action else None
            relative = "++" if "++" in action else None
            if action in pencil.path.move_action_aliases:
                coords, args = args[0], args[1:]
                self.moveto(*coords, relative)
            elif action in pencil.path.line_action_aliases:
                coords, args = args[0], args[1:]
                self.lineto(*coords, relative)
            elif action in pencil.path.control_action_aliases:
                coords, args = args[0], args[1:]
                self.controlpoint(coords, relative)
            elif action in pencil.path.curve_action_aliases:
                coords, args = args[0], args[1:]
                self.curveto(coords, relative)
            else:
                coords, args = args[0], args[1:]
                self.moveto(*coords)

    def decorate_position(self, path, position,
                          rotate=True, **transform_options):
        T = path._positional_transform(position, **transform_options)
        if T is None:
            return pencil.path()
        if rotate:
            path = self.path.transform(T)
        else:
            path = self.path.translate(T[-1, :-1])
        return path

    def decorate_equidistant(self, path, distance,
                             rotate=True, endpoints=False,
                             normalized_distance=False,
                             **transform_options):
        collective_path = pencil.path()  # will store all subpaths.
        length = path.full_length()
        if not normalized_distance:
            distance = distance / length
        if endpoints:
            start, end = 0, 1 + distance * 0.1
        else:
            start, end = distance, 1 - distance * 0.1
        positions = np.arange(start, end, step=distance, dtype=np.float64)

        for position in positions:
            T = path._positional_transform(position, **transform_options)
            if T is None:
                continue
            if rotate:
                subpath = self.path.transform(T)
            else:
                subpath = self.path.translate(T[-1, :-1])
            collective_path.append_path(subpath)

        return collective_path


class pdf_canvas:
    def __init__(self):
        self.content = bytearray()
        self.colors_rgb = {"black": (0, 0, 0),
                           "white": (1, 1, 1),
                           "red": (1, 0, 0),
                           "blue": (0, 1, 0),
                           "green": (0, 0, 1)}
        self.state = {"drawcolor": "black",
                      "fillcolor": "black"}
        self.saved_states = []

    def apply_path(self, path, **attributes):

        self.save_state()

        self.set(**attributes)

        coordinates = path.absolute_coordinates()
        for i, (action, coords) in enumerate(zip(path.action_list,
                                                 coordinates)):
            if pencil.path.move_action in action:
                content = (("{:.2f} " * len(coords)) + "{}\n")
                content = content.format(*coords, pencil.path.move_action)
            if pencil.path.line_action in action:
                content = (("{:.2f} " * len(coords)) + "{}\n")
                content = content.format(*coords, pencil.path.line_action)
            if pencil.path.control_action in action:
                content = (("{:.2f} " * len(coords)) + "{}\n")
                content = content.format(*coords, pencil.path.control_action)
            if pencil.path.curve_action in action:
                content = (("{:.2f} " * len(coords)) + "{}\n")
                content = content.format(*coords, pencil.path.curve_action)
            self.content += bytearray(content, "utf-8")

        self.content += bytearray("S\n", "utf-8")

        self.restore_state()

    def save_state(self):
        self.saved_states.append(dict(self.state))

    def restore_state(self):
        self.state = self.saved_states.pop()

    def set(self, **attributes):

        for key in list(attributes.keys()):
            if key == "color":
                color = attributes.pop(key)
                if color in self.colors_rgb:
                    color = self.colors_rgb[color]
                    attributes["drawcolor_rgb"] = color
                    attributes["fillcolor_rgb"] = color
                    self.state["drawcolor_rgb"] = color
                    self.state["fillcolor_rgb"] = color
            elif key == "drawcolor" or key == "fillcolor":
                color = attributes.pop(key)
                if color in self.colors_rgb:
                    color = self.colors_rgb[color]
                    attributes[key + "_rgb"] = color
                    self.state[key + "_rgb"] = color
            elif key == "color_rgb":
                color = attributes.pop(key)
                attributes["drawcolor_rgb"] = color
                attributes["fillcolor_rgb"] = color
                self.state["drawcolor_rgb"] = color
                self.state["fillcolor_rgb"] = color
            else:
                self.state[key] = attributes[key]

        if "linewidth" in attributes:
            linewidth = attributes.pop("linewidth")
            self.content += bytearray("{:.5f} w\n".format(linewidth), "utf-8")
        draw = "draw" in attributes and attributes["draw"]
        draw = draw or ("draw" in self.state and self.state["draw"])
        fill = "fill" in attributes and attributes["fill"]
        fill = fill or ("fill" in self.state and self.state["fill"])
        if "drawcolor_rgb" in attributes and draw:
            color = attributes.pop("drawcolor_rgb")
            content = "{:.5f} {:.5f} {:.5f} RG\n".format(*color)
            self.content += bytearray(content, "utf-8")
        if "fillcolor_rgb" in attributes and fill:
            color = attributes.pop("fillcolor_rgb")
            content = "{:.5f} {:.5f} {:.5f} rg\n".format(*color)
            self.content += bytearray(content, "utf-8")


class tikz_canvas:
    def __init__(self):
        self.content = bytearray()
        self.colors_rgb = {"black": (0, 0, 0),
                           "white": (1, 1, 1),
                           "red": (1, 0, 0),
                           "blue": (0, 1, 0),
                           "green": (0, 0, 1)}
        self.state = {"drawcolor": "black",
                      "fillcolor": "black"}
        self.saved_states = []

    def standalone_latex(self, filename=None):
        latex = "\\RequirePackage{{luatex85}}\n"
        latex += "\\documentclass[tikz]{{standalone}}\n"
        latex += "\\begin{{document}}\n"
        latex += "\\begin{{tikzpicture}}\n"
        latex += "{}\\end{{tikzpicture}}\n"
        latex += "\\end{{document}}\n"
        latex = latex.format(self.content.decode("utf-8"))

        if filename is not None:
            with open(filename, 'w') as file:
                file.write(latex)
        else:
            return latex

    def apply_path(self, path, **attributes):

        self.save_state()

        self.set(**attributes)

        content = "\\path[{}]".format(self.tikz_state)
        self.content += bytearray(content, "utf-8")
        in_curve = False
        prev = None
        for i, (action, coords) in enumerate(zip(path.action_list,
                                                 path.coordinate_list)):
            coords = np.array(coords)
            coords_content = ("(" + ("{:.5f}," * len(coords))[:-1] + ")")
            coords_content = coords_content.format(*coords)
            if pencil.path.move_action in action:
                content = " "
                if pencil.path.relative_action in action:
                    content += "++ "
                elif pencil.path.relative_return_action in action:
                    content += "+ "
                content += coords_content
                prev = coords
            if pencil.path.line_action in action:
                content = " -- "
                if pencil.path.relative_action in action:
                    content += "++ "
                elif pencil.path.relative_return_action in action:
                    content += "+ "
                content += coords_content
                prev = coords
            if pencil.path.control_action in action:
                if not in_curve:
                    content = " .. controls "
                    in_curve = True
                else:
                    content = " and "
                if pencil.path.relative_action in action:
                    coords += prev
                    prev = coords
                elif pencil.path.relative_return_action in action:
                    coords += prev
                coords_content = ("(" + ("{:.5f}," * len(coords))[:-1] + ")")
                coords_content = coords_content.format(*coords)
                content += coords_content
            if pencil.path.curve_action in action:
                content = " .. "
                if pencil.path.relative_action in action:
                    coords += prev
                    prev = coords
                elif pencil.path.relative_return_action in action:
                    coords += prev
                coords_content = ("(" + ("{:.5f}," * len(coords))[:-1] + ")")
                coords_content = coords_content.format(*coords)
                content += coords_content
                in_curve = False
            self.content += bytearray(content, "utf-8")

        self.content += bytearray(";\n", "utf-8")

        self.restore_state()

    def save_state(self):
        self.saved_states.append(dict(self.state))

    def restore_state(self):
        self.state = self.saved_states.pop()

    def set(self, **attributes):

        for key in list(attributes.keys()):
            if key == "color":
                color = attributes.pop(key)
                if color in self.colors_rgb:
                    color = self.colors_rgb[color]
                    attributes["drawcolor_rgb"] = color
                    attributes["fillcolor_rgb"] = color
                    self.state["drawcolor_rgb"] = color
                    self.state["fillcolor_rgb"] = color
            elif key == "drawcolor" or key == "fillcolor":
                color = attributes.pop(key)
                if color in self.colors_rgb:
                    color = self.colors_rgb[color]
                    attributes[key + "_rgb"] = color
                    self.state[key + "_rgb"] = color
            elif key == "color_rgb":
                color = attributes.pop(key)
                attributes["drawcolor_rgb"] = color
                attributes["fillcolor_rgb"] = color
                self.state["drawcolor_rgb"] = color
                self.state["fillcolor_rgb"] = color
            else:
                self.state[key] = attributes[key]

        self.tikz_state = ""

        if "linewidth" in attributes:
            linewidth = attributes.pop("linewidth")
            self.tikz_state += "{:.5f} w\n".format(linewidth)
        draw = "draw" in attributes and attributes["draw"]
        draw = draw or ("draw" in self.state and self.state["draw"])
        fill = "fill" in attributes and attributes["fill"]
        fill = fill or ("fill" in self.state and self.state["fill"])
        if "drawcolor_rgb" in attributes and draw:
            color = attributes.pop("drawcolor_rgb")
            content = "draw={{rgb,1:red,{:.5f}; green,{:.5f}; blue,{:.5f}}},"
            self.tikz_state += content.format(*color)
        if "fillcolor_rgb" in attributes and fill:
            color = attributes.pop("fillcolor_rgb")
            content = "fill={{rgb,1:red,{:.5f}; green,{:.5f}; blue,{:.5f}}},"
            self.tikz_state += content.format(*color)


def _rotation_matrix(R):
    if len(np.array(R).shape) == 3:  # 3D
        # Rodrigues' rotation formula
        angle = np.linalg.norm(R)
        R = R / angle
        K = np.array([[00000, -R[2], +R[1]],
                      [+R[2], 00000, -R[0]],
                      [-R[1], +R[0], 00000]], dtype=np.float64)
        return (np.sin(angle) + (1 - np.cos(angle)) * K).dot(K) + 1
    return np.array([[np.cos(R), -np.sin(R)],
                     [np.sin(R), np.cos(R)]], dtype=np.float64)


def _vector_vector_rotation(v0, v1):
    if len(v0) == 3:
        x = np.cross(v0, v1)
        X = np.linalg.norm(x)
        if X > 1e-12:
            x /= X
        angle = v0.dot(v1)
        angle /= np.linalg.norm(v0) * np.linalg.norm(v1)
        angle = np.arccos(angle)
        if abs(angle) < 1e-12:
            return np.zeros_like(v0)
        if abs(angle - np.pi) < 1e-12:
            if abs(v0[1]) > 1e-12 or abs(v0[2]) > 1e-12:
                x = np.cross(v0, (1, 0, 0))
            else:
                x = np.cross(v0, (0, 1, 0))
        return x * angle
    if len(v0) == 2:
        angle = v0.dot(v1)
        angle /= np.linalg.norm(v0) * np.linalg.norm(v1)
        angle = np.arccos(angle)
        return angle


class _approximate_linear_interpolator:
    def __init__(self, path, bezier_sample_count=None):
        self.segments = []
        self.lengths = []
        self.offsets = []
        if bezier_sample_count is None:
            bezier_sample_count = 100
        accumulated_length = 0
        absolute_coordinates = path.absolute_coordinates()
        p0, c0, c1 = None, None, None  # curve persistent coordinates
        previous = absolute_coordinates[0]
        for action, coordinates in zip(path.action_list[1:],
                                       absolute_coordinates[1:]):
            if pencil.path.line_action in action:
                length = np.linalg.norm(coordinates - previous)
                self.offsets.append(accumulated_length)
                self.lengths.append(length)
                self.segments.append((previous, coordinates))
                accumulated_length += length
            elif pencil.path.control_action in action:
                if c0 is None:
                    p0 = previous
                    c0 = coordinates
                elif c1 is None:
                    c1 = coordinates
                else:
                    raise RuntimeError("Too many control points to curve!")
            elif pencil.path.curve_action in action:
                if c0 is None or c1 is None:
                    raise RuntimeError("Too few control points to curve!")
                p1 = coordinates
                length = np.linalg.norm(p1 - p0) / 2
                length += np.linalg.norm(c0 - p0) / 6
                length += np.linalg.norm(c1 - p0) / 6
                length += np.linalg.norm(c0 - p1) / 6
                length += np.linalg.norm(c1 - p1) / 6
                length += np.linalg.norm(c1 - c0) / 6
                sample_count = bezier_sample_count * length
                bezier = _evaluate_bezier(p0, c0, c1, p1, sample_count)
                for p0, p1 in bezier:
                    length = np.linalg.norm(p1 - p0)
                    self.offsets.append(accumulated_length)
                    self.lengths.append(length)
                    self.segments.append((p0, p1))
                    accumulated_length += length
                c0, c1 = None, None
            previous = coordinates
        self.full_length = accumulated_length

    def eval(self, x):
        if x < 0:
            v = self.segments[0]
            return v[0] + x * (v[1] - v[0]) / self.lengths[0]
        elif x >= self.full_length:
            x = x - self.full_length
            v = self.segments[-1]
            return v[1] + x * (v[1] - v[0]) / self.lengths[0]
        for i, v in enumerate(self.segments):
            if x >= self.offsets[i] and x - self.offsets[i] < self.lengths[i]:
                x = x - self.offsets[i]
                return v[0] + x * (v[1] - v[0]) / self.lengths[0]
        raise RuntimeError("Should not reach this point!")

    def eval_tangent(self, x):
        prev = self.segments[0]
        prev = (prev[1] - prev[0]) / self.lengths[0]
        for i in range(len(self.segments)):
            if x <= self.offsets[i]:
                return prev
            prev = self.segments[i]
            prev = (prev[1] - prev[0]) / self.lengths[i]
        return prev


def _evaluate_bezier(p0, c0, c1, p1, bezier_sample_count=100):
    t = np.linspace(0, 1, bezier_sample_count)[:, np.newaxis]
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    coords = p0 * mt3 + 3 * c0 * mt2 * t + 3 * c1 * mt * t2 + p1 * t3
    return zip(coords[:-1], coords[1:])


if __name__ == '__main__':
    main()
