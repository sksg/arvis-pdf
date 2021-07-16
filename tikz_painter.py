import numpy as np
import pencil


class canvas:
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

    @staticmethod
    def tikz_coordinates(coordinates, action=""):
        fmt = (action + "(" + ("{:.5f}," * len(coordinates))[:-1] + ")")
        return fmt.format(*coordinates)

    def apply_path(self, path):

        self.save_state()
        self.apply_bare_path(path, **path.style)

        for subpath in path.subpaths:
            self.apply_path(subpath)

        self.restore_state()

    def apply_bare_path(self, path, **attributes):

        content = "\\path"

        self.save_state()
        self.set(**attributes)

        content += "[{}]".format(self.tikz_state)

        action_coordinates = zip(path.actions, path.coordinates)
        for i, (a, c) in enumerate(action_coordinates):
            action = ""
            if pencil.path_action.is_move(a):
                action += " "
            if pencil.path_action.is_line(a):
                action += " -- "
            if pencil.path_action.is_control(a):
                continue
            is_curve = pencil.path_action.is_curve(a)
            if is_curve:
                valid = i >= 2
                if valid:
                    c0, c1 = path.actions[i - 2], path.actions[i - 1]
                    valid &= pencil.path_action.is_control(c0)
                    valid &= pencil.path_action.is_control(c1)
                if not valid:
                    raise RuntimeError("Missing control points for curve!")
                abs_coordinates = path.absolute_coordinates()
                c0, c1 = abs_coordinates[i - 2], abs_coordinates[i - 1]
                c0 = canvas.tikz_coordinates(c0)
                c1 = canvas.tikz_coordinates(c1)
                action += " .. controls {} and {} .. ".format(c0, c1)
                c = abs_coordinates[i]
            if pencil.path_action.is_relative(a) and not is_curve:
                action += "++ "
            elif pencil.path_action.is_relative_and_back(a) and not is_curve:
                action += "+ "

            content += action + canvas.tikz_coordinates(c)
        self.content += bytearray(content + ";\n", "utf-8")

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
