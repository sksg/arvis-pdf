import numpy as np


class bezier:
    @staticmethod
    def linearize(p0, c0, c1, p1, bezier_sample_count=100):
        t = np.linspace(0, 1, bezier_sample_count)[:, np.newaxis]
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        coords = p0 * mt3 + 3 * c0 * mt2 * t + 3 * c1 * mt * t2 + p1 * t3
        return zip(coords[:-1], coords[1:])

    @staticmethod
    def rough_length(p0, c0, c1, p1):
        length = np.linalg.norm(p1 - p0) / 2
        length += np.linalg.norm(c0 - p0) / 6
        length += np.linalg.norm(c1 - p0) / 6
        length += np.linalg.norm(c0 - p1) / 6
        length += np.linalg.norm(c1 - p1) / 6
        length += np.linalg.norm(c1 - c0) / 6
        return length


class path_action:
    move = " "
    line = "-"
    control = "c"
    curve = "C"
    relative = "++"
    relative_and_back = "+<"

    @staticmethod
    def is_move(action):
        return path_action.move in action

    @staticmethod
    def is_line(action):
        return path_action.line in action

    @staticmethod
    def is_control(action):
        return path_action.control in action

    @staticmethod
    def is_curve(action):
        return path_action.curve in action

    @staticmethod
    def is_relative(action, any_relative=False):
        if not any_relative:
            return path_action.relative in action
        elif any_relative:
            relative = path_action.relative in action
            relative |= path_action.relative_and_back in action
            return relative

    @staticmethod
    def is_relative_and_back(action):
        return path_action.relative_and_back in action


class bare_path:
    """A bare path is an *undecorated* path of actions between coordinates."""

    def __init__(self, actions=[], coordinates=None, bezier_sample_count=100):
        self.actions = actions  # list of path_actions
        self.coordinates = coordinates  # array of (relative) coordinates
        self._abs_coordinates = None  # absolute--only computed if needed!
        self._approx_segments = None  # linearized--only computed if needed!
        self._approx_length = None  # linearized--only computed if needed!
        self._approx_lengths = None  # linearized--only computed if needed!
        self.bezier_sample_count = bezier_sample_count

    def _assert_origin(self):
        if len(self.coordinates) == 0:
            msg = "You are trying to add an action to a path.\n\n"
            msg += "This action needs an origin!!---Use moveto() before "
            msg += "any lineto(), curveto(), or controlpoint()."
            raise RuntimeError(msg)

    def append_action(self, point, action):
        if not path_action.is_move(action):
            self._assert_origin()
        self.actions.append(action)
        point = np.array(point, dtype=np.float64)[np.newaxis, :]
        point = point
        if self.coordinates is None:
            self.coordinates = point
        else:
            self.coordinates = np.append(self.coordinates, point, axis=0)
        return self

    def moveto(self, *coords, relative=""):
        return self.append_action(coords, relative + path_action.move)

    def lineto(self, *coords, relative=""):
        return self.append_action(coords, relative + path_action.line)

    def controlpoint(self, *coords, relative=""):
        return self.append_action(coords, relative + path_action.control)

    def curveto(self, *coords, relative=""):
        return self.append_action(coords, relative + path_action.curve)

    def rotate(self, R):
        if len(R.shape) == 1:
            if len(np.array(R).shape) == 3:  # 3D
                # Rodrigues' rotation formula
                angle = np.linalg.norm(R)
                K = np.array([[00000, -R[2], +R[1]],
                              [+R[2], 00000, -R[0]],
                              [-R[1], +R[0], 00000]], dtype=np.float64) / angle
                R = (np.sin(angle) + (1 - np.cos(angle)) * K).dot(K) + 1
            else:
                R = np.array([[np.cos(R), -np.sin(R)],
                              [np.sin(R), np.cos(R)]], dtype=np.float64)
        return bare_path(self.action_list.copy(),
                         R.dot(self.coordinate_list.T).T,
                         bezier_sample_count=self.bezier_sample_count)

    def translate(self, T):
        T = np.array(T, dtype=np.float64)[np.newaxis, :]
        return bare_path(self.action_list.copy(), T + self.coordinate_list,
                         bezier_sample_count=self.bezier_sample_count)

    def transform(self, T):
        R, T = T[:-1, :-1], T[np.newaxis, :-1, -1]
        return bare_path(self.action_list.copy(),
                         R.dot(self.coordinate_list.T).T + T,
                         bezier_sample_count=self.bezier_sample_count)

    def absolute_coordinates(self, recompute=False):
        recompute |= self._abs_coordinates is None
        if recompute and self.coordinates is not None:
            self._abs_coordinates = np.array(self.coordinates)
            previous = np.zeros_like(self.coordinates[0])
            for i, (a, c) in enumerate(zip(self.actions, self.coordinates)):
                if path_action.is_relative(a, any_relative=True):
                    self._abs_coordinates[i] += previous
                if not path_action.is_relative_and_back(a):
                    previous = self._abs_coordinates[i]
        return self._abs_coordinates

    def approximate_linear_coordinates(self, recompute=False):
        recompute |= self._approx_segments is None
        if recompute and self.coordinates is not None:
            coordinates = self.absolute_coordinates()
            approx_segments = coordinates[0, np.newaxis, :]
            previous = np.zeros_like(coordinates[0])
            self._approx_length = 0
            self._approx_lengths = []
            for i, (a, c) in enumerate(zip(self.actions, coordinates)):
                if path_action.is_move(a) or path_action.is_control(a):
                    continue
                if path_action.is_line(a):
                    segments = [(previous, c)]
                if path_action.is_curve(a):
                    valid = i >= 2
                    if valid:
                        c0, c1 = self.actions[i - 1], self.actions[i - 2]
                        valid &= path_action.is_control(c0)
                        valid &= path_action.is_control(c1)
                    if not valid:
                        raise RuntimeError("Missing control points for curve!")
                    c0, c1 = coordinates[i - 1], coordinates[i - 2]
                    bc = bezier.rough_length(previous, c0, c1, c)
                    bc *= self.bezier_sample_count
                    segments = bezier.linearize(previous, c0, c1, c, bc)
                for l0, l1 in segments:
                    self._approx_lengths.append(np.linalg.norm(l1 - l0))
                    self._approx_length += self._approx_lengths[-1]
                approx_segments = np.append(approx_segments, list(segments))
            self._approx_segments = np.array(approx_segments)
        return self._approx_segments

    def length(self, at=None, recompute=False):
        recompute |= self._approx_length is None
        if recompute and self.coordinates is not None:
            self.approximate_linear_coordinates()
        return self._approx_length

    def position(self, position, indices=slice(None)):
        if position == "begin" and self.coordinates is not None:
            return self.coordinates[0]
        elif position == "end" and self.coordinates is not None:
            return self.absolute_coordinates()[-1]
        elif position == "nodes" and self.coordinates is not None:
            return self.absolute_coordinates()[indices]

    def interpolate_equidistant(self, delta, absolute_length=False):
        if absolute_length:
            delta /= self.length()
        positions = np.arange(0, 1 + 1e-12, delta, dtype=np.float64)
        return self.interpolate(positions)

    def interpolate(self, position, absolute_length=False):
        if absolute_length:
            position /= self.length()
        position = np.array(position, dtype=np.float)[:, np.newaxis]
        offsets = self.offsets[np.newaxis, :]
        indices = (position >= offsets).sum(axis=1) - 1
        indices[indices < 0] = 0
        segments = self.segments[indices]
        delta = (position - offsets[indices]) / self._approx_lengths[indices]
        position = segments[:, 0]
        position += delta * (segments[:, 0])
        return segments[:, 0] + delta * (segments[:, 1] - segments[:, 0])


class path(bare_path):
    """A decorated bare_path, including styles and sub paths."""

    def __init__(self, actions=[], coordinates=None, subpaths=[],
                 style={}, bezier_sample_count=100):
        super(path, self).__init__(actions, coordinates, bezier_sample_count)
        self.subpaths = subpaths
        self.style = style

    def decorate_at_position(path, at=):
        pass
