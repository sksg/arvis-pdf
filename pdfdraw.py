import pencil
import tikz_painter


def main():
    canvas = tikz_painter.canvas()

    line = pencil.bare_path()
    line.moveto(0., 0.)
    line.controlpoint(10., 10., relative="++")
    line.controlpoint(10., -10., relative="++")
    line.curveto(10., 10., relative="++")

    canvas.apply_path(line, draw=True, color="red")

    canvas.standalone_latex("test.tex")


if __name__ == '__main__':
    main()
