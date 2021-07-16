import tempfile
import os.path
import subprocess
import shutil


def path(*args, **options):
    contents = "\path[{}]".format(",".join(fmt_options(**options)))

    curve_step = 0
    for arg in args:
        if arg[0] == "-" or (arg[0] == "~" and curve_step == 0):
            contents += "--"
        elif arg[0] == "~":
            contents += ".."
            curve_step = 0
        elif arg[0] == "." and curve_step == 0:
            contents += "..controls"
            curve_step += 1
        elif arg[0] == "." and curve_step == 1:
            contents += "and"
            curve_step += 1
        elif arg[0] == "." and curve_step > 1:
            raise RuntimeError("More than two control points not supported!")
        elif len(arg) == 2:
            contents += "({:f}, {:f})".format(*arg)
            continue
        else:
            raise RuntimeError("Unrecognized arg '{}'.".format(arg))
        if arg[1:] == "+":
            contents += "+"
        elif arg[1:] == "++":
            contents += "++"

    return contents + ";\n"


def fmt_options(**options):
    opt_list = ["draw=" + options.pop("draw", "black")]
    opt_list.append("fill=" + options.pop("fill", "none"))
    return opt_list


def compile_tikz(filename, tikz_contents=None, latex_contents=None,
                 working_directory=None, engine="LuaLatex", ignore_stdout=True,
                 keep_log="auto"):
    _working_directory = None
    if working_directory is None:
        _working_directory = tempfile.TemporaryDirectory()
        working_directory = _working_directory.name
    latex_filename = os.path.join(working_directory, "tikz")

    filename, ext = os.path.splitext(filename)
    if ext != "" and ext.lower() != ".pdf":
        raise RuntimeError("Currently only pdf output is supported!")

    args = None
    lua = False
    if engine.lower() == "lualatex":
        lua = True
        args = ["lualatex",
                "-interaction=nonstopmode",
                latex_filename + ".tex"]
    if engine.lower() == "pdflatex":
        args = ["pdflatex",
                "-interaction=nonstopmode",
                latex_filename + ".tex"]

    with open(latex_filename + ".tex", 'w') as latex_file:
        if latex_contents is None and tikz_contents is not None:
            latex_contents = ""
            if lua:
                latex_contents += "\\RequirePackage{{luatex85}}\n"
            latex_contents += "\\documentclass[tikz]{{standalone}}\n"
            latex_contents += "\\begin{{document}}\n"
            latex_contents += "\\begin{{tikzpicture}}\n"
            latex_contents += "{}\\end{{tikzpicture}}\n"
            latex_contents += "\\end{{document}}\n"
            latex_contents = latex_contents.format(tikz_contents)
        latex_file.write(latex_contents)

    stdout = None
    if ignore_stdout:
        stdout = subprocess.DEVNULL
        pass
    try:
        subprocess.run(args, cwd=working_directory,
                       stdout=stdout, check=True)
    except Exception as e:
        print("Error in latex/tikz compilation!")
        if keep_log is not False:
            shutil.copyfile(latex_filename + ".log", filename + ".log")
            shutil.copyfile(latex_filename + ".tex", filename + ".tex")
            print("See .tex and .log file!")
    shutil.copyfile(latex_filename + ".pdf", filename + ".pdf")
    if keep_log is True:
        shutil.copyfile(latex_filename + ".log", filename + ".log")
    if _working_directory is not None:
        _working_directory.cleanup()


if __name__ == '__main__':
    test_tikz = path((0, 0), "-", (1, 1), (1, 0.5), "~", (1, 0),
                     ".", (1, 1), "~", (0, 0))
    compile_tikz("test.pdf", test_tikz)
