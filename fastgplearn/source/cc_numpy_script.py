# -*- coding: utf-8 -*-

# @Time     : 2021/11/5 15:18
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

"""
This is script. For install version, don't include it.
"""

if __name__ == "__main__":
    import pathlib
    from mgetool.draft import DraftPyx

    # first
    path = pathlib.Path(__file__)
    file = path.parent.parent / "source" / "np_tool.pyx"
    file = str(file.absolute())
    bd = DraftPyx(file, language="c++", temps="np_temps")
    bd.write()
    c_numpy_backend_tool = bd.quick_import(build=True, with_html=True)
