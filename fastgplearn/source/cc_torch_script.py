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

    from mgetool.draft import TorchJit

    path = pathlib.Path(__file__)
    file = path.parent.parent / "source" / "torch_tool.cpp"
    file = str(file.absolute())
    bd = TorchJit(file, temps="torch_temp")
    bd.write(functions=["c_torch_score", "c_torch_cal"])
    c_torch_backend_tool = bd.quick_import(build=True, suffix=None)
