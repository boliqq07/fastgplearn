# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:29
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os
import pathlib

from mgetool.draft import DraftPyx, TorchJit
from mgetool.imports import BatchFile


class CLICommand:

    """
    Compile pyx or c++ code.

    Example:

        $ mgetool cc_torch
    """

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args, parser):
        # args = args.parse_args()

        # first
        path = pathlib.Path(__file__)
        file = path.parent.parent / "source" / "torch_tool.cpp"
        file = str(file.absolute())
        bd = TorchJit(file, temps="torch_temp")
        bd.write(functions=["c_torch_score", "c_torch_cal"])
        c_torch_backend_tool = bd.quick_import(build=True, suffix=None)




