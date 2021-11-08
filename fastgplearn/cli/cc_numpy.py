# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:29
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os
import pathlib

from mgetool.draft import DraftPyx
from mgetool.imports import BatchFile


class CLICommand:

    """
    Compile pyx or c++ code.

    Example:

        $ fastgplearn cc_numpy
    """

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args, parser):
        # args = args.parse_args()

        # first
        path = pathlib.Path(__file__)
        file = path.parent.parent / "source" / "np_tool.pyx"
        file = str(file.absolute())
        bd = DraftPyx(file, language="c++", temps="np_temps")
        bd.write()
        bd.quick_import(build=True, with_html=True)




