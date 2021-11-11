Install
==================
To suit the needs of different people, the installation is divided into 3 optional parts.


**1. Install with pip:** ::

    pip install fastgplearn

.. note::
    The package has been installed, but advise to use c++ compiler and torch for more power speed up, as 2 and 3. See more speed test: :doc:`../Guide/benchmark` .

**2. optional:** ::

    fastgplearn cc_numpy

.. note::
    For windows platform, C++14 or more needed (`Note help <https://wiki.python.org/moin/WindowsCompilers>`_,
    `VS Buildtools <https://visualstudio.microsoft.com/>`_, Proposed English version.)

**3. optional:** ::

    fastgplearn cc_torch

.. note::
    Torch is needed (`pytorch.org <https://pytorch.org/>`_),
    For linux, windows platform, C++14 or more needed (`Note help <https://wiki.python.org/moin/WindowsCompilers>`_,
    `VS Buildtools <https://visualstudio.microsoft.com/>`_, Proposed English version.)


Requirements
::::::::::::

Packages:

============= =================  ============
 Dependence   Name               Version
------------- -----------------  ------------
 necessary    python             >=3.6
 necessary    numpy              >=1.17
 recommend    torch              >=1.7
============= =================  ============