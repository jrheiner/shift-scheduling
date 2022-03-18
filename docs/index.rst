.. shift-scheduling documentation master file, created by
   sphinx-quickstart on Fri Feb 11 14:19:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to shift-scheduling's documentation!
************************************************

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

shift-scheduling is a Python package for creating shift schedules using graph coloring.


See repository https://github.com/jrheiner/shift-scheduling


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick-Start
===========
Install package: `pip install fuzzy-graph-coloring`

Try simple code:

.. code-block::

   python shift_scheduling.py test_input.json

Result: `({1: 2, 2: 3, 3: 3, 4: 1}, 0.918918918918919)`

Public functions
================
.. automodule:: shift_scheduling
   :members:

Bibliography
============
The project uses a lot of the by Keshavarz created basics:
E. Keshavarz, “Vertex-coloring of fuzzy graphs: A new approach,” Journal of Intelligent & Fuzzy Systems, vol. 30, pp. 883–893, 2016, issn: 1875-8967. https://doi.org/10.3233/IFS-151810

License
=======
This project is licensed under GNU General Public License v3.0 (GNU GPLv3). See `LICENSE` in the code repository.