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

Usage
=====
1. Get Python 3.8+
2. Create virtual environment: :code:`python -m venv .venv`
3. Activate virtual environment: :code:`.\.venv\Scripts\activate`
4. Install dependencies: :code:`pip install jsonschema networkx numpy fuzzy-graph-coloring`

Use as CLI-Application: :code:`python .\shift_scheduling.py -v default_input.json`

Help:

.. code-block::

   usage: shift_scheduling.py [-h] [-o [OUTPUT_FILE]] [-s] [-v] [-p] [input_file]

   positional arguments:
     input_file            Shift scheduling input file. Defaults to "default_input.json"

   optional arguments:
     -h, --help            show this help message and exit
     -o [OUTPUT_FILE], --output-file [OUTPUT_FILE]
                           Shift scheduling output csv file. Defaults to "schedule.csv"
     -s, --show-graph      Whether the graph should be shown
     -v, --verbose         Prints additional graph and solution information
     -p, --print-color-assignment
                           Prints additional graph and solution information




Public functions
================
.. automodule:: shift_scheduling
   :members:

License
=======
This project is licensed under GNU General Public License v3.0 (GNU GPLv3). See :code:`LICENSE` in the code repository.
