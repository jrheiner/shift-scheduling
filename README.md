# shift-scheduling

shift-scheduling is a Python package for creating shift schedules using graph coloring.

# Usage
1. Get Python 3.8+
2. Create virtual environment: `python -m venv .venv`
3. Activate virtual environment: `.\.venv\Scripts\activate`
4. Install dependencies: `pip install jsonschema networkx numpy fuzzy-graph-coloring`

Use as CLI-Application: `python .\shift_scheduling.py -v default_input.json`

Output:
```
Graph has 312 nodes and 2319 edges.
Alpha-cut using alpha=0.75
Graph coloring has score of 0.9881404174573055
Shift distribution: [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20]
Solution has a fairness score of -1.9230769230769231
Wrote staff timetable to 'schedule.csv'.
```
`schedule.csv`:
```
Day,Date,Shift,Position,Staff
We,2022-02-02,1,1,9
We,2022-02-02,1,2,10
We,2022-02-02,1,3,11
We,2022-02-02,2,1,3
We,2022-02-02,2,2,4
We,2022-02-02,2,3,5
Th,2022-02-03,1,1,12
Th,2022-02-03,1,2,13
Th,2022-02-03,1,3,14
Th,2022-02-03,2,1,0
Th,2022-02-03,2,2,1
Th,2022-02-03,2,3,2
Fr,2022-02-04,1,1,6
...
```


Help and parameter information:

```
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
```

# License
This project is licensed under GNU General Public License v3.0 (GNU GPLv3). See `LICENSE` in the code repository.


# Setup development environment
1. Get poetry https://python-poetry.org/docs/
2. Make sure, Python 3.8 is being used
3. `poetry install` in your system shell
4. `poetry run pre-commit install`

## Run pre-commit
`poetry run pre-commit run --all-files`

## Create documentation
`.\docs\make html`
