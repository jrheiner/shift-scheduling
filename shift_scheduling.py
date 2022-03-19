import argparse
import csv
import datetime
# import functools
import json
import pathlib
import warnings
from collections import Counter
# import timeit
from typing import Tuple

import fuzzy_graph_coloring as fgc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from jsonschema import validate


def _parse_input(input_path: str) -> dict:
    """
    Parses and validates the input file
    :param input_path: Path to the input file
    :return: input data as dict
    """
    with open(input_path, "r") as input_file, open("schema/input_schema.json") as schema_file:
        schema = json.load(schema_file)
        input_data = json.load(input_file)
        validate(instance=input_data, schema=schema)
    assert input_data["shifts"] * input_data["staff_per_shift"] <= input_data["total_staff"], \
        f"Invalid input. With only {input_data['total_staff']} staff members a schedule with " \
        f"{input_data['shifts']} shifts and {input_data['staff_per_shift']} staff members per shift is not possible."

    input_data["start_date"] = datetime.datetime.strptime(input_data["start_date"], "%Y-%m-%d")
    input_data["end_date"] = datetime.datetime.strptime(input_data["end_date"], "%Y-%m-%d")
    input_data["period"] = (input_data["end_date"] - input_data["start_date"]).days + 1
    assert input_data["period"] > 0, \
        f"Invalid input. Start date ({input_data['start_date'].strftime('%Y-%m-%d')}) is after " \
        f"end date ({input_data['end_date'].strftime('%Y-%m-%d')})."
    return input_data


def _alpha_cut(graph: nx.Graph, alpha: float) -> nx.Graph:
    """
    Alpha-cut for a given NetworkX Graph. Needs attribute "weight" on edges and removes nodes that are not connected
    anymore.
    :param graph: NetworkX Graph which edges have an attribute "weight"
    :param alpha: Float number for alpha-cut
    :return: Alpha-cut graph
    """
    g = nx.Graph()
    for u, v, a in graph.edges(data=True):
        if a["weight"] >= alpha:
            g.add_edge(u, v, **a)
    return g


def _draw_weighted_graph(graph: nx.Graph, shifts_per_day, node_colors=None, draw_weights: bool = False):
    """
    Plots a given NetworkX graph. Optionally colors nodes and labels edges according to their assigned weight.
    Note: Colormap only supports max. 20 different colors.

    :param graph: NetworkX graph
    :param shifts_per_day: Number of shifts each day
    :param node_colors: Array of node colors (as integers), actual colors are given by the colormap
    :param draw_weights: Whether edges should be labeled with their assigned weight
    :return: None
    """
    color_map = plt.cm.tab10
    if node_colors and len(set(node_colors)) > 10:
        color_map = plt.cm.tab20
    if node_colors and len(set(node_colors)) > 20:
        warnings.warn(f"Colormap only supports 20 different colors. Coloring has {len(set(node_colors))} colors.")
    pos = {s: (shifts_per_day * int(s.split(".")[0]) * 1.3 + int(s.split(".")[1]) + 0.3 * (int(s.split(".")[2]) % 2),
               int(s.split(".")[2]))
           for s in graph.nodes()}
    with_labels = graph.number_of_nodes() <= 30
    nx.draw(graph, pos,
            node_size=1e3 if with_labels else 1e2,
            width=1 if with_labels else .5,
            node_color=node_colors,
            cmap=color_map,
            with_labels=with_labels)
    if draw_weights:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.show()
    # plt.savefig("graph.png", dpi=300)


def create_schedule(input_path: str, show_graph: bool = False, verbose: bool = False,
                    print_color_assignment: bool = False, output_file: str = "schedule.csv"):
    """
    Create a work schedule based on the supplied input file.
    Writes schedule in output file 'schedule.csv' to disk.


    :param input_path: Path to the input file
    :param show_graph: Flag whether the generated and colored graphs should be shown. Recommended only for small graphs.
    :param verbose: Flag whether number of nodes, edges, coloring scores, and final alpha should be printed
    :param print_color_assignment: Flag whether the complete color assignment dictionary should be printed
    :param output_file: Output file path
    :return:
    """
    graph, input_data = generate_graph(input_path)
    if verbose:
        print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    if show_graph:
        _draw_weighted_graph(graph, input_data["shifts"])

    if graph.number_of_nodes() < input_data["total_staff"]:
        raise Exception("There are more members in your team than available shifts")

    try:
        fuzzy_coloring, score, alpha = fuzzy_color(graph, input_data["total_staff"], verbose=verbose)
    except fgc.NoSolutionException:
        raise Exception("Even by considering only hard constraints, a schedule is not possible. "
                        f"(A {input_data['total_staff']}-coloring does not exist.)")
    if show_graph:
        _draw_weighted_graph(graph, input_data["shifts"], node_colors=[fuzzy_coloring.get(node) for node in graph])

    if print_color_assignment:
        print("Color assignment: ", fuzzy_coloring)

    if verbose:
        print(f"Alpha-cut using alpha={alpha}")
        print(f"Graph coloring has score of {score}")
        print(f"Solution has a fairness score of {_calculate_fairness(fuzzy_coloring, print_distribution=verbose)}")
    interpret_graph(graph, fuzzy_coloring, input_data, output_file)


def generate_graph(input_path: str) -> Tuple[nx.Graph, dict]:
    """
    Builds a graph represented the unassigned shift schedule.
    Graph nodes are named following the convention: [D].[S].[P] e.g., 0.0.1 => 1st day, 1st shift, 2nd pos

    :param input_path: Path to the input file
    :return: Tuple(Graph, input_data)
    """
    input_data = _parse_input(input_path)
    current_day_id = 0
    while current_day_id < input_data["period"]:
        today = input_data["start_date"] + datetime.timedelta(days=current_day_id)
        yesterday = input_data["start_date"] + datetime.timedelta(days=current_day_id - 1)
        if _get_weekday(today) not in input_data["days_of_week"]:
            current_day_id += 1
            continue
        nodes = [f"{current_day_id}.{s}.{p}"
                 for s in range(input_data["shifts"])
                 for p in range(input_data["staff_per_shift"])]
        # [D].[S].[P] # 0.0.1 = Monika: 1st day, 1st shift, 2nd pos
        if current_day_id == 0:
            graph = nx.complete_graph(nodes)
            nx.set_edge_attributes(graph, 1, "weight")
            # Schicht: max(s)
        elif _get_weekday(yesterday) in input_data["days_of_week"]:
            # Schicht: 0
            connect_nodes = [f"{current_day_id - 1}.{input_data['shifts'] - 1}.{p}" for p in
                             range(input_data["staff_per_shift"])]
            connect_nodes.extend([f"{current_day_id}.0.{p}" for p in range(input_data["staff_per_shift"])])
            connect_days = nx.complete_graph(connect_nodes)
            nx.set_edge_attributes(connect_days, 1, "weight")
            graph2 = nx.complete_graph(nodes)
            nx.set_edge_attributes(graph2, 1, "weight")
            graph = nx.compose_all([graph2, graph, connect_days])

        else:
            graph2 = nx.complete_graph(nodes)
            nx.set_edge_attributes(graph2, 1, "weight")
            graph = nx.compose_all([graph2, graph])

        # add soft constraint "balanced_weekends"
        try:
            balanced_weekends_constraint = input_data["soft_constraints"]["balanced_weekends"]
        except KeyError:
            balanced_weekends_constraint = False
        if balanced_weekends_constraint and _get_weekday(today) in ["Sa", "So"]:
            future_weekends_summands = [(7, 0.75), (14, 0.5), (21, 0.25)]
            if _get_weekday(today) == "Sa" and "Su" in input_data["days_of_week"]:
                future_weekends_summands.extend([(8, 0.75), (15, 0.5), (22, 0.25)])
            elif _get_weekday(today) == "Su" and "Sa" in input_data["days_of_week"]:
                future_weekends_summands.extend([(6, 0.75), (13, 0.5), (20, 0.25)])

            # add the according IDs to list if period is not exceeded
            future_weekends = [(current_day_id + days, weight) for (days, weight) in future_weekends_summands if
                               current_day_id + days < input_data["period"]]
            future_weekends, weights = zip(*future_weekends) if future_weekends else ([], [])
            connect_nodes = [f"{future_day}.{s}.{p}"
                             for s in range(input_data['shifts'])
                             for p in range(input_data["staff_per_shift"])
                             for future_day in future_weekends]  # List of all nodes associated with these weekends
            graph2 = nx.Graph()
            graph2.add_weighted_edges_from([(f"{current_day_id}.{s}.{p}", v, w)
                                            for s in range(input_data['shifts'])
                                            for p in range(input_data["staff_per_shift"])
                                            for v in connect_nodes
                                            for w in
                                            weights])  # Add an edge of every node with the respective weight of today
            # ...to the future weekend nodes
            graph = nx.compose_all([graph2, graph])
        current_day_id += 1
    return graph, input_data


def fuzzy_color(graph: nx.Graph, k: int, verbose: bool = False):
    """
    Calls the fuzzy graph k-coloring algorithm.
    Tries to assign colors equitably. If no "fair" solutions is found a fallback algorithm is used.

    :param graph: NetworkX fuzzy graph
    :param k: k for a k-coloring
    :param verbose: Print information on which coloring method is used
    :return: Tuple(coloring, score, Optional[alpha])
    """
    # t = timeit.Timer(functools.partial(fgc.alpha_fuzzy_color, graph, k, fair=True))
    # r = t.repeat(100, 1)
    # print(r)
    # print(np.mean(r), np.std(r))
    try:
        return fgc.alpha_fuzzy_color(graph, k, fair=True, return_alpha=True)
    except fgc.NoSolutionException:
        if verbose:
            print("Failed to use a fair order of colors, i.e., team members. Try to use best-fit.")
        return fgc.alpha_fuzzy_color(graph, k, return_alpha=True)


def interpret_graph(graph: nx.Graph, coloring, input_data, output_file: str):
    """
    Interprets a colored graph as staff-shift-assignment and writes the final schedule csv file.
    The CSV files has the rows: Day, Date, Shift, Position
    Example line: We, 2022-02-02,1,1,6

    :param graph: Input graph
    :param coloring: Graph color assignment
    :param input_data: Input data
    :param output_file: Output file path
    :return: Nothing
    """
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Day', 'Date', 'Shift', 'Position', 'Staff'])
        # Day, Shift, Position, Staff
        for node in sorted(graph, key=_sort_key_nodes_factory(input_data)):
            d, s, p = node.split(".")
            d = int(d)
            s = int(s) + 1
            p = int(p) + 1
            assigned_staff = coloring.get(node)
            date = input_data["start_date"] + datetime.timedelta(days=d)
            csv_writer.writerow([_get_weekday(date), date.strftime("%Y-%m-%d"), s, p, assigned_staff])
    print(f"Wrote staff timetable to '{output_file}'.")


def _sort_key_nodes_factory(input_data):
    """
    Factory to loosely couple the input_data dictionary to the function that sorts nodes by their names.
    :param input_data: the input data dictionary
    :return: Sort key function for sorting nodes by their names.
    """

    def _sort_key_nodes(name: str):
        """
        Sort key for nodes by names with {d}.{s}.{p}
        :param name: node name
        :return: scalar value representing order
        """
        d, s, p = name.split('.')
        d, s, p = int(d), int(s), int(p)
        return (input_data['shifts'] + input_data["staff_per_shift"] + 2) * d \
            + (input_data["staff_per_shift"] + 1) * s \
            + p

    return _sort_key_nodes


def _get_weekday(date: datetime):
    """
    Returns the week day string abbreviation for a given datetime object
    :param date: Datetime object
    :return: week day string abbreviation
    """
    weekday = {
        0: "Mo",
        1: "Tu",
        2: "We",
        3: "Th",
        4: "Fr",
        5: "Sa",
        6: "Su"
    }
    return weekday.get(date.weekday())


def _calculate_fairness(coloring: dict, print_distribution: bool = False):
    """
    Gives a score for the fairness of a coloring
    :param coloring: Color assignment
    :param print_distribution: Whether the shift distribution should be printed
    :return: Negative coefficient of variation (as percentage) of assigned shifts per staff member
    """
    shift_dist = list(Counter(coloring.values()).values())
    if print_distribution:
        print(f"Shift distribution: {shift_dist}")
    return np.std(shift_dist) / np.mean(shift_dist) * - 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), nargs='?',
                        help='Shift scheduling input file. Defaults to "default_input.json"',
                        default="default_input.json")
    parser.add_argument('-o', '--output-file', type=pathlib.Path, nargs='?',
                        help='Shift scheduling output csv file. Defaults to "schedule.csv"',
                        default="schedule.csv")
    parser.add_argument('-s', '--show-graph', action='store_true', help='Whether the graph should be shown')
    parser.add_argument('-v', '--verbose', action='store_true', help='Prints additional graph and solution information')
    parser.add_argument('-p', '--print-color-assignment', action='store_true',
                        help='Prints additional graph and solution information')
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = "schedule.csv"

    create_schedule(args.input_file.name, show_graph=args.show_graph, verbose=args.verbose,
                    print_color_assignment=args.print_color_assignment, output_file=args.output_file)
