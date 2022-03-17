import csv
import datetime
# import functools
import json
from collections import Counter
# import timeit
from typing import Tuple

import fuzzy_graph_coloring as fgc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from jsonschema import validate


def _parse_input(config_path: str) -> dict:
    """
    Parses and validates the configuration file
    :param config_path: Path to the input file
    :return: input data as dict
    """
    with open(config_path, "r") as input_file, open("schema/input_schema.json") as schema_file:
        schema = json.load(schema_file)
        input_data = json.load(input_file)
        validate(instance=input_data, schema=schema)
    assert input_data["shifts"] * input_data["staff_per_shift"] <= input_data["total_staff"], \
        f"Invalid configuration. With only {input_data['total_staff']} staff members a schedule with " \
        f"{input_data['shifts']} shifts and {input_data['staff_per_shift']} staff members per shift is not possible."

    input_data["start_date"] = datetime.datetime.strptime(input_data["start_date"], "%Y-%m-%d")
    input_data["end_date"] = datetime.datetime.strptime(input_data["end_date"], "%Y-%m-%d")
    input_data["period"] = (input_data["end_date"] - input_data["start_date"]).days + 1
    assert input_data["period"] > 0, \
        f"Invalid configuration. Start date ({input_data['start_date'].strftime('%Y-%m-%d')}) is after " \
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


def _draw_weighted_graph(graph: nx.Graph, shifts_per_day, cm=None):
    """
    Plots a given NetworkX graph and labels edges according to their assigned weight.
    Note: Colormap only supports 10 different colors.

    :param graph: NetworkX graph
    :return: None
    """
    pos = {s: (shifts_per_day * int(s.split(".")[0]) * 1.3 + int(s.split(".")[1]) + 0.3 * (int(s.split(".")[2]) % 2),
               int(s.split(".")[2]))
           for s in graph.nodes()}
    nx.draw(graph, pos,
            labels={node: node for node in graph.nodes()},
            node_size=1e3,
            node_color=cm,
            cmap=plt.cm.tab10)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.show()


def create_schedule(config_path: str, show_graph: bool = False, print_stats: bool = False):
    """
    Create a work schedule based on the supplied configuration file.
    Writes schedule in output file 'schedule.csv' to disk.

    :param config_path: Path to the configuration file
    :param show_graph: Flag whether the generated and colored graphs should be shown. Recommended only for small graphs.
    :param print_stats: Flag whether number of nodes, edges, and coloring scores should be printed to the console
    :return:
    """
    graph, input_data = generate_graph(config_path)
    if print_stats:
        print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    if show_graph:
        _draw_weighted_graph(graph, input_data["shifts"])

    if graph.number_of_nodes() < input_data["total_staff"]:
        raise Exception("There are more members in your team than available shifts")

    try:
        fuzzy_coloring, score = fuzzy_color(graph, input_data["total_staff"])
    except fgc.NoSolutionException:
        raise Exception("Even by considering only hard constraints, a schedule is not possible. "
                        f"(A {input_data['total_staff']}-coloring does not exist.)")
    if show_graph:
        _draw_weighted_graph(graph, input_data["shifts"], cm=[fuzzy_coloring.get(node) for node in graph])

    if print_stats:
        print(score, fuzzy_coloring)
        print(_calculate_fairness(fuzzy_coloring))
    interpret_graph(graph, fuzzy_coloring, input_data)


def generate_graph(config_path: str) -> Tuple[nx.Graph, dict]:
    """
    Builds a graph represented the unassigned shift schedule.
    Graph nodes are named following the convention: [D].[S].[P] e.g., 0.0.1 => 1st day, 1st shift, 2nd pos

    :param config_path: Path to the configuration file
    :return: Tuple(Graph, config_data)
    """
    input_data = _parse_input(config_path)
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

            # add soft constraint "balanced_weekends"
            try:
                balanced_weekends_constraint = input_data["soft_constraints"]["balanced_weekends"]
            except KeyError:
                balanced_weekends_constraint = False
            if balanced_weekends_constraint and _get_weekday(today) in ["Sa", "So"]:
                future_weekends_summands = [7, 8, 14, 15, 21, 22] if _get_weekday(today) == "Sa" \
                    else [6, 7, 13, 14, 20, 21]  # If somebody works on a Saturday or Sunday,
                # ... working on the coming weekends is discouraged
                future_weekends = [current_day_id + days for days in future_weekends_summands if
                                   current_day_id + days < input_data[
                                       "period"]]  # add the according IDs to list if period is not exceeded
                connect_nodes = [f"{future_day}.{s}.{p}"
                                 for s in range(input_data['shifts'])
                                 for p in range(input_data["staff_per_shift"])
                                 for future_day in future_weekends]  # List of all nodes associated with these weekends
                graph2 = nx.Graph()
                graph2.add_edges_from([(f"{current_day_id}.{s}.{p}", v)
                                       for s in range(input_data['shifts'])
                                       for p in range(input_data["staff_per_shift"])
                                       for v in connect_nodes], weight=0.5)  # Add an edge of every node of today to
                # ... the future weekend nodes
                graph = nx.compose_all([graph2, graph])
        else:
            graph2 = nx.complete_graph(nodes)
            nx.set_edge_attributes(graph2, 1, "weight")
            graph = nx.compose_all([graph2, graph])
        current_day_id += 1
    return graph, input_data


def fuzzy_color(graph: nx.Graph, k: int):
    """
    Calls the fuzzy graph k-coloring algorithm.
    Tries to assign colors equitably. If no "fair" solutions is found a fallback algorithm is used.

    :param graph: NetworkX fuzzy graph
    :param k: k for a k-coloring
    :return: Tuple(coloring, score, Optional[alpha])
    """
    # t = timeit.Timer(functools.partial(fgc.alpha_fuzzy_color, graph, k, fair=True))
    # r = t.repeat(100, 1)
    # print(r)
    # print(np.mean(r), np.std(r))
    try:
        return fgc.alpha_fuzzy_color(graph, k, fair=True)
    except fgc.NoSolutionException:
        print("Unfair")
        return fgc.alpha_fuzzy_color(graph, k)


def interpret_graph(graph: nx.graph, coloring, input_data):
    """
    Interprets a colored graph as staff-shift-assignment and writes the final schedule as 'schedule.csv' file.
    The CSV files has the rows: Day, Date, Shift, Position
    Example line: We, 2022-02-02,1,1,6

    :param graph: Input graph
    :param coloring: Graph color assignment
    :param input_data: Configuration data
    :return:
    """
    with open('schedule.csv', 'w', newline='') as csvfile:
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


def _sort_key_nodes_factory(input_data):
    """
    Factory to loosely couple the input_data dictionary to the function that sorts nodes by their names
    :param input_data:
    :return:
    """
    def _sort_key_nodes(name: str):
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


def _calculate_fairness(coloring: dict):
    """
    Gives a score for the fairness of a coloring
    :param coloring: Color assignment
    :return: Negative standard deviation of assigned shifts per staff member
    """
    print(list(Counter(coloring.values()).values()))
    return - np.std(list(Counter(coloring.values()).values()))


if __name__ == '__main__':
    create_schedule("test_input.json", show_graph=True, print_stats=True)
