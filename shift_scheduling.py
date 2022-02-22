import csv
import datetime
import json
from typing import Tuple
from collections import Counter

import fuzzy_graph_coloring as fgc
import matplotlib.pyplot as plt
import networkx as nx
from jsonschema import validate, ValidationError
import numpy as np


def _parse_input(input_path: str) -> dict:
    """
    Parses and validates input file
    :param input_path: Path to the input file
    :return: input data as dict
    """
    with open(input_path, "r") as input_file, open("schema/input_schema.json") as schema_file:
        schema = json.load(schema_file)
        input_data = json.load(input_file)
        try:
            validate(instance=input_data, schema=schema)
        except ValidationError as ve:
            print(ve)
            raise ve
    input_data["start_day"] = datetime.datetime.strptime(input_data["start_day"], "%Y-%m-%d")
    input_data["start_end"] = datetime.datetime.strptime(input_data["start_end"], "%Y-%m-%d")
    input_data["period"] = (input_data["start_end"] - input_data["start_day"]).days + 1
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
            cmap=plt.cm.tab10)  # TODO Does just work until 10
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.show()


def create_schedule(input_path: str):
    graph, input_data = generate_graph(input_path)
    _draw_weighted_graph(graph, input_data["shifts"])
    print(max(nx.greedy_color(graph).values()))
    if graph.number_of_nodes() < input_data["total_staff"]:
        raise Exception("There are more members in your team than available shifts")

    try:
        fuzzy_coloring, score = fuzzy_color(graph, input_data["total_staff"])
    except fgc.NoSolutionException:
        raise Exception("Even by considering only hard constraints, a schedule is not possible. "
                        f"(A {input_data['total_staff']}-coloring does not exist.)")

    _draw_weighted_graph(graph, input_data["shifts"], cm=[fuzzy_coloring.get(node) for node in graph])
    print(score, fuzzy_coloring)
    print(_calculate_fairness(fuzzy_coloring))
    interpret_graph(graph, fuzzy_coloring, input_data)


def generate_graph(input_path: str) -> Tuple[nx.Graph, dict]:
    input_data = _parse_input(input_path)
    total_days = 0
    while total_days < input_data["period"]:
        if _get_weekday(input_data["start_day"] + datetime.timedelta(days=total_days)) \
                not in input_data["days_of_week"]:
            total_days += 1
            continue
        nodes = [f"{total_days}.{s}.{p}"
                 for s in range(input_data["shifts"])
                 for p in range(input_data["staff_per_shift"])]
        # [D].[S].[P] # 0.0.1 = Monika: 1st day, 1st shift, 2nd pos
        if total_days == 0:
            graph = nx.complete_graph(nodes)
            # Schicht: max(s)
        elif _get_weekday(input_data["start_day"] + datetime.timedelta(days=total_days - 1)) \
                in input_data["days_of_week"]:
            # Schicht: 0
            connect_nodes = [f"{total_days - 1}.{input_data['shifts'] - 1}.{p}" for p in
                             range(input_data["staff_per_shift"])]
            connect_nodes.extend([f"{total_days}.0.{p}" for p in range(input_data["staff_per_shift"])])
            connect_days = nx.complete_graph(connect_nodes)
            graph2 = nx.complete_graph(nodes)
            graph = nx.compose_all([graph2, graph, connect_days])
        else:
            graph2 = nx.complete_graph(nodes)
            graph = nx.compose_all([graph2, graph])
        total_days += 1
    nx.set_edge_attributes(graph, 1, "weight")

    # for d in range(input_data["period"]):
    #     nodes = [f"{d}.{s}.{p}"
    #              for s in range(input_data["shifts"])
    #              for p in range(input_data["staff_per_shift"])]
    #     # [D].[S].[P] # 0.0.1 = Monika: 1st day, 1st shift, 2nd pos
    #     if d == 0:
    #         graph = nx.complete_graph(nodes)
    #         # Schicht: max(s)
    #     else:
    #         # Schicht: 0
    #         connect_nodes = [f"{d - 1}.{input_data['shifts'] - 1}.{p}" for p in range(input_data["staff_per_shift"])]
    #         connect_nodes.extend([f"{d}.0.{p}" for p in range(input_data["staff_per_shift"])])
    #         connect_days = nx.complete_graph(connect_nodes)
    #         graph2 = nx.complete_graph(nodes)
    #         graph = nx.compose_all([graph2, graph, connect_days])
    # nx.set_edge_attributes(graph, 1, "weight")
    return graph, input_data


def fuzzy_color(graph: nx.Graph, k):
    try:
        return fgc.alpha_fuzzy_color(graph, k, fair=True)
    except fgc.NoSolutionException:
        return fgc.alpha_fuzzy_color(graph, k)


def interpret_graph(graph: nx.graph, coloring, input_data):
    with open('schedule.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Day', 'Date', 'Shift', 'Position', 'Staff'])
        # Day, Shift, Position, Staff
        for node in sorted(graph):
            d, s, p = node.split(".")
            d = int(d)
            s = int(s) + 1
            p = int(p) + 1
            assigned_staff = coloring.get(node)
            date = input_data["start_day"] + datetime.timedelta(days=d)
            csv_writer.writerow([_get_weekday(date), date.strftime("%Y-%m-%d"), s, p, assigned_staff])


def _get_weekday(date: datetime):
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
    :param coloring:
    :return: Negative standard deviation of assigned shifts per staff member
    """
    print(list(Counter(coloring.values()).values()))
    return - np.std(list(Counter(coloring.values()).values()))


if __name__ == '__main__':
    create_schedule("test_input.json")
