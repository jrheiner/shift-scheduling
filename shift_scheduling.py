from typing import Tuple, Any

from jsonschema import validate, ValidationError
import networkx as nx
import fuzzy_graph_coloring as fgc
import json
import matplotlib.pyplot as plt


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
    period = input_data["period"]
    if period[-1:] == "d":
        input_data["period"] = int(period[:-1])
    elif period[-1:] == "w":
        input_data["period"] = int(period[:-1]) * 7
    else:
        raise Exception("todo json schema validation")
    return input_data


def _check_hard_constraints() -> bool:
    return False


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
    if graph.number_of_nodes() < input_data["total_staff"]:
        raise Exception("There are more members in your team than available shifts")

    try:
        fuzzy_coloring, score = fuzzy_color(graph, input_data["total_staff"])
    except fgc.NoSolutionException:
        raise Exception("Even by considering only hard constraints, a schedule is not possible. "
                        f"(A {input_data['total_staff']}-coloring does not exist.)")

    _draw_weighted_graph(graph, input_data["shifts"], cm=[fuzzy_coloring.get(node) for node in graph])
    print(score, fuzzy_coloring)
    # interpret_graph()
    pass


def generate_graph(input_path: str) -> Tuple[nx.Graph, dict]:
    input_data = _parse_input(input_path)
    for d in range(input_data["period"]):
        nodes = [f"{d}.{s}.{p}"
                 for s in range(input_data["shifts"])
                 for p in range(input_data["staff_per_shift"])]
        # [D].[S].[P] # 0.0.1 = Monika: 1st day, 1st shift, 2nd pos
        if d == 0:
            graph = nx.complete_graph(nodes)
            # Schicht: max(s)
        else:
            # Schicht: 0
            connect_nodes = [f"{d - 1}.{input_data['shifts'] - 1}.{p}" for p in range(input_data["staff_per_shift"])]
            connect_nodes.extend([f"{d}.0.{p}" for p in range(input_data["staff_per_shift"])])
            connect_days = nx.complete_graph(connect_nodes)
            graph2 = nx.complete_graph(nodes)
            graph = nx.compose_all([graph2, graph, connect_days])
    nx.set_edge_attributes(graph, 1, "weight")
    return graph, input_data


def fuzzy_color(graph: nx.Graph, k):
    try:
        return fgc.alpha_fuzzy_color(graph, k, fair=True)
    except fgc.NoSolutionException:
        return fgc.alpha_fuzzy_color(graph, k)


def interpret_graph():
    pass


if __name__ == '__main__':
    create_schedule("test_input.json")
