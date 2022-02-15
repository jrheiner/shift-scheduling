from jsonschema import validate, ValidationError
import networkx as nx
import fuzzy_graph_coloring as fgc
import json


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
            g.add_edge(u, v, a)
    return g


def create_schedule(input_path: str):
    graph, input_data = generate_graph(input_path)
    crisp_coloring = nx.greedy_color(_alpha_cut(graph, alpha=1))
    k = max(crisp_coloring.values()) + 1
    if k > input_data["total_staff"]:
        raise Exception("Even by considering only hard constraints, a schedule is not possible."
                        f"(A {input_data['total_staff']}-coloring does not exist.)")
    fuzzy_coloring, score = fuzzy_color(graph, input_data["total_staff"])
    print(crisp_coloring)
    print(fuzzy_coloring, score)
    # interpret_graph()
    pass


def generate_graph(input_path: str) -> nx.Graph:
    # input_data = _parse_input(input_path)
    graph = nx.Graph()
    return graph


def fuzzy_color(graph: nx.Graph, k):
    return fgc.fuzzy_color(graph, k=k)


def interpret_graph():
    pass


if __name__ == '__main__':
    create_schedule("test_input.json")
