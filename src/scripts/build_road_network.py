from pathlib import Path

import osmnx as ox

from src.config import PROCESSED_DIR

OUTPUT_FILE = PROCESSED_DIR / "nova_iguacu_drive.graphml"

TARGET_PLACE = "Nova Iguaçu, Rio de Janeiro, Brazil"


def build_graph():
    print(f"downloading road network for: {TARGET_PLACE}")

    graph = ox.graph_from_place(
        TARGET_PLACE,
        network_type="drive",
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    return graph


def add_travel_times(graph):
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)

    return graph


def validate_graph(graph):
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    print()
    print("road network summary")
    print("--------------------")

    print(f"nodes: {node_count}")
    print(f"edges: {edge_count}")

    if node_count == 0:
        raise ValueError("graph does not have nodes")

    if edge_count == 0:
        raise ValueError("graph does not have edges")

    missing_length = 0
    missing_speed = 0
    missing_travel_time = 0

    for _, _, _, data in graph.edges(keys=True, data=True):
        if data.get("length") is None:
            missing_length += 1

        if data.get("speed_kph") is None:
            missing_speed += 1

        if data.get("travel_time") is None:
            missing_travel_time += 1

    print(f"edges without length: {missing_length}")

    print(f"edges without speed_kph: {missing_speed}")

    print(f"edges without travel_time: {missing_travel_time}")

    if missing_travel_time > 0:
        raise ValueError("Existem arestas sem tempo de viagem.")


def save_graph(graph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    ox.save_graphml(graph, path)
    print()
    print(f"saved: {path}")


def main():
    graph = build_graph()
    graph = add_travel_times(graph)
    validate_graph(graph)

    save_graph(graph, OUTPUT_FILE)


if __name__ == "__main__":
    main()
