import networkx as nx
import osmnx as ox
import polars as pl
import pyarrow.parquet as pq

from src.config import PROCESSED_DIR

GRAPH_FILE = PROCESSED_DIR / "nova_iguacu_drive.graphml"

SECTORS_FILE = PROCESSED_DIR / "sectors_network_nodes.parquet"

CANDIDATES_FILE = PROCESSED_DIR / "candidate_locations.parquet"

OUTPUT_FILE = PROCESSED_DIR / "sector_candidate_travel_times.parquet"


def load_data():
    graph = ox.load_graphml(GRAPH_FILE)

    sectors = pl.read_parquet(SECTORS_FILE)

    candidates = pl.read_parquet(CANDIDATES_FILE)

    return (graph, sectors, candidates)


def validate_inputs(sectors: pl.DataFrame, candidates: pl.DataFrame):
    sector_required = {
        "sector_id",
        "neighborhood_id",
        "network_node",
        "estimated_population_0_to_3_2025",
    }

    candidate_required = {"candidate_id", "network_node"}

    missing_sector = sector_required - set(sectors.columns)
    missing_candidate = candidate_required - set(candidates.columns)

    if missing_sector:
        raise ValueError(f"colunas faltantes de sectors:\n{missing_sector}")

    if missing_candidate:
        raise ValueError(f"colunas faltantes de candidates:\n{missing_candidate}")

    if sectors.get_column("network_node").null_count() > 0:
        raise ValueError("existem setores sem network_node")

    if candidates.get_column("network_node").null_count() > 0:
        raise ValueError("existem candidatos sem network_node")

    duplicate_candidates = (
        candidates.group_by("candidate_id").len().filter(pl.col("len") > 1)
    )

    if duplicate_candidates.height > 0:
        raise ValueError("candidate_id contém duplicatas")

    duplicate_nodes = (
        candidates.group_by("network_node").len().filter(pl.col("len") > 1)
    )

    if duplicate_nodes.height > 0:
        raise ValueError("existem candidatos duplicados no mesmo network_node")

    print()
    print("Input summary")
    print("-------------")

    print(f"sectors: {sectors.height}")

    print(f"candidates: {candidates.height}")

    print(f"expected pairs: {sectors.height * candidates.height}")


def build_sector_candidate_times(
    graph, sectors: pl.DataFrame, candidates: pl.DataFrame
):
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    reversed_graph = graph.reverse(copy=False)

    sector_ids = sectors.get_column("sector_id").to_list()

    neighborhood_ids = sectors.get_column("neighborhood_id").to_list()

    sector_nodes = sectors.get_column("network_node").to_list()

    populations = sectors.get_column("estimated_population_0_to_3_2025").to_list()

    expected_pairs = sectors.height * candidates.height

    candidate_ids = candidates.get_column("candidate_id").to_list()

    candidate_nodes = candidates.get_column("network_node").to_list()

    if len(candidate_ids) != candidates.height:
        raise ValueError("candidate_ids possui tamanho inesperado.")

    if len(candidate_nodes) != candidates.height:
        raise ValueError("candidate_nodes possui tamanho inesperado.")

    if len(candidate_ids) != len(candidate_nodes):
        raise ValueError("candidate_id e network_node possuem tamanhos diferentes.")

    writer = None

    total_rows = 0
    unreachable_pairs = 0

    min_time = None
    max_time = None
    time_sum = 0.0
    reachable_count = 0

    processed_candidates = 0
    total_rows = 0

    try:
        for index, (
            candidate_id,
            candidate_node,
        ) in enumerate(
            zip(
                candidate_ids,
                candidate_nodes,
            ),
            start=1,
        ):
            candidate_node = int(candidate_node)

            print(f"[{index}/{candidates.height}] candidate {candidate_id}")

            distances = nx.single_source_dijkstra_path_length(
                reversed_graph,
                candidate_node,
                weight="travel_time",
            )

            travel_times = [
                (float(distances[int(node)]) if int(node) in distances else None)
                for node in map(int, sector_nodes)
            ]

            if len(travel_times) != sectors.height:
                raise ValueError(
                    f"Candidato {candidate_id}: "
                    f"{len(travel_times)} tempos, "
                    f"esperado {sectors.height}."
                )

            unreachable_here = sum(value is None for value in travel_times)

            unreachable_pairs += unreachable_here

            reachable_values = [
                float(value) for value in travel_times if value is not None
            ]

            if reachable_values:
                current_min = min(reachable_values)

                current_max = max(reachable_values)

                min_time = (
                    current_min
                    if min_time is None
                    else min(
                        min_time,
                        current_min,
                    )
                )

                max_time = (
                    current_max
                    if max_time is None
                    else max(
                        max_time,
                        current_max,
                    )
                )

                time_sum += sum(reachable_values)

                reachable_count += len(reachable_values)

            batch = pl.DataFrame(
                {
                    "sector_id": [str(value) for value in sector_ids],
                    "neighborhood_id": [str(value) for value in neighborhood_ids],
                    "candidate_id": [str(candidate_id)] * sectors.height,
                    "travel_time_seconds": travel_times,
                    "estimated_population_0_to_3_2025": [
                        float(value) for value in populations
                    ],
                },
                schema={
                    "sector_id": pl.String,
                    "neighborhood_id": pl.String,
                    "candidate_id": pl.String,
                    "travel_time_seconds": pl.Float64,
                    "estimated_population_0_to_3_2025": pl.Float64,
                },
            ).with_columns(
                [
                    pl.col("sector_id").cast(pl.String),
                    pl.col("neighborhood_id").cast(pl.String),
                    pl.col("candidate_id").cast(pl.String),
                    pl.col("travel_time_seconds").cast(pl.Float64),
                    pl.col("estimated_population_0_to_3_2025").cast(pl.Float64),
                ]
            )

            if batch.height != sectors.height:
                raise ValueError(
                    f"Candidato {candidate_id}: "
                    f"batch possui {batch.height} linhas, "
                    f"esperado {sectors.height}."
                )

            arrow_table = batch.to_arrow()

            if writer is None:
                OUTPUT_FILE.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                writer = pq.ParquetWriter(
                    OUTPUT_FILE,
                    arrow_table.schema,
                    compression="zstd",
                )

            writer.write_table(arrow_table)

            total_rows += batch.height
            processed_candidates += 1

    finally:
        if writer is not None:
            writer.close()

    expected_pairs = sectors.height * candidates.height

    print()
    print("Sector-candidate travel matrix")
    print("------------------------------")

    print(f"sectors: {sectors.height}")

    print(f"candidates expected: {candidates.height}")

    print(f"candidates processed: {processed_candidates}")

    print(f"rows generated: {total_rows}")

    print(f"expected pairs: {expected_pairs}")

    print(f"difference: {expected_pairs - total_rows}")

    print(f"unreachable pairs: {unreachable_pairs}")

    if processed_candidates != candidates.height:
        raise ValueError("Quantidade de candidatos processados diferente do esperado.")

    if total_rows != expected_pairs:
        raise ValueError("Quantidade de pares diferente do esperado.")


def main() -> None:

    (
        graph,
        sectors,
        candidates,
    ) = load_data()

    validate_inputs(
        sectors,
        candidates,
    )

    build_sector_candidate_times(
        graph,
        sectors,
        candidates,
    )


if __name__ == "__main__":
    main()
