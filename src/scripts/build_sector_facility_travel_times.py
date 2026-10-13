import networkx as nx
import osmnx as ox
import polars as pl

from src.config import PROCESSED_DIR

GRAPH_FILE = PROCESSED_DIR / "nova_iguacu_drive.graphml"

SECTORS_FILE = PROCESSED_DIR / "sectors_network_nodes.parquet"

FACILITIES_FILE = PROCESSED_DIR / "facilities_network_nodes.parquet"

OUTPUT_FILE = PROCESSED_DIR / "sector_facility_travel_times.parquet"


def load_data():
    graph = ox.load_graphml(GRAPH_FILE)

    sectors = pl.read_parquet(SECTORS_FILE)

    facilities = pl.read_parquet(FACILITIES_FILE)

    return (
        graph,
        sectors,
        facilities,
    )


def build_travel_times(
    graph,
    sectors: pl.DataFrame,
    facilities: pl.DataFrame,
):
    reversed_graph = graph.reverse(copy=False)

    sector_nodes = sectors.select(
        [
            "sector_id",
            "neighborhood_id",
            "network_node",
            "estimated_population_0_to_3_2025",
        ]
    )

    rows = []
    for facility in facilities.iter_rows(named=True):
        facility_id = facility["facility_id"]
        facility_node = facility["network_node"]

        print(f"calculating travel times for facility {facility_id}")
        distances = nx.single_source_dijkstra_path_length(
            reversed_graph,
            facility_node,
            weight="travel_time",
        )

        for sector in sector_nodes.iter_rows(named=True):
            sector_node = sector["network_node"]

            travel_time = distances.get(sector_node)

            rows.append(
                {
                    "sector_id": (sector["sector_id"]),
                    "neighborhood_id": (sector["neighborhood_id"]),
                    "facility_id": (facility_id),
                    "travel_time_seconds": (travel_time),
                    "estimated_population_0_to_3_2025": (
                        sector["estimated_population_0_to_3_2025"]
                    ),
                }
            )

    return pl.DataFrame(rows)


def validate_travel_times(df: pl.DataFrame):
    expected = (
        df.select(pl.col("sector_id").n_unique()).item()
        * df.select(pl.col("facility_id").n_unique()).item()
    )

    print()
    print("travel-time matrix")
    print("------------------")

    print(f"rows: {df.height}")

    print(f"expected pairs: {expected}")

    unreachable = df.filter(pl.col("travel_time_seconds").is_null())

    print(f"unreachable pairs: {unreachable}")

    reachable = df.filter(pl.col("travel_time_seconds").is_not_null())

    print()
    print(
        reachable.select(
            [
                (pl.col("travel_time_seconds").min() / 60).alias("min_minutes"),
                (pl.col("travel_time_seconds").median() / 60).alias("median_minutes"),
                (pl.col("travel_time_seconds").mean() / 60).alias("mean_minutes"),
                (pl.col("travel_time_seconds").max() / 60).alias("max_minutes"),
            ]
        )
    )


def main():
    graph, sectors, facilities = load_data()

    travel_times = build_travel_times(graph, sectors, facilities)
    validate_travel_times(travel_times)

    travel_times.write_parquet(OUTPUT_FILE, compression="zstd")

    print()
    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
