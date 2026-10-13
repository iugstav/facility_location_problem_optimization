import osmnx as ox
import polars as pl

from src.config import PROCESSED_DIR

GRAPH_FILE = PROCESSED_DIR / "nova_iguacu_drive.graphml"

SECTORS_FILE = PROCESSED_DIR / "population_2025_by_sector.parquet"

FACILITIES_FILE = PROCESSED_DIR / "existing_facilities_locations_final.parquet"

SECTORS_OUTPUT = PROCESSED_DIR / "sectors_network_nodes.parquet"

FACILITIES_OUTPUT = PROCESSED_DIR / "facilities_network_nodes.parquet"


def load_graph():
    return ox.load_graphml(GRAPH_FILE)


def validate_coordinates(
    df: pl.DataFrame,
    id_column: str,
    latitude_column: str,
    longitude_column: str,
):
    missing = df.filter(
        pl.col(latitude_column).is_null() & pl.col(longitude_column).is_null()
    )

    if missing.height > 0:
        raise ValueError(f"{missing.height} registros de {id_column} sem coordenadas.")


def snap_points(graph, df: pl.DataFrame, latitude_column: str, longitude_column: str):
    longitudes = df.get_column(longitude_column).to_list()
    latitudes = df.get_column(latitude_column).to_list()

    nodes, distances = ox.distance.nearest_nodes(
        graph, X=longitudes, Y=latitudes, return_dist=True
    )

    return df.with_columns(
        [
            pl.Series("network_node", nodes, dtype=pl.Int64),
            pl.Series("snap_distance_m", distances, dtype=pl.Float64),
        ]
    )


def validate_snap_distances(
    df: pl.DataFrame, entity_name: str, warning_threshold_m: float = 500.0
):
    suspicious = df.filter(pl.col("snap_distance_m") > warning_threshold_m).sort(
        "snap_distance_m", descending=True
    )

    print()
    print(f"{entity_name} snap summary")
    print("-" * (len(entity_name) + len(" snap summary")))

    print(
        df.select(
            [
                pl.col("snap_distance_m").min().alias("min_m"),
                pl.col("snap_distance_m").median().alias("median_m"),
                pl.col("snap_distance_m").mean().alias("mean_m"),
                pl.col("snap_distance_m").max().alias("max_m"),
            ]
        )
    )

    print(f"points above {warning_threshold_m:.0f} m: {suspicious.height}")

    if suspicious.height > 0:
        print()
        print(suspicious)


def build_sector_nodes(graph):
    sectors = pl.read_parquet(SECTORS_FILE)

    required = {
        "sector_id",
        "neighborhood_id",
        "longitude",
        "latitude",
        "estimated_population_0_to_3_2025",
    }

    missing = required - set(sectors.columns)

    if missing:
        raise ValueError(f"Colunas ausentes nos setores: {missing}")

    validate_coordinates(
        sectors,
        id_column="sector_id",
        latitude_column="latitude",
        longitude_column="longitude",
    )

    return snap_points(
        graph,
        sectors,
        longitude_column="longitude",
        latitude_column="latitude",
    )


def build_facility_nodes(graph):
    facilities = pl.read_parquet(FACILITIES_FILE)

    required = {
        "facility_id",
        "facility_name",
        "final_capacity",
        "latitude_final",
        "longitude_final",
    }

    missing = required - set(facilities.columns)

    if missing:
        raise ValueError(f"colunas ausentes nas facilities: {missing}")

    validate_coordinates(
        facilities,
        id_column="facility_id",
        latitude_column="latitude_final",
        longitude_column="longitude_final",
    )

    return snap_points(
        graph,
        facilities,
        latitude_column="latitude_final",
        longitude_column="longitude_final",
    )


def main() -> None:
    print("loading road network")

    graph = load_graph()

    print("snapping sectors to network")

    sectors = build_sector_nodes(graph)

    validate_snap_distances(
        sectors,
        "sectors",
    )

    print()
    print("snapping facilities to network")

    facilities = build_facility_nodes(graph)

    validate_snap_distances(
        facilities,
        "facilities",
    )

    sectors.write_parquet(
        SECTORS_OUTPUT,
        compression="zstd",
    )

    facilities.write_parquet(
        FACILITIES_OUTPUT,
        compression="zstd",
    )

    print()
    print(f"saved: {SECTORS_OUTPUT}")

    print(f"saved: {FACILITIES_OUTPUT}")


if __name__ == "__main__":
    main()
