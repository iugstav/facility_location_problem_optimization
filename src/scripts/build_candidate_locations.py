import polars as pl

from src.config import PROCESSED_DIR

SECTORS_FILE = PROCESSED_DIR / "sectors_network_nodes.parquet"

FACILITIES_FILE = PROCESSED_DIR / "facilities_network_nodes.parquet"

OUTPUT_FILE = PROCESSED_DIR / "candidate_locations.parquet"


def build_candidates(sectors: pl.DataFrame, facilities: pl.DataFrame):
    existing_nodes = facilities.get_column("network_node").unique().to_list()

    candidates = (
        sectors.filter(pl.col("estimated_population_0_to_3_2025") > 0)
        .filter(~pl.col("network_node").is_in(existing_nodes))
        .group_by("network_node")
        .agg(
            pl.col("estimated_population_0_to_3_2025")
            .sum()
            .alias("local_population_0_to_3_2025"),
            pl.col("sector_id").unique().alias("represented_sector_ids"),
            pl.col("sector_id").n_unique().alias("represented_sector_count"),
            pl.col("neighborhood_id").unique().alias("represented_neighborhood_ids"),
            pl.col("neighborhood_name")
            .unique()
            .alias("represented_neighborhood_names"),
            pl.col("neighborhood_id")
            .n_unique()
            .alias("represented_neighborhood_count"),
            pl.col("latitude").mean().alias("latitude"),
            pl.col("longitude").mean().alias("longitude"),
        )
        .with_row_index("candidate_index")
        .with_columns(
            (pl.lit("candidate_") + pl.col("candidate_index").cast(pl.String)).alias(
                "candidate_id"
            ),
            (pl.col("represented_neighborhood_count") > 1).alias(
                "cross_neighborhood_candidate"
            ),
        )
    )

    return candidates


def validate_candidates(candidates: pl.DataFrame, facilities: pl.DataFrame):
    duplicate_nodes = (
        candidates.group_by("network_node").len().filter(pl.col("len") > 1)
    )

    if duplicate_nodes.height > 0:
        raise ValueError("existem candidatos associados ao mesmo nó viário")

    existing_nodes = set(facilities.get_column("network_node").to_list())
    candidate_nodes = set(candidates.get_column("network_node").to_list())
    overlap = existing_nodes & candidate_nodes

    if overlap:
        raise ValueError("existem candidatos que coincidem com facilities existentes")

    print()
    print("Candidate locations")
    print("-------------------")

    print(f"candidates: {candidates.height}")

    neighborhood_count = candidates.select(
        pl.col("represented_neighborhood_ids")
        .explode(empty_as_null=True)
        .n_unique()
        .alias("neighborhood_count")
    ).item()

    print(f"neighborhoods represented: {neighborhood_count}")

    print()

    print(
        candidates.select(
            [
                pl.col("local_population_0_to_3_2025").min().alias("min_population"),
                pl.col("local_population_0_to_3_2025")
                .median()
                .alias("median_population"),
                pl.col("local_population_0_to_3_2025").mean().alias("mean_population"),
                pl.col("local_population_0_to_3_2025").max().alias("max_population"),
            ]
        )
    )


def main():
    sectors = pl.read_parquet(SECTORS_FILE)
    facilities = pl.read_parquet(FACILITIES_FILE)

    candidates = build_candidates(sectors, facilities)

    validate_candidates(candidates, facilities)

    candidates.write_parquet(OUTPUT_FILE, compression="zstd")

    print()
    print(f"savec: {OUTPUT_FILE}")

    # print(
    #     candidates.sort(
    #         "local_population_0_to_3_2025",
    #         descending=True,
    #     )
    #     .select(
    #         [
    #             "candidate_id",
    #             "neighborhood_name",
    #             "local_population_0_to_3_2025",
    #             "represented_sector_count",
    #             "network_node",
    #         ]
    #     )
    #     .filter(pl.col("represented_sector_count") > 1)
    # )

    print(
        candidates.filter(pl.col("represented_neighborhood_count") > 1).select(
            [
                "candidate_id",
                "network_node",
                "represented_sector_count",
                "represented_neighborhood_count",
                "represented_neighborhood_names",
            ]
        )
    )


if __name__ == "__main__":
    main()
