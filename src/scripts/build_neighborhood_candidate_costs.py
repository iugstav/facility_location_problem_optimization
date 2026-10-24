import polars as pl

from src.config import PROCESSED_DIR

TRAVEL_TIMES_FILE = PROCESSED_DIR / "sector_candidate_travel_times.parquet"

CANDIDATES_FILE = PROCESSED_DIR / "candidate_locations.parquet"

OUTPUT_FILE = PROCESSED_DIR / "neighborhood_candidate_costs.parquet"


def build_cost_matrix():
    travel_times = pl.scan_parquet(TRAVEL_TIMES_FILE)

    costs = (
        travel_times.with_columns(
            (
                pl.col("estimated_population_0_to_3_2025")
                * pl.col("travel_time_seconds")
            ).alias("weighted_travel_time")
        )
        .group_by(["neighborhood_id", "candidate_id"])
        .agg(
            pl.col("weighted_travel_time").sum().alias("weighted_travel_time_sum"),
            pl.col("estimated_population_0_to_3_2025")
            .sum()
            .alias("neighborhood_population_0_to_3_2025"),
            pl.col("sector_id").n_unique().alias("sector_count"),
        )
        .with_columns(
            (
                pl.col("weighted_travel_time_sum")
                / pl.col("neighborhood_population_0_to_3_2025")
            ).alias("travel_time_seconds")
        )
        .with_columns(
            (pl.col("travel_time_seconds") / 60.0).alias("travel_time_minutes")
        )
        .select(
            [
                "neighborhood_id",
                "candidate_id",
                "neighborhood_population_0_to_3_2025",
                "sector_count",
                "travel_time_seconds",
                "travel_time_minutes",
            ]
        )
        .sort(["neighborhood_id", "candidate_id"])
        .collect()
    )

    return costs


def validate_cost_matrix(costs: pl.DataFrame):
    candidates = pl.read_parquet(CANDIDATES_FILE)

    neighborhood_count = costs.get_column("neighborhood_id").n_unique()

    candidate_count = costs.get_column("candidate_id").n_unique()

    expected_candidates = candidates.get_column("candidate_id").n_unique()

    expected_pairs = neighborhood_count * expected_candidates

    print()
    print("Neighborhood-candidate cost matrix")
    print("----------------------------------")

    print(f"neighborhoods: {neighborhood_count}")

    print(f"candidates: {candidate_count}")

    print(f"rows: {costs.height}")

    print(f"expected pairs: {expected_pairs}")

    if neighborhood_count != 67:
        raise ValueError("Quantidade inesperada de bairros.")

    if candidate_count != expected_candidates:
        raise ValueError("Nem todos os candidatos estão presentes.")

    if costs.height != expected_pairs:
        raise ValueError("A matriz bairro-candidato não está completa.")

    invalid_population = costs.filter(
        pl.col("neighborhood_population_0_to_3_2025") <= 0
    )

    if invalid_population.height > 0:
        raise ValueError("Existem bairros com população infantil não positiva.")

    invalid_time = costs.filter(
        pl.col("travel_time_seconds").is_null() | (pl.col("travel_time_seconds") < 0)
    )

    if invalid_time.height > 0:
        raise ValueError("Existem tempos de viagem inválidos.")

    print()
    print(
        costs.select(
            [
                pl.col("travel_time_minutes").min().alias("min_minutes"),
                pl.col("travel_time_minutes").median().alias("median_minutes"),
                pl.col("travel_time_minutes").mean().alias("mean_minutes"),
                pl.col("travel_time_minutes").max().alias("max_minutes"),
            ]
        )
    )


def main() -> None:
    print("building neighborhood-candidate cost matrix")

    costs = build_cost_matrix()

    validate_cost_matrix(costs)

    costs.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    print()
    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
