import polars as pl

from src.config import PROCESSED_DIR

TRAVEL_TIMES_FILE = PROCESSED_DIR / "sector_facility_travel_times.parquet"

OUTPUT_FILE = PROCESSED_DIR / "neighborhood_facility_costs.parquet"


def load_travel_times():
    df = pl.read_parquet(TRAVEL_TIMES_FILE)

    req = {
        "sector_id",
        "neighborhood_id",
        "facility_id",
        "travel_time_seconds",
        "estimated_population_0_to_3_2025",
    }

    missing = req - set(df.columns)

    if missing:
        raise ValueError(f"missing cols: {missing}")

    return df


def build_neighorhood_facility_costs(df: pl.DataFrame):
    return (
        df.with_columns(
            (
                pl.col("estimated_population_0_to_3_2025")
                * pl.col("travel_time_seconds")
            ).alias("weighted_travel_time")
        )
        .group_by(["neighborhood_id", "facility_id"])
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
        .with_columns((pl.col("travel_time_seconds") / 60).alias("travel_time_minutes"))
        .select(
            [
                "neighborhood_id",
                "facility_id",
                "neighborhood_population_0_to_3_2025",
                "sector_count",
                "travel_time_seconds",
                "travel_time_minutes",
            ]
        )
        .sort(["neighborhood_id", "facility_id"])
    )


def validate_cost_matrix(df: pl.DataFrame):
    neighborhoods = df.select(pl.col("neighborhood_id").n_unique()).item()

    facilities = df.select(pl.col("facility_id").n_unique()).item()

    expected_pairs = neighborhoods * facilities

    print()
    print("Neighborhood-facility cost matrix")
    print("---------------------------------")

    print(f"neighborhoods: {neighborhoods}")

    print(f"facilities: {facilities}")

    print(f"rows: {df.height}")

    print(f"expected pairs: {expected_pairs}")

    if df.height != expected_pairs:
        raise ValueError("a matriz bairro-facility não está completa")

    invalid_population = df.filter(pl.col("neighborhood_population_0_to_3_2025") <= 0)

    if invalid_population.height > 0:
        raise ValueError("existem bairros com população infantil não positiva")

    invalid_times = df.filter(
        pl.col("travel_time_seconds").is_null() | (pl.col("travel_time_seconds") < 0)
    )

    if invalid_times.height > 0:
        raise ValueError("Existem custos de viagem inválidos.")

    print()
    print(
        df.select(
            [
                pl.col("travel_time_minutes").min().alias("min_minutes"),
                pl.col("travel_time_minutes").median().alias("median_minutes"),
                pl.col("travel_time_minutes").mean().alias("mean_minutes"),
                pl.col("travel_time_minutes").max().alias("max_minutes"),
            ]
        )
    )


def main():
    travel_times = load_travel_times()

    costs = build_neighorhood_facility_costs(travel_times)

    validate_cost_matrix(costs)

    costs.write_parquet(OUTPUT_FILE, compression="zstd")

    print()
    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
