import polars as pl

from src.config import PROCESSED_DIR, IBGE_DIR

from src.data.population_projection import (
    MUNICIPAL_POPULATION_0_3_2025,
    add_neighborhood_density,
    agg_population_by_neighborhood,
    calculate_neighborhood_areas,
    load_neighborhood_geometries,
    load_population_points,
    project_sector_population_2025,
    validate_projection,
)

NEIGHBORHOODD_FILE = IBGE_DIR / "spatial/RJ_bairros_CD2022.gpkg"

POPULATION_POINTS_FILE = PROCESSED_DIR / "population_points.parquet"

SECTOR_OUTPUT_FILE = PROCESSED_DIR / "population_2025_by_sector.parquet"

NEIGHBORHOOD_OUTPUT_FILE = PROCESSED_DIR / "neighborhood_population_2025.parquet"

TARGET_COVERAGE_RATE = 0.6


def add_target_demand(df: pl.DataFrame, target_coverage_rate=TARGET_COVERAGE_RATE):
    if not 0 <= target_coverage_rate <= 1:
        raise ValueError("target_coverage_rate deve estar entre 0 e 1")

    return df.with_columns(
        (pl.col("estimated_population_0_to_3_2025") * target_coverage_rate).alias(
            "target_demand"
        )
    )


def print_summary(
    sector_population: pl.DataFrame, neighborhood_population: pl.DataFrame
):
    sector_total = sector_population.select(
        pl.col("estimated_population_0_to_3_2025").sum()
    ).item()

    neighborhood_total = neighborhood_population.select(
        pl.col("estimated_population_0_to_3_2025").sum()
    ).item()

    neighborhood_count = neighborhood_population.height

    sector_count = sector_population.height

    density_stats = neighborhood_population.select(
        pl.col("child_density_0_to_3_2025").min().alias("min"),
        pl.col("child_density_0_to_3_2025").median().alias("median"),
        pl.col("child_density_0_to_3_2025").mean().alias("mean"),
        pl.col("child_density_0_to_3_2025").max().alias("max"),
    )

    print()
    print("Population projection summary")
    print("-----------------------------")
    print(
        "Municipal population 0-3 (2025):",
        MUNICIPAL_POPULATION_0_3_2025,
    )
    print(
        "Projected sector total:",
        sector_total,
    )
    print(
        "Projected neighborhood total:",
        neighborhood_total,
    )
    print(
        "Sector count:",
        sector_count,
    )
    print(
        "Neighborhood count:",
        neighborhood_count,
    )

    print()
    print("Child density statistics")
    print("------------------------")
    print(density_stats)


def main():
    population_points = load_population_points(POPULATION_POINTS_FILE)
    sector_population = project_sector_population_2025(population_points)
    validate_projection(sector_population)

    sector_population.write_parquet(SECTOR_OUTPUT_FILE, compression="zstd")

    neighborhood_population = agg_population_by_neighborhood(sector_population)
    neighborhoods = load_neighborhood_geometries(NEIGHBORHOODD_FILE)
    neighborhood_areas = calculate_neighborhood_areas(neighborhoods)

    neighborhood_population = add_neighborhood_density(
        neighborhood_population, neighborhood_areas
    )

    neighborhood_population = add_target_demand(neighborhood_population)

    print(
        neighborhood_population.select(
            pl.col("target_demand").sum().alias("total_target_demand")
        )
    )

    print(neighborhood_population.select(pl.col("neighborhood_id").n_unique()))

    print(neighborhood_areas.select(pl.col("neighborhood_id").n_unique()))

    print(neighborhood_population.height)
    print(neighborhood_areas.height)

    final_total = neighborhood_population.select(
        pl.col("estimated_population_0_to_3_2025").sum()
    ).item()

    if abs(final_total - MUNICIPAL_POPULATION_0_3_2025) > 1e-6:
        raise ValueError(
            "neighborhood aggregation changed the municipal population total"
        )

    neighborhood_population.write_parquet(NEIGHBORHOOD_OUTPUT_FILE, compression="zstd")

    print_summary(sector_population, neighborhood_population)


if __name__ == "__main__":
    main()
