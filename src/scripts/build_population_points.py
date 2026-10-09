from pathlib import Path

import polars as pl

from src.config import AGG_DIR, PROCESSED_DIR

RAW_BASIC_FILE = Path(AGG_DIR / "Agregados_por_setores_basico_BR_utf8.csv")

POINTS_FILE = Path(PROCESSED_DIR / "demand_points_tabular.parquet")

DEMOGRAPHICS_FILE = Path(PROCESSED_DIR / "sector_demographics.parquet")

OUTPUT_FILE = Path(PROCESSED_DIR / "population_points.parquet")


def load_demand_points(path: Path):
    return pl.read_parquet(path).with_columns(
        pl.col("sector_id").cast(pl.String).str.strip_chars(),
        pl.col("neighborhood_id").cast(pl.String).str.strip_chars(),
    )


def load_demographics(path: Path):
    return pl.read_parquet(path).with_columns(
        pl.col("sector_id").cast(pl.String).str.strip_chars(),
    )


def build_population_points(points: pl.DataFrame, demographics: pl.DataFrame):
    demographics = demographics.with_columns(pl.lit(True).alias("_demographic_match"))
    return points.join(demographics, on="sector_id", how="left", validate="1:1")


def validate_unique_sectors(df: pl.DataFrame):
    duplicated = df.group_by("sector_id").len().filter(pl.col("len") > 1)
    if duplicated.height > 0:
        raise ValueError(f"foram encontrados {duplicated.height} setores duplicados")


def validate_coordinates(df: pl.DataFrame):
    missing = df.filter(pl.col("latitude").is_null() | pl.col("longitude").is_null())
    if missing.height > 0:
        raise ValueError(f"{missing.height} setores sem coordenadas")

    invalid = df.filter(
        ~pl.col("latitude").is_between(-90, 90)
        | ~pl.col("longitude").is_between(-180, 180)
    )
    if invalid.height > 0:
        raise ValueError(
            f"{missing.height} setores possuem coordenadas fora dos limites válidos"
        )


def validate_neighborhoods(
    df: pl.DataFrame,
) -> None:
    missing = df.filter(
        pl.col("neighborhood_id").is_null() | pl.col("neighborhood_name").is_null()
    )

    if missing.height > 0:
        raise ValueError(f"{missing.height} setores estão sem bairro associado.")


def validate_demographic_join(
    df: pl.DataFrame,
) -> None:
    unmatched_demographics = df.filter(pl.col("_demographic_match").is_null())

    print(
        "Setores sem correspondência demográfica:",
        unmatched_demographics.height,
    )

    if unmatched_demographics.height > 0:
        print(
            unmatched_demographics.select(
                [
                    "sector_id",
                    "neighborhood_id",
                    "neighborhood_name",
                ]
            ).head(50)
        )


def filter_sectors_by_missing_population(
    basic: pl.DataFrame, points: pl.DataFrame, demographics: pl.DataFrame
):
    points_without_population = points.join(basic, on="sector_id", how="left").filter(
        pl.col("total_population_basic") == 0
    )

    population_points = build_population_points(
        points,
        demographics,
    ).filter(
        ~pl.col("sector_id").is_in(
            points_without_population.select("sector_id").to_series()
        )
    )

    points_without_population = points.join(
        basic,
        on="sector_id",
        how="left",
    ).filter(pl.col("total_population_basic") == 0)

    return population_points, points_without_population


def build_neighbourhood_age_ratio(population_points: pl.DataFrame):
    neighborhood_age_ratio = (
        population_points.filter(
            pl.col("child_population_0_to_4").is_not_null()
            & pl.col("total_population").is_not_null()
            & (pl.col("total_population") > 0)
        )
        .group_by(["neighborhood_id", "neighborhood_name"])
        .agg(
            pl.col("child_population_0_to_4").sum().alias("known_children"),
            pl.col("total_population").sum().alias("total_population"),
        )
        .with_columns(
            (pl.col("known_children") / pl.col("total_population")).alias(
                "child_ratio_0_to_4"
            )
        )
    )

    return neighborhood_age_ratio


def print_summary(
    df: pl.DataFrame,
) -> None:
    summary = df.select(
        pl.len().alias("total_sectors"),
        pl.col("total_population").is_null().sum().alias("missing_total_population"),
        pl.col("child_population_0_to_4")
        .is_null()
        .sum()
        .alias("missing_child_population"),
        pl.col("child_population_0_to_4").sum().alias("known_children"),
    )

    print(summary)


def main() -> None:
    points = load_demand_points(POINTS_FILE)
    demographics = load_demographics(DEMOGRAPHICS_FILE)

    basic = (
        pl.read_csv(
            RAW_BASIC_FILE,
            separator=";",
            columns=[
                "CD_SETOR",
                "v0001",
            ],
            schema_overrides={
                "CD_SETOR": pl.String,
            },
            null_values=["X", "."],
            infer_schema_length=10_000,
        )
        .rename(
            {
                "CD_SETOR": "sector_id",
                "v0001": "total_population_basic",
            }
        )
        .with_columns(pl.col("total_population_basic").cast(pl.Int64, strict=False))
    )

    population_points, points_without_population = filter_sectors_by_missing_population(
        basic, points, demographics
    )

    demographics_without_points = demographics.join(
        points.select("sector_id"),
        on="sector_id",
        how="anti",
    )
    missing_ids = demographics_without_points.select("sector_id").to_series().to_list()
    excluded = demographics.filter(pl.col("sector_id").is_in(missing_ids)).with_columns(
        pl.lit("rural_sector_without_neighborhood").alias("exclusion_reason")
    )

    validate_unique_sectors(population_points)
    validate_coordinates(population_points)
    validate_neighborhoods(population_points)
    validate_demographic_join(population_points)
    print_summary(population_points)

    neighborhood_age_ratio = build_neighbourhood_age_ratio(population_points)

    population_points = population_points.join(
        neighborhood_age_ratio,
        on="neighborhood_id",
        how="left",
    )

    population_points = population_points.with_columns(
        pl.when(pl.col("total_population") == 0)
        .then(pl.lit(0))
        .when(pl.col("child_population_0_to_4").is_not_null())
        .then(pl.col("child_population_0_to_4"))
        .when(
            pl.col("child_ratio_0_to_4").is_not_null()
            & pl.col("total_population").is_not_null()
        )
        .then(
            (pl.col("total_population") * pl.col("child_ratio_0_to_4"))
            .round(0)
            .cast(pl.Int64)
        )
        .otherwise(None)
        .alias("child_population_0_to_4_final")
    )

    population_points = population_points.with_columns(
        pl.when(pl.col("total_population") == 0)
        .then(pl.lit("zero_population"))
        .when(pl.col("child_population_0_to_4").is_not_null())
        .then(pl.lit("observed"))
        .when(pl.col("child_population_0_to_4_final").is_not_null())
        .then(pl.lit("imputed_neighborhood_ratio"))
        .otherwise(pl.lit("unresolved"))
        .alias("child_population_source")
    )

    population_points = population_points.join(basic, on="sector_id", how="left")
    # unresolved_sectors = population_points.filter(
    #     pl.col("child_population_source") == "unresolved"
    # ).select(
    #     [
    #         "sector_id",
    #         "neighborhood_id",
    #         "neighborhood_name",
    #         "total_population",
    #         "total_population_basic",
    #         "child_ratio_0_to_4",
    #     ]
    # )

    population_points = population_points.with_columns(
        pl.coalesce(
            [pl.col("total_population"), pl.col("total_population_basic")]
        ).alias("total_population_final")
    )
    population_points = population_points.with_columns(
        (pl.col("total_population_final") * pl.col("child_ratio_0_to_4"))
        .round(0)
        .cast(pl.Int64)
        .alias("estimated_child_population_0_to_4_final")
    )

    population_points = population_points.with_columns(
        pl.when(pl.col("child_population_0_to_4_final").is_not_null())
        .then(pl.col("child_population_0_to_4_final"))
        .when(pl.col("estimated_child_population_0_to_4_final").is_not_null())
        .then(
            pl.max_horizontal(
                "estimated_child_population_0_to_4_final",
                "known_children",
            )
        )
        .otherwise(None)
        .alias("child_population_0_to_4_final")
    )

    population_points = population_points.with_columns(
        pl.when(pl.col("child_population_source") != "unresolved")
        .then(pl.col("child_population_source"))
        .when(pl.col("child_population_0_to_4_final").is_not_null())
        .then(pl.lit("imputed_basic_population"))
        .otherwise(pl.lit("unresolved"))
        .alias("child_population_source")
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    population_points.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    excluded.write_parquet(
        PROCESSED_DIR / "excluded_sectors.parquet",
        compression="zstd",
    )
    points_without_population.write_parquet(
        PROCESSED_DIR / "excluded_zero_population_sectors.parquet",
        compression="zstd",
    )

    print(f"Arquivo salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
