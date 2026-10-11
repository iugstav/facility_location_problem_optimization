from pathlib import Path

import polars as pl

from src.config import PROCESSED_DIR

POPULATION_POINTS_FILE = Path(PROCESSED_DIR / "population_points.parquet")

OUTPUT_FILE = Path(PROCESSED_DIR / "neighborhood_demographics.parquet")


def load_population_points(path: Path):
    return pl.read_parquet(path)


def agg_by_neighborhood(df: pl.DataFrame):
    return (
        df.group_by(["neighborhood_id", "neighborhood_name"])
        .agg(
            pl.col("total_population_final").sum().alias("total_population"),
            pl.col("child_population_0_to_4_final").sum().alias("child_population_0_to_4"),
            pl.len().alias("sector_count"),
            (pl.col("child_population_source") == "observed")
            .sum()
            .alias("observed_sector_count"),
            (pl.col("child_population_source").str.starts_with("imputed"))
            .sum()
            .alias("imputed_sector_count"),
        )
        .sort("neighborhood_name")
    )


def validate(df: pl.DataFrame):
    duplicated = df.group_by("neighborhood_id").len().filter(pl.col("len") > 1)
    if duplicated.height > 0:
        raise ValueError("existem bairros duplicados")

    invalid = df.filter(pl.col("child_population_0_to_4") > pl.col("total_population"))

    if invalid.height > 0:
        raise ValueError(
            "existem bairros com população infantil superior à população total"
        )

    missing = df.filter(
        pl.col("neighborhood_id").is_null() | pl.col("neighborhood_name").is_null()
    )

    if missing.height > 0:
        raise ValueError("existem registros sem bairro")


def print_summary(df: pl.DataFrame):
    print(
        df.select(
            pl.len().alias("neighborhood_count"),
            pl.col("total_population").sum().alias("total_population"),
            pl.col("child_population_0_to_4").sum().alias("child_population_0_to_4"),
            pl.col("sector_count").sum().alias("sector_count"),
            pl.col("imputed_sector_count").sum().alias("imputed_sector_count"),
        )
    )


def main():
    population_points = load_population_points(POPULATION_POINTS_FILE)

    neighborhoods = agg_by_neighborhood(population_points)

    validate(neighborhoods)

    print_summary(neighborhoods)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    neighborhoods.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    print(f"Arquivo salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
