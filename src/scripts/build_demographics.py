from pathlib import Path
import polars as pl
from src.config import PROCESSED_DIR, AGG_DIR, IBGE_NOVA_IGUACU_CODE

RAW_DEMOGRAPHICS_FILE = Path(AGG_DIR / "Agregados_por_setores_demografia_BR.csv")
OUTPUT_FILE = PROCESSED_DIR / "sector_demographics.parquet"


def load_demographics(path: Path):
    return pl.read_csv(
        path,
        separator=";",
        schema_overrides={"CD_setor": pl.String},
        null_values=["X"],
        infer_schema_length=10_000,
    )


def filter_nova_iguacu(df: pl.DataFrame):
    return df.filter(pl.col("CD_setor").str.starts_with(IBGE_NOVA_IGUACU_CODE))


def select_demographic_variables(df: pl.DataFrame):
    req_cols = {
        "CD_setor",
        "V01006",
        "V01009",
        "V01020",
    }

    missing_cols = req_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Colunas obrigatórias ausentes no arquivo de demografia: "
            f"{sorted(missing_cols)}"
        )

    return df.select(
        pl.col("CD_setor").alias("sector_id"),
        pl.col("V01006").cast(pl.Int64, strict=False).alias("total_population"),
        pl.col("V01009").cast(pl.Int64, strict=False).alias("male_population_0_to_4"),
        pl.col("V01020").cast(pl.Int64, strict=False).alias("female_population_0_to_4"),
    )


def build_child_population(df: pl.DataFrame):
    return df.with_columns(
        pl.when(
            pl.col("male_population_0_to_4").is_not_null()
            & pl.col("female_population_0_to_4").is_not_null()
        )
        .then(pl.col("male_population_0_to_4") + pl.col("female_population_0_to_4"))
        .otherwise(None)
        .alias("child_population_0_to_4")
    )


def add_missing_flag(df: pl.DataFrame):
    return df.with_columns(
        pl.col("child_population_0_to_4").is_null().alias("child_population_missing")
    )


def validate_demographics(df: pl.DataFrame):
    duplicated_sectors = df.group_by("sector_id").len().filter(pl.col("len") > 1)
    if duplicated_sectors.height > 0:
        raise ValueError(
            f"foram encontrados  {duplicated_sectors.height} setores duplicados."
        )

    invalid_child_population = df.filter(
        pl.col("child_population_0_to_4").is_not_null()
        & pl.col("total_population").is_not_null()
        & (pl.col("child_population_0_to_4") > pl.col("total_population"))
    )
    if invalid_child_population.height > 0:
        raise ValueError(
            "foram encontrados setores onde a população "
            "de 0 a 4 anos é maior que a população total"
        )


def print_summary(df: pl.DataFrame):
    total_sectors = df.height

    sectors_with_missing_child_population = df.filter(
        pl.col("child_population_missing")
    ).height

    summary = df.select(
        pl.col("total_population").sum().alias("total_population"),
        pl.col("child_population_0_to_4").sum().alias("known_child_population_0_to_4"),
    )

    print(f"setores de nova iguaçu: {total_sectors}")

    print(
        "setores com população 0-4 não divulgada: "
        f"{sectors_with_missing_child_population}"
    )

    print(summary)


def build_demographics(source: Path):
    df = load_demographics(source)

    df = filter_nova_iguacu(df)

    df = select_demographic_variables(df)

    df = build_child_population(df)

    df = add_missing_flag(df)

    validate_demographics(df)

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    demographics = build_demographics(RAW_DEMOGRAPHICS_FILE)
    demographics.write_parquet(OUTPUT_FILE, compression="zstd")

    print_summary(demographics)

    print(f"arquivo salvo em {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
