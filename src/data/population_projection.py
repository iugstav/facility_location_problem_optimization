from pathlib import Path

import polars as pl
import geopandas as gpd

from src.config import IBGE_NOVA_IGUACU_CODE, PROCESSED_DIR

POPULATION_POINTS_FILE = Path(PROCESSED_DIR / "population_points.parquet")

MUNICIPAL_POPULATION_0_3_2025 = 37_844

PROJECTED_CRS = "EPSG:31983"


def load_population_points(
    path: Path,
) -> pl.DataFrame:
    df = pl.read_parquet(path)

    required_columns = {
        "sector_id",
        "neighborhood_id",
        "neighborhood_name",
        "child_population_0_to_4_final",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes em population_points: {missing}")

    return df


def project_sector_population_2025(population_points: pl.DataFrame):
    total_population_0_4_2022 = population_points.select(
        pl.col("child_population_0_to_4_final").sum()
    ).item()

    population_points_2025 = population_points.with_columns(
        (pl.col("child_population_0_to_4_final") / total_population_0_4_2022).alias(
            "spatial_weight"
        )
    ).with_columns(
        (pl.col("spatial_weight") * MUNICIPAL_POPULATION_0_3_2025).alias(
            "estimated_population_0_to_3_2025"
        )
    )

    return population_points_2025


def validate_projection(
    projected: pl.DataFrame,
    expected_total: int = MUNICIPAL_POPULATION_0_3_2025,
    tolerance: float = 1e-6,
):
    weight_sum = projected.select(pl.col("spatial_weight").sum()).item()

    population_sum = projected.select(
        pl.col("estimated_population_0_to_3_2025").sum()
    ).item()

    if abs(weight_sum - 1.0) > tolerance:
        raise ValueError(f"os pesos espaciais não somam 1. resultado: {weight_sum}")

    if abs(population_sum - expected_total) > tolerance:
        raise ValueError(
            "a projeção não reproduziu o total municipal. "
            f"esperado: {expected_total}; "
            f"obtido: {population_sum}"
        )


def agg_population_by_neighborhood(projected: pl.DataFrame):
    return (
        projected.group_by("neighborhood_id")
        .agg(
            pl.col("neighborhood_name").first().alias("neighborhood_name"),
            pl.col("estimated_population_0_to_3_2025")
            .sum()
            .alias("estimated_population_0_to_3_2025"),
            pl.col("child_population_0_to_4_final")
            .sum()
            .alias("child_population_0_to_4_2022"),
            pl.col("spatial_weight").sum().alias("spatial_weight"),
            pl.len().alias("sector_count"),
        )
        .sort("neighborhood_name")
    )


def load_neighborhood_geometries(path: Path):
    neighborhoods = gpd.read_file(path)

    req_cols = {"CD_MUN", "CD_BAIRRO", "NM_BAIRRO", "geometry"}

    missing = req_cols - set(neighborhoods.columns)

    if missing:
        raise ValueError(f"Colunas ausentes na malha de bairros: {missing}")

    neighborhoods["CD_MUN"] = neighborhoods["CD_MUN"].astype(str)

    neighborhoods = neighborhoods[neighborhoods["CD_MUN"] == IBGE_NOVA_IGUACU_CODE].copy()

    if neighborhoods.empty:
        raise ValueError("Nenhum bairro de Nova Iguaçu foi encontrado.")

    return neighborhoods[
        [
            "CD_BAIRRO",
            "NM_BAIRRO",
            "geometry",
        ]
    ].copy()


def calculate_neighborhood_areas(neighborhoods: gpd.GeoDataFrame):
    projected = neighborhoods.to_crs(PROJECTED_CRS).copy()

    dissolved = neighborhoods.dissolve(
        by="CD_BAIRRO", aggfunc={"NM_BAIRRO": "first"}
    ).reset_index()

    projected = dissolved.to_crs(PROJECTED_CRS).copy()

    projected["area_km2"] = projected.geometry.area / 1_000_000

    area_df = projected[["CD_BAIRRO", "NM_BAIRRO", "area_km2"]].rename(
        columns={
            "CD_BAIRRO": "neighborhood_id",
            "NM_BAIRRO": "neighborhood_name",
        }
    )

    result = pl.from_pandas(area_df)
    return result.with_columns(pl.col("neighborhood_id").cast(pl.String))


def assert_unique_neighborhoods(
    df: pl.DataFrame,
    dataset_name: str,
) -> None:

    duplicates = df.group_by("neighborhood_id").len().filter(pl.col("len") > 1)

    if duplicates.height > 0:
        raise ValueError(f"{dataset_name} possui bairros duplicados:\n{duplicates}")


def add_neighborhood_density(
    neighborhood_population: pl.DataFrame, neighborhood_areas: pl.DataFrame
):
    population = neighborhood_population.with_columns(
        pl.col("neighborhood_id").cast(pl.String)
    )

    assert_unique_neighborhoods(
        neighborhood_population,
        "neighborhood_population",
    )

    assert_unique_neighborhoods(
        neighborhood_areas,
        "neighborhood_areas",
    )

    result = population.join(
        neighborhood_areas.select(["neighborhood_id", "area_km2"]),
        on="neighborhood_id",
        how="left",
        validate="1:1",
    )

    missing_area = result.filter(pl.col("area_km2").is_null())
    if missing_area.height:
        raise ValueError(
            "existem bairros sem área correspondente: \n"
            f"{missing_area.select(['neighborhood_id', 'neighborhood_name'])}"
        )

    invalid_area = result.filter(pl.col("area_km2") == 0).height
    if invalid_area > 0:
        raise ValueError("existem bairros com área nula ou negativa")

    return result.with_columns(
        (pl.col("estimated_population_0_to_3_2025") / pl.col("area_km2")).alias(
            "child_density_0_to_3_2025"
        )
    ).sort("neighborhood_name")
