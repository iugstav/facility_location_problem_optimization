from pathlib import Path
import geopandas as gpd
import pandas as pd
from src.config import IBGE_NOVA_IGUACU_CODE


def load_nova_iguacu_neighborhoods(path: Path) -> gpd.GeoDataFrame | pd.Series:
    gdf = gpd.read_file(path)

    print("colunas dos bairros:")
    print(gdf.columns.tolist())

    municipality_columns = [
        "CD_MUN",
        "CD_MUN_1",
        "CD_GEOCMU",
    ]

    municipality_column = next(
        (column for column in municipality_columns if column in gdf.columns), None
    )
    if not municipality_column:
        raise ValueError(
            "não foi possível encontrar uma coluna com o código do município"
        )

    nova_iguacu = gdf[
        gdf[municipality_column].astype(str) == IBGE_NOVA_IGUACU_CODE
    ].copy()
    if nova_iguacu.empty:
        raise ValueError("nenhum bairro de nova iguaçu foi encontrado")

    return nova_iguacu


def load_nova_iguacu_sectors(path: Path) -> gpd.GeoDataFrame | pd.Series:
    gdf = gpd.read_file(path)

    print("colunas dos setores:")
    print(gdf.columns.tolist())

    municipality_columns = [
        "CD_MUN",
        "CD_MUN_1",
        "CD_GEOCMU",
    ]

    municipality_column = next(
        (column for column in municipality_columns if column in gdf.columns), None
    )
    if not municipality_column:
        raise ValueError(
            "não foi possível encontrar uma coluna com o código do município"
        )

    nova_iguacu = gdf[
        gdf[municipality_column].astype(str) == IBGE_NOVA_IGUACU_CODE
    ].copy()
    if nova_iguacu.empty:
        raise ValueError("nenhum setor de nova iguaçu foi encontrado")

    return nova_iguacu

