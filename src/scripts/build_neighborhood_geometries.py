import geopandas as gpd

from src.config import PROCESSED_DIR, SPATIAL_DIR, IBGE_NOVA_IGUACU_CODE

SECTORS_FILE = SPATIAL_DIR / "RJ_setores_CD2022.gpkg"

OUTPUT_FILE = PROCESSED_DIR / "neighborhoods.geojson"


def load_sectors() -> gpd.GeoDataFrame:
    if not SECTORS_FILE.exists():
        raise FileNotFoundError(f"Sector geometry file not found: {SECTORS_FILE}")

    sectors = gpd.read_file(SECTORS_FILE)

    print()
    print("Input geometry")
    print("--------------")
    print(f"rows: {len(sectors)}")
    print(f"CRS: {sectors.crs}")

    return sectors


def filter_nova_iguacu(
    sectors: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    required_columns = {
        "CD_SETOR",
        "CD_MUN",
        "CD_BAIRRO",
        "NM_BAIRRO",
        "geometry",
    }

    missing = required_columns - set(sectors.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sectors = sectors.copy()

    sectors["CD_MUN"] = sectors["CD_MUN"].astype(str).str.strip()

    nova_iguacu = sectors[sectors["CD_MUN"] == IBGE_NOVA_IGUACU_CODE].copy()

    print()
    print("Nova Iguaçu")
    print("-----------")
    print(f"sectors: {len(nova_iguacu)}")

    print(f"identified neighborhoods: {nova_iguacu['CD_BAIRRO'].nunique(dropna=True)}")

    print(f"sectors without neighborhood: {nova_iguacu['CD_BAIRRO'].isna().sum()}")

    return nova_iguacu


def clean_sectors(
    sectors: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    sectors = sectors.copy()

    # Remove sectors without official neighborhood.
    sectors = sectors[sectors["CD_BAIRRO"].notna()].copy()

    sectors = sectors[sectors["NM_BAIRRO"].notna()].copy()

    # Remove empty geometries.
    sectors = sectors[sectors.geometry.notna()].copy()

    sectors = sectors[~sectors.geometry.is_empty].copy()

    sectors["CD_SETOR"] = sectors["CD_SETOR"].astype(str).str.strip()

    sectors["CD_BAIRRO"] = sectors["CD_BAIRRO"].astype(str).str.strip()

    sectors["NM_BAIRRO"] = sectors["NM_BAIRRO"].astype(str).str.strip()

    return sectors


def build_neighborhoods(
    sectors: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    neighborhoods = sectors.dissolve(
        by=[
            "CD_BAIRRO",
            "NM_BAIRRO",
        ],
        as_index=False,
    )

    neighborhoods = neighborhoods[
        [
            "CD_BAIRRO",
            "NM_BAIRRO",
            "geometry",
        ]
    ].rename(
        columns={
            "CD_BAIRRO": "neighborhood_id",
            "NM_BAIRRO": "neighborhood_name",
        }
    )

    return neighborhoods


# ============================================================
# Validation
# ============================================================


def validate_neighborhoods(
    neighborhoods: gpd.GeoDataFrame,
) -> None:
    print()
    print("Neighborhood geometry")
    print("---------------------")

    print(f"neighborhoods: {len(neighborhoods)}")

    print(f"invalid geometries: {(~neighborhoods.geometry.is_valid).sum()}")

    print(f"empty geometries: {neighborhoods.geometry.is_empty.sum()}")

    if len(neighborhoods) != 67:
        raise ValueError(f"Expected 67 neighborhoods, found {len(neighborhoods)}.")

    if neighborhoods.geometry.is_empty.any():
        raise ValueError("There are empty neighborhood geometries.")


def repair_geometries(
    neighborhoods: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    neighborhoods = neighborhoods.copy()

    invalid = ~neighborhoods.geometry.is_valid

    if invalid.any():
        print()
        print("Repairing invalid geometries...")

        neighborhoods.loc[
            invalid,
            "geometry",
        ] = neighborhoods.loc[
            invalid,
            "geometry",
        ].buffer(0)

    remaining_invalid = ~neighborhoods.geometry.is_valid

    if remaining_invalid.any():
        raise ValueError(f"{remaining_invalid.sum()} invalid geometries remain.")

    return neighborhoods


def save_geojson(
    neighborhoods: gpd.GeoDataFrame,
) -> None:
    neighborhoods = neighborhoods.to_crs("EPSG:4326")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    neighborhoods.to_file(
        OUTPUT_FILE,
        driver="GeoJSON",
    )

    print()
    print(f"saved to: {OUTPUT_FILE}")


def main():
    sectors = load_sectors()

    sectors = filter_nova_iguacu(sectors)

    sectors = clean_sectors(sectors)

    neighborhoods = build_neighborhoods(sectors)

    validate_neighborhoods(neighborhoods)

    neighborhoods = repair_geometries(neighborhoods)

    save_geojson(neighborhoods)


if __name__ == "__main__":
    main()
