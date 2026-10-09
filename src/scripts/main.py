from pathlib import Path

from src.config import PROCESSED_DIR, SPATIAL_DIR
from src.data.load_boundaries import (
    load_nova_iguacu_neighborhoods,
    load_nova_iguacu_sectors,
)
from src.data.build_demand_points import (
    assign_sectors_to_largest_overlap,
    build_sector_points,
)

bairros = Path(SPATIAL_DIR / "RJ_bairros_CD2022.gpkg")
setores = Path(SPATIAL_DIR / "RJ_setores_CD2022.gpkg")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    neighborhoods = load_nova_iguacu_neighborhoods(bairros)
    sectors = load_nova_iguacu_sectors(setores)

    sectors_with_neighborhood = assign_sectors_to_largest_overlap(
        sectors, neighborhoods
    )
    demand_points = build_sector_points(sectors_with_neighborhood)
    print(demand_points.columns)
    tabular_demand_points = demand_points[
        ["CD_SETOR", "CD_BAIRRO", "NM_BAIRRO", "longitude", "latitude"]
    ].rename(
        columns={
            "CD_SETOR": "sector_id",
            "CD_BAIRRO": "neighborhood_id",
            "NM_BAIRRO": "neighborhood_name",
        }
    )

    neighborhoods.to_parquet(PROCESSED_DIR / "neighborhoods.parquet")
    sectors.to_parquet(PROCESSED_DIR / "sectors.parquet")
    demand_points.to_parquet(PROCESSED_DIR / "demand_points.parquet")
    tabular_demand_points.to_parquet(PROCESSED_DIR / "demand_points_tabular.parquet")

    print(f"bairros: {len(neighborhoods)}")
    print(f"pontos de demanda: {len(demand_points)}")


if __name__ == "__main__":
    main()
