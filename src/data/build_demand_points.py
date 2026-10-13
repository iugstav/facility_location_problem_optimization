import geopandas as gpd


def assign_sectors_to_largest_overlap(
    sectors: gpd.GeoDataFrame, neighborhoods: gpd.GeoDataFrame
):
    if sectors.crs != neighborhoods.crs:
        neighborhoods = neighborhoods.to_crs(sectors.crs)

    sectors = sectors.copy()
    neighborhoods = neighborhoods.copy()

    sectors = sectors[
        [
            "CD_SETOR",
            "geometry",
        ]
    ].copy()

    neighborhoods = neighborhoods[
        [
            "CD_BAIRRO",
            "NM_BAIRRO",
            "geometry",
        ]
    ].copy()

    sectors["sector_id_internal"] = range(len(sectors))

    intersection = gpd.overlay(
        sectors,
        neighborhoods,
        how="intersection",
        keep_geom_type=False,
    )

    projected_crs = "EPSG:31983"
    intersection_projected = intersection.to_crs(projected_crs)
    intersection["intersection_area"] = intersection_projected.geometry.area

    best_match = intersection.sort_values(
        "intersection_area", ascending=False
    ).drop_duplicates("sector_id_internal")

    return best_match


def build_sector_points(sectors: gpd.GeoDataFrame):
    projected = sectors.to_crs("EPSG:31983").copy()
    projected["geometry"] = projected.geometry.representative_point()

    points = projected.to_crs("EPSG:4326")
    points["longitude"] = points.geometry.x
    points["latitude"] = points.geometry.y

    return points
