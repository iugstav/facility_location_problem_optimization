import polars as pl

from src.config import PROCESSED_DIR

OUTPUT_FILE = PROCESSED_DIR / "facility_types.parquet"

FACILITY_TYPES = [
    {
        "facility_type": "type_2",
        "facility_type_name": "FNDE Creche Pré-Escola Tipo 2",
        "creche_class_count": 4,
        "capacity": 76.8,
        "construction_cost": 547_746.81,
        "reference_land_area_m2": 1800.0,
    },
    {
        "facility_type": "type_1",
        "facility_type_name": "FNDE Creche Pré-Escola Tipo 1",
        "creche_class_count": 6,
        "capacity": 115.2,
        "construction_cost": 671_576.50,
        "reference_land_area_m2": 2925.0,
    },
]


def build_facility_types():
    return pl.DataFrame(
        FACILITY_TYPES,
        schema={
            "facility_type": pl.String,
            "facility_type_name": pl.String,
            "creche_class_count": pl.Int64,
            "capacity": pl.Float64,
            "construction_cost": pl.Float64,
            "reference_land_area_m2": pl.Float64,
        },
    )


def validate_facility_types(df: pl.DataFrame):
    if df.height != 2:
        raise ValueError("esperava exatamente duas tipologias de facility")

    if df.get_column("facility_type").n_unique() != df.height:
        raise ValueError("facility_type contém duplicatas")

    invalid = df.filter(
        (pl.col("capacity") <= 0)
        | (pl.col("construction_cost") <= 0)
        | (pl.col("reference_land_area_m2") <= 0)
        | (pl.col("creche_class_count") <= 0)
    )

    if invalid.height != 0:
        raise ValueError("existem parâmetros não positivos")

    print()
    print("Facility types")
    print("--------------")
    print(df)

    print()
    print(
        df.select(
            [
                (pl.col("construction_cost") / pl.col("capacity")).alias(
                    "cost_per_modeled_place"
                )
            ]
        )
    )


def main():
    facility_types = build_facility_types()

    validate_facility_types(facility_types)

    facility_types.write_parquet(OUTPUT_FILE, compression="zstd")

    print()
    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
