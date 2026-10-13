from pathlib import Path

import polars as pl

from src.config import INEP_DIR, PROCESSED_DIR

FACILITIES_FILE = PROCESSED_DIR / "existing_facilities_with_capacity.parquet"

SCHOOLS_FILE = INEP_DIR / "Análise - Tabela da lista das escolas - Detalhado.csv"

OUTPUT_FILE = PROCESSED_DIR / "existing_facilities_geocoded.parquet"


def load_school_locations(path: Path):
    schools = pl.read_csv(
        path,
        separator=",",
        schema_overrides={
            "Código INEP": pl.String,
            "Latitude": pl.Float64,
            "Longitute": pl.Float64,
        },
    )

    return schools.select(
        [
            "Código INEP",
            "Escola",
            "Município",
            "Localização",
            "Endereço",
            "Dependência Administrativa",
            "Latitude",
            "Longitude",
        ]
    ).rename(
        {
            "Código INEP": "facility_id",
            "Escola": "school_name_source",
            "Município": "municipality",
            "Localização": "school_location_type",
            "Endereço": "address",
            "Dependência Administrativa": ("administrative_dependency"),
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )


def validate_school_locations(schools: pl.DataFrame):
    duplicates = schools.group_by("facility_id").len().filter(pl.col("len") > 1)

    if duplicates.height > 0:
        raise ValueError(f"existem códigos inep duplicados:\n{duplicates}")


def build_geocoded_facilities(facilities: pl.DataFrame, school_locations: pl.DataFrame):
    facilities = facilities.with_columns(pl.col("facility_id").cast(pl.String))

    result = facilities.join(
        school_locations, on="facility_id", how="left", validate="1:1"
    )

    return result


def validate_result(df: pl.DataFrame):
    missing_location = df.filter(
        pl.col("latitude").is_null() | pl.col("longitude").is_null()
    )

    print()
    print(f"facilities: {df.height}")
    print(f"missing coordinates: {missing_location.height}")

    if missing_location.height > 0:
        print()
        print(missing_location.select(["facility_id", "facility_name"]))

    invalid_municipality = df.filter(pl.col("municipality") != "Nova Iguaçu")

    if invalid_municipality.height > 0:
        raise ValueError(
            "Existem instalações associadas a outro município:\n"
            f"{
                invalid_municipality.select(
                    [
                        'facility_id',
                        'facility_name',
                        'municipality',
                    ]
                )
            }"
        )


def print_summary(
    df: pl.DataFrame,
) -> None:
    print()
    print("Geocoded facilities")
    print("--------------------")

    print(
        df.select(
            [
                "facility_id",
                "facility_name",
                "address",
                "latitude",
                "longitude",
                "final_capacity",
            ]
        )
    )


def main():
    facilities = pl.read_parquet(FACILITIES_FILE)
    school_locations = load_school_locations(SCHOOLS_FILE)

    validate_school_locations(school_locations)

    result = build_geocoded_facilities(facilities, school_locations)

    validate_result(result)

    result.write_parquet(OUTPUT_FILE, compression="zstd")

    print_summary(result)

    print()
    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
