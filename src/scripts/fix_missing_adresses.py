from pathlib import Path

import polars as pl

from src.config import PROCESSED_DIR, INEP_DIR

FACILITIES_FILE = PROCESSED_DIR / "existing_facilities_geocoded_final.parquet"

OVERRIDES_FILE = INEP_DIR / "facility_location_overrides.csv"

OUTPUT_FILE = PROCESSED_DIR / "existing_facilities_locations_final.parquet"


def load_overrides(path: Path):
    return (
        pl.read_csv(
            path,
            schema_overrides={
                "facility_id": pl.String,
                "latitude": pl.Float64,
                "longitude": pl.Float64,
            },
        )
        .select(
            [
                "facility_id",
                "address",
                "latitude",
                "longitude",
                "location_source",
                "location_notes",
            ]
        )
        .rename(
            {
                "address": "override_address",
                "latitude": "override_latitude",
                "longitude": "override_longitude",
            }
        )
    )


def apply_location_overrides(facilities: pl.DataFrame, overrides: pl.DataFrame):
    result = (
        facilities.with_columns(pl.col("facility_id").cast(pl.String))
        .join(overrides, on="facility_id", how="left", validate="1:1")
        .with_columns(
            pl.coalesce([pl.col("latitude"), pl.col("override_latitude")]).alias(
                "latitude_final"
            ),
            pl.coalesce([pl.col("longitude"), pl.col("override_longitude")]).alias(
                "longitude_final"
            ),
            pl.coalesce([pl.col("address"), pl.col("override_address")]).alias(
                "address_final"
            ),
        )
    ).with_columns(
        pl.when(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
        .then(pl.col("coordinate_source"))
        .when(
            pl.col("override_latitude").is_not_null()
            & pl.col("override_longitude").is_not_null()
        )
        .then(pl.col("location_source"))
        .otherwise(pl.lit("unresolved"))
        .alias("coordinate_source_final")
    )

    return result


def validate_final_locations(
    df: pl.DataFrame,
) -> None:
    unresolved = df.filter(
        pl.col("latitude_final").is_null() | pl.col("longitude_final").is_null()
    )

    print(f"facilities: {df.height}")

    print(f"unresolved locations: {unresolved.height}")

    if unresolved.height > 0:
        print(
            unresolved.select(
                [
                    "facility_id",
                    "facility_name",
                    "address_final",
                    "coordinate_source_final",
                ]
            )
        )


def main():
    facilities = pl.read_parquet(FACILITIES_FILE)
    overrides = load_overrides(OVERRIDES_FILE)

    result = apply_location_overrides(facilities, overrides)

    validate_final_locations(result)

    result.write_parquet(OUTPUT_FILE, compression="zstd")

    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
