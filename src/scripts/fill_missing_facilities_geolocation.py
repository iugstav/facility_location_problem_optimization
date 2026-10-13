import time

import polars as pl
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from src.config import PROCESSED_DIR

INPUT_FILE = PROCESSED_DIR / "existing_facilities_geocoded.parquet"

OUTPUT_FILE = PROCESSED_DIR / "existing_facilities_geocoded_final.parquet"

CITY = "Nova Iguaçu"
STATE = "Rio de Janeiro"
COUNTRY = "Brasil"


def build_query(
    address: str | None,
) -> str | None:
    if address is None:
        return None

    address = address.strip()

    if not address:
        return None

    return f"{address}, {CITY}, {STATE}, {COUNTRY}"


def geocode_missing_locations(df: pl.DataFrame):
    geolocator = Nominatim(user_agent="nova-iguacu-creche-location-project")

    geocode = RateLimiter(
        geolocator.geocode, min_delay_seconds=1.1, swallow_exceptions=True
    )

    rows = []

    for row in df.iter_rows(named=True):
        lat = row["latitude"]
        lon = row["longitude"]

        if lat is not None and lon is not None:
            row["coordinate_source"] = "school_table"
            rows.append(row)
            continue

        query = build_query(row.get("address"))

        if query is None:
            row["coordinate_source"] = "missing_address"
            rows.append(row)
            continue

        print(f"geocoding: {row['facility_name']}")

        print(f"query: {query}")

        location = geocode(query)

        if location is None:
            row["coordinate_source"] = "geocoding_failed"

            rows.append(row)
            continue

        row["latitude"] = float(location.latitude)

        row["longitude"] = float(location.longitude)

        row["coordinate_source"] = "geocoded_address"

        print(f"found: {location.latitude}, {location.longitude}")

        time.sleep(0.2)

        rows.append(row)

    return pl.DataFrame(rows)


def validate_coordinates(
    df: pl.DataFrame,
) -> None:

    missing = df.filter(pl.col("latitude").is_null() | pl.col("longitude").is_null())

    print()
    print(f"facilities: {df.height}")

    print(f"missing coordinates: {missing.height}")

    if missing.height > 0:
        print()
        print(
            missing.select(
                [
                    "facility_id",
                    "facility_name",
                    "address",
                    "coordinate_source",
                ]
            )
        )


def print_coordinate_sources(
    df: pl.DataFrame,
) -> None:

    print()
    print("Coordinate sources")
    print("------------------")

    print(df.group_by("coordinate_source").len().sort("coordinate_source"))


def main() -> None:

    facilities = pl.read_parquet(INPUT_FILE)

    facilities = facilities.with_columns(
        [
            pl.col("latitude").cast(pl.Float64, strict=False),
            pl.col("longitude").cast(pl.Float64, strict=False),
        ]
    )

    result = geocode_missing_locations(facilities)

    validate_coordinates(result)

    print_coordinate_sources(result)

    result.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    print()
    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
