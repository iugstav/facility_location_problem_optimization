import polars as pl

from src.config import PROCESSED_DIR

TRAVEL_TIMES_FILE = PROCESSED_DIR / "sector_facility_travel_times.parquet"

OUTPUT_FILE = PROCESSED_DIR / "neighborhood_facility_costs.parquet"


def load_travel_times():
    df = pl.read_parquet(TRAVEL_TIMES_FILE)
