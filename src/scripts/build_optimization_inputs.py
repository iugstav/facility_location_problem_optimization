import polars as pl

from src.config import PROCESSED_DIR

NEIGHBORHOOD_POPULATION_FILE = PROCESSED_DIR / "neighborhood_population_2025.parquet"

EXISTING_FACILITIES_FILE = PROCESSED_DIR / "existing_facilities_with_capacity.parquet"

EXISTING_COSTS_FILE = PROCESSED_DIR / "neighborhood_facility_costs.parquet"

CANDIDATE_COSTS_FILE = PROCESSED_DIR / "neighborhood_candidate_costs.parquet"

CANDIDATES_FILE = PROCESSED_DIR / "candidate_locations.parquet"

FACILITY_TYPES_FILE = PROCESSED_DIR / "facility_types.parquet"

OUTPUT_DIR = PROCESSED_DIR / "optimization"


TARGET_COVERAGE = 0.60


def load_inputs():
    neighborhoods = pl.read_parquet(NEIGHBORHOOD_POPULATION_FILE)

    existing_facilities = pl.read_parquet(EXISTING_FACILITIES_FILE)

    existing_costs = pl.read_parquet(EXISTING_COSTS_FILE)

    candidate_costs = pl.read_parquet(CANDIDATE_COSTS_FILE)

    candidates = pl.read_parquet(CANDIDATES_FILE)

    facility_types = pl.read_parquet(FACILITY_TYPES_FILE)

    return (
        neighborhoods,
        existing_facilities,
        existing_costs,
        candidate_costs,
        candidates,
        facility_types,
    )


def build_demand(neighborhoods: pl.DataFrame):
    population_column = "estimated_population_0_to_3_2025"

    if population_column not in neighborhoods.columns:
        raise ValueError(f"coluna {population_column} faltante em neighborhoods")

    demand = (
        neighborhoods.select(
            ["neighborhood_id", "neighborhood_name", population_column]
        )
        .with_columns((pl.col(population_column) * TARGET_COVERAGE).alias("demand"))
        .sort("neighborhood_id")
    )

    return demand


def validate_inputs(
    demand: pl.DataFrame,
    existing_facilities: pl.DataFrame,
    existing_costs: pl.DataFrame,
    candidate_costs: pl.DataFrame,
    candidates: pl.DataFrame,
    facility_types: pl.DataFrame,
) -> None:

    n_neighborhoods = demand.get_column("neighborhood_id").n_unique()

    n_existing = existing_facilities.get_column("facility_id").n_unique()

    n_candidates = candidates.get_column("candidate_id").n_unique()

    n_types = facility_types.get_column("facility_type").n_unique()

    expected_existing_pairs = n_neighborhoods * n_existing

    expected_candidate_pairs = n_neighborhoods * n_candidates

    if existing_costs.height != expected_existing_pairs:
        raise ValueError("Existing cost matrix is incomplete.")

    if candidate_costs.height != expected_candidate_pairs:
        raise ValueError("Candidate cost matrix is incomplete.")

    if n_types != 2:
        raise ValueError("Expected exactly two facility types.")

    if demand.filter(pl.col("demand") <= 0).height > 0:
        raise ValueError("Invalid non-positive demand.")

    if existing_facilities.filter(pl.col("final_capacity") <= 0).height > 0:
        raise ValueError("Invalid existing facility capacity.")

    if (
        facility_types.filter(
            (pl.col("capacity") <= 0) | (pl.col("construction_cost") <= 0)
        ).height
        > 0
    ):
        raise ValueError("Invalid facility type parameters.")

    print()
    print("Optimization input summary")
    print("--------------------------")

    print(f"neighborhoods: {n_neighborhoods}")

    print(f"existing facilities: {n_existing}")

    print(f"candidate locations: {n_candidates}")

    print(f"facility types: {n_types}")

    print(f"total demand: {demand['demand'].sum():.4f}")

    print(f"existing capacity: {existing_facilities['final_capacity'].sum():.4f}")


def save_inputs(
    demand: pl.DataFrame,
    existing_facilities: pl.DataFrame,
    existing_costs: pl.DataFrame,
    candidate_costs: pl.DataFrame,
    candidates: pl.DataFrame,
    facility_types: pl.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    demand.write_parquet(OUTPUT_DIR / "demand.parquet")

    existing_facilities.write_parquet(OUTPUT_DIR / "existing_facilities.parquet")

    existing_costs.write_parquet(OUTPUT_DIR / "existing_costs.parquet")

    candidate_costs.write_parquet(OUTPUT_DIR / "candidate_costs.parquet")

    candidates.write_parquet(OUTPUT_DIR / "candidates.parquet")

    facility_types.write_parquet(OUTPUT_DIR / "facility_types.parquet")


def main() -> None:

    (
        neighborhoods,
        existing_facilities,
        existing_costs,
        candidate_costs,
        candidates,
        facility_types,
    ) = load_inputs()

    demand = build_demand(neighborhoods)

    validate_inputs(
        demand,
        existing_facilities,
        existing_costs,
        candidate_costs,
        candidates,
        facility_types,
    )

    save_inputs(
        demand,
        existing_facilities,
        existing_costs,
        candidate_costs,
        candidates,
        facility_types,
    )

    print()
    print(f"saved optimization inputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
