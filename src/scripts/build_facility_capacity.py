from pathlib import Path

import polars as pl

from src.config import PROCESSED_DIR

INPUT_FILE = PROCESSED_DIR / "existing_facilities.parquet"

OUTPUT_FILE = PROCESSED_DIR / "existing_facilities_with_capacity.parquet"


def load_existing_facilities(path: Path):
    df = pl.read_parquet(path)

    req_cols = {
        "facility_id",
        "facility_name",
        "creche_enrollment",
        "creche_class_count",
    }

    missing = req_cols - set(df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes em existing_facilities: {missing}")

    return df


def validate_existing_facilities(
    df: pl.DataFrame,
) -> None:
    duplicated_facilities = df.group_by("facility_id").len().filter(pl.col("len") > 1)

    if duplicated_facilities.height > 0:
        raise ValueError(f"existem facility_id duplicados:\n{duplicated_facilities}")

    negative_enrollment = df.filter(pl.col("creche_enrollment") < 0).height

    if negative_enrollment > 0:
        raise ValueError("existem matrículas negativas.")

    negative_classes = df.filter(pl.col("creche_class_count") < 0).height

    if negative_classes > 0:
        raise ValueError("existem números de turmas negativos.")


def add_capacity_status(df: pl.DataFrame):
    return df.with_columns(
        pl.when(pl.col("creche_class_count") > 0)
        .then(pl.lit("resolved"))
        .when((pl.col("creche_enrollment") > 0) & (pl.col("creche_class_count") == 0))
        .then(pl.lit("capacity_unresolved"))
        .otherwise(pl.lit("no_creche_activity"))
        .alias("capacity_status")
    )


def add_observed_class_size(df: pl.DataFrame):
    return df.with_columns(
        pl.when(pl.col("creche_class_count") > 0)
        .then(pl.col("creche_enrollment") / pl.col("creche_class_count"))
        .otherwise(None)
        .alias("observed_students_per_class")
    )


def estimate_capacity_scenarios(df: pl.DataFrame):
    class_size = df.filter(pl.col("creche_class_count") > 0).with_columns(
        (pl.col("creche_enrollment") / pl.col("creche_class_count")).alias(
            "students_per_class"
        )
    )

    stats = class_size.select(
        pl.col("students_per_class").quantile(0.75).alias("q75"),
        pl.col("students_per_class").quantile(0.90).alias("q90"),
        pl.col("students_per_class").quantile(0.95).alias("q95"),
    ).row(0, named=True)

    return {
        "conservative": stats["q75"],
        "central": stats["q90"],
        "expanded": stats["q95"],
    }


def add_capacity_scenarios(df: pl.DataFrame, cap_scenarios: dict[str, float]):
    expressions = []
    for scenario_name, cap_per_class in cap_scenarios.items():
        expressions.append(
            pl.when(pl.col("creche_class_count") > 0)
            .then(pl.col("creche_class_count") * cap_per_class)
            .otherwise(None)
            .alias(f"estimated_capacity_{scenario_name}")
        )

    return df.with_columns(expressions)


def add_utilization_rates(df: pl.DataFrame, cap_scenarios: dict[str, float]):
    expressions = []
    for scenario_name in cap_scenarios:
        cap_col = f"estimated_capacity_{scenario_name}"

        expressions.append(
            pl.when(pl.col(cap_col).is_not_null() & (pl.col(cap_col) > 0))
            .then(pl.col("creche_enrollment") / pl.col(cap_col))
            .otherwise(None)
            .alias(f"utilization_{scenario_name}")
        )

    return df.with_columns(expressions)


def add_final_capacity(
    df: pl.DataFrame,
) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("capacity_status") == "resolved")
        .then(pl.col("estimated_capacity_central"))
        .when(pl.col("capacity_status") == "capacity_unresolved")
        .then(pl.col("creche_enrollment"))
        .otherwise(None)
        .alias("final_capacity")
    )


def add_capacity_source(
    df: pl.DataFrame,
) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("capacity_status") == "resolved")
        .then(pl.lit("empirical_q90"))
        .when(pl.col("capacity_status") == "capacity_unresolved")
        .then(pl.lit("observed_enrollment_fallback"))
        .otherwise(pl.lit("unresolved"))
        .alias("capacity_source")
    )


def build_capacity_dataset(
    facilities: pl.DataFrame, cap_scenarios: dict[str, float]
) -> pl.DataFrame:
    result = (
        facilities.pipe(add_capacity_status)
        .pipe(add_observed_class_size)
        .pipe(lambda df: add_capacity_scenarios(df, cap_scenarios))
        .pipe(lambda df: add_utilization_rates(df, cap_scenarios))
        .pipe(add_final_capacity)
        .pipe(add_capacity_source)
    )

    return result


def validate_capacity_dataset(
    df: pl.DataFrame, cap_scenarios: dict[str, float]
) -> None:
    """
    Verifica se capacidades estimadas são válidas.
    """

    for scenario_name in cap_scenarios:
        capacity_column = f"estimated_capacity_{scenario_name}"

        invalid_capacity = df.filter(
            pl.col(capacity_column).is_not_null() & (pl.col(capacity_column) <= 0)
        ).height

        if invalid_capacity > 0:
            raise ValueError(f"Existem capacidades inválidas em {capacity_column}.")


def print_summary(df: pl.DataFrame, cap_scenarios: dict[str, float]) -> None:
    print()
    print("Facility capacity summary")
    print("-------------------------")

    print(df.group_by("capacity_status").len().sort("capacity_status"))

    print()
    print("Observed class size")
    print("-------------------")

    print(
        df.filter(pl.col("observed_students_per_class").is_not_null()).select(
            pl.col("observed_students_per_class").min().alias("min"),
            pl.col("observed_students_per_class").median().alias("median"),
            pl.col("observed_students_per_class").mean().alias("mean"),
            pl.col("observed_students_per_class").max().alias("max"),
        )
    )

    print()
    print("Capacity scenarios")
    print("------------------")

    for scenario_name, capacity_per_class in cap_scenarios.items():
        capacity_column = f"estimated_capacity_{scenario_name}"

        utilization_column = f"utilization_{scenario_name}"

        total_capacity = df.select(pl.col(capacity_column).sum()).item()

        overloaded_units = df.filter(pl.col(utilization_column) > 1).height

        print()
        print(f"{scenario_name}: {capacity_per_class} crianças/turma")
        print(f"total estimated capacity: {total_capacity}")
        print(f"units above estimated capacity: {overloaded_units}")


def main() -> None:
    print("building facility capacity dataset")

    facilities = load_existing_facilities(INPUT_FILE)

    validate_existing_facilities(facilities)

    cap_scenarios = estimate_capacity_scenarios(facilities)

    facilities_with_capacity = build_capacity_dataset(facilities, cap_scenarios)

    print(facilities_with_capacity.columns)

    validate_capacity_dataset(facilities_with_capacity, cap_scenarios)

    facilities_with_capacity.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    print_summary(facilities_with_capacity, cap_scenarios)

    print(
        facilities_with_capacity.filter(pl.col("utilization_central") > 1)
        .select(
            [
                "facility_id",
                "facility_name",
                "creche_enrollment",
                "creche_class_count",
                "estimated_capacity_central",
                "utilization_central",
            ]
        )
        .sort(
            "utilization_central",
            descending=True,
        )
    )

    print()
    print("saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
