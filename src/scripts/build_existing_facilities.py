from pathlib import Path

from src.config import INEP_DIR, PROCESSED_DIR
from src.data.inep import (
    load_schools,
    load_enrollments,
    load_classes,
    prepare_creche_schools,
    build_existing_facilities,
    add_class_statistics,
)

import polars as pl

SCHOOLS_FILE = Path(INEP_DIR / "Tabela_Escola_2025_V2.csv")
ENROLLMENTS_FILE = Path(INEP_DIR / "Tabela_Matricula_2025_V2.csv")
CLASSES_FILE = Path(INEP_DIR / "Tabela_Turma_2025_V2.csv")

OUTPUT_FILE = Path(PROCESSED_DIR / "existing_facilities.parquet")


def main():
    schools = prepare_creche_schools(load_schools(SCHOOLS_FILE))
    enrollments = load_enrollments(ENROLLMENTS_FILE)
    classes = load_classes(CLASSES_FILE)

    facilities = build_existing_facilities(schools, enrollments, classes)
    facilities = add_class_statistics(facilities)

    # print("creches municipais: ", facilities.height)
    # print(
    #     facilities.select(
    #         pl.col("creche_enrollment").sum().alias("total_creche_enrollment"),
    #         pl.col("creche_class_count").sum().alias("total_creche_classes"),
    #     )
    # )

    facilities = facilities.with_columns(
        ((pl.col("creche_enrollment") > 0) & (pl.col("creche_class_count") == 0)).alias(
            "class_count_inconsistent"
        )
    )

    valid_facilities = facilities.filter(~pl.col("class_count_inconsistent"))

    print(
        valid_facilities.select(
            pl.col("students_per_creche_class").quantile(0.75).alias("q75"),
            pl.col("students_per_creche_class").quantile(0.90).alias("q90"),
            pl.col("students_per_creche_class").quantile(0.95).alias("q95"),
        )
    )

    facilities.write_parquet(OUTPUT_FILE, compression="zstd")


if __name__ == "__main__":
    main()
