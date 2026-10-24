import itertools

import polars as pl

from src.config import PROCESSED_DIR

INPUT_DIR = PROCESSED_DIR / "optimization" / "sensitivity"

SUMMARY_FILE = INPUT_DIR / "k_sensitivity_summary.parquet"

OUTPUT_FILE = INPUT_DIR / "k_sensivity_analysis.parquet"

JACCARD_FILE = INPUT_DIR / "k_sensivity_jaccard.parquet"

REFERENCE_K = 200


def load_summary():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"file {SUMMARY_FILE} not found")

    return pl.read_parquet(SUMMARY_FILE)


def validate_summary(df: pl.DataFrame):
    required = {
        "k",
        "p",
        "coverage",
        "average_travel_minutes",
        "investment",
        "type_1_count",
        "type_2_count",
        "new_capacity",
        "served_demand",
        "unmet_demand",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def add_reference_deltas(df: pl.DataFrame):
    reference = df.filter(pl.col("k") == REFERENCE_K).select(
        [
            "p",
            pl.col("coverage").alias("coverage_reference"),
            pl.col("average_travel_minutes").alias("travel_reference"),
            pl.col("new_capacity").alias("capacity_reference"),
        ]
    )

    result = (
        df.join(reference, on="p", how="left")
        .with_columns(
            (pl.col("coverage") - pl.col("coverage_reference")).alias("delta_coverage"),
            (pl.col("average_travel_minutes") - pl.col("travel_reference")).alias(
                "delta_travel_minutes"
            ),
            (pl.col("new_capacity") - pl.col("capacity_reference")).alias(
                "delta_capacity"
            ),
            ((pl.col("coverage") - pl.col("coverage_reference")) * 100).alias(
                "delta_coverage_pp"
            ),
        )
        .sort(["p", "k"])
    )

    return result


def load_opened_candidates(k: int, p: int):
    file = INPUT_DIR / f"k_{k:03d}" / f"p_{p:02d}" / "opened_facilities.parquet"

    if not file.exists():
        raise FileNotFoundError(f"Opened facilities not found: {file}")

    df = pl.read_parquet(file)

    return set(df.get_column("candidate_id").cast(pl.String).to_list())


def jaccard(a: set[str], b: set[str]):
    union = a | b
    if not union:
        return 1.0

    return len(a & b) / len(union)


def build_jaccard_table(summary: pl.DataFrame):
    k_values = summary.get_column("k").unique().sort().to_list()
    p_values = summary.get_column("p").unique().sort().to_list()

    rows = []
    for p in p_values:
        candidate_sets = {k: load_opened_candidates(k, p) for k in k_values}

        for k1, k2 in itertools.combinations(k_values, 2):
            set_1 = candidate_sets[k1]
            set_2 = candidate_sets[k2]

            intersection = set_1 & set_2
            union = set_1 | set_2

            rows.append(
                {
                    "p": p,
                    "k_1": k1,
                    "k_2": k2,
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "jaccard": jaccard(set_1, set_2),
                }
            )

    return pl.DataFrame(rows).sort(["p", "k_1", "k_2"])


def build_reference_jaccard(jaccard_df: pl.DataFrame):
    return jaccard_df.filter(pl.col("k_2") == REFERENCE_K).sort(["p", "k_1"])


def report(
    analysis: pl.DataFrame,
    jaccard_df: pl.DataFrame,
):

    print()
    print("=" * 70)
    print("K-NEAREST SENSITIVITY ANALYSIS")
    print("=" * 70)

    print()
    print(f"reference K: {REFERENCE_K}")

    print()
    print("Metric comparison against reference")
    print("----------------------------")

    print(
        analysis.select(
            [
                "p",
                "k",
                "coverage",
                "delta_coverage_pp",
                "average_travel_minutes",
                "delta_travel_minutes",
                "new_capacity",
                "delta_capacity",
            ]
        )
    )

    print()
    print("Jaccard against reference")
    print("-------------------------")

    print(build_reference_jaccard(jaccard_df))


def main():

    summary = load_summary()

    validate_summary(summary)

    analysis = add_reference_deltas(summary)

    jaccard_df = build_jaccard_table(summary)

    analysis.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    analysis.write_csv(INPUT_DIR / "k_sensitivity_analysis.csv")

    jaccard_df.write_parquet(
        JACCARD_FILE,
        compression="zstd",
    )

    jaccard_df.write_csv(INPUT_DIR / "k_sensitivity_jaccard.csv")

    report(
        analysis,
        jaccard_df,
    )


if __name__ == "__main__":
    main()
