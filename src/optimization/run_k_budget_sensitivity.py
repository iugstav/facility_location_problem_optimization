import polars as pl

from src.config import PROCESSED_DIR

from solve_pmedian_scenarios import run_scenarios

BUDGET_VALUES = [
    15_000_000.0,
    17_000_000.0,
    20_000_000.0,
    23_000_000.0,
    25_000_000.0,
    30_000_000.0,
    # 35_000_000.0,
    # 40_000_000.0,
    # 50_000_000.0,
    # 60_000_000.0,
    # 70_000_000.0,
]

K_VALUES = [
    50,
    100,
    # 150,
    # 200,
]

# P_VALUES = [
#     5,
#     10,
#     15,
#     20,
#     22,
#     23,
#     25,
#     27,
# ]

OUTPUT_DIR = PROCESSED_DIR / "optimization" / "full_sensitivity"

SUMMARY_FILE = OUTPUT_DIR / "full_sensitivity_summary.parquet"


def budget_slug(budget: float):
    value = budget / 1_000_000
    if value.is_integer():
        return f"{int(value)}m"

    return f"{value:.1f}".replace(".", "_") + "m"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for budget in BUDGET_VALUES:
        for k in K_VALUES:
            slug = budget_slug(budget)
            results_dir = OUTPUT_DIR / f"budget_{slug}" / f"k_{k:03d}"

            print()
            print("=" * 80)
            print(f"BUDGET=R$ {budget:,.2f} | K={k}")
            print("=" * 80)

            summary = run_scenarios(
                k_nearest_candidates=k,
                budget=budget,
                p_values=None,
                results_dir=results_dir,
            )

            summary = summary.with_columns(
                pl.lit(budget).cast(pl.Float64).alias("experiment_budget"),
                pl.lit(k).cast(pl.Int64).alias("experiment_k"),
            )

            summaries.append(summary)

    sensitivity = pl.concat(
        summaries,
        how="vertical_relaxed",
    ).sort(
        [
            "experiment_budget",
            "experiment_k",
            "p",
        ]
    )

    sensitivity.write_parquet(
        SUMMARY_FILE,
        compression="zstd",
    )

    # sensitivity.write_csv(OUTPUT_DIR / "full_sensitivity_summary.csv")

    print()
    print("=" * 80)
    print("FULL SENSITIVITY COMPLETED")
    print("=" * 80)

    print()
    print(
        sensitivity.select(
            [
                "experiment_budget",
                "experiment_k",
                "p",
                "coverage",
                "average_travel_minutes",
                "investment",
                "type_1_count",
                "type_2_count",
                "new_capacity",
            ]
        )
    )


if __name__ == "__main__":
    main()
