import polars as pl

from src.config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "optimization" / "results" / "scenario_summary.parquet"

OUTPUT_DIR = PROCESSED_DIR / "optimization" / "analysis"

ANALYSIS_FILE = OUTPUT_DIR / "scenario_analysis.parquet"

PARETO_FILE = OUTPUT_DIR / "pareto_frontier.parquet"


# ============================================================
# Loading
# ============================================================


def load_scenarios() -> pl.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Scenario summary not found: {INPUT_FILE}")

    scenarios = pl.read_parquet(INPUT_FILE)

    if scenarios.height == 0:
        raise ValueError("scenario_summary.parquet is empty.")

    return scenarios


# ============================================================
# Validation
# ============================================================


def validate_scenarios(
    scenarios: pl.DataFrame,
) -> None:

    required_columns = {
        "p",
        "budget",
        "investment",
        "budget_usage",
        "type_1_count",
        "type_2_count",
        "new_capacity",
        "total_demand",
        "served_demand",
        "unmet_demand",
        "coverage",
        "weighted_travel_minutes",
        "average_travel_minutes",
    }

    missing = required_columns - set(scenarios.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if scenarios.get_column("p").n_unique() != scenarios.height:
        raise ValueError("There are duplicate p scenarios.")

    if scenarios.filter(pl.col("investment") <= 0).height > 0:
        raise ValueError("There are scenarios with non-positive investment.")

    if scenarios.filter((pl.col("coverage") < 0) | (pl.col("coverage") > 1)).height > 0:
        raise ValueError("Coverage outside [0, 1].")


# ============================================================
# Derived indicators
# ============================================================


def add_indicators(
    scenarios: pl.DataFrame,
) -> pl.DataFrame:

    return scenarios.with_columns(
        # -----------------------------------------------
        # Investment in millions
        # -----------------------------------------------
        (pl.col("investment") / 1_000_000).alias("investment_million"),
        # -----------------------------------------------
        # Capacity obtained per R$ 1 million
        # -----------------------------------------------
        (pl.col("new_capacity") / (pl.col("investment") / 1_000_000)).alias(
            "capacity_per_million"
        ),
        # -----------------------------------------------
        # Served demand per R$ 1 million
        # -----------------------------------------------
        (pl.col("served_demand") / (pl.col("investment") / 1_000_000)).alias(
            "served_per_million"
        ),
        # -----------------------------------------------
        # Remaining budget
        # -----------------------------------------------
        (pl.col("budget") - pl.col("investment")).alias("remaining_budget"),
        # -----------------------------------------------
        # Share of Type 1 facilities
        # -----------------------------------------------
        (pl.col("type_1_count") / pl.col("p")).alias("type_1_share"),
        # -----------------------------------------------
        # Share of Type 2 facilities
        # -----------------------------------------------
        (pl.col("type_2_count") / pl.col("p")).alias("type_2_share"),
    ).sort("p")


# ============================================================
# Marginal indicators
# ============================================================


def add_marginal_indicators(
    scenarios: pl.DataFrame,
) -> pl.DataFrame:

    scenarios = scenarios.sort("p")

    return scenarios.with_columns(
        # ---------------------------------------------------
        # Change in coverage when increasing p
        # ---------------------------------------------------
        (pl.col("coverage") - pl.col("coverage").shift(1)).alias("delta_coverage"),
        # ---------------------------------------------------
        # Change in served demand
        # ---------------------------------------------------
        (pl.col("served_demand") - pl.col("served_demand").shift(1)).alias(
            "delta_served_demand"
        ),
        # ---------------------------------------------------
        # Change in capacity
        # ---------------------------------------------------
        (pl.col("new_capacity") - pl.col("new_capacity").shift(1)).alias(
            "delta_new_capacity"
        ),
        # ---------------------------------------------------
        # Change in average travel time
        # ---------------------------------------------------
        (
            pl.col("average_travel_minutes") - pl.col("average_travel_minutes").shift(1)
        ).alias("delta_average_travel_minutes"),
        # ---------------------------------------------------
        # Change in investment
        # ---------------------------------------------------
        (pl.col("investment") - pl.col("investment").shift(1)).alias(
            "delta_investment"
        ),
    )


# ============================================================
# Pareto frontier
# ============================================================


def build_pareto_frontier(
    scenarios: pl.DataFrame,
) -> pl.DataFrame:
    """
    A scenario A dominates B if:

        coverage_A >= coverage_B

    and

        travel_A <= travel_B

    with at least one strict inequality.

    We therefore want:
        maximize coverage
        minimize average travel time
    """

    rows = scenarios.to_dicts()

    pareto_rows = []

    for current in rows:
        dominated = False

        current_coverage = float(current["coverage"])

        current_travel = float(current["average_travel_minutes"])

        for other in rows:
            if other["p"] == current["p"]:
                continue

            other_coverage = float(other["coverage"])

            other_travel = float(other["average_travel_minutes"])

            no_worse = (
                other_coverage >= current_coverage and other_travel <= current_travel
            )

            strictly_better = (
                other_coverage > current_coverage or other_travel < current_travel
            )

            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            pareto_rows.append(current)

    return pl.DataFrame(pareto_rows).sort("p")


# ============================================================
# Rankings
# ============================================================


def get_best_coverage(
    scenarios: pl.DataFrame,
    n: int = 5,
) -> pl.DataFrame:

    return scenarios.sort(
        [
            "coverage",
            "average_travel_minutes",
        ],
        descending=[
            True,
            False,
        ],
    ).head(n)


def get_best_accessibility(
    scenarios: pl.DataFrame,
    n: int = 5,
) -> pl.DataFrame:

    return scenarios.sort(
        [
            "average_travel_minutes",
            "coverage",
        ],
        descending=[
            False,
            True,
        ],
    ).head(n)


def get_best_capacity_efficiency(
    scenarios: pl.DataFrame,
    n: int = 5,
) -> pl.DataFrame:

    return scenarios.sort(
        "capacity_per_million",
        descending=True,
    ).head(n)


# ============================================================
# Structural break
# ============================================================


def find_coverage_peak(
    scenarios: pl.DataFrame,
) -> dict:

    best = scenarios.sort(
        [
            "coverage",
            "average_travel_minutes",
        ],
        descending=[
            True,
            False,
        ],
    ).row(
        0,
        named=True,
    )

    return best


def find_first_coverage_decline(
    scenarios: pl.DataFrame,
):

    decline = scenarios.filter(pl.col("delta_coverage") < 0).sort("p")

    if decline.height == 0:
        return None

    return decline.row(
        0,
        named=True,
    )


# ============================================================
# Reporting
# ============================================================


def print_selected_columns(
    df: pl.DataFrame,
) -> None:

    columns = [
        "p",
        "investment",
        "type_1_count",
        "type_2_count",
        "new_capacity",
        "coverage",
        "average_travel_minutes",
        "capacity_per_million",
    ]

    print(df.select(columns))


def report(
    scenarios: pl.DataFrame,
    pareto: pl.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("P-MEDIAN SCENARIO ANALYSIS")
    print("=" * 70)

    print()
    print(f"number of scenarios: {scenarios.height}")

    print(f"p range: {scenarios['p'].min()} to {scenarios['p'].max()}")

    # --------------------------------------------------------
    # Peak coverage
    # --------------------------------------------------------

    peak = find_coverage_peak(scenarios)

    print()
    print("Maximum coverage")
    print("----------------")

    print(f"p: {peak['p']}")

    print(f"coverage: {peak['coverage']:.4%}")

    print(f"served demand: {peak['served_demand']:.4f}")

    print(f"unmet demand: {peak['unmet_demand']:.4f}")

    print(f"investment: R$ {peak['investment']:,.2f}")

    print(f"type 1: {peak['type_1_count']}")

    print(f"type 2: {peak['type_2_count']}")

    print(f"average travel: {peak['average_travel_minutes']:.4f} min")

    # --------------------------------------------------------
    # First coverage decline
    # --------------------------------------------------------

    decline = find_first_coverage_decline(scenarios)

    print()
    print("First coverage decline")
    print("----------------------")

    if decline is None:
        print("No decrease in coverage was detected.")

    else:
        print(f"first decline at p={decline['p']}")

        print(f"delta coverage: {decline['delta_coverage']:.6f}")

        print(f"delta capacity: {decline['delta_new_capacity']:.4f}")

        print(f"type 1: {decline['type_1_count']}")

        print(f"type 2: {decline['type_2_count']}")

    # --------------------------------------------------------
    # Rankings
    # --------------------------------------------------------

    print()
    print("Top 5 by coverage")
    print("-----------------")

    print_selected_columns(get_best_coverage(scenarios))

    print()
    print("Top 5 by accessibility")
    print("----------------------")

    print_selected_columns(get_best_accessibility(scenarios))

    print()
    print("Top 5 by capacity efficiency")
    print("----------------------------")

    print_selected_columns(get_best_capacity_efficiency(scenarios))

    # --------------------------------------------------------
    # Pareto frontier
    # --------------------------------------------------------

    print()
    print("Pareto frontier")
    print("---------------")

    print_selected_columns(pareto)


# ============================================================
# Save
# ============================================================


def save_results(
    scenarios: pl.DataFrame,
    pareto: pl.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios.write_parquet(
        ANALYSIS_FILE,
        compression="zstd",
    )

    scenarios.write_csv(OUTPUT_DIR / "scenario_analysis.csv")

    pareto.write_parquet(
        PARETO_FILE,
        compression="zstd",
    )

    pareto.write_csv(OUTPUT_DIR / "pareto_frontier.csv")


# ============================================================
# Main
# ============================================================


def main():

    scenarios = load_scenarios()

    validate_scenarios(scenarios)

    scenarios = add_indicators(scenarios)

    scenarios = add_marginal_indicators(scenarios)

    pareto = build_pareto_frontier(scenarios)

    report(
        scenarios,
        pareto,
    )

    save_results(
        scenarios,
        pareto,
    )

    print()
    print(f"analysis saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
