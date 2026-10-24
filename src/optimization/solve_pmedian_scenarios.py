import shutil

import polars as pl
import pulp

from src.config import PROCESSED_DIR

INPUT_DIR = PROCESSED_DIR / "optimization"

RESULTS_DIR = INPUT_DIR / "results"

SUMMARY_FILE = RESULTS_DIR / "scenario_summary.parquet"


BUDGET = 15_000_000.0

k_nearest_candidates = 100
P_VALUES = None
EPSILON = 1e-4
SOLVER_MSG = True
CLEAR_PREVIOUS_RESULTS = True


def load_inputs():
    demand = pl.read_parquet(INPUT_DIR / "demand.parquet")
    existing_facilities = pl.read_parquet(INPUT_DIR / "existing_facilities.parquet")
    existing_costs = pl.read_parquet(INPUT_DIR / "existing_costs.parquet")
    candidate_costs = pl.read_parquet(INPUT_DIR / "candidate_costs.parquet")
    candidates = pl.read_parquet(INPUT_DIR / "candidates.parquet")
    facility_types = pl.read_parquet(INPUT_DIR / "facility_types.parquet")

    return (
        demand,
        existing_facilities,
        existing_costs,
        candidate_costs,
        candidates,
        facility_types,
    )


def build_parameter_maps(
    demand: pl.DataFrame,
    existing_facilities: pl.DataFrame,
    existing_costs: pl.DataFrame,
    candidate_costs: pl.DataFrame,
    facility_types_df: pl.DataFrame,
    k_nearest_candidates: int,
):
    neighborhoods = demand.get_column("neighborhood_id").cast(pl.String).to_list()

    existing_ids = (
        existing_facilities.get_column("facility_id").cast(pl.String).to_list()
    )
    candidate_ids = (
        candidate_costs.get_column("candidate_id")
        .cast(pl.String)
        .unique()
        .sort()
        .to_list()
    )
    facility_types = (
        facility_types_df.get_column("facility_type").cast(pl.String).to_list()
    )
    q = dict(
        zip(
            demand.get_column("neighborhood_id").cast(pl.String).to_list(),
            demand.get_column("demand").cast(pl.Float64).to_list(),
        )
    )
    existing_capacity = dict(
        zip(
            existing_facilities.get_column("facility_id").cast(pl.String).to_list(),
            existing_facilities.get_column("final_capacity").cast(pl.Float64).to_list(),
        )
    )
    type_capacity = dict(
        zip(
            facility_types_df.get_column("facility_type").cast(pl.String).to_list(),
            facility_types_df.get_column("capacity").cast(pl.Float64).to_list(),
        )
    )
    type_cost = dict(
        zip(
            facility_types_df.get_column("facility_type").cast(pl.String).to_list(),
            facility_types_df.get_column("construction_cost")
            .cast(pl.Float64)
            .to_list(),
        )
    )

    existing_neighborhoods = (
        existing_costs.get_column("neighborhood_id").cast(pl.String).to_list()
    )
    existing_facility_column = (
        existing_costs.get_column("facility_id").cast(pl.String).to_list()
    )
    existing_times = (
        existing_costs.get_column("travel_time_minutes").cast(pl.Float64).to_list()
    )

    d_existing = {
        (i, e): float(d)
        for i, e, d in zip(
            existing_neighborhoods, existing_facility_column, existing_times
        )
    }

    # candidate_neighborhoods = (
    #     candidate_costs.get_column("neighborhood_id").cast(pl.String).to_list()
    # )
    # candidate_column = (
    #     candidate_costs.get_column("candidate_id").cast(pl.String).to_list()
    # )
    # candidate_times = (
    #     candidate_costs.get_column("travel_time_minutes").cast(pl.Float64).to_list()
    # )

    nearest_costs, nearest_candidates = build_neares_candidate_sets(
        candidate_costs, k_nearest_candidates
    )

    nearest_neighborhoods = nearest_costs.get_column("neighborhood_id").to_list()

    nearest_candidate_ids = nearest_costs.get_column("candidate_id").to_list()

    nearest_times = nearest_costs.get_column("travel_time_minutes").to_list()

    d_candidate = {
        (i, j): float(d)
        for i, j, d in zip(
            nearest_neighborhoods,
            nearest_candidate_ids,
            nearest_times,
        )
    }
    return {
        "neighborhoods": neighborhoods,
        "existing_ids": existing_ids,
        "candidate_ids": candidate_ids,
        "facility_types": facility_types,
        "q": q,
        "existing_capacity": existing_capacity,
        "type_capacity": type_capacity,
        "type_cost": type_cost,
        "d_existing": d_existing,
        "d_candidate": d_candidate,
        "nearest_candidates": nearest_candidates,
    }


def validate_parameters(params, k_nearest_candidates: int):
    neighborhoods = params["neighborhoods"]
    existing_ids = params["existing_ids"]
    candidate_ids = params["candidate_ids"]
    nearest_candidates = params["nearest_candidates"]

    q = params["q"]
    existing_capacity = params["existing_capacity"]
    type_capacity = params["type_capacity"]
    type_cost = params["type_cost"]

    d_existing = params["d_existing"]
    d_candidate = params["d_candidate"]

    if len(neighborhoods) != 67:
        raise ValueError(f"Expected 67 neighborhoods, found {len(neighborhoods)}.")

    if len(existing_ids) != 28:
        raise ValueError(f"Expected 28 existing facilities, found {len(existing_ids)}.")

    if len(candidate_ids) != 1561:
        raise ValueError(f"Expected 1561 candidates, found {len(candidate_ids)}.")

    expected_existing_pairs = len(neighborhoods) * len(existing_ids)

    expected_candidate_arcs = sum(len(nearest_candidates[i]) for i in neighborhoods)

    if len(d_existing) != expected_existing_pairs:
        raise ValueError("Existing cost matrix is incomplete.")

    if len(d_candidate) != expected_candidate_arcs:
        raise ValueError(
            "Reduced candidate cost matrix has an unexpected number of arcs. "
            f"Expected {expected_candidate_arcs}, found {len(d_candidate)}."
        )

    if any(value <= 0 for value in q.values()):
        raise ValueError("Demand contains non-positive values.")

    if any(value <= 0 for value in existing_capacity.values()):
        raise ValueError("Existing capacity contains non-positive values.")

    if any(value <= 0 for value in type_capacity.values()):
        raise ValueError("Facility type capacity contains non-positive values.")

    if any(value <= 0 for value in type_cost.values()):
        raise ValueError("Facility type cost contains non-positive values.")

    for i in neighborhoods:
        if i not in nearest_candidates:
            raise ValueError(f"bairro {i} não tem arcos candidatos")

        if len(nearest_candidates[i]) == 0:
            raise ValueError(f"bairro {i} tem zero arcos candidatos")

        if len(nearest_candidates[i]) > k_nearest_candidates:
            raise ValueError(
                f"bairro {i} tem mais do que k_nearest_candidates arcos candidatos."
            )

        for j in nearest_candidates[i]:
            if (i, j) not in d_candidate:
                raise ValueError(
                    f"Missing travel cost for neighborhood {i} and candidate {j}."
                )

    total_candidate_arcs = sum(len(values) for values in nearest_candidates.values())

    print()
    print("Parameter validation")
    print("--------------------")
    print(f"neighborhoods: {len(neighborhoods)}")
    print(f"existing facilities: {len(existing_ids)}")
    print(f"candidate locations: {len(candidate_ids)}")
    print(f"facility types: {len(params['facility_types'])}")
    print(f"total demand: {sum(q.values()):.4f}")
    print(f"existing capacity: {sum(existing_capacity.values()):.4f}")
    print(f"candidate service arcs: {total_candidate_arcs}")


def get_p_values(candidate_ids, type_cost, budget):
    cheapest_cost = min(type_cost.values())
    p_max_budget = int(budget // cheapest_cost)
    p_max = min(p_max_budget, len(candidate_ids))

    if p_max < 1:
        raise ValueError(
            "o orçamento é insuficiente até para construir a facility mais barata"
        )

    if P_VALUES is None:
        p_values = list(range(1, p_max + 1))
    else:
        p_values = sorted(set(P_VALUES))
        invalid = [p for p in p_values if p < 1 or p > p_max]

        if invalid:
            raise ValueError(
                f"valores de P inviáveis: {invalid}. O P máximo viável abaixo do orçamento é {p_max}"
            )

    print()
    print("Scenario configuration")
    print("----------------------")

    print(f"budget: R$ {budget:,.2f}")

    print(f"cheapest facility: R$ {cheapest_cost:,.2f}")

    print(f"maximum P under budget: {p_max}")

    print(f"P values: {p_values}")

    return p_values


def build_neares_candidate_sets(candidate_costs: pl.DataFrame, k: int):
    if k <= 0:
        raise ValueError("k_nearest_candidates dever ser positivo")

    nearest = (
        candidate_costs.select(
            [
                pl.col("neighborhood_id").cast(pl.String),
                pl.col("candidate_id").cast(pl.String),
                pl.col("travel_time_minutes").cast(pl.Float64),
            ]
        )
        .sort(["neighborhood_id", "travel_time_minutes", "candidate_id"])
        .group_by("neighborhood_id", maintain_order=True)
        .head(k)
    )

    nearest_candidates = {}
    for row in nearest.iter_rows(named=True):
        i = row["neighborhood_id"]
        j = row["candidate_id"]

        nearest_candidates.setdefault(i, []).append(j)

    return (nearest, nearest_candidates)


def optimal_type_mix(p, budget, type_cost):
    c1 = type_cost["type_1"]
    c2 = type_cost["type_2"]

    min_cost = p * c2
    if min_cost > budget:
        raise ValueError(f"p={p} é inviável com base no orçamento")

    additional_budget = budget - min_cost
    additional_cost_type_1 = c1 - c2

    max_type_1 = int(additional_budget // additional_cost_type_1)

    n_type_1 = min(p, max_type_1)
    n_type_2 = p - n_type_1

    return (n_type_1, n_type_2)


def build_model(params, budget, p):
    neighborhoods = params["neighborhoods"]
    existing_ids = params["existing_ids"]
    candidate_ids = params["candidate_ids"]
    nearest_candidates = params["nearest_candidates"]
    facility_types = params["facility_types"]

    q = params["q"]
    existing_capacity = params["existing_capacity"]
    type_capacity = params["type_capacity"]
    type_cost = params["type_cost"]

    d_existing = params["d_existing"]
    d_candidate = params["d_candidate"]

    n_type_1, n_type_2 = optimal_type_mix(p, budget, type_cost)

    total_demand = sum(q.values())

    total_existing_capacity = sum(existing_capacity.values())

    new_capacity = (
        n_type_1 * type_capacity["type_1"] + n_type_2 * type_capacity["type_2"]
    )

    optimal_unmet = max(
        0.0,
        total_demand - total_existing_capacity - new_capacity,
    )

    print(
        f"p={p}: "
        f"type_1={n_type_1}, "
        f"type_2={n_type_2}, "
        f"optimal unmet={optimal_unmet:.4f}"
    )

    model = pulp.LpProblem(
        f"Capacitated_p_Median_p_{p}",
        pulp.LpMinimize,
    )

    x = model.add_variable_dicts(
        "open",
        (
            candidate_ids,
            facility_types,
        ),
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    z_existing = model.add_variable_dicts(
        "serve_existing",
        (
            neighborhoods,
            existing_ids,
        ),
        lowBound=0,
        cat=pulp.LpContinuous,
    )

    z_candidate = {
        (i, j): model.add_variable(
            f"serve_candidate_{i}_{j}",
            lowBound=0,
            cat=pulp.LpContinuous,
        )
        for i in neighborhoods
        for j in nearest_candidates[i]
    }

    unmet = model.add_variable_dicts(
        "unmet",
        neighborhoods,
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    for i in neighborhoods:
        model += (
            (
                pulp.lpSum(z_existing[i][e] for e in existing_ids)
                + pulp.lpSum(z_candidate[i, j] for j in nearest_candidates[i])
                + unmet[i]
                == q[i]
            ),
            f"demand_{i}",
        )

    for e in existing_ids:
        model += (
            (
                pulp.lpSum(z_existing[i][e] for i in neighborhoods)
                <= existing_capacity[e]
            ),
            f"existing_capacity_{e}",
        )

    candidate_neighborhoods = {j: [] for j in candidate_ids}

    for i in neighborhoods:
        for j in nearest_candidates[i]:
            candidate_neighborhoods[j].append(i)

    for j in candidate_ids:
        model += (
            pulp.lpSum(z_candidate[i, j] for i in candidate_neighborhoods[j])
            <= pulp.lpSum(type_capacity[r] * x[j][r] for r in facility_types),
            f"candidate_capacity_{j}",
        )

    for j in candidate_ids:
        model += (pulp.lpSum(x[j][r] for r in facility_types) <= 1), f"one_type_{j}"

    model += (
        (pulp.lpSum(unmet[i] for i in neighborhoods) <= optimal_unmet + EPSILON),
        "optimal_coverage",
    )

    # model += (
    #     (pulp.lpSum(x[j][r] for j in candidate_ids for r in facility_types) == p),
    #     "number_of_new_faiclities",
    model += (
        (pulp.lpSum(x[j]["type_1"] for j in candidate_ids) == n_type_1),
        "number_type_1",
    )

    model += (
        (pulp.lpSum(x[j]["type_2"] for j in candidate_ids) == n_type_2),
        "number_type_2",
    )  # )

    model += (
        (
            pulp.lpSum(
                type_cost[r] * x[j][r] for j in candidate_ids for r in facility_types
            )
            <= budget
        ),
        "budget",
    )

    total_unmet = pulp.lpSum(unmet[i] for i in neighborhoods)
    travel_existing = pulp.lpSum(
        d_existing[i, e] * z_existing[i][e] for i in neighborhoods for e in existing_ids
    )
    travel_candidate = pulp.lpSum(
        d_candidate[i, j] * z_candidate[i, j]
        for i in neighborhoods
        for j in nearest_candidates[i]
    )
    total_travel = travel_existing + travel_candidate

    print()
    print("MODEL SIZE")
    print("----------")

    print(f"opening variables: {len(candidate_ids) * len(facility_types)}")

    print(f"existing flow variables: {len(neighborhoods) * len(existing_ids)}")

    print(f"candidate flow variables: {len(z_candidate)}")

    print(f"unmet variables: {len(neighborhoods)}")

    print(f"model variables: {len(model.variables())}")

    print(f"model constraints: {len(model.constraints())}")

    return {
        "model": model,
        "x": x,
        "z_existing": z_existing,
        "z_candidate": z_candidate,
        "unmet": unmet,
        "total_unmet": total_unmet,
        "total_travel": total_travel,
    }


def solve_model(
    objects,
):
    model = objects["model"]

    total_unmet = objects["total_unmet"]

    total_travel = objects["total_travel"]

    solver = pulp.COIN_CMD(
        msg=SOLVER_MSG,
    )

    print()
    print("solving p-median distance model")

    model.objective = total_travel

    stats = model.solve(solver)

    print(f"status: {stats.status_str}")

    if not stats.has_solution:
        raise RuntimeError(f"Solver failed: {stats.status_str}")

    final_unmet = float(pulp.value(total_unmet))

    final_travel = float(pulp.value(total_travel))

    return {
        "status": stats.status_str,
        "unmet": final_unmet,
        "weighted_travel": final_travel,
    }


# def solve_lexicographic(objects):
#     model = objects["model"]
#     total_unmet = objects["total_unmet"]
#     total_travel = objects["total_travel"]
#
#     solver = pulp.COIN_CMD(msg=SOLVER_MSG)
#
#     print()
#     print("estágio 1: minimizar demanda não atendida")
#
#     model.objective = total_travel
#     stats_1 = model.solve(solver)
#
#     print(f"status do estágio 1: {stats_1.status_str}")
#
#     if not stats_1.has_solution:
#         raise RuntimeError(f"o estágio 1 falhou: {stats_1.status_str}")
#
#     optimal_unmet = float(pulp.value(total_unmet))
#
#     print(f"demanda mínima não atendida: {optimal_unmet:.6f}")
#
#     model += (total_unmet <= optimal_unmet + EPSILON), "fix_optimal_coverage"
#
#     print()
#     print("estágio 2: minimizar a distância ponderada dos caminhos da população")
#
#     model.objective = total_travel
#
#     stats_2 = model.solve(solver)
#
#     print(f"status do estágio 2: {stats_2.status_str}")
#
#     if not stats_2.has_solution:
#         raise RuntimeError(f"o estágio 2 falhou: {stats_2.status_str}")
#
#     final_unmet = float(pulp.value(total_unmet))
#
#     final_travel = float(pulp.value(total_travel))
#
#     return {
#         "status": stats_2.status_str,
#         "unmet": final_unmet,
#         "weighted_travel": final_travel,
#     }


def extract_solution(params, objects, budget, p, solution_values):
    neighborhoods = params["neighborhoods"]
    existing_ids = params["existing_ids"]
    candidate_ids = params["candidate_ids"]
    nearest_candidates = params["nearest_candidates"]
    facility_types = params["facility_types"]

    q = params["q"]
    type_capacity = params["type_capacity"]
    type_cost = params["type_cost"]

    x = objects["x"]
    z_existing = objects["z_existing"]
    z_candidate = objects["z_candidate"]
    unmet = objects["unmet"]

    opened_rows = []

    investment = 0.0
    new_capacity = 0.0

    type_counts = {r: 0 for r in facility_types}

    for j in candidate_ids:
        for r in facility_types:
            value = pulp.value(x[j][r])

            if value is not None and value > 0.5:
                cost = float(type_cost[r])
                capacity = float(type_capacity[r])

                investment += cost
                new_capacity += capacity

                type_counts[r] += 1

                opened_rows.append(
                    {
                        "candidate_id": j,
                        "facility_type": r,
                        "capacity": capacity,
                        "construction_cost": cost,
                    }
                )

    unmet_rows = []
    total_unmet = 0.0

    for i in neighborhoods:
        value = float(pulp.value(unmet[i]) or 0.0)
        total_unmet += value

        unmet_rows.append(
            {"neighborhood_id": i, "demand": float(q[i]), "unmet_demand": value}
        )

    existing_flow_rows = []
    existing_served = 0.0

    for i in neighborhoods:
        for e in existing_ids:
            value = pulp.value(z_existing[i][e])

            if value is not None and value > 1e-7:
                value = float(value)

                existing_served += value

                existing_flow_rows.append(
                    {"neighborhood_id": i, "facility_id": e, "served_demand": value}
                )

    candidate_flow_rows = []
    new_served = 0.0

    for i in neighborhoods:
        for j in nearest_candidates[i]:
            value = pulp.value(z_candidate[i, j])

            if value is not None and value > 1e-7:
                value = float(value)

                new_served += value

                candidate_flow_rows.append(
                    {"neighborhood_id": i, "candidate_id": j, "served_demand": value}
                )

    total_demand = float(sum(q.values()))
    served_demand = existing_served + new_served

    coverage = served_demand / total_demand if total_demand > 0 else 0.0

    weighted_travel = solution_values["weighted_travel"]

    avg_travel_minutes = weighted_travel / served_demand if served_demand > 0 else None

    budget_usage = investment / budget if budget > 0 else 0.0

    summary = {
        "p": p,
        "budget": budget,
        "investment": investment,
        "budget_usage": budget_usage,
        "type_1_count": (
            type_counts.get(
                "type_1",
                0,
            )
        ),
        "type_2_count": (
            type_counts.get(
                "type_2",
                0,
            )
        ),
        "new_capacity": new_capacity,
        "total_demand": total_demand,
        "existing_served": (existing_served),
        "new_served": new_served,
        "served_demand": (served_demand),
        "unmet_demand": (total_unmet),
        "coverage": coverage,
        "weighted_travel_minutes": (weighted_travel),
        "average_travel_minutes": (avg_travel_minutes),
    }

    return {
        "opened": opened_rows,
        "unmet": unmet_rows,
        "existing_flows": existing_flow_rows,
        "candidate_flows": candidate_flow_rows,
        "summary": summary,
    }


def save_scenario(
    p,
    result,
):

    scenario_dir = RESULTS_DIR / f"p_{p:02d}"

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    opened = result["opened"]
    unmet = result["unmet"]
    existing_flows = result["existing_flows"]
    candidate_flows = result["candidate_flows"]

    if opened:
        pl.DataFrame(opened).write_parquet(scenario_dir / "opened_facilities.parquet")

    pl.DataFrame(unmet).write_parquet(scenario_dir / "unmet_demand.parquet")

    if existing_flows:
        pl.DataFrame(existing_flows).write_parquet(
            scenario_dir / "existing_service_flows.parquet"
        )

    if candidate_flows:
        pl.DataFrame(candidate_flows).write_parquet(
            scenario_dir / "candidate_service_flows.parquet"
        )

    pl.DataFrame([result["summary"]]).write_parquet(scenario_dir / "summary.parquet")


# ============================================================
# Reporting
# ============================================================


def print_scenario_summary(
    summary,
):

    print()
    print(f"Scenario p={summary['p']}")
    print("------------------------")

    print(f"investment: R$ {summary['investment']:,.2f}")

    print(f"budget usage: {summary['budget_usage']:.2%}")

    print(f"type 1: {summary['type_1_count']}")

    print(f"type 2: {summary['type_2_count']}")

    print(f"new capacity: {summary['new_capacity']:.2f}")

    print(f"served demand: {summary['served_demand']:.4f}")

    print(f"unmet demand: {summary['unmet_demand']:.4f}")

    print(f"coverage: {summary['coverage']:.4%}")

    print(f"average travel: {summary['average_travel_minutes']:.4f} min")


def save_scenario_to_dir(
    scenario_dir,
    result,
):

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    opened = result["opened"]
    unmet = result["unmet"]
    existing_flows = result["existing_flows"]
    candidate_flows = result["candidate_flows"]

    if opened:
        pl.DataFrame(opened).write_parquet(scenario_dir / "opened_facilities.parquet")

    pl.DataFrame(unmet).write_parquet(scenario_dir / "unmet_demand.parquet")

    if existing_flows:
        pl.DataFrame(existing_flows).write_parquet(
            scenario_dir / "existing_service_flows.parquet"
        )

    if candidate_flows:
        pl.DataFrame(candidate_flows).write_parquet(
            scenario_dir / "candidate_service_flows.parquet"
        )

    pl.DataFrame([result["summary"]]).write_parquet(scenario_dir / "summary.parquet")


def get_feasible_p_values(
    budget: float,
    type_cost: dict[str, float],
) -> list[int]:

    cheapest_cost = min(type_cost.values())

    p_max = int(budget // cheapest_cost)

    return list(
        range(
            1,
            p_max + 1,
        )
    )


def run_scenarios(
    k_nearest_candidates: int,
    budget: float,
    p_values: list[int] | None = None,
    results_dir=None,
):

    if results_dir is None:
        results_dir = RESULTS_DIR

    if results_dir.exists():
        shutil.rmtree(results_dir)

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        demand,
        existing_facilities,
        existing_costs,
        candidate_costs,
        candidates,
        facility_types_df,
    ) = load_inputs()

    params = build_parameter_maps(
        demand,
        existing_facilities,
        existing_costs,
        candidate_costs,
        facility_types_df,
        k_nearest_candidates,
    )

    validate_parameters(
        params,
        k_nearest_candidates,
    )

    if p_values is None:
        scenario_p_values = get_feasible_p_values(
            budget=budget,
            type_cost=params["type_cost"],
        )

    else:
        cheapest_cost = min(params["type_cost"].values())
        p_max_budget = int(budget // cheapest_cost)

        scenario_p_values = [p for p in sorted(set(p_values)) if 1 <= p <= p_max_budget]
        skipped = [p for p in p_values if p > p_max_budget]

        if skipped:
            print(
                f"pulando valores de p inviáveis para o orçamento R$ {budget:,.2f}: {skipped}"
            )

    scenario_summaries = []

    for scenario_index, p in enumerate(
        scenario_p_values,
        start=1,
    ):
        print()
        print("=" * 70)

        print(
            f"K={k_nearest_candidates} | "
            f"SCENARIO "
            f"{scenario_index}/"
            f"{len(scenario_p_values)} | "
            f"p={p}"
        )

        print("=" * 70)

        objects = build_model(
            params=params,
            budget=budget,
            p=p,
        )

        solution_values = solve_model(objects)

        result = extract_solution(
            params=params,
            objects=objects,
            budget=budget,
            p=p,
            solution_values=solution_values,
        )

        scenario_dir = results_dir / f"p_{p:02d}"

        save_scenario_to_dir(
            scenario_dir=scenario_dir,
            result=result,
        )

        summary = {
            **result["summary"],
            "k": k_nearest_candidates,
        }

        scenario_summaries.append(summary)

        print_scenario_summary(summary)

    summary_df = pl.DataFrame(scenario_summaries).sort(
        [
            "k",
            "p",
        ]
    )

    summary_df.write_parquet(
        results_dir / "scenario_summary.parquet",
        compression="zstd",
    )

    summary_df.write_csv(results_dir / "scenario_summary.csv")

    return summary_df


def main():

    run_scenarios(
        k_nearest_candidates=100,
        p_values=None,
        results_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
