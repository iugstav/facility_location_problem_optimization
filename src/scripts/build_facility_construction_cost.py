import polars as pl

from src.config import FNDE_DIR
from src.data.extract_facility_construction_cost import load_costs_workbook, get_costs

COSTS_FILE = [
    FNDE_DIR / "obra_valores_creche_tipo1.xlsx",
    FNDE_DIR / "obra_valores_creche_tipo2.xlsx",
]


def main():
    for file in COSTS_FILE:
        ws = load_costs_workbook(file)
        result = get_costs(ws)
        df = pl.DataFrame(result, schema=["section", "value"])

        print(file.name)
        print(
            df.with_columns(pl.col("value").cast(pl.Float64).alias("value")).select(
                pl.col("value").sum().alias("total_cost")
            )
        )


if __name__ == "__main__":
    main()
