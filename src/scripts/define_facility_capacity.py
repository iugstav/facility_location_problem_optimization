import polars as pl

from src.config import PROCESSED_DIR

facilities = pl.read_parquet(
    PROCESSED_DIR / "existing_facilities_with_capacity.parquet"
)

print(facilities.columns)
stats = facilities.filter(pl.col("creche_class_count") > 0).select(
    [
        pl.col("creche_class_count").min().alias("min"),
        pl.col("creche_class_count").quantile(0.25).alias("q25"),
        pl.col("creche_class_count").median().alias("median"),
        pl.col("creche_class_count").quantile(0.75).alias("q75"),
        pl.col("creche_class_count").max().alias("max"),
    ]
)

existing_capacity = facilities.select(pl.col("final_capacity").sum()).item()

print(f"existing capacity: {existing_capacity}")

print(stats)
