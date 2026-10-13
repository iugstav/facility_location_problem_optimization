from pathlib import Path
import polars as pl
from src.config import IBGE_NOVA_IGUACU_CODE


def load_sector_demographics(path: Path):
    return (
        pl.read_csv(
            path,
            separator=";",
            schema_overrides={"CD_setor": pl.String},
            null_values=["X"],
            infer_schema_length=10_000,
        )
        .filter(pl.col("CD_setor").str.starts_with(IBGE_NOVA_IGUACU_CODE))
        .select(
            "CD_setor",
            pl.col("V01006").cast(pl.Int64, strict=False).alias("total_population"),
            (
                pl.col("V01009").cast(pl.Int64, strict=False)
                + pl.col("V01020").cast(pl.Int64, strict=False)
            ).alias("child_population_0_to_4"),
        )
    )
