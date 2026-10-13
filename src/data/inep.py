from pathlib import Path
from src.config import IBGE_NOVA_IGUACU_CODE

import polars as pl

MUNICIPAL_DEP = 3
NOVA_IGUACU_CODE = int(IBGE_NOVA_IGUACU_CODE)


def load_schools(path: Path):
    cols = [
        "CO_MUNICIPIO",
        "CO_ENTIDADE",
        "NO_ENTIDADE",
        "TP_DEPENDENCIA",
        "TP_LOCALIZACAO",
        "TP_SITUACAO_FUNCIONAMENTO",
        "IN_COMUM_CRECHE",
        "IN_ESP_EXCLUSIVA_CRECHE",
        "QT_SALAS_UTILIZADAS",
    ]

    return pl.read_csv(
        path,
        separator=";",
        columns=cols,
        infer_schema_length=10_000,
        encoding="utf8-lossy",
    )


def filter_municipal_schools(df: pl.DataFrame):
    return df.filter(
        (pl.col("CO_MUNICIPIO") == NOVA_IGUACU_CODE)
        & (pl.col("TP_DEPENDENCIA") == MUNICIPAL_DEP)
        & (pl.col("TP_SITUACAO_FUNCIONAMENTO") == 1)
    )


def filter_creche_schools(df: pl.DataFrame):
    return df.filter(
        (pl.col("IN_COMUM_CRECHE") == 1) | (pl.col("IN_ESP_EXCLUSIVA_CRECHE") == 1)
    )


def load_enrollments(path: Path):
    return (
        pl.read_csv(
            path,
            separator=";",
            columns=[
                "CO_MUNICIPIO",
                "CO_ENTIDADE",
                "TP_DEPENDENCIA",
                "QT_MAT_INF_CRE",
            ],
            infer_schema_length=10_000,
            encoding="utf8-lossy",
        )
        .filter(
            (pl.col("CO_MUNICIPIO") == NOVA_IGUACU_CODE)
            & (pl.col("TP_DEPENDENCIA") == MUNICIPAL_DEP)
        )
        .select(
            pl.col("CO_ENTIDADE").cast(pl.String).alias("facility_id"),
            pl.col("QT_MAT_INF_CRE")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("creche_enrollment"),
        )
    )


def load_classes(path: Path):
    return (
        pl.read_csv(
            path,
            separator=";",
            columns=[
                "CO_MUNICIPIO",
                "CO_ENTIDADE",
                "TP_DEPENDENCIA",
                "QT_TUR_INF_CRE",
            ],
            infer_schema_length=10_000,
            encoding="utf8-lossy",
        )
        .filter(
            (pl.col("CO_MUNICIPIO") == NOVA_IGUACU_CODE)
            & (pl.col("TP_DEPENDENCIA") == MUNICIPAL_DEP)
        )
        .select(
            pl.col("CO_ENTIDADE").cast(pl.String).alias("facility_id"),
            pl.col("QT_TUR_INF_CRE")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("creche_class_count"),
        )
    )


def prepare_creche_schools(df: pl.DataFrame):
    return (
        df.pipe(filter_municipal_schools)
        .pipe(filter_creche_schools)
        .select(
            pl.col("CO_ENTIDADE").cast(pl.String).alias("facility_id"),
            pl.col("NO_ENTIDADE").alias("facility_name"),
            pl.col("TP_LOCALIZACAO").alias("location_type"),
            pl.col("QT_SALAS_UTILIZADAS")
            .cast(pl.Int64, strict=False)
            .alias("school_rooms_used"),
            "IN_COMUM_CRECHE",
            "IN_ESP_EXCLUSIVA_CRECHE",
        )
    )


def build_existing_facilities(
    schools: pl.DataFrame, enrollments: pl.DataFrame, classes: pl.DataFrame
):
    return (
        schools.join(enrollments, on="facility_id", how="left", validate="1:1")
        .join(classes, on="facility_id", how="left", validate="1:1")
        .with_columns(
            pl.col("creche_enrollment").fill_null(0),
            pl.col("creche_class_count").fill_null(0),
        )
    )


def add_class_statistics(df: pl.DataFrame):
    return df.with_columns(
        pl.when(pl.col("creche_class_count") > 0)
        .then(pl.col("creche_enrollment") / pl.col("creche_class_count"))
        .otherwise(None)
        .alias("students_per_creche_class")
    )
