import json
import hashlib
import colorsys

import folium
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from streamlit_folium import st_folium

from src.config import PROCESSED_DIR

OPTIMIZATION_DIR = PROCESSED_DIR / "optimization"
RESULTS_DIR = OPTIMIZATION_DIR / "results"
FULL_SENSITIVITY_DIR = OPTIMIZATION_DIR / "full_sensitivity"

FULL_SENSITIVITY_SUMMARY_FILE = (
    OPTIMIZATION_DIR / "full_sensitivity" / "full_sensitivity_summary.parquet"
)
SUMMARY_FILE = RESULTS_DIR / "scenario_summary.parquet"
CANDIDATES_FILE = OPTIMIZATION_DIR / "candidates.parquet"
EXISTING_LOCATIONS_FILE = PROCESSED_DIR / "existing_facilities_locations_final.parquet"
NEIGHBORHOODS_GEOJSON_FILE = PROCESSED_DIR / "neighborhoods.geojson"
SECTOR_EXISTING_TRAVEL_FILE = PROCESSED_DIR / "sector_facility_travel_times.parquet"
SECTOR_CANDIDATE_TRAVEL_FILE = PROCESSED_DIR / "sector_candidate_travel_times.parquet"

st.set_page_config(
    page_title="Otimização da Localização de Creches em Nova Iguaçu", layout="wide"
)


def deterministic_color(
    value: str,
) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()

    hue = (
        int(
            digest[:8],
            16,
        )
        % 360
    )

    saturation = 58
    lightness = 52

    r, g, b = colorsys.hls_to_rgb(
        hue / 360,
        lightness / 100,
        saturation / 100,
    )

    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def safe_float(
    value,
    default: float = 0.0,
) -> float:
    if value is None:
        return default

    return float(value)


@st.cache_data
def load_summary():
    return pl.read_parquet(SUMMARY_FILE).sort("p")


@st.cache_data
def load_full_sensitivity_summary():
    if not FULL_SENSITIVITY_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {FULL_SENSITIVITY_SUMMARY_FILE}"
        )

    return pl.read_parquet(FULL_SENSITIVITY_SUMMARY_FILE).sort(
        [
            "experiment_budget",
            "experiment_k",
            "p",
        ]
    )


@st.cache_data
def load_candidates():
    return pl.read_parquet(CANDIDATES_FILE)


@st.cache_data
def load_existing_locations():
    return pl.read_parquet(EXISTING_LOCATIONS_FILE)


@st.cache_data
def load_sector_existing_times():
    return pl.read_parquet(SECTOR_EXISTING_TRAVEL_FILE)


@st.cache_data
def load_sector_candidate_times():
    return pl.read_parquet(SECTOR_CANDIDATE_TRAVEL_FILE)


@st.cache_data
def load_neighborhood_service(
    budget: float,
    k: int,
    p: int,
):
    scenario_dir = get_scenario_directory(
        budget,
        k,
        p,
    )

    existing = pl.read_parquet(scenario_dir / "existing_service_flows.parquet")

    new = pl.read_parquet(scenario_dir / "candidate_service_flows.parquet")

    unmet = pl.read_parquet(scenario_dir / "unmet_demand.parquet")

    existing_by_neighborhood = existing.group_by("neighborhood_id").agg(
        pl.col("served_demand").sum().alias("existing_served")
    )

    new_by_neighborhood = new.group_by("neighborhood_id").agg(
        pl.col("served_demand").sum().alias("new_served")
    )

    return (
        unmet.join(
            existing_by_neighborhood,
            on="neighborhood_id",
            how="left",
        )
        .join(
            new_by_neighborhood,
            on="neighborhood_id",
            how="left",
        )
        .with_columns(
            pl.col("existing_served").fill_null(0.0),
            pl.col("new_served").fill_null(0.0),
        )
        .with_columns(
            (pl.col("existing_served") + pl.col("new_served")).alias("served_demand")
        )
        .with_columns(
            pl.when(pl.col("demand").is_not_null() & (pl.col("demand") > 0))
            .then(pl.col("served_demand") / pl.col("demand"))
            .otherwise(0.0)
            .alias("coverage_rate")
        )
    )


def empty_existing_flows():
    return pl.DataFrame(
        schema={
            "neighborhood_id": pl.String,
            "facility_id": pl.String,
            "served_demand": pl.Float64,
        }
    )


def empty_candidate_flows():
    return pl.DataFrame(
        schema={
            "neighborhood_id": pl.String,
            "candidate_id": pl.String,
            "served_demand": pl.Float64,
        }
    )


@st.cache_data
def load_scenario_flows(
    budget: float,
    k: int,
    p: int,
):
    scenario_dir = get_scenario_directory(
        budget,
        k,
        p,
    )

    existing_file = scenario_dir / "existing_service_flows.parquet"

    candidate_file = scenario_dir / "candidate_service_flows.parquet"

    unmet_file = scenario_dir / "unmet_demand.parquet"

    existing = (
        pl.read_parquet(existing_file)
        if existing_file.exists()
        else empty_existing_flows()
    )

    new = (
        pl.read_parquet(candidate_file)
        if candidate_file.exists()
        else empty_candidate_flows()
    )

    if not unmet_file.exists():
        raise FileNotFoundError(f"Unmet demand não encontrado: {unmet_file}")

    unmet = pl.read_parquet(unmet_file)

    return (
        existing,
        new,
        unmet,
    )


@st.cache_data
def load_opened_facilities(
    budget: float,
    k: int,
    p: int,
):
    scenario_dir = get_scenario_directory(
        budget,
        k,
        p,
    )

    file = scenario_dir / "opened_facilities.parquet"

    if not file.exists():
        raise FileNotFoundError(f"Resultado não encontrado: {file}")

    return pl.read_parquet(file)


@st.cache_data
def build_sector_nearest_facility(
    budget: float,
    k: int,
    p: int,
) -> pl.DataFrame:

    opened = load_opened_facilities(
        budget,
        k,
        p,
    )

    open_candidates = opened.get_column("candidate_id").cast(pl.String).to_list()

    existing_times = (
        load_sector_existing_times()
        .with_columns(
            pl.col("sector_id").cast(pl.String),
            pl.col("facility_id").cast(pl.String),
        )
        .select(
            [
                "sector_id",
                pl.col("facility_id"),
                "travel_time_minutes",
            ]
        )
        .with_columns(pl.lit("existing").alias("facility_kind"))
    )

    candidate_times = (
        load_sector_candidate_times()
        .with_columns(
            pl.col("sector_id").cast(pl.String),
            pl.col("candidate_id").cast(pl.String),
        )
        .filter(pl.col("candidate_id").is_in(open_candidates))
        .select(
            [
                "sector_id",
                pl.col("candidate_id").alias("facility_id"),
                "travel_time_minutes",
            ]
        )
        .with_columns(pl.lit("new").alias("facility_kind"))
    )

    combined = pl.concat(
        [
            existing_times,
            candidate_times,
        ],
        how="vertical",
    )

    return (
        combined.sort(
            [
                "sector_id",
                "travel_time_minutes",
            ]
        )
        .group_by(
            "sector_id",
            maintain_order=True,
        )
        .first()
        .rename(
            {
                "facility_id": "nearest_facility_id",
                "facility_kind": "nearest_facility_kind",
            }
        )
    )


@st.cache_data
def load_dominant_facility_by_neighborhood(
    budget: float,
    k: int,
    p: int,
) -> pl.DataFrame:

    (
        existing,
        new,
        _,
    ) = load_scenario_flows(
        budget,
        k,
        p,
    )

    existing_standard = existing.select(
        [
            pl.col("neighborhood_id").cast(pl.String),
            pl.col("facility_id").cast(pl.String).alias("facility_id"),
            pl.col("served_demand").cast(pl.Float64),
        ]
    ).with_columns(pl.lit("existing").alias("facility_kind"))

    new_standard = new.select(
        [
            pl.col("neighborhood_id").cast(pl.String),
            pl.col("candidate_id").cast(pl.String).alias("facility_id"),
            pl.col("served_demand").cast(pl.Float64),
        ]
    ).with_columns(pl.lit("new").alias("facility_kind"))

    flows = pl.concat(
        [
            existing_standard,
            new_standard,
        ],
        how="vertical",
    )

    if flows.height == 0:
        return pl.DataFrame(
            schema={
                "neighborhood_id": pl.String,
                "dominant_facility_id": pl.String,
                "dominant_facility_kind": pl.String,
                "dominant_served": pl.Float64,
            }
        )

    return (
        flows.sort(
            [
                "neighborhood_id",
                "served_demand",
            ],
            descending=[
                False,
                True,
            ],
        )
        .group_by(
            "neighborhood_id",
            maintain_order=True,
        )
        .first()
        .rename(
            {
                "facility_id": "dominant_facility_id",
                "facility_kind": "dominant_facility_kind",
                "served_demand": "dominant_served",
            }
        )
    )


def get_scenario_summary(summary: pl.DataFrame, p: int):
    scenario = summary.filter(pl.col("p") == p)
    if scenario.height != 1:
        raise ValueError(
            f"esperava exatamente um cenário para p={p}, mas foi encontrado {scenario.height}"
        )

    return scenario.row(0, named=True)


def prepare_new_facilities(opened: pl.DataFrame, candidates: pl.DataFrame):
    return opened.join(candidates, on="candidate_id", how="left")


def get_candidate_coordinate_columns(candidates: pl.DataFrame):
    lat_candidates = ["latitude", "lat"]
    lon_candidates = ["longitude", "lon", "lng"]

    lat_col = next((col for col in lat_candidates if col in candidates.columns), None)
    lon_col = next((col for col in lon_candidates if col in candidates.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError(
            "não foi possível identificar as colunas de latitude e longitude em candidates.parquet. "
            f"colunas disponíveis: {candidates.columns}"
        )

    return (lat_col, lon_col)


def budget_slug(
    budget: float,
) -> str:

    value = budget / 1_000_000

    if value.is_integer():
        return f"{int(value)}m"

    return f"{value:.1f}".replace(".", "_") + "m"


def get_scenario_directory(
    budget: float,
    k: int,
    p: int,
):

    return (
        FULL_SENSITIVITY_DIR
        / f"budget_{budget_slug(budget)}"
        / f"k_{k:03d}"
        / f"p_{p:02d}"
    )


def compute_map_center(
    existing: pl.DataFrame,
    new_facilities: pl.DataFrame,
):
    new_latitude, new_longitude = get_candidate_coordinate_columns(new_facilities)

    lat = []
    lon = []

    if "latitude_final" in existing.columns:
        lat.extend(existing.get_column("latitude_final").drop_nulls().to_list())

    if "longitude_final" in existing.columns:
        lon.extend(existing.get_column("longitude_final").drop_nulls().to_list())

    lat.extend(new_facilities.get_column(new_latitude).drop_nulls().to_list())

    lon.extend(new_facilities.get_column(new_longitude).drop_nulls().to_list())

    if len(lat) == 0 or len(lon) == 0:
        raise ValueError("nenhuma coordenada válida encontrada")

    return (sum(lat) / len(lat), sum(lon) / len(lon))


def coverage_color(
    rate: float,
) -> str:

    if rate >= 0.75:
        return "#1a9850"

    if rate >= 0.50:
        return "#91cf60"

    if rate >= 0.25:
        return "#fee08b"

    if rate >= 0.10:
        return "#fc8d59"

    return "#d73027"


def add_neighborhood_coverage_layer(
    map_: folium.Map, neighborhood_service: pl.DataFrame
):
    if not NEIGHBORHOODS_GEOJSON_FILE.exists():
        raise FileNotFoundError(
            f"GeoJSON de bairros não encontrado: {NEIGHBORHOODS_GEOJSON_FILE}"
        )

    with open(NEIGHBORHOODS_GEOJSON_FILE, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    service_map = {
        str(row["neighborhood_id"]): row
        for row in neighborhood_service.iter_rows(named=True)
    }

    for feature in geojson["features"]:
        properties = feature["properties"]

        neighborhood_id = str(properties["neighborhood_id"])

        row = service_map.get(neighborhood_id)
        if row is None:
            properties["coverage_rate"] = 0.0
            properties["coverage_percent"] = 0.0
            properties["demand"] = 0.0
            properties["existing_served"] = 0.0
            properties["new_served"] = 0.0
            properties["served_demand"] = 0.0
            properties["unmet_demand"] = 0.0

            continue

        coverage = safe_float(row.get("coverage_rate"))
        properties["coverage_rate"] = coverage
        properties["coverage_percent"] = round(
            coverage * 100,
            2,
        )
        properties["demand"] = round(
            safe_float(row.get("demand")),
            1,
        )
        properties["existing_served"] = round(
            safe_float(row.get("existing_served")),
            1,
        )
        properties["new_served"] = round(
            safe_float(row.get("new_served")),
            1,
        )
        properties["served_demand"] = round(
            safe_float(row.get("served_demand")),
            1,
        )
        properties["unmet_demand"] = round(
            safe_float(row.get("unmet_demand")),
            1,
        )

    def style_function(feature):
        rate = safe_float(feature["properties"].get("coverage_rate"))
        return {
            "fillColor": coverage_color(rate),
            "color": "#555555",
            "weight": 1,
            "fillOpacity": 0.55,
        }

    folium.GeoJson(
        geojson,
        name="Cobertura por bairro",
        style_function=style_function,
        highlight_function=lambda feature: {
            "weight": 3,
            "color": "#222222",
            "fillOpacity": 0.70,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "neighborhood_name",
                "coverage_percent",
            ],
            aliases=[
                "Bairro:",
                "Cobertura (%):",
            ],
            localize=True,
            sticky=False,
        ),
        popup=folium.GeoJsonPopup(
            fields=[
                "neighborhood_name",
                "demand",
                "existing_served",
                "new_served",
                "served_demand",
                "unmet_demand",
                "coverage_percent",
            ],
            aliases=[
                "Bairro:",
                "Demanda alvo:",
                "Rede existente:",
                "Rede nova:",
                "Total atendido:",
                "Não atendido:",
                "Cobertura (%):",
            ],
            localize=True,
        ),
    ).add_to(map_)


def add_dominant_facility_layer(
    map_: folium.Map,
    neighborhood_service: pl.DataFrame,
    dominant: pl.DataFrame,
):

    if not NEIGHBORHOODS_GEOJSON_FILE.exists():
        return

    with open(
        NEIGHBORHOODS_GEOJSON_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        geojson = json.load(file)

    dominant_map = {
        str(row["neighborhood_id"]): row for row in dominant.iter_rows(named=True)
    }

    service_map = {
        str(row["neighborhood_id"]): row
        for row in neighborhood_service.iter_rows(named=True)
    }

    for feature in geojson["features"]:
        properties = feature["properties"]

        neighborhood_id = str(properties["neighborhood_id"])

        dominant_row = dominant_map.get(neighborhood_id)

        service_row = service_map.get(neighborhood_id)

        if dominant_row is None:
            facility_id = "Sem atendimento"

            facility_kind = "-"

            color = "#cccccc"

        else:
            facility_id = str(dominant_row["dominant_facility_id"])

            facility_kind = dominant_row["dominant_facility_kind"]

            color = deterministic_color(facility_id)

        properties["dominant_facility_id"] = facility_id

        properties["dominant_facility_kind"] = facility_kind

        properties["cluster_color"] = color

        coverage = 0.0

        if service_row is not None:
            coverage = safe_float(service_row.get("coverage_rate"))

        properties["coverage_percent"] = round(
            coverage * 100,
            2,
        )

    def style_function(
        feature,
    ):

        return {
            "fillColor": feature["properties"]["cluster_color"],
            "color": "#555555",
            "weight": 1,
            "fillOpacity": 0.50,
        }

    folium.GeoJson(
        geojson,
        name=("Facility dominante por bairro"),
        style_function=(style_function),
        highlight_function=(
            lambda feature: {
                "weight": 3,
                "color": "#222222",
                "fillOpacity": 0.70,
            }
        ),
        tooltip=(
            folium.GeoJsonTooltip(
                fields=[
                    "neighborhood_name",
                    "dominant_facility_id",
                    "dominant_facility_kind",
                    "coverage_percent",
                ],
                aliases=[
                    "Bairro:",
                    "Facility dominante:",
                    "Rede:",
                    "Cobertura (%):",
                ],
                sticky=False,
            )
        ),
    ).add_to(map_)


# def add_sector_cluster_layer(
#     map_: folium.Map,
#     nearest_facilities: pl.DataFrame,
# ):
#
#     if not SECTORS_GEOJSON.exists():
#         st.warning("O mapa de setores requer `nova_iguacu_sectors.geojson`.")
#
#         return
#
#     with open(
#         SECTORS_GEOJSON,
#         "r",
#         encoding="utf-8",
#     ) as file:
#         geojson = json.load(file)
#
#     nearest_map = {
#         str(row["sector_id"]): row for row in nearest_facilities.iter_rows(named=True)
#     }
#
#     for feature in geojson["features"]:
#         properties = feature["properties"]
#
#         sector_id = str(properties["sector_id"])
#
#         row = nearest_map.get(sector_id)
#
#         if row is None:
#             properties["nearest_facility_id"] = "-"
#
#             properties["nearest_facility_kind"] = "-"
#
#             properties["travel_time_minutes"] = None
#
#             properties["cluster_color"] = "#cccccc"
#
#             continue
#
#         facility_id = str(row["nearest_facility_id"])
#
#         properties["nearest_facility_id"] = facility_id
#
#         properties["nearest_facility_kind"] = row["nearest_facility_kind"]
#
#         properties["travel_time_minutes"] = round(
#             safe_float(row.get("travel_time_minutes")),
#             2,
#         )
#
#         properties["cluster_color"] = deterministic_color(facility_id)
#
#     def style_function(
#         feature,
#     ):
#
#         return {
#             "fillColor": feature["properties"]["cluster_color"],
#             "color": "#777777",
#             "weight": 0.4,
#             "fillOpacity": 0.48,
#         }
#
#     folium.GeoJson(
#         geojson,
#         name=("Setores por facility mais próxima"),
#         style_function=(style_function),
#         tooltip=(
#             folium.GeoJsonTooltip(
#                 fields=[
#                     "sector_id",
#                     "nearest_facility_id",
#                     "nearest_facility_kind",
#                     "travel_time_minutes",
#                 ],
#                 aliases=[
#                     "Setor:",
#                     "Facility mais próxima:",
#                     "Rede:",
#                     "Tempo (min):",
#                 ],
#                 sticky=False,
#             )
#         ),
#     ).add_to(map_)


def add_existing_facilities(
    map_: folium.Map,
    existing: pl.DataFrame,
):

    layer = folium.FeatureGroup(
        name="Creches existentes",
        show=True,
    )

    for row in existing.iter_rows(named=True):
        latitude = row.get("latitude_final")

        longitude = row.get("longitude_final")

        if latitude is None or longitude is None:
            continue

        name = row.get("facility_name") or "Creche existente"

        capacity = safe_float(row.get("final_capacity"))

        popup = f"""
        <div style="
            min-width: 220px;
            font-family: Arial;
        ">
            <b>{name}</b>
            <br><br>
            <b>Rede:</b>
            Existente
            <br>
            <b>Capacidade:</b>
            {capacity:.1f}
        </div>
        """

        folium.CircleMarker(
            location=[
                latitude,
                longitude,
            ],
            radius=5,
            color="#333333",
            fill=True,
            fill_color="#333333",
            fill_opacity=0.85,
            weight=1,
            tooltip=name,
            popup=folium.Popup(
                popup,
                max_width=300,
            ),
        ).add_to(layer)

    layer.add_to(map_)


def add_new_facilities(
    map_: folium.Map,
    facilities: pl.DataFrame,
):

    latitude_column, longitude_column = get_candidate_coordinate_columns(facilities)

    type_1_layer = folium.FeatureGroup(
        name="Novas — Tipo 1",
        show=True,
    )

    type_2_layer = folium.FeatureGroup(
        name="Novas — Tipo 2",
        show=True,
    )

    for row in facilities.iter_rows(named=True):
        latitude = row.get(latitude_column)

        longitude = row.get(longitude_column)

        if latitude is None or longitude is None:
            continue

        facility_type = row.get("facility_type")

        candidate_id = row.get("candidate_id")

        capacity = safe_float(row.get("capacity"))

        cost = safe_float(row.get("construction_cost"))

        if facility_type == "type_1":
            color = "#2563eb"

            radius = 9

            layer = type_1_layer

        else:
            color = "#f59e0b"

            radius = 8

            layer = type_2_layer

        popup = f"""
        <div style="
            min-width: 220px;
            font-family: Arial;
        ">
            <b>Nova creche</b>
            <br><br>
            <b>Candidato:</b>
            {candidate_id}
            <br>
            <b>Tipologia:</b>
            {facility_type}
            <br>
            <b>Capacidade:</b>
            {capacity:.1f}
            <br>
            <b>Custo:</b>
            R$ {cost:,.2f}
        </div>
        """

        folium.CircleMarker(
            location=[
                latitude,
                longitude,
            ],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            weight=2,
            tooltip=(f"{facility_type} | {candidate_id}"),
            popup=folium.Popup(
                popup,
                max_width=300,
            ),
        ).add_to(layer)

    type_1_layer.add_to(map_)

    type_2_layer.add_to(map_)


def add_coverage_legend(
    map_: folium.Map,
):

    html = """
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;
        background: white;
        padding: 12px 14px;
        border: 1px solid #cccccc;
        border-radius: 6px;
        font-family: Arial, sans-serif;
        font-size: 12px;
    ">

        <b>Cobertura da demanda</b>
        <br><br>

        <span style="color:#1a9850;">■</span>
        ≥ 75%
        <br>

        <span style="color:#91cf60;">■</span>
        50% – 75%
        <br>

        <span style="color:#fee08b;">■</span>
        25% – 50%
        <br>

        <span style="color:#fc8d59;">■</span>
        10% – 25%
        <br>

        <span style="color:#d73027;">■</span>
        &lt; 10%

    </div>
    """

    map_.get_root().html.add_child(folium.Element(html))


def build_map(
    existing: pl.DataFrame,
    new_facilities: pl.DataFrame,
    neighborhood_service: pl.DataFrame,
    map_mode: str,
    budget: float,
    k: int,
    p: int,
):

    center = compute_map_center(
        existing,
        new_facilities,
    )

    map_ = folium.Map(
        location=center,
        zoom_start=11,
        tiles=None,
        control_scale=True,
    )

    # --------------------------------------------------------
    # Base layer without API key
    # --------------------------------------------------------

    folium.TileLayer(
        tiles=("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
        attr=("© OpenStreetMap contributors"),
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(map_)

    # --------------------------------------------------------
    # Territorial layer
    # --------------------------------------------------------

    if map_mode == "Cobertura por bairro":
        add_neighborhood_coverage_layer(
            map_,
            neighborhood_service,
        )

        add_coverage_legend(map_)

    elif map_mode == "Facility dominante por bairro":
        dominant = load_dominant_facility_by_neighborhood(
            budget,
            k,
            p,
        )

        add_dominant_facility_layer(
            map_,
            neighborhood_service,
            dominant,
        )

    # elif map_mode == "Setores por facility mais próxima":
    #     nearest = build_sector_nearest_facility(
    #         budget,
    #         k,
    #         p,
    #     )
    #
    #     add_sector_cluster_layer(
    #         map_,
    #         nearest,
    #     )

    # --------------------------------------------------------
    # Facilities always over territorial layer
    # --------------------------------------------------------

    add_existing_facilities(
        map_,
        existing,
    )

    add_new_facilities(
        map_,
        new_facilities,
    )

    folium.LayerControl(collapsed=False).add_to(map_)

    return map_


# Charts


def build_coverage_by_budget_chart(
    sensitivity: pl.DataFrame,
    selected_k: int,
    selected_budget: float,
    selected_p: int,
):
    data = sensitivity.filter(pl.col("experiment_k") == selected_k)

    figure = go.Figure()

    budgets = data.get_column("experiment_budget").unique().sort().to_list()

    for budget in budgets:
        current = data.filter(pl.col("experiment_budget") == budget).sort("p")

        figure.add_trace(
            go.Scatter(
                x=current["p"].to_list(),
                y=(current["coverage"] * 100).to_list(),
                mode=("lines+markers"),
                name=(f"R$ {budget / 1_000_000:.1f} mi"),
            )
        )

    selected = data.filter(
        (pl.col("experiment_budget") == selected_budget) & (pl.col("p") == selected_p)
    )

    if selected.height == 1:
        selected_row = selected.row(
            0,
            named=True,
        )

        figure.add_trace(
            go.Scatter(
                x=[selected_row["p"]],
                y=[selected_row["coverage"] * 100],
                mode="markers",
                marker={
                    "size": 15,
                    "symbol": "diamond",
                },
                name=("Cenário selecionado"),
            )
        )

    selected_budget_data = data.filter(pl.col("experiment_budget") == selected_budget)

    best = selected_budget_data.sort(
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

    figure.add_trace(
        go.Scatter(
            x=[best["p"]],
            y=[best["coverage"] * 100],
            mode="markers",
            marker={
                "size": 17,
                "symbol": "star",
            },
            name=(f"p* = {best['p']}"),
        )
    )

    figure.update_layout(
        title=("Cobertura × número de novas creches"),
        xaxis_title=("Número de novas creches (p)"),
        yaxis_title=("Cobertura (%)"),
        hovermode="x unified",
    )

    return figure


def build_travel_by_budget_chart(
    sensitivity: pl.DataFrame,
    selected_k: int,
):

    data = sensitivity.filter(pl.col("experiment_k") == selected_k)

    figure = go.Figure()

    budgets = data.get_column("experiment_budget").unique().sort().to_list()

    for budget in budgets:
        current = data.filter(pl.col("experiment_budget") == budget).sort("p")

        figure.add_trace(
            go.Scatter(
                x=current["p"].to_list(),
                y=current["average_travel_minutes"].to_list(),
                mode=("lines+markers"),
                name=(f"R$ {budget / 1_000_000:.1f} mi"),
            )
        )

    figure.update_layout(
        title=("Tempo médio de deslocamento"),
        xaxis_title="p",
        yaxis_title="Minutos",
        hovermode="x unified",
    )

    return figure


def build_budget_p_heatmap(
    sensitivity: pl.DataFrame,
    selected_k: int,
):

    data = sensitivity.filter(pl.col("experiment_k") == selected_k)

    p_values = data.get_column("p").unique().sort().to_list()

    budgets = data.get_column("experiment_budget").unique().sort().to_list()

    matrix = []

    for budget in budgets:
        current = data.filter(pl.col("experiment_budget") == budget)

        values = {
            row["p"]: (safe_float(row["coverage"]) * 100)
            for row in current.iter_rows(named=True)
        }

        matrix.append([values.get(p) for p in p_values])

    figure = go.Figure(
        data=go.Heatmap(
            x=p_values,
            y=[budget / 1_000_000 for budget in budgets],
            z=matrix,
            colorbar={"title": "Cobertura (%)"},
            hovertemplate=(
                "p=%{x}<br>Budget=R$ %{y:.1f} mi<br>Cobertura=%{z:.2f}%<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=(f"Cobertura por orçamento e p — K={selected_k}"),
        xaxis_title=("Número de novas creches"),
        yaxis_title=("Orçamento (R$ milhões)"),
    )

    return figure


def build_k_p_heatmap(
    sensitivity: pl.DataFrame,
    selected_budget: float,
):

    data = sensitivity.filter(pl.col("experiment_budget") == selected_budget)

    p_values = data.get_column("p").unique().sort().to_list()

    k_values = data.get_column("experiment_k").unique().sort().to_list()

    matrix = []

    for k in k_values:
        current = data.filter(pl.col("experiment_k") == k)

        values = {
            row["p"]: (safe_float(row["coverage"]) * 100)
            for row in current.iter_rows(named=True)
        }

        matrix.append([values.get(p) for p in p_values])

    figure = go.Figure(
        data=go.Heatmap(
            x=p_values,
            y=k_values,
            z=matrix,
            colorbar={"title": "Cobertura (%)"},
            hovertemplate=("p=%{x}<br>K=%{y}<br>Cobertura=%{z:.2f}%<extra></extra>"),
        )
    )

    figure.update_layout(
        title=("Sensibilidade ao K-nearest"),
        xaxis_title="p",
        yaxis_title="K",
    )

    return figure


def build_type_composition_chart(
    sensitivity: pl.DataFrame,
    selected_budget: float,
    selected_k: int,
):

    data = sensitivity.filter(
        (pl.col("experiment_budget") == selected_budget)
        & (pl.col("experiment_k") == selected_k)
    ).sort("p")

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=data["p"].to_list(),
            y=data["type_1_count"].to_list(),
            name="Tipo 1",
        )
    )

    figure.add_trace(
        go.Bar(
            x=data["p"].to_list(),
            y=data["type_2_count"].to_list(),
            name="Tipo 2",
        )
    )

    figure.update_layout(
        title=("Composição das novas unidades"),
        xaxis_title="p",
        yaxis_title=("Quantidade de unidades"),
        barmode="stack",
    )

    return figure


def build_neighborhood_chart(
    neighborhood_service: pl.DataFrame,
):

    data = neighborhood_service.sort("coverage_rate")

    y_column = (
        "neighborhood_name"
        if "neighborhood_name" in data.columns
        else "neighborhood_id"
    )

    figure = go.Figure(
        go.Bar(
            x=(data["coverage_rate"] * 100).to_list(),
            y=data[y_column].to_list(),
            orientation="h",
            customdata=[
                [
                    safe_float(row.get("demand")),
                    safe_float(row.get("served_demand")),
                    safe_float(row.get("unmet_demand")),
                ]
                for row in data.iter_rows(named=True)
            ],
            hovertemplate=(
                "%{y}"
                "<br>"
                "Cobertura: %{x:.2f}%"
                "<br>"
                "Demanda: %{customdata[0]:.1f}"
                "<br>"
                "Atendida: %{customdata[1]:.1f}"
                "<br>"
                "Não atendida: %{customdata[2]:.1f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=("Cobertura da demanda por bairro"),
        xaxis_title=("Demanda atendida (%)"),
        yaxis_title="Bairro",
        height=1100,
    )

    return figure


def build_recommended_p_table(
    sensitivity: pl.DataFrame,
    selected_k: int,
):

    data = (
        sensitivity.filter(pl.col("experiment_k") == selected_k)
        .sort(
            [
                "experiment_budget",
                "coverage",
                "average_travel_minutes",
            ],
            descending=[
                False,
                True,
                False,
            ],
        )
        .group_by(
            "experiment_budget",
            maintain_order=True,
        )
        .first()
        .sort("experiment_budget")
    )

    return data


def build_recommended_p_chart(
    sensitivity: pl.DataFrame,
    selected_k: int,
):

    data = build_recommended_p_table(
        sensitivity,
        selected_k,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=(data["experiment_budget"] / 1_000_000).to_list(),
            y=data["p"].to_list(),
            mode="lines+markers",
            name="p recomendado",
        )
    )

    figure.update_layout(
        title=("Número recomendado de novas creches por orçamento"),
        xaxis_title=("Orçamento (R$ milhões)"),
        yaxis_title=("p de máxima cobertura"),
    )

    return figure


# ============================================================
# Main dashboard
# ============================================================


def main():
    sensitivity = load_full_sensitivity_summary()

    candidates = load_candidates()

    existing = load_existing_locations()

    # Sidebar
    st.sidebar.header("Cenário")

    # Budget
    available_budgets = (
        sensitivity.get_column("experiment_budget").unique().sort().to_list()
    )

    default_budget = (
        15_000_000.0 if 15_000_000.0 in available_budgets else available_budgets[0]
    )

    selected_budget = st.sidebar.selectbox(
        "Orçamento disponível",
        options=available_budgets,
        index=(available_budgets.index(default_budget)),
        format_func=lambda value: f"R$ {value / 1_000_000:.1f} mi",
    )

    available_k = (
        sensitivity.filter(pl.col("experiment_budget") == selected_budget)
        .get_column("experiment_k")
        .unique()
        .sort()
        .to_list()
    )

    default_k = 100 if 100 in available_k else available_k[0]

    with st.sidebar.expander(
        "Parâmetros avançados",
        expanded=False,
    ):
        selected_k = st.selectbox(
            "K nearest candidates",
            options=available_k,
            index=(available_k.index(default_k)),
        )

        st.caption(
            "K controla a quantidade de "
            "candidatos de atendimento mantidos "
            "por bairro na formulação esparsa."
        )

    budget_k_scenarios = sensitivity.filter(
        (pl.col("experiment_budget") == selected_budget)
        & (pl.col("experiment_k") == selected_k)
    ).sort("p")

    if budget_k_scenarios.height == 0:
        st.error("Não existem cenários para a combinação de orçamento e K selecionada.")

        st.stop()

    available_p = budget_k_scenarios.get_column("p").unique().sort().to_list()

    p_max = max(available_p)

    best_scenario = budget_k_scenarios.sort(
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

    recommended_p = int(best_scenario["p"])

    selected_p = st.sidebar.select_slider(
        "Número de novas creches",
        options=available_p,
        value=recommended_p,
    )

    st.sidebar.caption(f"★ p recomendado: {recommended_p}")

    st.sidebar.caption(f"p máximo viável: {p_max}")
    st.sidebar.divider()

    st.sidebar.subheader("Mapa")

    map_mode = st.sidebar.radio(
        "Visualização territorial",
        options=[
            "Cobertura por bairro",
            "Facility dominante por bairro",
            "Setores por facility mais próxima",
        ],
    )

    scenario_df = sensitivity.filter(
        (pl.col("experiment_budget") == selected_budget)
        & (pl.col("experiment_k") == selected_k)
        & (pl.col("p") == selected_p)
    )

    if scenario_df.height != 1:
        st.error("A combinação selecionada não possui exatamente um resultado.")

        st.stop()

    scenario = scenario_df.row(
        0,
        named=True,
    )

    opened = load_opened_facilities(
        selected_budget,
        selected_k,
        selected_p,
    )

    new_facilities = prepare_new_facilities(
        opened,
        candidates,
    )

    neighborhood_service = load_neighborhood_service(
        selected_budget,
        selected_k,
        selected_p,
    )

    st.title("Otimização da Rede Municipal de Creches")

    st.caption("Exploração espacial e operacional dos cenários de expansão da rede.")

    st.caption(
        f"Cenário atual: "
        f"B = R$ "
        f"{selected_budget / 1_000_000:.1f} mi "
        f"| p = {selected_p} "
        f"| K = {selected_k}"
    )

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_4, metric_5, metric_6 = st.columns(3)

    metric_1.metric(
        "Cobertura",
        f"{scenario['coverage']:.2%}",
    )

    metric_2.metric(
        "Demanda atendida",
        (f"{scenario['served_demand']:.0f}"),
    )

    metric_3.metric(
        "Tempo médio",
        (f"{scenario['average_travel_minutes']:.2f} min"),
    )

    metric_4.metric(
        "Investimento",
        (f"R$ {scenario['investment'] / 1_000_000:.2f} mi"),
    )

    metric_5.metric(
        "Capacidade nova",
        (f"{scenario['new_capacity']:.0f}"),
    )

    metric_6.metric(
        "Uso do orçamento",
        (f"{scenario['budget_usage']:.1%}"),
    )

    st.divider()

    st.subheader("Distribuição espacial")

    if map_mode == "Setores por facility mais próxima":
        st.caption(
            "Esta camada representa regiões "
            "de proximidade derivadas da solução. "
            "Ela não representa diretamente os "
            "fluxos de atendimento do modelo."
        )

    map_ = build_map(
        existing=existing,
        new_facilities=(new_facilities),
        neighborhood_service=(neighborhood_service),
        map_mode=map_mode,
        budget=selected_budget,
        k=selected_k,
        p=selected_p,
    )

    st_folium(
        map_,
        width=None,
        height=650,
        returned_objects=[],
    )

    st.divider()

    st.subheader("Análise de cenários")

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.plotly_chart(
            build_coverage_by_budget_chart(
                sensitivity,
                selected_k,
                selected_budget,
                selected_p,
            ),
            use_container_width=True,
        )

    with chart_col_2:
        st.plotly_chart(
            build_travel_by_budget_chart(
                sensitivity,
                selected_k,
            ),
            use_container_width=True,
        )

    heat_col_1, heat_col_2 = st.columns(2)

    with heat_col_1:
        st.plotly_chart(
            build_budget_p_heatmap(
                sensitivity,
                selected_k,
            ),
            use_container_width=True,
        )

        st.caption(
            "Células vazias representam valores de p inviáveis para o orçamento correspondente."
        )

    with heat_col_2:
        st.plotly_chart(
            build_k_p_heatmap(
                sensitivity,
                selected_budget,
            ),
            use_container_width=True,
        )

    st.subheader("Escalonamento do número de unidades")

    st.plotly_chart(
        build_recommended_p_chart(
            sensitivity,
            selected_k,
        ),
        use_container_width=True,
    )

    st.caption(
        "p* representa o número de novas "
        "creches que maximiza a cobertura "
        "para cada orçamento, considerando "
        f"K={selected_k}."
    )

    st.subheader("Composição da expansão")

    st.plotly_chart(
        build_type_composition_chart(
            sensitivity,
            selected_budget,
            selected_k,
        ),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Atendimento territorial")

    st.caption(
        "A cobertura de cada bairro é calculada "
        "como demanda atendida dividida pela "
        "demanda-alvo daquele bairro."
    )

    st.plotly_chart(
        build_neighborhood_chart(neighborhood_service),
        use_container_width=True,
    )

    st.subheader("Detalhes do cenário")

    detail_1, detail_2, detail_3, detail_4 = st.columns(4)
    detail_1.metric("Tipo 1", scenario["type_1_count"])
    detail_2.metric("Tipo 2", scenario["type_2_count"])
    detail_3.metric("Demanda não atendida", (f"{scenario['unmet_demand']:.0f}"))
    detail_4.metric("p", selected_p)

    decision_1, decision_2, decision_3 = st.columns(3)
    decision_1.metric("p selecionado", selected_p)
    decision_2.metric(
        "p recomendado", recommended_p, delta=(selected_p - recommended_p)
    )
    decision_3.metric("p máximo viável", p_max)

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader("Interpretação do cenário")

    if selected_p == recommended_p:
        st.success(
            f"Para orçamento de "
            f"R$ {selected_budget / 1_000_000:.1f} mi "
            f"e K={selected_k}, "
            f"p={selected_p} apresenta a maior "
            f"cobertura entre os valores de p "
            f"avaliados."
        )
    elif selected_p < recommended_p:
        coverage_gap = best_scenario["coverage"] - scenario["coverage"]
        st.info(
            f"O cenário selecionado utiliza "
            f"p={selected_p}, enquanto a maior "
            f"cobertura ocorre em "
            f"p={recommended_p}. "
            f"A diferença de cobertura é de "
            f"{coverage_gap * 100:.2f} "
            f"ponto(s) percentual(is)."
        )
    else:
        coverage_gap = best_scenario["coverage"] - scenario["coverage"]
        st.warning(
            f"O cenário utiliza p={selected_p}, "
            f"acima do p recomendado "
            f"({recommended_p}). "
            f"A cobertura é "
            f"{coverage_gap * 100:.2f} "
            f"ponto(s) percentual(is) inferior "
            f"ao máximo encontrado para este "
            f"orçamento."
        )


if __name__ == "__main__":
    main()
