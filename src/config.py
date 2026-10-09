from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

IBGE_DIR = RAW_DIR / "ibge"
SPATIAL_DIR = IBGE_DIR / "spatial"
AGG_DIR = IBGE_DIR / "agg"

IBGE_NOVA_IGUACU_CODE = "3303500"

