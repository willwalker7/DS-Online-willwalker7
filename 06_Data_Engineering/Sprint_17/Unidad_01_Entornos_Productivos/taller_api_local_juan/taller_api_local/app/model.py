#metodo load_model, train_model, predict, load_metadata

import json
import io
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Advertising.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_dataframe(csv_bytes: bytes | None = None) -> pd.DataFrame:
    """Carga el dataset desde bytes subidos por el usuario o desde el CSV por defecto."""
    if csv_bytes is not None:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    else:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset en {DATA_PATH}. Sube un CSV o colócalo en esa ruta."
            )
        df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Al dataset le faltan columnas requeridas: {missing}")

    return df[FEATURES + [TARGET]].dropna()


def train_model(csv_bytes: bytes | None = None, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Entrena (o reentrena) el modelo y lo guarda en disco junto a sus métricas."""
    df = _load_dataframe(csv_bytes)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(mean_squared_error(y_test, y_pred) ** 0.5, 4),
    }

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "coefficients": dict(zip(FEATURES, [round(c, 4) for c in model.coef_])),
        "intercept": round(model.intercept_, 4),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    return metadata


def load_model() -> LinearRegression:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No hay un modelo entrenado todavía. Llama a /train primero.")
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict | None:
    if not METADATA_PATH.exists():
        return None
    return json.loads(METADATA_PATH.read_text())


def predict(tv: float, radio: float, newspaper: float) -> tuple[float, str]:
    model = load_model()
    metadata = load_metadata()
    X = pd.DataFrame([[tv, radio, newspaper]], columns=FEATURES)
    pred = float(model.predict(X)[0])
    version = metadata["model_version"] if metadata else "unknown"
    return pred, version
