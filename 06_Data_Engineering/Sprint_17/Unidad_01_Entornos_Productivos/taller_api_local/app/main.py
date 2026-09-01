from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from app import modell as ml
from app.schemas import TrainResponse, PredictResponse, MetricsResponse


app = FastAPI(
    title  = "Advertising Sales Predicto API",
    descrption = (
        "API simple de Machine Learning (Regresion Lineal) entrenada con el dataset"
        "Advertising.csv (TC, Radio, Newspaper -> Sales). Soporta reentrenamiento y predicción"
    ),
    version = "0.0.1"
)

@app.on_event("startup")
def _ensure_model_exists():
    "Si no existe el modelo guardado, entrena uno con el dataser por defecto"
    try:
        ml.load_model()
    except FileNotFoundError:
        # No hay dataset por defecto, el usuario deberá llamar /train con un csv
        pass
    
@app.get("/", tags=["General"])
def root():
    return {
        "mensaje": "Advertising Sales Pedictor API",
        "docs": "/docs",
        "endpoints": ["/health", "/train", "/predict", "/metrics"]
    }
    
@app.get("/health", tags=["General"])
def health():
    return {"status": "ok"}

app.post("/train", response_model=TrainResponse, tags=["Modelo"])
async def train(file: UploadFile | None = File(default=None)):
    """
    Reentrena el modelo.
    
    - Sin archivo: Usa el dataset por defecto
    - Con archivo: Sube el CSV con las columnas necesarias y reentrena
    
    """
    
    csv_bytes = await file.read() if file is not None else None
    
    try:
        metadata = ml.train_model(csv_bytes=csv_bytes)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return TrainResponse(
        mensaje = "Modelo entrenado y guardado",
        model_version = metadata["model_version"],
        n_samples = metadata["n_samples"],
        metrics = metadata["metrics"]
    )

@app.post("/predict", response_model=PredictResponse, tags=["Modelo"])
def predict(payload: PredictResponse):
    try:
        pred, version = ml.predict(payload.TV, payload.Radio, payload.Newspaper)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(predict_sales=round(pred, 3), model_version=version)

@app.post("/metrics", response_model=MetricsResponse, tags=["Modelo"])
def metrics():
    metadata = ml.load_metadata()
    if metadata is None:
        return JSONResponse(
            status_code=404,
            content={"detalles": "No hay modelo entrenado todavía. LLama a /train"}
        )