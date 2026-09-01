# crear TrainResponse, MetricsResponce, PredictResponce

from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    TV: float = Field(..., ge=0, description="Inversión en publicidad TV (miles de $)")
    Radio: float = Field(..., ge=0, description="Inversión en publicidad Radio (miles de $)")
    Newspaper: float = Field(..., ge=0, description="Inversión en publicidad Newspaper (miles de $)")

    class Config:
        json_schema_extra = {
            "example": {"TV": 230.1, "Radio": 37.8, "Newspaper": 69.2}
        }


class PredictResponse(BaseModel):
    predicted_sales: float
    model_version: str


class TrainResponse(BaseModel):
    message: str
    model_version: str
    n_samples: int
    metrics: dict


class MetricsResponse(BaseModel):
    model_version: str | None
    trained_at: str | None
    n_samples: int | None
    metrics: dict | None
