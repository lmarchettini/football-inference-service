from pydantic import BaseModel


class PredictionRequest(BaseModel):

    model_path: str

    market: str

    features: list[float]