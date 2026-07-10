from pydantic import BaseModel


class PredictionResponse(BaseModel):

    market: str

    probability: float

    negative_probability: float