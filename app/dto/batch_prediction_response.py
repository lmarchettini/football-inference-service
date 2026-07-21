from pydantic import BaseModel


class BatchPredictionResult(BaseModel):

    fixture_id: int

    market: str

    probability: float

    negative_probability: float


class BatchPredictionFailure(BaseModel):

    fixture_id: int

    market: str

    code: str

    message: str


class BatchPredictionResponse(BaseModel):

    predictions: list[
        BatchPredictionResult
    ]

    failures: list[
        BatchPredictionFailure
    ]