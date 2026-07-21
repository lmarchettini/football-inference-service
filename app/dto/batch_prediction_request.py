from pydantic import (
    BaseModel,
    Field,
)

from app.dto.prediction_request import (
    PredictionRequest,
)


class BatchPredictionItem(
    PredictionRequest
):

    fixture_id: int = Field(
        gt=0
    )


class BatchPredictionRequest(BaseModel):

    predictions: list[
        BatchPredictionItem
    ] = Field(
        min_length=1,
        max_length=5000,
    )