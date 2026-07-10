from fastapi import FastAPI

from app.dto.prediction_request import (
    PredictionRequest
)

from app.service.prediction_service import (
    predict
)

app = FastAPI()


@app.post("/predict")
def predict_endpoint(
        request: PredictionRequest
):
    return predict(request)