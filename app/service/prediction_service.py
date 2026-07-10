import joblib

from app.dto.prediction_response import (
    PredictionResponse
)


def predict(request):

    model = joblib.load(
        request.model_path
    )

    probabilities = model.predict_proba(
        [request.features]
    )[0]

    return PredictionResponse(
        market=request.market,

        probability=round(
            float(probabilities[1]),
            4
        ),

        negative_probability=round(
            float(probabilities[0]),
            4
        )
    )