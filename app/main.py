import logging

from fastapi import (
    FastAPI,
    HTTPException,
)

from app.dto.prediction_request import (
    PredictionRequest,
)

from app.dto.prediction_response import (
    PredictionResponse,
)

from app.service.prediction_service import (
    ModelCompatibilityError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionExecutionError,
    PredictionValidationError,
    predict,
    MarketDisabledError
)

from app.dto.batch_prediction_request import (
    BatchPredictionRequest,
)

from app.dto.batch_prediction_response import (
    BatchPredictionResponse,
)

from app.service.batch_prediction_service import (
    predict_batch,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Football Inference Service"
)


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_endpoint(
    request: PredictionRequest,
) -> PredictionResponse:

    try:
        return predict(
            request
        )
        

    except PredictionValidationError as exc:
        logger.warning(
            "Invalid prediction request: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
        
    except MarketDisabledError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ModelNotFoundError as exc:
        logger.error(
            "Prediction model not found: %s",
            exc,
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ModelCompatibilityError as exc:
        logger.error(
            "Model compatibility error: %s",
            exc,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except ModelLoadError as exc:
        logger.exception(
            "Model loading failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except PredictionExecutionError as exc:
        logger.exception(
            "Prediction execution failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unhandled error in /predict"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected internal "
                "prediction error"
            ),
        ) from exc
        
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch_endpoint(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:

    try:
        return predict_batch(
            request
        )

    except Exception as exc:
        logger.exception(
            "Unhandled error in /predict/batch"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected internal batch "
                "prediction error"
            ),
        ) from exc